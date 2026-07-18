"""
streamlit_app.py
=================

Streamlit version of the two-way conversation aid:

    Deaf person   -> upload a sign clip -> Sign Recognition Model -> text
    Hearing person -> speak into mic (or upload audio) -> Whisper -> text

Both directions write into one shared transcript.


DESIGN NOTES (why this differs slightly from the Gradio version):

  - Streamlit's built-in camera widget (st.camera_input) only captures a
    single still PHOTO, not a video clip -- it cannot support our
    sequence-based sign model, which needs a run of frames over time.
    So the sign side uses file upload (record a clip on your phone/
    webcam software separately, then upload it here) rather than a
    live in-browser recorder. This avoids an entire category of
    browser-recording fragility for a comparatively rare capability.

  - st.audio_input IS a genuine, built-in live microphone recorder
    (not just a photo-style limitation), so speech recognition supports
    live recording directly, same as before.

  - Whisper defaults to "tiny" here (not "base") because Streamlit
    Community Cloud enforces a hard 1 GiB RAM limit per app -- "tiny"
    has a much smaller memory footprint, trading a little transcription
    accuracy for headroom against that ceiling.

  - WebRTC ICE servers come from turn.py's get_ice_servers(), which uses
    Twilio's TURN service. This is required, not optional -- Streamlit
    Community Cloud's infrastructure does not reliably allow WebRTC
    connections using STUN alone (confirmed directly in streamlit-webrtc's
    own official sample code/docs).
"""

import json
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import av
import numpy as np
import streamlit as st
import torch

from slr.landmarks import HolisticLandmarkExtractor, iter_video_landmarks, FEATURE_DIM
from slr.model import SignLanguageArcFaceTCN
from stt.speech_to_text import SpeechToText
from turn import get_ice_servers

try:
    from streamlit_webrtc import webrtc_streamer, RTCConfiguration
except ImportError as e:
    raise ImportError(
        "streamlit-webrtc is required for live webcam sign recognition: "
        "pip install streamlit-webrtc av"
    ) from e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent / "models"
CHECKPOINT_PATH = MODELS_DIR / "best_model.pt"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
PROTOTYPES_PATH = MODELS_DIR / "prototypes.npy"

WHISPER_MODEL_SIZE = "tiny"  # see DESIGN NOTES above re: 1 GiB RAM limit
SIMILARITY_THRESHOLD = 0.3

DEVICE = torch.device("cpu")  # small model -- CPU is fine, avoids any GPU/quota concerns

# Live sign recognition tuning
WINDOW_SECONDS = 2.0        # how many seconds of recent frames to keep in the buffer
INFER_EVERY_N_FRAMES = 5    # run inference every N incoming frames (lower = more responsive, more CPU)
MOTION_THRESHOLD = 0.003    # skip inference when the window is nearly static (idle hands)
ASSUMED_FPS = 15            # used to size the frame buffer; browsers vary, this is a rough estimate


# ---------------------------------------------------------------------------
# Model loading -- cached so this only happens once per running instance,
# not on every Streamlit script rerun (Streamlit reruns the whole script
# top-to-bottom on every interaction by default).
# ---------------------------------------------------------------------------
@st.cache_resource
def load_sign_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if not any(k.startswith("arc_head.") for k in ckpt["model_state_dict"]):
        raise RuntimeError(
            "models/best_model.pt does not look like an ArcFace checkpoint "
            "(no arc_head.* keys found). This app expects a model trained "
            "with --use_arcface."
        )
    model = SignLanguageArcFaceTCN(
        input_dim=ckpt["input_dim"],
        num_classes=ckpt["num_classes"],
        d_model=ckpt["d_model"],
        tcn_channels=tuple(ckpt["tcn_channels"]),
        kernel_size=ckpt["kernel_size"],
        embedding_dim=ckpt.get("embedding_dim", 256),
        arc_scale=ckpt.get("arc_scale", 30.0),
        arc_margin=ckpt.get("arc_margin", 0.3),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    prototypes = torch.from_numpy(np.load(PROTOTYPES_PATH)).to(DEVICE)

    return model, ckpt["max_len"], label_map, idx_to_label, prototypes


@st.cache_resource
def load_speech_model():
    return SpeechToText(model_size=WHISPER_MODEL_SIZE, device="cpu")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _prepare_sequence(seq: np.ndarray, max_len: int):
    T = seq.shape[0]
    if T > max_len:
        idxs = np.linspace(0, T - 1, max_len).round().astype(int)
        seq = seq[idxs]
        length = max_len
    else:
        pad = np.zeros((max_len - T, seq.shape[1]), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
        length = T
    seq_t = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(DEVICE)
    len_t = torch.tensor([length], dtype=torch.long).to(DEVICE)
    return seq_t, len_t


def recognize_sign(video_bytes: bytes, suffix: str) -> tuple[str, list]:
    """Returns (predicted_word_or_message, top5_list_of_(word, score))."""
    model, max_len, label_map, idx_to_label, prototypes = load_sign_model()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        frames = list(iter_video_landmarks(tmp_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not frames:
        return "(no landmarks detected -- try again with hands/face visible)", []

    seq = np.stack(frames, axis=0)
    seq_t, len_t = _prepare_sequence(seq, max_len)

    with torch.no_grad():
        emb = model.embed(seq_t, len_t)
        sims = (emb @ prototypes.t()).squeeze(0)
        best_sim, best_idx = sims.max(dim=0)
        top5_vals, top5_idxs = sims.topk(min(5, sims.shape[0]))

    top5 = [(idx_to_label.get(i.item(), "?"), round(v.item(), 4))
            for v, i in zip(top5_vals, top5_idxs)]

    if best_sim.item() < SIMILARITY_THRESHOLD:
        return "(sign not recognized confidently -- try again)", top5
    return idx_to_label.get(best_idx.item(), "?"), top5


def recognize_speech(audio_bytes: bytes, suffix: str) -> str:
    stt = load_speech_model()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = stt.transcribe_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result["text"] or "(no speech detected)"


# ---------------------------------------------------------------------------
# Live webcam sign recognition (streamlit-webrtc)
# ---------------------------------------------------------------------------
class LiveSignState:
    """
    Thread-safe shared state between the WebRTC video callback (runs in its
    own forked thread) and the main Streamlit script thread. streamlit-webrtc
    callbacks cannot call st.* methods directly, so the callback only writes
    into this container under a lock, and the main script thread reads from
    it in a polling loop to update the UI.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.buffer = deque(maxlen=int(WINDOW_SECONDS * ASSUMED_FPS))
        self.frame_count = 0
        self.last_prediction_text = ""
        self.last_prediction_time = 0.0
        self.last_appended_word = None  # avoids spamming the transcript with repeats

    def motion_score(self) -> float:
        if len(self.buffer) < 2:
            return 0.0
        arr = np.stack(list(self.buffer), axis=0)
        return float(np.abs(np.diff(arr, axis=0)).mean())


@st.cache_resource
def get_live_state():
    return LiveSignState()


@st.cache_resource
def get_landmark_extractor():
    return HolisticLandmarkExtractor()


def make_video_frame_callback(state: LiveSignState):
    model, max_len, label_map, idx_to_label, prototypes = load_sign_model()
    extractor = get_landmark_extractor()

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")

        vec, annotated = extractor.process(img, draw=True)

        with state.lock:
            state.buffer.append(vec)
            state.frame_count += 1
            do_infer = (
                state.frame_count % INFER_EVERY_N_FRAMES == 0
                and len(state.buffer) >= max(5, state.buffer.maxlen // 3)
            )
            motion = state.motion_score() if do_infer else 0.0

        if do_infer and motion >= MOTION_THRESHOLD:
            with state.lock:
                seq = np.stack(list(state.buffer), axis=0)
            seq_t, len_t = _prepare_sequence(seq, max_len)
            with torch.no_grad():
                emb = model.embed(seq_t, len_t)
                sims = (emb @ prototypes.t()).squeeze(0)
                best_sim, best_idx = sims.max(dim=0)
            if best_sim.item() >= SIMILARITY_THRESHOLD:
                word = idx_to_label.get(best_idx.item(), "?")
                with state.lock:
                    state.last_prediction_text = f"{word} ({best_sim.item():.2f})"
                    state.last_prediction_time = time.time()

        # Overlay the current prediction directly on the video feed, same
        # as the local live_inference.py script -- gives immediate visual
        # feedback without needing to look elsewhere on the page.
        with state.lock:
            display_text = state.last_prediction_text if (
                time.time() - state.last_prediction_time
            ) < 2.5 else ""
        if display_text:
            import cv2
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(annotated, display_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    return video_frame_callback


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sign <-> Speech Conversation Aid", layout="wide")
st.title("Eswatini Sign Language \u2194 Speech Conversation Aid")
st.write(
    "A Deaf person uploads a signed clip and it appears as text below. "
    "A hearing person speaks into the microphone (or uploads audio) and it "
    "appears as text too. Both people read the same shared transcript."
)

if "history" not in st.session_state:
    st.session_state.history = []  # list of (speaker, avatar, text) tuples

col1, col2 = st.columns(2)

with col1:
    st.subheader("\U0001F9CF Deaf person: sign here")
    live_tab, upload_tab = st.tabs(["\U0001F534 Live webcam", "\U0001F4C1 Upload clip"])

    with live_tab:
        st.caption(
            "Sign in view of your camera. The recognized word appears overlaid on "
            "the video and gets added to the transcript below automatically."
        )
        live_state = get_live_state()
        webrtc_ctx = webrtc_streamer(
            key="live-sign-recognition",
            video_frame_callback=make_video_frame_callback(live_state),
            rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Poll the shared state (written by the video callback thread) and
        # sync any new confident prediction into the transcript. This loop
        # only runs while the stream is actually playing.
        live_status_placeholder = st.empty()
        if webrtc_ctx.state.playing:
            with live_state.lock:
                current_text = live_state.last_prediction_text
                current_word = current_text.split(" (")[0] if current_text else None
                is_recent = (time.time() - live_state.last_prediction_time) < 2.5

            if current_word and is_recent and current_word != live_state.last_appended_word:
                st.session_state.history.append(
                    ("Deaf person (signed)", "\U0001F9CF", current_word)
                )
                live_state.last_appended_word = current_word
                live_status_placeholder.success(f"Recognized: {current_text}")
                st.rerun()
            else:
                live_status_placeholder.info("Watching for signs...")

    with upload_tab:
        st.caption("Prefer to record separately and upload the clip instead.")
        sign_file = st.file_uploader(
            "Upload a short clip of one sign", type=["mp4", "mov", "webm", "avi"]
        )
        if st.button("Recognize sign", type="primary") and sign_file is not None:
            with st.spinner("Extracting landmarks and matching..."):
                suffix = Path(sign_file.name).suffix or ".mp4"
                word, top5 = recognize_sign(sign_file.getvalue(), suffix)
            st.session_state.history.append(("Deaf person (signed)", "\U0001F9CF", word))
            st.success(f"Recognized: {word}")
            if top5:
                st.caption("Top 5 matches: " + ", ".join(f"{w} ({s:.2f})" for w, s in top5))

with col2:
    st.subheader("\U0001F5E3\uFE0F Hearing person: speak here")
    audio_value = st.audio_input("Record speech")
    audio_upload = st.file_uploader(
        "...or upload an audio file instead", type=["wav", "mp3", "m4a", "ogg"]
    )
    if st.button("Transcribe speech", type="primary"):
        source = audio_value if audio_value is not None else audio_upload
        if source is not None:
            with st.spinner("Transcribing..."):
                suffix = Path(getattr(source, "name", "audio.wav")).suffix or ".wav"
                text = recognize_speech(source.getvalue(), suffix)
            st.session_state.history.append(("Hearing person (spoke)", "\U0001F5E3\uFE0F", text))
            st.success(f"Transcribed: {text}")

st.divider()
st.subheader("Shared conversation transcript")
if not st.session_state.history:
    st.caption("Nothing yet -- try recognizing a sign or transcribing some speech above.")
for speaker, avatar, text in st.session_state.history:
    with st.chat_message(speaker, avatar=avatar):
        st.write(text)

if st.button("Clear conversation"):
    st.session_state.history = []
    st.rerun()
