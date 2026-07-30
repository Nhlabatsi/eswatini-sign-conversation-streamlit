"""
streamlit_app.py
=================

Streamlit version of the two-way conversation aid:

    Camera / Microphone
           |
    +------+------+
    |             |
Sign Recognition  Speech Recognition
    (SLR)              (STT)
    |             |
    +------+------+
           |
    Conversation Manager      <-- Stage 1 of the improved architecture
           |
    +------+------+------+
    |      |             |
Phrase   Context &    Intent &
Builder   Memory       Topics
    |      |             |
    +------+------+------+
           |
    Shared Conversation Transcript
           |
    Text Display (Text-to-Speech: not yet implemented -- later stage)

DESIGN NOTES (why this differs slightly from the Gradio version):

  - Streamlit's built-in camera widget (st.camera_input) only captures a
    single still PHOTO, not a video clip -- it cannot support our
    sequence-based sign model, which needs a run of frames over time.
    So the sign side uses file upload (record a clip on your phone/
    webcam software separately, then upload it here) as well as live
    webcam via streamlit-webrtc.

  - st.audio_input IS a genuine, built-in live microphone recorder,
    so speech recognition supports live recording directly.

  - Whisper defaults to "tiny" here (not "base") because Streamlit
    Community Cloud enforces a hard 1 GiB RAM limit per app.

  - WebRTC ICE servers come from turn.py's get_ice_servers(), which uses
    Twilio's TURN service -- required, not optional, on Streamlit
    Community Cloud (confirmed in streamlit-webrtc's own docs).

  - CONVERSATION MANAGER (see conversation/ package): individually
    recognized sign words are no longer dumped straight into the
    transcript one at a time. They're buffered into phrases (a pause
    or a turn-change finalizes the phrase), tagged with a lightweight
    rule-based intent/topic, and tracked in a short-term context
    history -- see conversation/manager.py's docstring for exactly
    what's implemented at this stage vs. deferred to later stages
    (Grammar & Sentence Builder, Text-to-Speech).
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
from conversation import ConversationManager, SIGN_SPEAKER, SPEECH_SPEAKER

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
#
# The old fixed WINDOW_SECONDS + INFER_EVERY_N_FRAMES design ran inference
# on an arbitrary rolling window every N frames, regardless of whether a
# sign was actually in progress. That's a real mismatch with training:
# best_model.pt was trained on complete, trimmed single-sign clips, so
# feeding it "tail of previous sign + idle + start of next sign" (which a
# fixed window does constantly) gives it inputs it never saw, and explains
# a lot of the live-vs-upload accuracy gap. Replaced with a small
# state machine that detects the actual start/end of a sign via motion,
# and classifies once per completed sign -- exactly how infer_webcam.py
# (the local/offline version of this same idea) already works.
MOTION_START_THRESHOLD = 0.006   # motion needed to declare "a sign has begun"
MOTION_STOP_THRESHOLD = 0.003    # motion below this counts toward "sign has ended"
STOP_GRACE_SECONDS = 0.4         # how long motion must stay below the stop threshold
                                  # before the sign is considered finished -- absorbs
                                  # brief natural pauses mid-sign without cutting it off
PRE_ROLL_SECONDS = 0.15          # frames kept from just BEFORE motion crossed the start
                                  # threshold, so the very beginning of a sign (which is
                                  # often slow/low-motion) isn't clipped off
MIN_SIGN_SECONDS = 0.25          # shorter than this is almost certainly noise/jitter,
                                  # not a real sign -- discarded rather than classified
MAX_SIGN_SECONDS = 4.0           # safety cap in case motion never drops (e.g. continuous
                                  # movement) -- forces a classification rather than
                                  # buffering forever
ASSUMED_FPS = 15                 # only used as a fallback before enough real frame
                                  # timestamps have arrived to estimate actual FPS --
                                  # see LiveSignState._update_fps_estimate below. Browser
                                  # webcam frame rates vary a lot, so measuring beats
                                  # assuming once the stream is actually running.

# Conversation Manager tuning
PHRASE_PAUSE_SECONDS = 3.0  # gap since the last sign word before auto-finalizing a phrase
PHRASE_MAX_WORDS = 8        # safety cap so a phrase can't grow forever

_AVATAR_BY_SPEAKER = {SIGN_SPEAKER: "\U0001F9CF", SPEECH_SPEAKER: "\U0001F5E3\uFE0F"}


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


def recognize_sign(video_bytes: bytes, suffix: str) -> tuple[str, list, float]:
    """Returns (predicted_word_or_message, top5_list_of_(word, score), confidence)."""
    model, max_len, label_map, idx_to_label, prototypes = load_sign_model()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        frames = list(iter_video_landmarks(tmp_path))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not frames:
        return "(no landmarks detected -- try again with hands/face visible)", [], 0.0

    seq = np.stack(frames, axis=0)
    seq_t, len_t = _prepare_sequence(seq, max_len)

    with torch.no_grad():
        emb = model.embed(seq_t, len_t)
        sims = (emb @ prototypes.t()).squeeze(0)
        best_sim, best_idx = sims.max(dim=0)
        top5_vals, top5_idxs = sims.topk(min(5, sims.shape[0]))

    top5 = [(idx_to_label.get(i.item(), "?"), round(v.item(), 4))
            for v, i in zip(top5_vals, top5_idxs)]

    threshold = st.session_state.get("similarity_threshold", SIMILARITY_THRESHOLD)
    if best_sim.item() < threshold:
        return "(sign not recognized confidently -- try again)", top5, best_sim.item()
    return idx_to_label.get(best_idx.item(), "?"), top5, best_sim.item()


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
    it in a polling loop to feed the Conversation Manager (which lives in
    st.session_state and is only ever touched from the main script thread).

    Segmentation state machine (replaces the old fixed-window approach):

        IDLE --[motion >= start threshold]--> SIGNING
        SIGNING --[motion < stop threshold for STOP_GRACE_SECONDS]--> IDLE (classify)
        SIGNING --[buffer exceeds MAX_SIGN_SECONDS]--> IDLE (classify, forced)

    A short pre-roll of recent frames is kept continuously during IDLE so
    that when a sign is detected, the buffer used for classification
    includes the slow/low-motion frames right at the start of the sign,
    not just the frames from the moment motion crossed the threshold.
    """

    def __init__(self):
        self.lock = threading.Lock()

        self.state = "IDLE"
        self.pre_roll = deque(maxlen=1)   # resized once real FPS is known -- see _update_fps_estimate
        self.active_buffer: list = []
        self.prev_vec = None
        self.low_motion_since: float | None = None
        self.sign_start_time: float | None = None

        # Running estimate of actual incoming frame rate, since browser
        # webcam FPS varies a lot and ASSUMED_FPS is only a seed value.
        self._last_frame_time: float | None = None
        self.estimated_fps = float(ASSUMED_FPS)

        self.frame_count = 0
        self.last_prediction_word = None
        self.last_prediction_confidence = 0.0
        self.last_prediction_time = 0.0
        self.last_consumed_word = None  # avoids feeding the same held sign repeatedly

        # Small counter so the UI can say "saw motion but couldn't recognize it"
        # rather than looking identical to "nothing happened" -- not required,
        # but cheap and useful feedback while a signer is calibrating distance/lighting.
        self.last_rejected_time: float = 0.0

    def _update_fps_estimate(self, now: float):
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if 0 < dt < 1.0:  # ignore stalls/pauses so they don't skew the estimate
                instantaneous_fps = 1.0 / dt
                # Exponential moving average -- smooths jitter without lagging
                # too far behind a genuine sustained change in frame rate.
                self.estimated_fps = 0.9 * self.estimated_fps + 0.1 * instantaneous_fps
        self._last_frame_time = now

        target_pre_roll_len = max(1, int(PRE_ROLL_SECONDS * self.estimated_fps))
        if self.pre_roll.maxlen != target_pre_roll_len:
            self.pre_roll = deque(self.pre_roll, maxlen=target_pre_roll_len)

    def process_frame(self, vec: np.ndarray, now: float):
        """
        Feeds one frame's landmark vector through the segmentation state
        machine. Returns a completed clip (np.ndarray, shape (T, FEATURE_DIM))
        if a sign just finished and should be classified, else None.
        Must be called under self.lock.
        """
        self._update_fps_estimate(now)
        self.frame_count += 1

        motion = 0.0 if self.prev_vec is None else float(np.abs(vec - self.prev_vec).mean())
        self.prev_vec = vec

        if self.state == "IDLE":
            self.pre_roll.append(vec)
            if motion >= MOTION_START_THRESHOLD:
                self.state = "SIGNING"
                self.active_buffer = list(self.pre_roll) + [vec]
                self.sign_start_time = now
                self.low_motion_since = None
            return None

        # state == "SIGNING"
        self.active_buffer.append(vec)
        max_frames = int(MAX_SIGN_SECONDS * self.estimated_fps)
        elapsed = now - (self.sign_start_time or now)

        force_end = len(self.active_buffer) >= max_frames or elapsed >= MAX_SIGN_SECONDS
        natural_end = False
        if motion < MOTION_STOP_THRESHOLD:
            if self.low_motion_since is None:
                self.low_motion_since = now
            elif (now - self.low_motion_since) >= STOP_GRACE_SECONDS:
                natural_end = True
        else:
            self.low_motion_since = None  # motion resumed -- sign is still going

        if force_end or natural_end:
            clip = np.stack(self.active_buffer, axis=0)
            self.state = "IDLE"
            self.active_buffer = []
            self.low_motion_since = None
            self.sign_start_time = None
            self.pre_roll.clear()

            min_frames = max(1, int(MIN_SIGN_SECONDS * self.estimated_fps))
            if clip.shape[0] < min_frames:
                return None  # almost certainly noise, not a real sign -- discard silently
            return clip

        return None


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
        now = time.time()

        with state.lock:
            completed_clip = state.process_frame(vec, now)
            current_fsm_state = state.state

        if completed_clip is not None:
            # A full sign (start-to-end, via the motion state machine) just
            # finished -- classify it exactly the way training data looked:
            # one complete, variable-length clip, not a fixed-size window.
            threshold = st.session_state.get("similarity_threshold", SIMILARITY_THRESHOLD)
            seq_t, len_t = _prepare_sequence(completed_clip, max_len)
            with torch.no_grad():
                emb = model.embed(seq_t, len_t)
                sims = (emb @ prototypes.t()).squeeze(0)
                best_sim, best_idx = sims.max(dim=0)
            if best_sim.item() >= threshold:
                word = idx_to_label.get(best_idx.item(), "?")
                with state.lock:
                    state.last_prediction_word = word
                    state.last_prediction_confidence = best_sim.item()
                    state.last_prediction_time = now
            else:
                # Motion looked like a sign, but nothing matched confidently --
                # surfaced in the UI as distinct from "no motion detected yet"
                # (see _poll_live_sign), rather than silently doing nothing.
                with state.lock:
                    state.last_rejected_time = now

        # Overlay the current prediction and FSM state directly on the video
        # feed -- always reflects live recognition, independent of whether
        # it's been fed into the Conversation Manager's phrase buffer yet.
        with state.lock:
            show_word = state.last_prediction_word if (
                time.time() - state.last_prediction_time
            ) < 2.5 else None
            show_conf = state.last_prediction_confidence
        display_text = f"{show_word} ({show_conf:.2f})" if show_word else f"[{current_fsm_state}]"
        text_color = (0, 255, 0) if show_word else (0, 200, 200)
        import cv2
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(annotated, display_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    return video_frame_callback


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Sign <-> Speech Conversation Aid", layout="wide")
st.title("Eswatini Sign Language \u2194 Speech Conversation Aid")
st.write(
    "A Deaf person signs (live on camera or an uploaded clip) and it appears as "
    "text below. A hearing person speaks into the microphone (or uploads audio) "
    "and it appears as text too. Both people read the same shared transcript."
)

if "conversation_manager" not in st.session_state:
    st.session_state.conversation_manager = ConversationManager(
        phrase_pause_seconds=PHRASE_PAUSE_SECONDS, phrase_max_words=PHRASE_MAX_WORDS
    )
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = SIMILARITY_THRESHOLD

with st.sidebar:
    st.subheader("Recognition tuning")
    st.session_state.similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=0.0, max_value=1.0,
        value=st.session_state.similarity_threshold, step=0.05,
        help="Minimum cosine similarity to accept a prediction. There's no "
             "single correct value baked into the code -- the right number "
             "depends on your model's actual embedding distribution. Start "
             "around what evaluate_holdout.py's score spread suggests, then "
             "raise it here if you're seeing confident-looking wrong guesses, "
             "or lower it if correct signs are getting rejected as "
             "\"not recognized\".",
    )
    st.caption(
        f"Live segmentation: motion \u2265 {MOTION_START_THRESHOLD} starts a sign, "
        f"motion < {MOTION_STOP_THRESHOLD} for {STOP_GRACE_SECONDS}s ends it "
        f"(min {MIN_SIGN_SECONDS}s, max {MAX_SIGN_SECONDS}s per sign)."
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("\U0001F9CF Deaf person: sign here")
    live_tab, upload_tab = st.tabs(["\U0001F534 Live webcam", "\U0001F4C1 Upload clip"])

    with live_tab:
        st.caption(
            "Sign in view of your camera. Words are buffered into a phrase as you "
            "sign; pause for a few seconds (or switch to speech) to commit the "
            "phrase to the shared transcript below."
        )
        live_state = get_live_state()
        webrtc_ctx = webrtc_streamer(
            key="live-sign-recognition",
            video_frame_callback=make_video_frame_callback(live_state),
            rtc_configuration=RTCConfiguration({"iceServers": get_ice_servers()}),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        live_status_placeholder = st.empty()

        @st.fragment(run_every=1.0)
        def _poll_live_sign():
            manager = st.session_state.conversation_manager

            if not webrtc_ctx.state.playing:
                live_status_placeholder.info("Camera not connected yet.")
                return

            with live_state.lock:
                word = live_state.last_prediction_word
                confidence = live_state.last_prediction_confidence
                pred_time = live_state.last_prediction_time
                rejected_time = live_state.last_rejected_time
                fsm_state = live_state.state
                is_recent = (time.time() - pred_time) < 2.5
                was_just_rejected = (time.time() - rejected_time) < 2.5 and not is_recent

            # Feed a genuinely new, confident, recent word into the phrase
            # buffer (not on every poll tick -- only once per distinct sign).
            if word and is_recent and word != live_state.last_consumed_word:
                manager.add_sign_word(word, confidence=confidence, timestamp=pred_time)
                live_state.last_consumed_word = word

            # Check whether the buffered phrase should be committed (pause
            # elapsed, or it hit the max-words safety cap).
            finalized = manager.maybe_finalize_sign_phrase()

            pending = manager.pending_sign_phrase_preview()
            if finalized:
                live_status_placeholder.success(f"Added to transcript: \u201c{finalized['text']}\u201d")
                st.rerun()
            elif pending:
                live_status_placeholder.info(f"Signing: {pending} \u2026")
            elif was_just_rejected:
                live_status_placeholder.warning(
                    "Saw a sign-like motion but couldn't match it confidently "
                    "-- try again, or lower the similarity threshold in the sidebar."
                )
            elif fsm_state == "SIGNING":
                live_status_placeholder.info("Detecting a sign in progress\u2026")
            else:
                live_status_placeholder.info("Watching for signs...")

        _poll_live_sign()

    with upload_tab:
        st.caption(
            "Prefer to record separately and upload the clip instead. Each upload "
            "is treated as one complete phrase and added to the transcript immediately."
        )
        sign_file = st.file_uploader(
            "Upload a short clip of one sign", type=["mp4", "mov", "webm", "avi"]
        )
        if st.button("Recognize sign", type="primary") and sign_file is not None:
            with st.spinner("Extracting landmarks and matching..."):
                suffix = Path(sign_file.name).suffix or ".mp4"
                word, top5, confidence = recognize_sign(sign_file.getvalue(), suffix)

            manager = st.session_state.conversation_manager
            manager.add_sign_word(word, confidence=confidence)
            finalized = manager.force_finalize_sign_phrase()

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

            # add_speech_text() also force-finalizes any pending sign phrase
            # first, so turns stay in chronological order (see manager.py).
            st.session_state.conversation_manager.add_speech_text(text)
            st.success(f"Transcribed: {text}")

st.divider()
st.subheader("Shared conversation transcript")
transcript = st.session_state.conversation_manager.get_transcript()
if not transcript:
    st.caption("Nothing yet -- try recognizing a sign or transcribing some speech above.")
for turn in transcript:
    avatar = _AVATAR_BY_SPEAKER.get(turn["speaker"], "\U0001F4AC")
    with st.chat_message(turn["speaker"], avatar=avatar):
        st.write(turn["sentence"])
        details = []
        if turn["modality"] == "sign" and turn["sentence"] != turn["text"]:
            details.append(f"raw signs: {turn['text']}")
        if turn.get("intent"):
            details.append(f"intent: {turn['intent']}")
        if details:
            st.caption(" \u00b7 ".join(details))

if st.button("Clear conversation"):
    st.session_state.conversation_manager.reset()
    st.rerun()
