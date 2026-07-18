"""
landmarks.py (Tasks API version)
=================================

Rewritten to use MediaPipe's modern Tasks API (PoseLandmarker,
HandLandmarker, FaceLandmarker) instead of the legacy
mp.solutions.holistic.Holistic, which is either broken or entirely
absent (AttributeError: module 'mediapipe' has no attribute
'solutions') in every mediapipe release that has prebuilt wheels for
recent Python versions (0.10.30+). The Tasks API is what all current
and future mediapipe releases support, so this fixes the problem at
the root rather than fighting version pins that keep breaking.

IMPORTANT CAVEATS (read before trusting this blindly):
  - Model files are downloaded on first use from Google's public
    mediapipe-models bucket. I verified these exact URLs appear
    consistently across multiple independent, reputable sources
    (Google's own Android tutorials, npm docs, community write-ups),
    but I could not directly test the actual download/inference in my
    own sandbox (no network access to storage.googleapis.com there).
    Test this for real before fully trusting it.
  - Hand left/right assignment now comes from HandLandmarker's
    "handedness" classification, which may use a DIFFERENT convention
    than the legacy API's results.left_hand_landmarks/
    right_hand_landmarks did. If your training data was extracted with
    the old API, there's a real (if likely modest) risk of a
    left/right slot mismatch between training and live inference.
    Worth explicitly re-validating recognition accuracy after this
    change, not just assuming it transfers cleanly.
  - Uses IMAGE running mode (each frame processed independently) for
    simplicity/robustness, rather than VIDEO/LIVE_STREAM mode (which
    would enable internal frame-to-frame tracking but requires
    timestamp bookkeeping). This may be marginally slower per-frame
    than the legacy API's internal ROI carry-over, but is simpler and
    less fragile.

Feature layout (per frame) is UNCHANGED from before -- same 426-dim
vector, same normalization -- so trained checkpoints and prototypes.npy
remain compatible; only the extraction backend changed.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions,
        HandLandmarker, HandLandmarkerOptions,
        FaceLandmarker, FaceLandmarkerOptions,
        RunningMode,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mediapipe is required: pip install mediapipe opencv-python-headless"
    ) from e


_LIPS_IDX = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91,
    95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
    312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
]
_LEFT_EYEBROW_IDX = [46, 52, 53, 55, 63, 65, 66, 70]
_RIGHT_EYEBROW_IDX = [276, 282, 283, 285, 293, 295, 296, 300]
FACE_IDX = _LIPS_IDX + _LEFT_EYEBROW_IDX + _RIGHT_EYEBROW_IDX  # 56 points

N_POSE = 33
N_HAND = 21
N_FACE = len(FACE_IDX)

POSE_DIM = N_POSE * 4
HAND_DIM = N_HAND * 3
FACE_DIM = N_FACE * 3
FEATURE_DIM = POSE_DIM + 2 * HAND_DIM + FACE_DIM  # = 426

_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mp_models")
MODEL_URLS = {
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def _ensure_model(name: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.task")
    if not os.path.exists(path):
        urllib.request.urlretrieve(MODEL_URLS[name], path)
    return path


class HolisticLandmarkExtractor:
    """
    Drop-in replacement for the old Holistic-based extractor, same
    public interface (process(), close(), context manager), same
    output feature vector layout -- just backed by three separate
    Tasks API landmarkers instead of one legacy Holistic solution.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_hands: int = 2,
    ):
        self._pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model("pose")),
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
        ))
        self._hand = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model("hand")),
            running_mode=RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        ))
        self._face = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model("face")),
            running_mode=RunningMode.IMAGE,
            min_face_detection_confidence=min_detection_confidence,
        ))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._pose.close()
        self._hand.close()
        self._face.close()

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _pose_to_array(pose_result) -> np.ndarray:
        arr = np.zeros((N_POSE, 4), dtype=np.float32)
        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks[0]
            for i, lm in enumerate(lms):
                if i >= N_POSE:
                    break
                vis = getattr(lm, "visibility", None)
                vis = vis if vis is not None else 1.0
                arr[i] = (lm.x, lm.y, lm.z, vis)
        return arr

    @staticmethod
    def _hands_to_arrays(hand_result) -> tuple[np.ndarray, np.ndarray]:
        left = np.zeros((N_HAND, 3), dtype=np.float32)
        right = np.zeros((N_HAND, 3), dtype=np.float32)
        if hand_result.hand_landmarks:
            for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                label = handedness[0].category_name if handedness else None
                target = left if label == "Left" else right
                for i, lm in enumerate(hand_lms):
                    if i >= N_HAND:
                        break
                    target[i] = (lm.x, lm.y, lm.z)
        return left, right

    @staticmethod
    def _face_subset_to_array(face_result) -> np.ndarray:
        arr = np.zeros((N_FACE, 3), dtype=np.float32)
        if face_result.face_landmarks:
            lms = face_result.face_landmarks[0]
            for out_i, idx in enumerate(FACE_IDX):
                if idx < len(lms):
                    lm = lms[idx]
                    arr[out_i] = (lm.x, lm.y, lm.z)
        return arr

    @staticmethod
    def _normalize(pose, left_hand, right_hand, face):
        l_sh, r_sh = pose[_LEFT_SHOULDER, :2], pose[_RIGHT_SHOULDER, :2]
        shoulder_present = pose[_LEFT_SHOULDER, 3] > 0 or pose[_RIGHT_SHOULDER, 3] > 0
        center = (l_sh + r_sh) / 2.0
        scale = np.linalg.norm(l_sh - r_sh)
        if not shoulder_present or scale < 1e-6:
            center = np.array([0.5, 0.5], dtype=np.float32)
            scale = 1.0

        def _norm_xy(arr):
            out = arr.copy()
            out[:, 0:2] = (arr[:, 0:2] - center) / scale
            if arr.shape[1] >= 3:
                out[:, 2] = arr[:, 2] / scale
            return out

        return (
            _norm_xy(pose),
            _norm_xy(left_hand),
            _norm_xy(right_hand),
            _norm_xy(face),
        )

    # -- public API ---------------------------------------------------------

    def process(self, frame_bgr: np.ndarray, draw: bool = False):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = self._pose.detect(mp_image)
        hand_result = self._hand.detect(mp_image)
        face_result = self._face.detect(mp_image)

        pose = self._pose_to_array(pose_result)
        left_hand, right_hand = self._hands_to_arrays(hand_result)
        face = self._face_subset_to_array(face_result)

        pose_n, left_n, right_n, face_n = self._normalize(pose, left_hand, right_hand, face)

        feature_vec = np.concatenate(
            [pose_n.reshape(-1), left_n.reshape(-1), right_n.reshape(-1), face_n.reshape(-1)]
        ).astype(np.float32)

        annotated = frame_bgr
        if draw:
            annotated = frame_bgr.copy()
            h, w = annotated.shape[:2]
            for lm_arr, color in [
                (pose[:, :2], (0, 255, 0)),
                (left_hand[:, :2], (255, 0, 0)),
                (right_hand[:, :2], (0, 0, 255)),
            ]:
                for x, y in lm_arr:
                    if x == 0 and y == 0:
                        continue
                    cv2.circle(annotated, (int(x * w), int(y * h)), 3, color, -1)

        return feature_vec, annotated


def iter_video_landmarks(video_path: str, extractor: HolisticLandmarkExtractor = None):
    owns_extractor = extractor is None
    if owns_extractor:
        extractor = HolisticLandmarkExtractor()

    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            vec, _ = extractor.process(frame, draw=False)
            yield vec
    finally:
        cap.release()
        if owns_extractor:
            extractor.close()


def iter_camera_landmarks(camera_index: int = 0, extractor: HolisticLandmarkExtractor = None,
                           draw: bool = False):
    owns_extractor = extractor is None
    if owns_extractor:
        extractor = HolisticLandmarkExtractor()

    cap = cv2.VideoCapture(camera_index)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            vec, annotated = extractor.process(frame, draw=draw)
            yield vec, annotated
    finally:
        cap.release()
        if owns_extractor:
            extractor.close()
