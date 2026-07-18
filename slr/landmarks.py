
"""
landmarks.py
============
Wraps MediaPipe Holistic and turns each video frame into a fixed-size,
translation/scale-normalized landmark feature vector.
Feature layout (per frame), all float32:
    pose        : 33 landmarks x (x, y, z, visibility) = 132
    left_hand   : 21 landmarks x (x, y, z)              = 63
    right_hand  : 21 landmarks x (x, y, z)               = 63
    face_subset : 56 landmarks x (x, y, z)               = 168   (lips + eyebrows)
    -------------------------------------------------------------
    TOTAL                                                = 426
Any landmark group MediaPipe fails to detect in a frame (e.g. a hand
leaves frame) is filled with zeros rather than dropped, so every frame
always produces a vector of the same length -- important for feeding
fixed-shape tensors into the model later.
Normalization: all (x, y) coordinates are re-centered on the midpoint
of the shoulders and scaled by shoulder width, so the features are
roughly invariant to where the signer stands and how close they are
to the camera. z is scaled by the same factor.
"""

from __future__ import annotations

import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mediapipe is required: pip install mediapipe opencv-python"
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

POSE_DIM = N_POSE * 4          # x,y,z,visibility
HAND_DIM = N_HAND * 3          # x,y,z
FACE_DIM = N_FACE * 3          # x,y,z
FEATURE_DIM = POSE_DIM + 2 * HAND_DIM + FACE_DIM  # = 426

_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12


class HolisticLandmarkExtractor:
    """
    Thin wrapper around mediapipe.solutions.holistic.Holistic that
    converts a raw BGR frame into a normalized (FEATURE_DIM,) vector.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._mp_holistic = mp.solutions.holistic
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self.holistic = self._mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.holistic.close()

    @staticmethod
    def _landmarks_to_array(landmark_list, n_points, with_visibility=False):
        dim = 4 if with_visibility else 3
        arr = np.zeros((n_points, dim), dtype=np.float32)
        if landmark_list is not None:
            for i, lm in enumerate(landmark_list.landmark):
                if i >= n_points:
                    break
                if with_visibility:
                    arr[i] = (lm.x, lm.y, lm.z, lm.visibility)
                else:
                    arr[i] = (lm.x, lm.y, lm.z)
        return arr

    @staticmethod
    def _face_subset_to_array(face_landmark_list):
        arr = np.zeros((N_FACE, 3), dtype=np.float32)
        if face_landmark_list is not None:
            lms = face_landmark_list.landmark
            for out_i, idx in enumerate(FACE_IDX):
                if idx < len(lms):
                    lm = lms[idx]
                    arr[out_i] = (lm.x, lm.y, lm.z)
        return arr

    @staticmethod
    def _normalize(pose, left_hand, right_hand, face):
        """Center on shoulder midpoint, scale by shoulder width."""
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

    def process(self, frame_bgr: np.ndarray, draw: bool = False):
        """
        Run Holistic on one BGR frame.
        Returns:
            feature_vec: np.float32 array of shape (FEATURE_DIM,)
            annotated_frame: frame with landmarks drawn (only if draw=True,
                              else the original frame is returned unchanged)
        """
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)

        pose = self._landmarks_to_array(
            results.pose_landmarks, N_POSE, with_visibility=True
        )
        left_hand = self._landmarks_to_array(results.left_hand_landmarks, N_HAND)
        right_hand = self._landmarks_to_array(results.right_hand_landmarks, N_HAND)
        face = self._face_subset_to_array(results.face_landmarks)

        pose_n, left_n, right_n, face_n = self._normalize(pose, left_hand, right_hand, face)

        feature_vec = np.concatenate(
            [pose_n.reshape(-1), left_n.reshape(-1), right_n.reshape(-1), face_n.reshape(-1)]
        ).astype(np.float32)

        annotated = frame_bgr
        if draw:
            annotated = frame_bgr.copy()
            self._mp_drawing.draw_landmarks(
                annotated, results.face_landmarks,
                self._mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mp_styles.get_default_face_mesh_contours_style(),
            )
            self._mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, self._mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
            )
            self._mp_drawing.draw_landmarks(
                annotated, results.left_hand_landmarks, self._mp_holistic.HAND_CONNECTIONS
            )
            self._mp_drawing.draw_landmarks(
                annotated, results.right_hand_landmarks, self._mp_holistic.HAND_CONNECTIONS
            )

        return feature_vec, annotated


def iter_video_landmarks(video_path: str, extractor: HolisticLandmarkExtractor = None):
    """
    Generator that yields one feature vector (np.float32, shape (FEATURE_DIM,))
    per frame of a video file.
    """
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
    """
    Generator that yields (feature_vec, frame) pairs read live from a camera.
    """
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
