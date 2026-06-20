"""
poor_mans_kinect.py — Webcam-based full-body motion controller + dodge mini-game.

Uses MediaPipe Pose for real-time skeleton tracking (33 landmarks).
Falls back to MOG2 motion density when no person is detected.

Dependencies: opencv-python, numpy, mediapipe
Run: python poor_mans_kinect.py
"""

import os
import sys

# pip opencv only ships the xcb plugin — xcb runs via XWayland on GNOME/Wayland
os.environ["QT_QPA_PLATFORM"]  = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.text.*=false"
os.environ["OPENCV_LOG_LEVEL"]     = "ERROR"
os.environ["GLOG_minloglevel"]     = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import csv
import time
import random
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque

# ─────────────────────────────────────────────
# TUNABLE CONSTANTS
# ─────────────────────────────────────────────

# Camera & frame
CAMERA_INDEX        = 0
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
TARGET_FPS          = 30

# Calibration phase
CALIBRATION_SECS    = 3

# Background subtraction (fallback when pose not detected)
GAUSSIAN_BLUR_K     = 21
MOG2_HISTORY        = 200
MOG2_VAR_THRESHOLD  = 40
MIN_MOTION_PIXELS   = 400
JUMP_ZONE_RATIO     = 0.38
JUMP_PIXEL_THRESHOLD = 600
LATERAL_PIXEL_MIN   = 300
LEFT_ZONE_RATIO     = 0.38
RIGHT_ZONE_RATIO    = 0.62

# Pose-based command thresholds (normalized 0–1 coords)
# Lateral: use nose X to decide LEFT / RIGHT
POSE_LEFT_THRESHOLD  = 0.40   # nose.x < this → LEFT
POSE_RIGHT_THRESHOLD = 0.60   # nose.x > this → RIGHT
# Jump: wrist Y must be above shoulder Y by this fraction of frame height
JUMP_WRIST_ABOVE     = 0.10   # wrist.y < shoulder.y - 0.10 → JUMP
# Minimum pose visibility to trust a landmark
MIN_VISIBILITY       = 0.50

# Persistence filter (frames)
PERSIST_LATERAL      = 3
PERSIST_JUMP         = 1

# Latency logging
LOG_FILENAME         = "latency_log.csv"
LATENCY_WINDOW       = 60

# ─────────────────────────────────────────────
# GAME CONSTANTS
# ─────────────────────────────────────────────

GAME_WIN_W          = 480
GAME_WIN_H          = 600
PLAYER_W            = 50
PLAYER_H            = 50
PLAYER_SPEED_MIN    = 4        # px/frame when barely leaning past the zone boundary
PLAYER_SPEED_MAX    = 32       # px/frame when leaning as far as possible
JUMP_HEIGHT         = 160      # peak jump height in pixels
JUMP_DURATION       = 28       # frames for full jump arc
OBSTACLE_W          = 45
OBSTACLE_H          = 30
OBS_INIT_SPEED      = 4.0
OBS_SPEED_INCREMENT = 0.0015
OBS_SPAWN_INTERVAL  = 60
PLAYER_COLOR        = (50, 200, 50)
OBSTACLE_COLOR      = (30, 30, 220)
BG_COLOR            = (20, 20, 20)
TEXT_COLOR          = (230, 230, 230)
SCORE_COLOR         = (255, 215, 0)

# ─────────────────────────────────────────────
# COMMAND LABELS
# ─────────────────────────────────────────────

CMD_NONE    = "NEUTRAL"
CMD_LEFT    = "LEFT"
CMD_RIGHT   = "RIGHT"
CMD_JUMP    = "JUMP"
CMD_SPECIAL = "SPECIAL"

# MediaPipe landmark indices used for skeleton + commands
MP_NOSE         = 0
MP_L_SHOULDER   = 11
MP_R_SHOULDER   = 12
MP_L_ELBOW      = 13
MP_R_ELBOW      = 14
MP_L_WRIST      = 15
MP_R_WRIST      = 16
MP_L_HIP        = 23
MP_R_HIP        = 24
MP_L_KNEE       = 25
MP_R_KNEE       = 26
MP_L_ANKLE      = 27
MP_R_ANKLE      = 28

# Skeleton bone connections (pairs of landmark indices)
SKELETON_BONES = [
    # Head–torso
    (MP_NOSE,       MP_L_SHOULDER),
    (MP_NOSE,       MP_R_SHOULDER),
    (MP_L_SHOULDER, MP_R_SHOULDER),
    (MP_L_SHOULDER, MP_L_HIP),
    (MP_R_SHOULDER, MP_R_HIP),
    (MP_L_HIP,      MP_R_HIP),
    # Left arm
    (MP_L_SHOULDER, MP_L_ELBOW),
    (MP_L_ELBOW,    MP_L_WRIST),
    # Right arm
    (MP_R_SHOULDER, MP_R_ELBOW),
    (MP_R_ELBOW,    MP_R_WRIST),
    # Left leg
    (MP_L_HIP,   MP_L_KNEE),
    (MP_L_KNEE,  MP_L_ANKLE),
    # Right leg
    (MP_R_HIP,   MP_R_KNEE),
    (MP_R_KNEE,  MP_R_ANKLE),
]

# Joints to highlight as circles
JOINT_LANDMARKS = [
    MP_NOSE,
    MP_L_SHOULDER, MP_R_SHOULDER,
    MP_L_ELBOW,    MP_R_ELBOW,
    MP_L_WRIST,    MP_R_WRIST,
    MP_L_HIP,      MP_R_HIP,
    MP_L_KNEE,     MP_R_KNEE,
    MP_L_ANKLE,    MP_R_ANKLE,
]


# ═════════════════════════════════════════════
# LATENCY LOGGER
# ═════════════════════════════════════════════

class LatencyLogger:
    def __init__(self, filepath: str):
        self._window = deque(maxlen=LATENCY_WINDOW)
        self._file   = open(filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["frame", "capture_ts", "emit_ts", "latency_ms"])

    def record(self, frame_idx: int, capture_ts: float, emit_ts: float):
        ms = (emit_ts - capture_ts) * 1000.0
        self._writer.writerow([frame_idx, f"{capture_ts:.6f}", f"{emit_ts:.6f}", f"{ms:.2f}"])
        self._window.append(ms)

    @property
    def rolling_avg(self) -> float:
        return sum(self._window) / len(self._window) if self._window else 0.0

    def close(self):
        self._file.flush()
        self._file.close()


# ═════════════════════════════════════════════
# POSE TRACKER  (MediaPipe Tasks API ≥ 0.10)
# ═════════════════════════════════════════════

# Path to the bundled pose landmarker model (downloaded alongside the script)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "pose_landmarker_lite.task")


class PoseTracker:
    """
    Wraps MediaPipe PoseLandmarker (Tasks API) to extract 33 body landmarks
    and draw a custom stick figure over the webcam frame.
    """

    def __init__(self):
        base_opts = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options   = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker  = mp_vision.PoseLandmarker.create_from_options(options)
        self._ts_ms       = 0          # monotonic timestamp for VIDEO mode
        self.landmarks    = None       # List[NormalizedLandmark] for pose 0, or None
        self.px_coords    = {}         # {landmark_idx: (px, py)}

    def process(self, frame_bgr: np.ndarray):
        """Run pose estimation on one frame. Updates self.landmarks / px_coords."""
        h, w = frame_bgr.shape[:2]
        self._ts_ms += 1               # Tasks VIDEO mode requires strictly increasing ts

        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker.detect_for_video(mp_image, self._ts_ms)

        self.px_coords = {}
        if result.pose_landmarks:
            self.landmarks = result.pose_landmarks[0]   # first (and only) pose
            for idx, lm in enumerate(self.landmarks):
                if lm.visibility >= MIN_VISIBILITY:
                    self.px_coords[idx] = (int(lm.x * w), int(lm.y * h))
        else:
            self.landmarks = None

    def draw_skeleton(self, frame: np.ndarray, command: str) -> np.ndarray:
        """Return frame with stick figure drawn on top."""
        if not self.px_coords:
            return frame

        out = frame.copy()
        bone_color = {
            CMD_NONE:    (0,   230,   0),
            CMD_LEFT:    (50,  220, 255),
            CMD_RIGHT:   (255, 180,  50),
            CMD_JUMP:    (80,  255, 120),
            CMD_SPECIAL: (255,  80, 200),
        }.get(command, (0, 230, 0))

        # Bones
        for (a, b) in SKELETON_BONES:
            if a in self.px_coords and b in self.px_coords:
                cv2.line(out, self.px_coords[a], self.px_coords[b],
                         bone_color, 3, cv2.LINE_AA)

        # Joints
        for idx in JOINT_LANDMARKS:
            if idx not in self.px_coords:
                continue
            pt = self.px_coords[idx]
            r  = 10 if idx == MP_NOSE else 7
            cv2.circle(out, pt, r + 2, (0, 0, 0),    -1)
            cv2.circle(out, pt, r,     bone_color,    -1)
            cv2.circle(out, pt, r,     (255,255,255),  1)

        # Head circle
        if MP_NOSE in self.px_coords:
            nose = self.px_coords[MP_NOSE]
            cv2.circle(out, nose, 22, (0,   0,   0),  3)
            cv2.circle(out, nose, 22, bone_color,      2, cv2.LINE_AA)

        return out

    def get_lm(self, idx: int):
        """Return NormalizedLandmark or None if not visible / not detected."""
        if not self.landmarks:
            return None
        lm = self.landmarks[idx]
        return lm if lm.visibility >= MIN_VISIBILITY else None

    def close(self):
        self._landmarker.close()


# ═════════════════════════════════════════════
# MOTION DETECTOR  (fallback when no pose)
# ═════════════════════════════════════════════

class MotionDetector:
    def __init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )
        self._prev_gray = None

    def apply(self, gray: np.ndarray) -> np.ndarray:
        mask = self.mog2.apply(gray)
        self._prev_gray = gray.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=3)
        return mask


# ═════════════════════════════════════════════
# COMMAND CLASSIFIER
# ═════════════════════════════════════════════

class CommandClassifier:
    """
    Primary path: landmark-based (pose detected).
      JUMP  — both wrists rise above shoulders by JUMP_WRIST_ABOVE
      LEFT  — nose X < POSE_LEFT_THRESHOLD
      RIGHT — nose X > POSE_RIGHT_THRESHOLD

    Fallback path: motion-mask density (no person in frame).
      JUMP  — foreground pixels in top zone > JUMP_PIXEL_THRESHOLD
      LEFT/RIGHT — weighted X centroid of all foreground pixels
    """

    def __init__(self, frame_w: int, frame_h: int):
        self._w         = frame_w
        self._h         = frame_h
        self._jump_cut  = int(frame_h * JUMP_ZONE_RATIO)
        self._left_cut  = int(frame_w * LEFT_ZONE_RATIO)
        self._right_cut = int(frame_w * RIGHT_ZONE_RATIO)

        self._lat_pending:  list[str] = []
        self._jump_pending: list[str] = []
        self._confirmed = CMD_NONE

    def classify(self, pose: PoseTracker, mask: np.ndarray) -> tuple[str, float]:
        """
        Returns (confirmed_command, intensity).
        intensity is 0.0–1.0:
          0.0 = barely crossing the zone threshold (slow)
          1.0 = body at the far edge of the frame (fast)
        Jump is always intensity 1.0.
        """
        if pose.landmarks:
            raw, intensity = self._raw_pose(pose)
        else:
            raw, intensity = self._raw_mask(mask)

        confirmed = self._filter(raw)

        # Keep the live intensity even during the persistence window
        # so motion feels responsive from the first frame
        if confirmed == CMD_NONE:
            intensity = 0.0
        return confirmed, intensity

    # ── Pose-based (landmark coordinates) ───────────────────────────────────

    def _raw_pose(self, pose: PoseTracker) -> tuple[str, float]:
        nose    = pose.get_lm(MP_NOSE)
        l_wrist = pose.get_lm(MP_L_WRIST)
        r_wrist = pose.get_lm(MP_R_WRIST)
        l_shoul = pose.get_lm(MP_L_SHOULDER)
        r_shoul = pose.get_lm(MP_R_SHOULDER)

        # JUMP: either wrist above its shoulder
        if l_wrist and l_shoul and r_wrist and r_shoul:
            if (l_wrist.y < l_shoul.y - JUMP_WRIST_ABOVE or
                    r_wrist.y < r_shoul.y - JUMP_WRIST_ABOVE):
                return CMD_JUMP, 1.0

        if nose:
            if nose.x < POSE_LEFT_THRESHOLD:
                # intensity: 0 at the boundary, 1 at x=0
                intensity = min(1.0, (POSE_LEFT_THRESHOLD - nose.x) / POSE_LEFT_THRESHOLD)
                return CMD_LEFT, intensity
            if nose.x > POSE_RIGHT_THRESHOLD:
                intensity = min(1.0, (nose.x - POSE_RIGHT_THRESHOLD) / (1.0 - POSE_RIGHT_THRESHOLD))
                return CMD_RIGHT, intensity

        return CMD_NONE, 0.0

    # ── Mask-based (fallback) ────────────────────────────────────────────────

    def _raw_mask(self, mask: np.ndarray) -> tuple[str, float]:
        total_px = int(np.count_nonzero(mask))
        if total_px < MIN_MOTION_PIXELS:
            return CMD_NONE, 0.0

        top_px = int(np.count_nonzero(mask[:self._jump_cut, :]))
        if top_px >= JUMP_PIXEL_THRESHOLD:
            return CMD_JUMP, 1.0

        lower_mask = mask[self._jump_cut:, :]
        if int(np.count_nonzero(lower_mask)) < LATERAL_PIXEL_MIN:
            return CMD_NONE, 0.0

        ys, xs = np.where(lower_mask > 0)
        cx = int(np.mean(xs))
        if cx < self._left_cut:
            intensity = min(1.0, (self._left_cut - cx) / self._left_cut)
            return CMD_LEFT, intensity
        if cx > self._right_cut:
            intensity = min(1.0, (cx - self._right_cut) / (self._w - self._right_cut))
            return CMD_RIGHT, intensity
        return CMD_NONE, 0.0

    # ── Persistence filter ───────────────────────────────────────────────────

    def _filter(self, raw: str) -> str:
        self._jump_pending.append(raw)
        if len(self._jump_pending) > PERSIST_JUMP:
            self._jump_pending.pop(0)

        lat = raw if raw in (CMD_LEFT, CMD_RIGHT, CMD_NONE) else CMD_NONE
        self._lat_pending.append(lat)
        if len(self._lat_pending) > PERSIST_LATERAL:
            self._lat_pending.pop(0)

        if (len(self._jump_pending) == PERSIST_JUMP
                and all(c == CMD_JUMP for c in self._jump_pending)):
            self._confirmed = CMD_JUMP
        elif (len(self._lat_pending) == PERSIST_LATERAL
              and len(set(self._lat_pending)) == 1):
            self._confirmed = self._lat_pending[0]

        return self._confirmed


# ═════════════════════════════════════════════
# OVERLAY RENDERER
# ═════════════════════════════════════════════

class OverlayRenderer:
    CMD_LABELS = {
        CMD_NONE:    ("NEUTRAL",       (160, 160, 160)),
        CMD_LEFT:    ("<  LEFT",       (50,  220, 255)),
        CMD_RIGHT:   ("RIGHT  >",      (255, 180,  50)),
        CMD_JUMP:    ("^  JUMP",       (80,  255, 120)),
        CMD_SPECIAL: ("!! SPECIAL !!", (255,  80, 200)),
    }

    def __init__(self, frame_w: int, frame_h: int):
        self._w       = frame_w
        self._h       = frame_h
        self._jump_y  = int(frame_h * JUMP_ZONE_RATIO)
        self._left_x  = int(frame_w * LEFT_ZONE_RATIO)
        self._right_x = int(frame_w * RIGHT_ZONE_RATIO)

    def draw(
        self,
        frame: np.ndarray,
        command: str,
        intensity: float,
        pose_detected: bool,
        fps: float,
        avg_latency_ms: float,
        calibrating: bool,
        calib_remaining: float,
    ) -> np.ndarray:
        out = frame.copy()

        _, color = self.CMD_LABELS.get(command, ("", (120, 120, 120)))
        bgr = color[::-1]  # RGB→BGR for opencv

        # ── Zone tint ─────────────────────────────────────────────────────────
        if command == CMD_JUMP:
            self._tint_region(out, 0, 0, self._w, self._jump_y, bgr, 0.20)
        elif command == CMD_LEFT:
            self._tint_region(out, 0, self._jump_y, self._left_x, self._h, bgr, 0.20)
        elif command == CMD_RIGHT:
            self._tint_region(out, self._right_x, self._jump_y, self._w, self._h, bgr, 0.20)

        # ── Zone grid lines ───────────────────────────────────────────────────
        cv2.line(out, (0, self._jump_y), (self._w, self._jump_y), (0, 220, 255), 1)
        cv2.line(out, (self._left_x,  self._jump_y), (self._left_x,  self._h), (0, 200, 255), 1)
        cv2.line(out, (self._right_x, self._jump_y), (self._right_x, self._h), (0, 200, 255), 1)

        # Zone labels
        self._put(out, "^ RAISE ARMS ^",
                  (self._w // 2 - 75, self._jump_y - 6), 0.55, (0, 220, 255), 1)
        self._put(out, "LEFT",  (self._left_x // 2 - 20, self._jump_y + 22), 0.55, (0, 200, 255), 1)
        self._put(out, "RIGHT", (self._right_x + (self._w - self._right_x) // 2 - 22, self._jump_y + 22),
                  0.55, (0, 200, 255), 1)

        # ── Tracking mode indicator ───────────────────────────────────────────
        mode_text  = "POSE"  if pose_detected else "MOTION"
        mode_color = (80, 255, 80) if pose_detected else (80, 180, 255)
        self._put(out, mode_text, (10, 28), 0.6, mode_color, 2)

        # ── Active command label ──────────────────────────────────────────────
        label, lcolor = self.CMD_LABELS.get(command, ("?", (255, 255, 255)))
        cv2.putText(out, label, (15, self._h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, label, (15, self._h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, lcolor, 3, cv2.LINE_AA)

        # ── Stats ─────────────────────────────────────────────────────────────
        self._put(out, f"FPS: {fps:.1f}",          (self._w - 130, 28), 0.65, (200, 255, 200), 2)
        self._put(out, f"Lat: {avg_latency_ms:.1f} ms", (self._w - 155, 54), 0.65, (200, 255, 200), 2)

        # ── Speed bar (bottom-center) ─────────────────────────────────────────
        if command in (CMD_LEFT, CMD_RIGHT) and intensity > 0:
            bar_total = 160
            bar_filled = int(bar_total * intensity)
            bx = self._w // 2 - bar_total // 2
            by = self._h - 12
            cv2.rectangle(out, (bx, by), (bx + bar_total, by + 8), (60, 60, 60), -1)
            _, bar_color = self.CMD_LABELS[command]
            cv2.rectangle(out, (bx, by), (bx + bar_filled, by + 8), bar_color[::-1], -1)
            spd = int(PLAYER_SPEED_MIN + (PLAYER_SPEED_MAX - PLAYER_SPEED_MIN) * intensity)
            self._put(out, f"spd {spd}", (bx + bar_total + 6, by + 8), 0.45, (200, 200, 200), 1)

        # ── Calibration overlay ───────────────────────────────────────────────
        if calibrating:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (self._w, self._h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
            cv2.putText(out, "CALIBRATING...",
                        (self._w // 2 - 120, self._h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 255), 3, cv2.LINE_AA)
            cv2.putText(out, f"Stand still  {calib_remaining:.1f}s",
                        (self._w // 2 - 110, self._h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

        return out

    @staticmethod
    def _tint_region(img, x1, y1, x2, y2, bgr, alpha):
        roi = img[y1:y2, x1:x2]
        tint = np.full_like(roi, bgr)
        cv2.addWeighted(tint, alpha, roi, 1 - alpha, 0, roi)
        img[y1:y2, x1:x2] = roi

    @staticmethod
    def _put(img, text, pos, scale, color, thickness):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)


# ═════════════════════════════════════════════
# MINI DODGE GAME
# ═════════════════════════════════════════════

class DodgeGame:
    STATE_PLAYING  = "playing"
    STATE_GAMEOVER = "gameover"

    def __init__(self):
        self.reset()

    def reset(self):
        self.state          = self.STATE_PLAYING
        self.px             = GAME_WIN_W // 2 - PLAYER_W // 2
        self.py             = GAME_WIN_H - PLAYER_H - 10
        self.py_base        = self.py
        self.jump_frames    = 0
        self.obstacles: list[dict] = []
        self.score          = 0
        self.frame_count    = 0
        self.obs_speed      = OBS_INIT_SPEED
        self.spawn_timer    = 0
        self.spawn_interval = OBS_SPAWN_INTERVAL

    def update(self, command: str, intensity: float = 1.0):
        if self.state == self.STATE_GAMEOVER:
            return

        self.frame_count += 1
        self.score        = self.frame_count // 10
        self.obs_speed   += OBS_SPEED_INCREMENT

        # Analog speed: linearly mapped from MIN to MAX based on body lean intensity
        speed = int(PLAYER_SPEED_MIN + (PLAYER_SPEED_MAX - PLAYER_SPEED_MIN) * intensity)

        if command == CMD_LEFT:
            self.px = max(0, self.px - speed)
        elif command == CMD_RIGHT:
            self.px = min(GAME_WIN_W - PLAYER_W, self.px + speed)

        if command == CMD_JUMP and self.jump_frames == 0:
            self.jump_frames = JUMP_DURATION

        if self.jump_frames > 0:
            # Parabolic arc via sine curve
            t = 1.0 - (self.jump_frames / JUMP_DURATION)
            self.py = self.py_base - int(JUMP_HEIGHT * np.sin(t * np.pi))
            self.jump_frames -= 1
            if self.jump_frames == 0:
                self.py = self.py_base

        # Spawn obstacles
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer    = 0
            self.spawn_interval = max(25, OBS_SPAWN_INTERVAL - self.frame_count // 200)
            ox = random.randint(0, GAME_WIN_W - OBSTACLE_W)
            self.obstacles.append({"x": ox, "y": -OBSTACLE_H, "speed": self.obs_speed})

        alive = []
        player_rect = (self.px, self.py, PLAYER_W, PLAYER_H)
        for obs in self.obstacles:
            obs["y"] += obs["speed"]
            if obs["y"] > GAME_WIN_H:
                continue
            if self._hit(player_rect, (obs["x"], int(obs["y"]), OBSTACLE_W, OBSTACLE_H)):
                self.state = self.STATE_GAMEOVER
                return
            alive.append(obs)
        self.obstacles = alive

    def render(self) -> np.ndarray:
        canvas = np.full((GAME_WIN_H, GAME_WIN_W, 3), BG_COLOR, dtype=np.uint8)

        if self.state == self.STATE_PLAYING:
            for obs in self.obstacles:
                oy = int(obs["y"])
                cv2.rectangle(canvas, (obs["x"], oy),
                              (obs["x"] + OBSTACLE_W, oy + OBSTACLE_H), OBSTACLE_COLOR, -1)
                cv2.rectangle(canvas, (obs["x"], oy),
                              (obs["x"] + OBSTACLE_W, oy + OBSTACLE_H), (100, 100, 255), 2)

            # Player
            cv2.rectangle(canvas, (self.px, self.py),
                          (self.px + PLAYER_W, self.py + PLAYER_H), PLAYER_COLOR, -1)
            ey = self.py + 12
            for ex in (self.px + 13, self.px + 33):
                cv2.circle(canvas, (ex,     ey), 5, (255, 255, 255), -1)
                cv2.circle(canvas, (ex + 2, ey), 2, (0, 0, 0),       -1)

            # Jump indicator arc on player
            if self.jump_frames > 0:
                t = 1.0 - (self.jump_frames / JUMP_DURATION)
                height_pct = np.sin(t * np.pi)
                bar_w = int((PLAYER_W - 4) * height_pct)
                cv2.rectangle(canvas,
                              (self.px + 2, self.py + PLAYER_H + 4),
                              (self.px + 2 + bar_w, self.py + PLAYER_H + 8),
                              (80, 255, 120), -1)

            cv2.putText(canvas, f"Score: {self.score}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, SCORE_COLOR, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Speed: {self.obs_speed:.1f}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(canvas, "Lean LEFT/RIGHT | Raise arms = JUMP",
                        (GAME_WIN_W // 2 - 175, GAME_WIN_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (100, 100, 100), 1, cv2.LINE_AA)
        else:
            cx, cy = GAME_WIN_W // 2, GAME_WIN_H // 2
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (GAME_WIN_W, GAME_WIN_H), (0, 0, 80), -1)
            cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
            cv2.putText(canvas, "GAME OVER",
                        (cx - 130, cy - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (50, 50, 255), 4, cv2.LINE_AA)
            cv2.putText(canvas, f"Score: {self.score}",
                        (cx - 70, cy + 10),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, SCORE_COLOR, 2, cv2.LINE_AA)
            cv2.putText(canvas, "Press  R  to restart",
                        (cx - 120, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)
            cv2.putText(canvas, "Press  Q  to quit",
                        (cx - 95,  cy + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)

        return canvas

    @staticmethod
    def _hit(r1, r2) -> bool:
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1)


# ═════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path   = os.path.join(script_dir, LOG_FILENAME)

    print("[1/4] Loading MediaPipe pose model... (pode demorar 2-3s)")
    pose_tracker = PoseTracker()
    print("[2/4] Pose model OK.")

    print("[3/4] Opening webcam...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam at index {CAMERA_INDEX}.", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    print("[4/4] Webcam OK. Opening windows...")
    motion_det   = MotionDetector()
    classifier   = CommandClassifier(FRAME_WIDTH, FRAME_HEIGHT)
    renderer     = OverlayRenderer(FRAME_WIDTH, FRAME_HEIGHT)
    game         = DodgeGame()
    logger       = LatencyLogger(log_path)

    WIN_CAM  = "Poor Man's Kinect — Camera"
    WIN_GAME = "Poor Man's Kinect — Dodge Game"
    cv2.namedWindow(WIN_CAM,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_GAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CAM,  FRAME_WIDTH, FRAME_HEIGHT)
    cv2.resizeWindow(WIN_GAME, GAME_WIN_W,  GAME_WIN_H)
    # Position windows side by side so they don't overlap
    cv2.moveWindow(WIN_CAM,  30,  50)
    cv2.moveWindow(WIN_GAME, FRAME_WIDTH + 50, 50)
    print("[INFO] *** Duas janelas devem aparecer na tela agora. ***")
    print("[INFO] Se nao ver, minimize o terminal e procure na barra de tarefas.")

    total_commands = 0
    frame_idx      = 0
    fps_times: deque = deque(maxlen=30)
    calib_start    = time.time()
    calibrating    = True

    print("[INFO] Calibrating — step into frame in 3 seconds.")
    print(f"[INFO] Skeleton tracking via MediaPipe Pose.")
    print(f"[INFO] Latency log → {log_path}")
    print("[INFO] Press Q / ESC to quit, R to restart after game-over.")

    while True:
        capture_ts = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)                          # mirror for natural feel
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        fps_times.append(capture_ts)
        fps = (len(fps_times) / (fps_times[-1] - fps_times[0] + 1e-9)
               if len(fps_times) > 1 else 0.0)

        elapsed_calib = time.time() - calib_start
        if calibrating and elapsed_calib >= CALIBRATION_SECS:
            calibrating = False
            print("[INFO] Go! Lean left/right to move, raise arms to jump.")

        # ── Preprocessing (for fallback motion detector) ──────────────────────
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (GAUSSIAN_BLUR_K, GAUSSIAN_BLUR_K), 0)
        mask    = motion_det.apply(blurred)

        # ── Pose estimation ───────────────────────────────────────────────────
        pose_tracker.process(frame)
        pose_detected = pose_tracker.landmarks is not None

        # ── Command classification ────────────────────────────────────────────
        if calibrating:
            command, intensity = CMD_NONE, 0.0
        else:
            command, intensity = classifier.classify(pose_tracker, mask)

        emit_ts = time.perf_counter()
        logger.record(frame_idx, capture_ts, emit_ts)

        if command != CMD_NONE and not calibrating:
            total_commands += 1

        # ── Draw skeleton on camera feed ──────────────────────────────────────
        display_frame = pose_tracker.draw_skeleton(frame, command)

        # ── Draw HUD overlay ─────────────────────────────────────────────────
        calib_remaining = max(0.0, CALIBRATION_SECS - elapsed_calib)
        display_frame = renderer.draw(
            display_frame, command, intensity, pose_detected,
            fps, logger.rolling_avg, calibrating, calib_remaining,
        )
        cv2.imshow(WIN_CAM, display_frame)

        # ── Game ─────────────────────────────────────────────────────────────
        if not calibrating:
            game.update(command, intensity)
        cv2.imshow(WIN_GAME, game.render())

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r") and game.state == DodgeGame.STATE_GAMEOVER:
            game.reset()

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    pose_tracker.close()
    logger.close()

    print("\n" + "═" * 50)
    print("  SESSION SUMMARY")
    print("═" * 50)
    print(f"  Frames processed   : {frame_idx}")
    print(f"  Total commands     : {total_commands}")
    print(f"  Avg latency        : {logger.rolling_avg:.2f} ms")
    print(f"  Final game score   : {game.score}")
    print(f"  Latency log        : {log_path}")
    print("═" * 50)


if __name__ == "__main__":
    main()
