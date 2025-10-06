cat <<'PY' > air_interact.py
import argparse
import collections
import datetime
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def current_timestamps() -> Tuple[float, str]:
    ts = time.time()
    iso = datetime.datetime.fromtimestamp(ts).isoformat()
    return ts, iso


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class EventLogger:
    def __init__(self, log_path: Optional[str]):
        self.log_path = log_path
        self.handle = open(log_path, "a", encoding="utf-8") if log_path else None

    def emit(self, payload: Dict):
        ts, iso = current_timestamps()
        payload.setdefault("timestamp", ts)
        payload.setdefault("iso_time", iso)
        line = json.dumps(payload, ensure_ascii=False)
        print(line, flush=True)
        if self.handle:
            self.handle.write(line + "\n")
            self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.close()


class DLTracker:
    def __init__(self, width: int, height: int, pinch_thresh: float, hysteresis: float, smooth_alpha: float = 0.35):
        import mediapipe as mp

        self.width = width
        self.height = height
        self.diagonal = float(math.hypot(width, height))
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pinch_thresh = pinch_thresh
        self.hysteresis = hysteresis
        self.smooth_alpha = smooth_alpha
        self.prev_pen_state = "up"
        self.smoothed_tip: Optional[Tuple[float, float]] = None
        self.last_hand = None

    def _pixel_from_landmark(self, landmark) -> Tuple[int, int]:
        return int(landmark.x * self.width), int(landmark.y * self.height)

    def _count_fingers(self, hand_landmarks, handedness_label: str) -> int:
        fingers = 0
        landmarks = hand_landmarks.landmark
        tips = [4, 8, 12, 16, 20]
        pip = [3, 6, 10, 14, 18]
        # Thumb uses x comparison depending on handedness
        if handedness_label == "Right":
            fingers += int(landmarks[tips[0]].x < landmarks[pip[0]].x)
        else:
            fingers += int(landmarks[tips[0]].x > landmarks[pip[0]].x)
        for i in range(1, 5):
            fingers += int(landmarks[tips[i]].y < landmarks[pip[i]].y)
        return fingers

    def process(self, frame: np.ndarray) -> Dict:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        info = {
            "pen_state": "up",
            "index_point": None,
            "fingers": 0,
            "landmarks": None,
            "handedness": None,
        }
        if not results.multi_hand_landmarks:
            self.prev_pen_state = "up"
            self.smoothed_tip = None
            return info

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label
        self.last_hand = hand_landmarks
        index_tip = self._pixel_from_landmark(hand_landmarks.landmark[8])
        thumb_tip = self._pixel_from_landmark(hand_landmarks.landmark[4])
        if self.smoothed_tip is None:
            self.smoothed_tip = index_tip
        else:
            alpha = self.smooth_alpha
            self.smoothed_tip = (
                alpha * index_tip[0] + (1 - alpha) * self.smoothed_tip[0],
                alpha * index_tip[1] + (1 - alpha) * self.smoothed_tip[1],
            )
        pinch_distance = distance(index_tip, thumb_tip) / max(self.diagonal, 1.0)
        pen_state = self.prev_pen_state
        if pen_state == "up" and pinch_distance < self.pinch_thresh:
            pen_state = "down"
        elif pen_state == "down" and pinch_distance > (self.pinch_thresh + self.hysteresis):
            pen_state = "up"
        fingers = self._count_fingers(hand_landmarks, handedness)
        info.update(
            {
                "pen_state": pen_state,
                "index_point": tuple(map(float, self.smoothed_tip)),
                "fingers": fingers,
                "landmarks": hand_landmarks,
                "handedness": handedness,
                "pinch_distance": pinch_distance,
            }
        )
        self.prev_pen_state = pen_state
        return info

    def close(self):
        self.hands.close()


class CVTracker:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
        self.prev_point: Optional[Tuple[float, float]] = None
        self.prev_pen_state = "up"
        self.motion_threshold = 800
        self.idle_counter = 0
        self.idle_reset = 6

    def _skin_mask(self, frame: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.medianBlur(mask, 5)
        return mask

    def _count_fingers(self, frame: np.ndarray) -> int:
        mask = self._skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 2000:
            return 0
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            return 0
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0
        finger_gaps = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = contour[s][0]
            end = contour[e][0]
            far = contour[f][0]
            a = distance(start, end)
            b = distance(start, far)
            c = distance(end, far)
            angle = math.degrees(math.acos(max(min((b**2 + c**2 - a**2) / (2 * b * c + 1e-5), 1), -1)))
            if angle <= 90 and d > 2000:
                finger_gaps += 1
        return min(finger_gaps + 1, 5)

    def process(self, frame: np.ndarray) -> Dict:
        fgmask = self.bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.medianBlur(thresh, 5)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        info = {
            "pen_state": "up",
            "index_point": None,
            "fingers": self._count_fingers(frame),
        }
        if not contours:
            self.prev_point = None
            self.prev_pen_state = "up"
            return info
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.motion_threshold:
            self.idle_counter += 1
            if self.idle_counter > self.idle_reset:
                self.prev_point = None
                self.prev_pen_state = "up"
            return info
        self.idle_counter = 0
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        current_point = (float(topmost[0]), float(topmost[1]))
        pen_state = "down"
        if self.prev_point is not None and distance(current_point, self.prev_point) < 2.0:
            pen_state = "up"
        info.update({"pen_state": pen_state, "index_point": current_point})
        self.prev_point = current_point
        self.prev_pen_state = pen_state
        return info

    def close(self):
        pass


class StrokeBuffer:
    def __init__(self, idle_frames: int = 12, min_dist: float = 2.5, dot_length: float = 25.0):
        self.idle_frames = idle_frames
        self.min_dist = min_dist
        self.dot_length = dot_length
        self._points: Deque[Tuple[float, float]] = collections.deque()
        self._idle_counter = 0
        self._total_len = 0.0
        self._last_point: Optional[Tuple[float, float]] = None

    def add_point(self, point: Tuple[float, float]):
        if self._last_point is not None:
            seg_len = distance(point, self._last_point)
            if seg_len < self.min_dist:
                return
            self._total_len += seg_len
        self._points.append(point)
        self._last_point = point
        self._idle_counter = 0

    def tick_idle(self) -> Optional[List[Tuple[float, float]]]:
        if not self._points:
            return None
        self._idle_counter += 1
        if self._idle_counter >= self.idle_frames:
            return self.end_stroke()
        return None

    def end_stroke(self) -> Optional[List[Tuple[float, float]]]:
        if not self._points:
            self.reset()
            return None
        pts = list(self._points)
        total_len = self._total_len
        self.reset()
        return pts if pts else None

    def reset(self):
        self._points.clear()
        self._idle_counter = 0
        self._last_point = None
        self._total_len = 0.0

    def has_points(self) -> bool:
        return bool(self._points)


class DollarOne:
    def __init__(self, size: int = 64, square_size: int = 250):
        self.size = size
        self.square_size = square_size
        self.templates: Dict[str, List[List[Tuple[float, float]]]] = collections.defaultdict(list)
        self._seed_default_templates()

    def _seed_default_templates(self):
        # Basic primitives approximated as polylines
        primitives = {
            "check": [(0, 100), (60, 160), (200, 20)],
            "cross": [(20, 20), (200, 200), (110, 110), (200, 20), (20, 200)],
            "plus": [(120, 10), (120, 230), (120, 120), (10, 120), (230, 120)],
            "minus": [(10, 120), (230, 120)],
            "arrow": [(40, 200), (200, 120), (40, 40), (120, 120), (40, 200)],
            "zigzag": [(20, 20), (120, 120), (20, 220), (220, 20), (120, 120), (220, 220)],
        }
        for label, pts in primitives.items():
            self.add_template(label, pts)

    def add_template(self, label: str, points: Iterable[Tuple[float, float]]):
        processed = self._normalize(list(points))
        self.templates[label].append(processed)

    def _normalize(self, points: List[Tuple[float, float]]):
        pts = self._resample(points, self.size)
        indicative_angle = self._indicative_angle(pts)
        pts = self._rotate_by(pts, -indicative_angle)
        pts = self._scale_to_square(pts, self.square_size)
        pts = self._translate_to_origin(pts)
        return pts

    def _path_length(self, points: List[Tuple[float, float]]) -> float:
        return sum(distance(points[i - 1], points[i]) for i in range(1, len(points)))

    def _resample(self, points: List[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
        if len(points) < 2:
            return points * n
        path_len = self._path_length(points)
        if path_len == 0:
            return [points[0]] * n
        interval = path_len / (n - 1)
        D = 0.0
        new_points = [points[0]]
        i = 1
        while i < len(points):
            prev = points[i - 1]
            curr = points[i]
            d = distance(prev, curr)
            if d == 0:
                i += 1
                continue
            if D + d >= interval:
                t = (interval - D) / d
                new_x = prev[0] + t * (curr[0] - prev[0])
                new_y = prev[1] + t * (curr[1] - prev[1])
                new_points.append((new_x, new_y))
                points.insert(i, (new_x, new_y))
                D = 0.0
            else:
                D += d
                i += 1
        if len(new_points) < n:
            new_points.extend([points[-1]] * (n - len(new_points)))
        return new_points

    def _indicative_angle(self, points: List[Tuple[float, float]]) -> float:
        centroid = self._centroid(points)
        return math.atan2(centroid[1] - points[0][1], centroid[0] - points[0][0])

    def _rotate_by(self, points: List[Tuple[float, float]], angle: float):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        centroid = self._centroid(points)
        return [
            (
                (p[0] - centroid[0]) * cos_a - (p[1] - centroid[1]) * sin_a + centroid[0],
                (p[0] - centroid[0]) * sin_a + (p[1] - centroid[1]) * cos_a + centroid[1],
            )
            for p in points
        ]

    def _scale_to_square(self, points: List[Tuple[float, float]], size: int):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)
        scale = size / max(width, height)
        return [((p[0] - min_x) * scale, (p[1] - min_y) * scale) for p in points]

    def _translate_to_origin(self, points: List[Tuple[float, float]]):
        centroid = self._centroid(points)
        return [(p[0] - centroid[0], p[1] - centroid[1]) for p in points]

    def _centroid(self, points: List[Tuple[float, float]]):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def recognize(self, points: List[Tuple[float, float]]) -> Tuple[Optional[str], float]:
        if not self.templates:
            return None, 0.0
        normalized = self._normalize(points)
        best_score = -float("inf")
        best_label = None
        for label, template_list in self.templates.items():
            for template in template_list:
                score = -self._distance_at_best_angle(normalized, template)
                if score > best_score:
                    best_score = score
                    best_label = label
        similarity = 1 - best_score / (0.5 * math.sqrt(2 * (self.square_size**2)))
        return best_label, similarity

    def _distance_at_best_angle(self, points: List[Tuple[float, float]], template: List[Tuple[float, float]]):
        start = -math.pi / 4
        end = math.pi / 4
        phi = 0.5 * (-1 + math.sqrt(5))
        threshold = math.radians(2)

        x1 = phi * start + (1 - phi) * end
        f1 = self._distance_at_angle(points, template, x1)
        x2 = (1 - phi) * start + phi * end
        f2 = self._distance_at_angle(points, template, x2)

        while abs(end - start) > threshold:
            if f1 < f2:
                end = x2
                x2 = x1
                f2 = f1
                x1 = phi * start + (1 - phi) * end
                f1 = self._distance_at_angle(points, template, x1)
            else:
                start = x1
                x1 = x2
                f1 = f2
                x2 = (1 - phi) * start + phi * end
                f2 = self._distance_at_angle(points, template, x2)
        return min(f1, f2)

    def _distance_at_angle(self, points: List[Tuple[float, float]], template: List[Tuple[float, float]], angle: float):
        rotated = self._rotate_by(points, angle)
        return self._path_distance(rotated, template)

    def _path_distance(self, a: List[Tuple[float, float]], b: List[Tuple[float, float]]):
        return sum(distance(a[i], b[i]) for i in range(len(a))) / len(a)


class AirRecognizer:
    def __init__(self, frame_shape: Tuple[int, int, int], dot_threshold: float = 25.0):
        self.frame_shape = frame_shape
        self.dot_threshold = dot_threshold
        self.dollar = DollarOne()
        self.last_shape = ""
        self.last_points: List[Tuple[float, float]] = []
        self.oscillation: Deque[int] = collections.deque(maxlen=20)
        self.osc_last_point: Optional[Tuple[float, float]] = None
        self.canvas = np.zeros(frame_shape, dtype=np.uint8)

    def clear(self):
        self.canvas[:] = 0
        self.last_points = []
        self.last_shape = ""
        self.oscillation.clear()
        self.osc_last_point = None

    def render_mask(self, points: List[Tuple[float, float]], shape: Tuple[int, int, int]) -> np.ndarray:
        if isinstance(shape, tuple) and len(shape) == 3:
            height, width = shape[0], shape[1]
        else:
            height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        if not points:
            return mask
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=6)
        cv2.dilate(mask, np.ones((3, 3), np.uint8), mask)
        return mask

    def update_canvas(self, points: List[Tuple[float, float]]):
        mask = self.render_mask(points, self.frame_shape)
        colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.canvas = cv2.addWeighted(self.canvas, 0.6, colored, 0.4, 0)

    def _path_length(self, points: List[Tuple[float, float]]):
        if len(points) < 2:
            return 0.0
        return sum(distance(points[i - 1], points[i]) for i in range(1, len(points)))

    def update_oscillation(self, point: Optional[Tuple[float, float]]) -> bool:
        if point is None:
            self.oscillation.clear()
            self.osc_last_point = None
            return False
        if self.osc_last_point is not None:
            dx = point[0] - self.osc_last_point[0]
            if abs(dx) > 4:
                sign = 1 if dx > 0 else -1
                self.oscillation.append(sign)
        self.osc_last_point = point
        pos = self.oscillation.count(1)
        neg = self.oscillation.count(-1)
        if pos > 3 and neg > 3 and abs(pos - neg) < 3:
            self.oscillation.clear()
            self.osc_last_point = None
            return True
        return False

    def classify(self, points: List[Tuple[float, float]]) -> str:
        if not points:
            return "unknown"
        path_len = self._path_length(points)
        if path_len < self.dot_threshold:
            self.last_shape = "dot"
            self.last_points = points
            self.update_canvas(points)
            return "dot"

        mask = self.render_mask(points, self.frame_shape)
        line_label = self._detect_line(mask, path_len, points)
        if line_label:
            self.last_shape = line_label
            self.last_points = points
            self.update_canvas(points)
            return line_label

        circle_label = self._detect_circle(mask)
        if circle_label:
            self.last_shape = circle_label
            self.last_points = points
            self.update_canvas(points)
            return circle_label

        polygon_label = self._detect_polygon(mask)
        if polygon_label:
            self.last_shape = polygon_label
            self.last_points = points
            self.update_canvas(points)
            return polygon_label

        label, score = self.dollar.recognize(points)
        if label and score >= 0.72:
            final_label = label
        else:
            final_label = "unknown"
        self.last_shape = final_label
        self.last_points = points
        self.update_canvas(points)
        return final_label

    def _detect_line(self, mask: np.ndarray, path_len: float, points: List[Tuple[float, float]]) -> Optional[str]:
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=10)
        if lines is None:
            return None
        best = None
        longest = 0
        for x1, y1, x2, y2 in lines[:, 0]:
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len > longest:
                best = (x1, y1, x2, y2, seg_len)
                longest = seg_len
        if not best:
            return None
        straightness = longest / max(path_len, 1e-5)
        if straightness >= 0.9:
            return "line"
        return None

    def _detect_circle(self, mask: np.ndarray) -> Optional[str]:
        blurred = cv2.medianBlur(mask, 5)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=100, param2=20, minRadius=10, maxRadius=300)
        if circles is None:
            return None
        return "circle"

    def _detect_polygon(self, mask: np.ndarray) -> Optional[str]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) == 3:
            return "triangle"
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ratio = w / float(h) if h != 0 else 0
            if 0.8 <= ratio <= 1.2:
                return "square"
            return "rectangle"
        return None

    def save_template(self, label: str):
        if not self.last_points:
            return False
        self.dollar.add_template(label, self.last_points)
        return True


@dataclass
class HUDState:
    mode: str
    recording: bool
    pen_state: str
    shape: str
    fingers: int
    fps: float
    pinch_distance: Optional[float] = None


class FaceVisualizer:
    def __init__(self, mode: str):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.mode = mode

    def cycle(self):
        modes = ["off", "box", "pip"]
        idx = modes.index(self.mode) if self.mode in modes else 0
        self.mode = modes[(idx + 1) % len(modes)]
        return self.mode

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == "off":
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return frame
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        if self.mode == "box":
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif self.mode == "pip":
            face_crop = frame[y : y + h, x : x + w]
            if face_crop.size != 0:
                resized = cv2.resize(face_crop, (160, 160))
                frame[20 : 20 + 160, 20 : 20 + 160] = resized
                cv2.rectangle(frame, (20, 20), (180, 180), (0, 255, 0), 2)
        return frame


def draw_hud(frame: np.ndarray, hud: HUDState):
    text = f"Mode: {hud.mode} | REC: {'ON' if hud.recording else 'OFF'} | Pen: {hud.pen_state.upper()} | Shape: {hud.shape} | Fingers: {hud.fingers} | FPS: {hud.fps:.1f}"
    if hud.pinch_distance is not None:
        text += f" | Pinch: {hud.pinch_distance:.3f}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    help_text = "Keys: R toggle REC | F face view | C clear | T template | ESC quit"
    cv2.putText(frame, help_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time air interaction system")
    parser.add_argument("--mode", choices=["dl", "cv"], default="dl")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--record-start", choices=["on", "off"], default="off")
    parser.add_argument("--face", choices=["off", "box", "pip"], default="off")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--pinch-threshold", type=float, default=0.05)
    parser.add_argument("--pinch-hysteresis", type=float, default=0.02)
    parser.add_argument("--idle-frames", type=int, default=12)
    parser.add_argument("--dot-threshold", type=float, default=25.0)
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(json.dumps({"type": "error", "message": "Unable to access camera"}))
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    logger = EventLogger(args.log_file)
    face_visualizer = FaceVisualizer(args.face)
    stroke_buffer = StrokeBuffer(idle_frames=args.idle_frames, dot_length=args.dot_threshold)
    recognizer = AirRecognizer((height, width, 3), dot_threshold=args.dot_threshold)

    tracker = None
    if args.mode == "dl":
        tracker = DLTracker(width, height, args.pinch_threshold, args.pinch_hysteresis)
    else:
        tracker = CVTracker(width, height)

    recording = args.record_start == "on"
    pen_state = "up"
    last_pen_state = "up"
    last_emit_shape = ""
    last_fingers = 0
    last_timestamp = time.time()
    fps = 0.0
    cv2.namedWindow("Air Interaction")
    logger.emit({"type": "session_start", "mode": args.mode, "recording": recording})
    if args.face != "off":
        logger.emit({"type": "face_mode", "mode": args.face})

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            dt = now - last_timestamp
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            last_timestamp = now

            tracker_info = tracker.process(frame)
            pen_state = tracker_info.get("pen_state", "up")
            index_point = tracker_info.get("index_point")
            pinch_distance = tracker_info.get("pinch_distance")
            last_fingers = tracker_info.get("fingers", 0)

            erase_triggered = recognizer.update_oscillation(index_point)
            if erase_triggered and recording:
                logger.emit({"type": "gesture", "gesture": "erase"})
                recognizer.clear()

            if args.mode == "dl":
                if pen_state == "down" and recording and index_point is not None:
                    stroke_buffer.add_point(index_point)
                elif pen_state == "up":
                    completed = stroke_buffer.end_stroke()
                    if completed and recording:
                        label = recognizer.classify(completed)
                        last_emit_shape = label
                        logger.emit({"type": "stroke_end", "shape": label})
            else:
                if index_point is not None and recording:
                    stroke_buffer.add_point(index_point)
                else:
                    completed = stroke_buffer.tick_idle()
                    if completed and recording:
                        label = recognizer.classify(completed)
                        last_emit_shape = label
                        logger.emit({"type": "stroke_end", "shape": label})

            if args.mode == "dl" and pen_state != last_pen_state:
                event_type = "pen_down" if pen_state == "down" else "pen_up"
                logger.emit({"type": event_type})
            last_pen_state = pen_state

            hud = HUDState(
                mode=args.mode.upper(),
                recording=recording,
                pen_state=pen_state,
                shape=last_emit_shape,
                fingers=last_fingers,
                fps=fps,
                pinch_distance=pinch_distance if args.mode == "dl" else None,
            )

            display = frame.copy()
            if recognizer.last_points:
                for i in range(1, len(recognizer.last_points)):
                    cv2.line(
                        display,
                        (int(recognizer.last_points[i - 1][0]), int(recognizer.last_points[i - 1][1])),
                        (int(recognizer.last_points[i][0]), int(recognizer.last_points[i][1])),
                        (0, 0, 255),
                        2,
                    )
            if index_point is not None:
                cv2.circle(display, (int(index_point[0]), int(index_point[1])), 8, (255, 0, 0), -1)

            display = face_visualizer.annotate(display)
            draw_hud(display, hud)

            cv2.imshow("Air Interaction", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                logger.emit({"type": "session_end"})
                break
            if key in (ord("r"), ord("R")):
                recording = not recording
                logger.emit({"type": "recording", "recording": recording})
                if not recording:
                    stroke_buffer.reset()
            if key in (ord("f"), ord("F")):
                mode = face_visualizer.cycle()
                logger.emit({"type": "face_mode", "mode": mode})
            if key in (ord("c"), ord("C")):
                recognizer.clear()
                stroke_buffer.reset()
                logger.emit({"type": "canvas_cleared"})
                last_emit_shape = ""
            if key in (ord("t"), ord("T")):
                if recognizer.last_shape:
                    success = recognizer.save_template(recognizer.last_shape)
                    if success:
                        logger.emit({"type": "template_saved", "label": recognizer.last_shape})

    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.close()


if __name__ == "__main__":
    main()
