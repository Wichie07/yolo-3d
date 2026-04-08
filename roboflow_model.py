import base64
import cv2
import numpy as np
import requests


class RoboflowDetector:
    """
    Drop-in replacement for ObjectDetector that uses the Roboflow hosted inference API.
    Returns detections in the same format: [[bbox, score, class_id, obj_id], ...]

    model_id format: "<project-slug>/<version>"  e.g. "my-tools/1"
    The API endpoint used is: https://detect.roboflow.com/<model_id>?api_key=<key>
    """

    def __init__(self, api_key: str, model_id: str, confidence: float = 0.4,
                 api_url: str = "https://serverless.roboflow.com"):
        """
        Args:
            api_key:    Roboflow API key
            model_id:   "<project>/<version>" e.g. "dekracoating/1"
            confidence: Minimum confidence threshold (0–1)
            api_url:    Roboflow inference server (default: serverless hosted API)
        """
        self.api_key = api_key
        self.model_id = model_id
        self.confidence = int(confidence * 100)
        self._endpoint = f"{api_url.rstrip('/')}/{model_id}"

        # Built lazily from the first inference response
        self._class_names: dict[int, str] = {}

        # Simple centroid tracker: {obj_id: (cx, cy)}
        self._next_id = 0
        self._tracks: dict[int, tuple[float, float]] = {}
        self._max_dist = 80  # pixels — max centroid movement to keep same ID

        # Masks from the last detect() call: list of (polygon_pts, class_id)
        self._last_masks: list[tuple[np.ndarray, int]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, track: bool = True):
        """
        Run inference on a single BGR frame.

        Returns:
            list of [bbox, score, class_id, obj_id]
            where bbox = [x1, y1, x2, y2]  (int pixel coordinates)
        """
        # Encode frame to base64 JPEG
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return []
        img_b64 = base64.b64encode(buf).decode("utf-8")

        resp = requests.post(
            self._endpoint,
            params={"api_key": self.api_key, "confidence": self.confidence},
            data=img_b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        resp.raise_for_status()
        predictions = resp.json().get("predictions", [])

        self._last_masks = []
        detections = []
        for pred in predictions:
            cx = pred["x"]
            cy = pred["y"]
            w = pred["width"]
            h = pred["height"]

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            score = float(pred["confidence"])
            class_name = pred["class"]
            class_id = int(pred.get("class_id", self._name_to_id(class_name)))
            self._class_names[class_id] = class_name  # always keep in sync

            obj_id = self._assign_id((cx, cy)) if track else None
            detections.append([[x1, y1, x2, y2], score, class_id, obj_id])

            if "points" in pred and pred["points"]:
                pts = np.array([(int(p["x"]), int(p["y"])) for p in pred["points"]], dtype=np.int32)
                self._last_masks.append((pts, class_id, class_name))

        # Remove stale tracks (objects not seen this frame)
        if track:
            seen_ids = {d[3] for d in detections if d[3] is not None}
            stale = [k for k in self._tracks if k not in seen_ids]
            for k in stale:
                del self._tracks[k]

        # Draw detections on frame to match ObjectDetector.detect() return signature
        for bbox, score, class_id, obj_id in detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self._class_names.get(class_id, str(class_id))} {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, detections

    def draw_masks(self, frame: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay segmentation masks from the last detect() call onto frame.
        Each class gets a consistent colour; masks are semi-transparent fills
        with a solid outline.

        Args:
            frame: BGR image to draw on (modified in-place).
            alpha: Opacity of the filled mask (0 = invisible, 1 = opaque).

        Returns:
            The same frame with masks drawn.
        """
        if not self._last_masks:
            return frame

        overlay = frame.copy()
        for pts, class_id, class_name in self._last_masks:
            color = self._class_color(class_id, class_name)
            cv2.fillPoly(overlay, [pts], color)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw solid outlines on top of the blended result
        for pts, class_id, class_name in self._last_masks:
            color = self._class_color(class_id, class_name)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        return frame

    def get_class_names(self) -> dict:
        """Return {class_id: class_name} mapping built from inference responses."""
        return self._class_names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _class_color(self, class_id: int, class_name: str = "") -> tuple[int, int, int]:
        """Return a BGR colour for a class, derived from its Dutch colour name if possible."""
        dutch_to_bgr = {
            'blauw':  (220,  80,   0),   # blue
            'oranje': (  0, 140, 255),   # orange
            'groen':  (  0, 200,  50),   # green
            'rood':   (  0,   0, 220),   # red
            'geel':   (  0, 220, 220),   # yellow
            'wit':    (230, 230, 230),   # white
            'zwart':  ( 50,  50,  50),   # black
            'paars':  (200,   0, 200),   # purple
            'roze':   (180, 105, 255),   # pink
            'bruin':  ( 42,  75, 120),   # brown
            'grijs':  (128, 128, 128),   # grey
        }
        name_lower = class_name.lower()
        for dutch, bgr in dutch_to_bgr.items():
            if name_lower.startswith(dutch):
                return bgr
        # Fallback: stable colour from palette indexed by class_id
        palette = [
            (255, 56,  56),  (255, 157, 151), (255, 112,  31),
            ( 31, 112, 255), ( 56, 255,  56), (151, 255, 157),
            (255, 178,  29), ( 29, 178, 255), (207,  52, 255),
            ( 52, 207, 255),
        ]
        return palette[class_id % len(palette)]

    def _name_to_id(self, name: str) -> int:
        """Assign a stable integer ID to a class name (built incrementally)."""
        for cid, cname in self._class_names.items():
            if cname == name:
                return cid
        new_id = len(self._class_names)
        self._class_names[new_id] = name
        return new_id

    def _assign_id(self, centroid: tuple[float, float]) -> int:
        """
        Match centroid to an existing track or create a new one.
        Uses nearest-neighbour matching with a distance threshold.
        """
        cx, cy = centroid
        best_id = None
        best_dist = self._max_dist

        for tid, (tx, ty) in self._tracks.items():
            dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = tid

        if best_id is None:
            best_id = self._next_id
            self._next_id += 1

        self._tracks[best_id] = (cx, cy)
        return best_id
