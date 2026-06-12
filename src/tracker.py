from ultralytics import YOLO
import cv2

class FootballTracker:
    def __init__(self, model_path='model/yolo11n.onnx'):
        self.model = YOLO(model_path)
        self.ball_class_id = 32

    def detect_and_track(self, frame):
        # detecting bodies with activating persist tracking and verbose mode off
        results = self.model.track(frame, persist=True, verbose=False)[0]
        frame_data = {"players": [], "ball": None}

        if results.boxes is None: return frame_data
       
        for idx, box in enumerate(results.boxes): # type: ignore
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            coords = box.xyxy[0].tolist()

            if cls_id == self.ball_class_id:
                frame_data["ball"] = {"coords": coords}
            elif cls_id == 0: # Person class
                kp = []
                if hasattr(results, 'keypoints') and results.keypoints is not None:
                    kp = results.keypoints.xy[idx].tolist()

                frame_data["players"].append({
                    "track_id": track_id,
                    "coords": coords,
                    "keypoints": kp
                })

        return frame_data
