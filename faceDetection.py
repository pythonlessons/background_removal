import cv2
import typing
import numpy as np
import mediapipe as mp

class MPFaceDetection:
    """Object to create and do mediapipe face detection, more about it:
    https://google.github.io/mediapipe/solutions/face_detection.html
    """
    def __init__(
        self,
        model_selection: bool = 1,
        confidence: float = 0.5,
        mp_drawing_utils: bool = True,
        color: typing.Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        ) -> None:
        """
        Args:
            model_selection: (bool) - 1 - for low distance, 0 - for far distance face detectors.
            confidence: (float) - confidence for face detector, when detection are confirmed, range (0.0-1.0).
            mp_drawing_utils: (bool) - bool option whether to use mp_drawing utils or or own, Default to True.
            color: (typing.Tuple[int, int, int]) - Color for drawing the annotation. Default to the white color.
            thickness: (int) - Thickness for drawing the annotation. Default to 2 pixels.
        """
        self.mp_drawing_utils = mp_drawing_utils
        self.color = color
        self.thickness = thickness
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=confidence)


    def tlbr(self, frame: np.ndarray, mp_detections: typing.List) -> np.ndarray:
        """Return coorinates in typing.Iterable([[Top, Left, Bottom, Right]])

        Args:
            frame: (np.ndarray) - frame on which we want to apply detections
            mp_detections: (typing.List) - list of media pipe detections

        Returns:
            detections: (np.ndarray) - list of detection in [Top, Left, Bottom, Right] coordinates
        """
        detections = []
        frame_height, frame_width, _ = frame.shape
        for detection in mp_detections:
            height = int(detection.location_data.relative_bounding_box.height * frame_height)
            width = int(detection.location_data.relative_bounding_box.width * frame_width)
            left = int(detection.location_data.relative_bounding_box.xmin * frame_width)
            top = int(detection.location_data.relative_bounding_box.ymin * frame_height)

            detections.append([top, left, top + height, left + width])

        return np.array(detections)


    def __call__(self, frame: np.ndarray, return_tlbr: bool = False) -> np.ndarray:
        """Main function to do face detection

        Args:
            frame: (np.ndarray) - frame to excecute face detection on
            return_tlbr: (bool) - bool option to return coordinates instead of frame with drawn detections

        Returns:
            typing.Union[
                frame: (np.ndarray) - processed frame with detected faces,
                detections: (typing.List) - detections in [Top, Left, Bottom, Right]
                ]
        """
        results = self.face_detection.process(frame)

        if results.detections:
            if return_tlbr:
                return self.tlbr(frame, results.detections)

            if self.mp_drawing_utils:
                # Draw face detections of each face using media pipe drawing utils.
                for detection in results.detections:
                    self.mp_drawing.draw_detection(frame, detection)
            
            else:
                # Draw face detections of each face using our own tlbr and cv2.rectangle
                for tlbr in self.tlbr(frame, results.detections):
                    cv2.rectangle(frame, tlbr[:2][::-1], tlbr[2:][::-1], self.color, self.thickness)

        return frame