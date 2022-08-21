import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

class MPFaceDetection:
    """Object to create and do mediapipe face detection, more about it:
    https://google.github.io/mediapipe/solutions/face_detection.html
    """
    def __init__(
        self,
        model_selection: bool = 1,
        confidence: float = 0.5
        ) -> None:
        """
        Args:
            model_selection: (bool) - 1 - for low distance, 0 - for far distance face detectors
            confidence: (float) - confidence for face detector, when detection are confirmed
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=confidence)


    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Main function to do face detection

        Args:
            frame: (np.ndarray) - frame to excecute face detection on

        Returns:
            frame: (np.ndarray) - processed frame with face detection
        """
        results = self.face_detection.process(frame)

        if results.detections:
            # Draw face detections of each face.
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)
 
        return frame