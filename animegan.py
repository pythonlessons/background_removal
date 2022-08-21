import os
import cv2
import typing
import numpy as np
import onnxruntime as ort

class AnimeGAN:
    """Object to create and do mediapipe face detection, more about it:
    https://github.com/TachibanaYoshino/AnimeGANv2
    """
    def __init__(
        self,
        model_path: str = '',
        ) -> None:
        """
        Args:
            model_selection: (bool) - 1 - for low distance, 0 - for far distance face detectors
            confidence: (float) - confidence for face detector, when detection are confirmed
        """
        if not os.path.exists(model_path):
            raise Exception(f"Model doesn't exists in {model_path}")

        providers = ['CUDAExecutionProvider'] if ort.get_device() == "GPU" else ['CPUExecutionProvider']

        self.ort_sess = ort.InferenceSession(model_path, providers=providers)

    def to_32s(self, x):
        return 256 if x < 256 else x - x%32

    def process_frame(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        h, w = frame.shape[:2]
        if x32: # resize image to multiple of 32s
            frame = cv2.resize(frame, (self.to_32s(w), self.to_32s(h)))
        frame = frame.astype(np.float32) / 127.5 - 1.0
        return frame

    def post_process(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
        frame = (frame.squeeze() + 1.) / 2 * 255
        frame = frame.astype(np.uint8)
        frame = cv2.resize(frame, (wh[0], wh[1]))
        return frame

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Main function to process selfie semgentation on each call

        Args:
            frame: (np.ndarray) - frame to excecute face detection on

        Returns:
            frame: (np.ndarray) - processed frame with face detection
        """
        image = self.process_frame(frame)
        outputs = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: np.expand_dims(image, axis=0)})
        frame = self.post_process(outputs[0], frame.shape[:2][::-1])
 
        return frame
