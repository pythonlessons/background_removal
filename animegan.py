import os
import cv2
import typing
import numpy as np
import onnxruntime as ort

class AnimeGAN:
    """ Object to image animation using AnimeGAN models
    https://github.com/TachibanaYoshino/AnimeGANv2

    onnx models:
    'https://docs.google.com/uc?export=download&id=1VPAPI84qaPUCHKHJLHiMK7BP_JE66xNe' AnimeGAN_Hayao.onnx
    'https://docs.google.com/uc?export=download&id=17XRNQgQoUAnu6SM5VgBuhqSBO4UAVNI1' AnimeGANv2_Hayao.onnx
    'https://docs.google.com/uc?export=download&id=10rQfe4obW0dkNtsQuWg-szC4diBzYFXK' AnimeGANv2_Shinkai.onnx
    'https://docs.google.com/uc?export=download&id=1X3Glf69Ter_n2Tj6p81VpGKx7U4Dq-tI' AnimeGANv2_Paprika.onnx

    """
    def __init__(
        self,
        model_path: str = '',
        downsize_ratio: float = 1.0,
        ) -> None:
        """
        Args:
            model_path: (str) - path to onnx model file
            downsize_ratio: (float) - ratio to downsize input frame for faster inference
        """
        if not os.path.exists(model_path):
            raise Exception(f"Model doesn't exists in {model_path}")
        
        self.downsize_ratio = downsize_ratio

        providers = ['CUDAExecutionProvider'] if ort.get_device() == "GPU" else ['CPUExecutionProvider']

        self.ort_sess = ort.InferenceSession(model_path, providers=providers)

    def to_32s(self, x):
        return 256 if x < 256 else x - x%32

    def process_frame(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        """ Function to process frame to fit model input as 32 multiplier and resize to fit model input

        Args:
            frame: (np.ndarray) - frame to process
            x32: (bool) - if True, resize frame to 32 multiplier

        Returns:
            frame: (np.ndarray) - processed frame
        """
        h, w = frame.shape[:2]
        if x32: # resize image to multiple of 32s
            frame = cv2.resize(frame, (self.to_32s(int(w*self.downsize_ratio)), self.to_32s(int(h*self.downsize_ratio))))
        frame = frame.astype(np.float32) / 127.5 - 1.0
        return frame

    def post_process(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
        """ Convert model float output to uint8 image resized to original frame size

        Args:
            frame: (np.ndarray) - AnimeGaAN output frame
            wh: (typing.Tuple[int, int]) - original frame size

        Returns:
            frame: (np.ndarray) - original size animated image
        """
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
