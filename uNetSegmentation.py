import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import stow
import typing
import numpy as np
# import mediapipe as mp
import onnxruntime as ort

class unetSegmentation:
    """Object to create and do mediapipe selfie segmentation, more about it:
    https://google.github.io/mediapipe/solutions/selfie_segmentation.html
    """
    def __init__(
        self,
        bg_blur_ratio: typing.Tuple[int, int] = (35, 35),
        bg_image: typing.Optional[np.ndarray] = None,
        threshold: float = 0.5,
        # onnx_model_path: str = "models/202210251749/model.onnx", 
        onnx_model_path: str = "models/modnet.onnx", 
        #model_selection: bool = 1,
        bg_images_path: str = None,
        bg_color : typing.Tuple[int, int, int] = None,
        force_cpu: bool = False,
        ) -> None:
        """
        Args:
            bg_blur_ratio: (typing.Tuple) = (35, 35) - ratio to apply for cv2.GaussianBlur
            bg_image: (typing.Optional) = None - background color to use instead of gray color in background
            threshold: (float) = 0.5 - accuracy border threshold separating background and foreground, necessary to play to get the best results
            model_selection: (bool) = 1 - general or landscape model selection for segmentations mask
            bg_images_path: (str) = None - path to folder for background images
            bg_color: (typing.Tuple[int, int, int]) = None - color to replace background with
        """
        # self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        # self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection)

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        providers = providers if ort.get_device() == "GPU" and not force_cpu else providers[::-1]

        self.ort_sess = ort.InferenceSession(onnx_model_path, providers=providers)

        # self.input_shape = self.ort_sess._inputs_meta[0].shape[1:3]
        self.input_shape = self.ort_sess._inputs_meta[0].shape[1:]

        self.bg_blur_ratio = bg_blur_ratio
        self.bg_image = bg_image
        self.threshold = threshold
        self.bg_color = bg_color

        if bg_images_path:
            self.bg_images = [cv2.imread(image.path) for image in stow.ls(bg_images_path)]
            self.bg_image = self.bg_images[0]

    def change_image(self, prevOrNext: bool = True) -> bool:
        """Change image to next or previous ir they are provided

        Args:
            prevOrNext: (bool) - argument to change image to next or previous in given list

        Returns:
            bool - Return True if successfully changed background image
        """
        if not self.bg_images:
            return False

        if prevOrNext:
            self.bg_images = self.bg_images[1:] + [self.bg_images[0]]
        else:
            self.bg_images = [self.bg_images[-1]] + self.bg_images[:-1]
        self.bg_image = self.bg_images[0]

        return True

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Main function to process selfie semgentation on each call

        Args:
            frame: (np.ndarray) - frame to excecute selfie segmentation on

        Returns:
            frame: (np.ndarray) - processed frame with selfie segmentation
        """
        # results = self.selfie_segmentation.process(frame)
        img = cv2.resize(frame, (512, 288), cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.expand_dims(img, axis=0)
        mask = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: img})[0][0][0]

        segmentation_mask = cv2.resize(mask*255, frame.shape[:2][::-1]).astype(np.uint8)

        # img_input = np.expand_dims(cv2.resize(frame, self.input_shape[1:]).astype(dtype="float32"), axis=0)

        # mask = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: img_input/255})[0][0]

        # segmentation_mask = cv2.resize(mask*255, frame.shape[:2][::-1]).astype(np.uint8)

        condition = np.stack((segmentation_mask,) * 3, axis=-1) > self.threshold

        if self.bg_image is not None:
            background = self.bg_image
        elif self.bg_color:
            background = np.ones(frame.shape, np.uint8)[...,:] * self.bg_color
        else:
            background = cv2.GaussianBlur(frame, self.bg_blur_ratio, 0)

        frame = np.where(condition, frame, cv2.resize(background, frame.shape[:2][::-1]))
 
        return frame.astype(np.uint8)