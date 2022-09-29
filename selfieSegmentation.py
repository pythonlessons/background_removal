import cv2
import stow
import typing
import numpy as np
import mediapipe as mp

class MPSegmentation:
    """Object to create and do mediapipe selfie segmentation, more about it:
    https://google.github.io/mediapipe/solutions/selfie_segmentation.html
    """
    def __init__(
        self,
        bg_blur_ratio: typing.Tuple[int, int] = (35, 35),
        bg_image: typing.Optional[np.ndarray] = None,
        threshold: float = 0.5,
        model_selection: bool = 1,
        bg_images_path: str = None,
        bg_color : typing.Tuple[int, int, int] = None,
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
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection)

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
        results = self.selfie_segmentation.process(frame)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > self.threshold

        if self.bg_image is not None:
            background = self.bg_image
        elif self.bg_color:
            background = np.ones(frame.shape, np.uint8)[...,:] * self.bg_color
        else:
            background = cv2.GaussianBlur(frame, self.bg_blur_ratio, 0)

        frame = np.where(condition, frame, cv2.resize(background, frame.shape[:2][::-1]))
 
        return frame.astype(np.uint8)