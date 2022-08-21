import cv2
import stow
import typing
import numpy as np
import mediapipe as mp

class MPSegmentations:
    """Object to create and do mediapipe selfie segmentation, more about it:
    https://google.github.io/mediapipe/solutions/selfie_segmentation.html
    """
    def __init__(
        self,
        bg_color: typing.Tuple[int, int, int] = (192, 192, 192), # gray
        bg_image: typing.Optional[np.ndarray] = None,
        threshold: float = 0.5,
        model_selection: bool = 1,
        bg_images_path: str = None,
        ) -> None:
        """
        Args:
            bg_color: (typing.Tuple) = (192, 192, 192) - background RGB color for removed background, gray is default
            bg_image: (typing.Optional) = None - background color to use instead of gray color in background
            threshold: (float) = 0.5 - accuracy border threshold seperating background and foreground, need to play to get best results
            model_selection: (bool) = 1 - generas or landscape model selection for segmentations mask
            bg_images_path: (str) = None - path to folder for background images
        """
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection)

        self.bg_color = bg_color
        self.bg_image = bg_image
        self.threshold = threshold

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

        if self.bg_image is None:
            self.bg_image = np.zeros(frame.shape, dtype=np.uint8)
            self.bg_image[:] = self.bg_color
    
        frame = np.where(condition, frame, cv2.resize(self.bg_image, frame.shape[:2][::-1]))
 
        return frame