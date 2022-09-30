import numpy as np
import cv2

class Adjust_gamma:
    def __init__(self, gamma: float = 1.0) -> None:
        """build a lookup table mapping the pixel values [0, 255] to
        their adjusted gamma values
        
        Args:
            gamma: (float) - value to adjust gamma with
        """
        self.invGamma = 1.0 / abs(gamma)
        self.table = np.array([((i / 255.0) ** self.invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def __call__(self, image: np.ndarray):
        """apply gamma correction using the lookup table

        Args:
            image: (np.ndarray) - image to which to apply the gamma adjust

        Return:
            image: (np.ndarray) - image with adjusted gamma
        """    
        return cv2.LUT(image, self.table)