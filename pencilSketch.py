import numpy as np
import cv2
import typing

class PencilSketch:
    """Apply pencil sketch effect to an image
    """
    def __init__(
        self,
        blur_simga: int = 5,
        ksize: typing.Tuple[int, int] = (0, 0),
        ) -> None:
        """
        Args:
            blur_simga: (int) - sigma ratio to apply for cv2.GaussianBlur
            ksize: (float) - ratio to apply for cv2.GaussianBlur
        """
        self.blur_simga = blur_simga
        self.ksize = ksize

    def dodge(self, front: np.ndarray, back: np.ndarray) -> np.ndarray:
        """The formula comes from http://www.adobe.com/devnet/pdf/pdfs/blend_modes.pdf
        Args:
            front: (np.ndarray) - front image to be applied to dodge algorithm
            back: (np.ndarray) - back image to be applied to dodge algorithm
        """
        result = back*255.0 / (255.0-front) 
        result[result>255] = 255
        result[back==255] = 255
        return result.astype('uint8')

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Main function to do pencil sketch

        Args:
            frame: (np.ndarray) - frame to excecute pencil sketch on

        Returns:
            frame: (np.ndarray) - processed frame that is pencil sketch type
        """
        grayscale = np.array(np.dot(frame[...,:3], [0.299, 0.587, 0.114]), dtype=np.uint8)
        grayscale = np.stack((grayscale,) * 3, axis=-1) # convert 1 channel grayscale image to 3 channels grayscale

        inverted_img = 255 - grayscale

        blur_img = cv2.GaussianBlur(inverted_img, ksize=self.ksize, sigmaX=self.blur_simga)

        final_img = self.dodge(blur_img, grayscale)
 
        return final_img