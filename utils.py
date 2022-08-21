import numpy as np
import typing
import time
import cv2

class FPSmetric:
    """ Measure FPS between calls of this object
    """
    def __init__(
        self, 
        range_average: int = 30,
        position: typing.Tuple[int, int] = (7, 70),
        fontFace: int = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale: int = 3,
        color: typing.Tuple[int, int, int] = (100, 255, 0),
        thickness: int = 3,
        lineType: int = cv2.LINE_AA,
        ):
        """
        Args:
            range_average: (int) = 30 - number of how many call should be averaged for a result
            position: (typing.Tuple[int, int]) = (7, 70) - position in a frame where to put text
            fontFace: (int) = cv2.FONT_HERSHEY_SIMPLEX - cv2 font for text
            fontScale: (int) = 3 - size of font
            color: (typing.Tuple[int, int, int]) = (100, 255, 0) - RGB color for text
            thickness: (int) = 3 - chickness for text
            lineType: (int) = cv2.LINE_AA - text line type
        """
        self._range_average = range_average
        self._frame_time = 0
        self._prev_frame_time = 0
        self._fps_list = []

        self.position = position
        self.fontFace = fontFace
        self.fontScale = fontScale
        self.color = color
        self.thickness = thickness
        self.lineType = lineType

    def __call__(self, frame: np.ndarray = None) -> typing.Union[bool, np.ndarray]:
        """Measure duration between each call and return calculated FPS or frame with added FPS on it

        Args:
            frame: (np.ndarray) - frame to add FPS text if wanted

        Returns:
            fps: (float) - fps number if frame not given otherwise return frame (np.ndarray)
        """
        self._prev_frame_time = self._frame_time
        self._frame_time = time.time()
        if not self._prev_frame_time:
            return 0
        self._fps_list.append(1/(self._frame_time - self._prev_frame_time))
        self._fps_list = self._fps_list[-self._range_average:]
        
        fps = float(np.average(self._fps_list))

        if frame is None:
            return fps

        cv2.putText(frame, str(int(fps)), self.position, self.fontFace, self.fontScale, self.color, self.thickness, self.lineType)
        return frame