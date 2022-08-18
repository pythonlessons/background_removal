import numpy as np
import typing
import time
import cv2

class FPSmetric:
    """ Measure FPS between calls of this funtion
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

    def __call__(self, frame=None) -> float:
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

def process_image(img: np.ndarray, x32=True) -> np.ndarray:
    h, w = img.shape[:2]
    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def post_precess(img: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
    img = (img.squeeze()+1.) / 2 * 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (wh[0], wh[1]))
    return img