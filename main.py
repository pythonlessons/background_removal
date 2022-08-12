# https://google.github.io/mediapipe/solutions/selfie_segmentation.html
# https://github.com/cvzone/cvzone/blob/master/cvzone/SelfiSegmentationModule.py
import cv2
import mediapipe as mp
import numpy as np
import time
import onnxruntime as ort
import tf2onnx
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="selfie_segmentation.tflite")
interpreter.allocate_tensors()


class Fps_metric:
    """ Measure FPS between calls of this funtion
    """
    def __init__(self, range_average: int = 30):
        self._range_average = range_average
        self._frame_time = 0
        self._prev_frame_time = 0
        self._fps_list = []

    def __call__(self):
        self._prev_frame_time = self._frame_time
        self._frame_time = time.time()
        if not self._prev_frame_time:
            return 0
        self._fps_list.append(1/(self._frame_time - self._prev_frame_time))
        self._fps_list = self._fps_list[-self._range_average:]
        return float(np.average(self._fps_list))

# mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# For webcam input:
BG_COLOR = (192, 192, 192) # gray
cap = cv2.VideoCapture(0)
fps = Fps_metric()
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
# tf2onnx
# tf2onnx.convert.from_tflite()

ort_sess = ort.InferenceSession('selfie_segmentation.onnx', providers=['CPUExecutionProvider'])

# with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
bg_image = None
while cap.isOpened():   
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.flip(image, 1)
    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    # image.flags.writeable = False
    # results = selfie_segmentation.process(image)
    # image.flags.writeable = False
    results = selfie_segmentation.process(image)
    r = ort_sess.run(None, {'animeganv3_input:0': image})
    # image.flags.writeable = True

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    threshold = 0.5
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > threshold

    if bg_image is None:
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
    output_image = np.where(condition, image, bg_image)
    

    cv2.putText(output_image, str(int(fps())), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    # condition = np.stack(
    #   (results.segmentation_mask,) * 3, axis=-1) > 0.1
    # The background can be customized.
    #   a) Load an image (with the same width and height of the input image) to
    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
    #   b) Blur the input image by applying image filtering, e.g.,
    #      bg_image = cv2.GaussianBlur(image,(55,55),0)
    # if bg_image is None:
    #   bg_image = np.zeros(image.shape, dtype=np.uint8)
    #   bg_image[:] = BG_COLOR
    # output_image = np.where(condition, image, bg_image)

    cv2.imshow('MediaPipe Selfie Segmentation', output_image)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        print('q')
        break
    elif k & 0xFF == ord('a'):
        print('a')
    elif k & 0xFF == ord('d'):
        print('d')
    # if cv2.waitKey(5) & 0xFF == 27:
    #     break

cap.release()