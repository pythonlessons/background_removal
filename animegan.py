# https://github.com/TachibanaYoshino/AnimeGANv3
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import mediapipe as mp
import time

import numpy as np
import onnxruntime as ort


def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def post_precess(img, wh):
    img = (img.squeeze()+1.) / 2 * 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (wh[0], wh[1]))
    return img

providers = ['CUDAExecutionProvider']# , 'CPUExecutionProvider']

# ort_sess = ort.InferenceSession('Shinkai_53.onnx', providers=providers)
ort_sess = ort.InferenceSession('AnimeGANv3_PortraitSketch_25.onnx', providers=providers)
# outputs = ort_sess.run(None, {'input': x.numpy()})

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# For webcam input:
BG_COLOR = (192, 192, 192) # gray
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX
frame_time = 0
prev_frame_time = 0
fps_list = []
with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    bg_image = None
    while cap.isOpened():   
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        # image.flags.writeable = False

        image.flags.writeable = False
        results = selfie_segmentation.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        threshold = 0.5
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > threshold

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)

        new_img = process_image(output_image)
        # outputs = ort_sess.run(None, {'generator_input:0': np.expand_dims(new_img, axis=0)})
        outputs = ort_sess.run(None, {'animeganv3_input:0': np.expand_dims(new_img, axis=0)})
        anime_img = post_precess(outputs[0], (output_image.shape[1], output_image.shape[0]))

        anime_img = cv2.cvtColor(anime_img, cv2.COLOR_RGB2BGR)
        output_image = np.where(condition, anime_img, bg_image)
        
        prev_frame_time = frame_time
        frame_time = time.time()
        fps = int(1/(frame_time - prev_frame_time))
        fps_list.append(fps)
        fps_list = fps_list[-30:]
        cv2.putText(output_image, str(int(np.average(fps_list))), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('MediaPipe Selfie Segmentation', output_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()