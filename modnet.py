



import cv2
import stow
import numpy as np
from tqdm import tqdm
import threading


def normalise(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # https://blog.csdn.net/qq_40035462/article/details/123786809
    image = image.astype(np.float32)
    image = image / 255.0
    image = (image - mean) / std
    return image

def preprocess_frame(frame):
    w = 960
    h = 544

    # results = self.selfie_segmentation.process(frame)
    # img = cv2.resize(frame, (w, h), cv2.INTER_AREA)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = normalise(img)
    img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    return img

import onnxruntime as ort
def create_ort_session():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    onnx_model_path: str = "models/modnet.onnx"
    ort_sess = ort.InferenceSession(onnx_model_path, providers=providers)

    return ort_sess

ort_sess1 = create_ort_session()
# ort_sess2 = create_ort_session()
# ort_sess3 = create_ort_session()
# ort_sess4 = create_ort_session()

# def threading_predict(ort_sess, frames):
#     preds = ort_sess.run(None, {ort_sess._inputs_meta[0].name: frames})[0]

#     return preds

def process_frame_batch(frame_batch):
    processed_frames = []
    frames_to_predict = np.array([preprocess_frame(frame) for frame in frame_batch]).astype(np.float32)
    


    preds = ort_sess1.run(None, {ort_sess1._inputs_meta[0].name: frames_to_predict[0]})[0] # [0][0]
    
    for frame, pred in zip(frame_batch, preds):
        
        mask = pred[0]

        matting = cv2.resize(mask, (frame.shape[1], frame.shape[0]), cv2.INTER_NEAREST)
        matting = np.expand_dims(matting, axis=-1)

        background = np.ones(frame.shape, np.uint8)[...,:] * (0,255,0)
        bg_frame = cv2.resize(background, frame.shape[:2][::-1])

        final_frame = matting * frame + (1 - matting) * bg_frame

        processed_frames.append(final_frame.astype(np.uint8))

    return processed_frames




def do_the_job(video_path, batch_size=8):
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if not cap.isOpened():
        raise Exception(f"Error opening video stream or file {video_path}")

    # Capture video details
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer in the same location as original video
    output_path = video_path.replace(f".{stow.extension(video_path)}", f"_output.avi")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, (width, height))

    frame_batch = []
    for fnum in tqdm(range(frames)):

        success, frame = cap.read()
        if not success:
            break

        # out.write(frame)
        frame_batch.append(frame)

        if len(frame_batch) == batch_size:
            processed_frames = process_frame_batch(frame_batch)
            for processed_frame in processed_frames:
                out.write(processed_frame)
                frame_batch = []
        else:
            continue

    if len(frame_batch) > 0:
        processed_frames = process_frame_batch(frame_batch)
        for processed_frame in processed_frames:
            out.write(processed_frame)

    cap.release()
    out.release()

do_the_job("C:/Users/rokas/Videos/2023-01-05 15-31-49-webcam.mkv")