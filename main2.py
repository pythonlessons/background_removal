from faceDetection import MPFaceDetection
from faceNet.faceNet import FaceNet
from locker import Locker

import cv2

locker = Locker()
facenet = FaceNet(
    detector=MPFaceDetection(),
    onnx_model_path="models/faceNet.onnx",
    anchors="faces",
    threshold=0.3,
    force_cpu=True,
)

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    while cap.isOpened():  
        success, frame = cap.read()
        if not success or frame is None:
            print("Ignoring empty camera frame.")
            continue

        frame, face_crops = facenet(frame, draw=True)

        cv2.imshow('Video', frame)

        locker.onFaceNetPipeline(face_crops)
        if locker.deviceLocked:
            print("Device locked")

            # save last frame
            print("Saving last frame")
            # cv2.imwrite("last_frame.jpg", frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting")
            # Save last quiting frame
            cv2.destroyAllWindows()
            break

    cap.release()

