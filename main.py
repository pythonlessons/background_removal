import argparse
from utils import FPSmetric
from engine import Engine
from faceDetection import MPFaceDetection
from faceNet.faceNet import FaceNet
from locker import Locker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', default=False, type=bool, help='Show camera feed')
    args = parser.parse_args()

    facenet = FaceNet(
        locker=Locker(),
        detector=MPFaceDetection(),
        onnx_model_path="models/faceNet.onnx",
        anchors="faces",
        threshold=0.3,
        force_cpu=True,
    )
    engine = Engine(webcam_id=0, show=args.qshow, custom_objects=[facenet, FPSmetric()])

    engine.run()
