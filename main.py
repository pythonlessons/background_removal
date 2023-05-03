import argparse
import threading
from utils import FPSmetric
from engine import Engine
from faceDetection import MPFaceDetection
from faceNet.faceNet import FaceNet
from locker import Locker

def update_show(engine):
    while True:
        input_str = input('Enter "s" to toggle camera feed on/off: ')
        if input_str.lower() == 's':
            engine.show = not engine.show

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
    engine = Engine(webcam_id=0, show=args.show, custom_objects=[facenet, FPSmetric()])

    show_thread = threading.Thread(target=update_show, args=(engine,))
    show_thread.daemon = True
    show_thread.start()

    engine.run()
