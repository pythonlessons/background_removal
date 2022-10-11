from utils import FPSmetric
from engine import Engine
from faceDetection import MPFaceDetection
from faceNet.faceNet import FaceNet

if __name__ == '__main__':
    facenet = FaceNet(
        detector = MPFaceDetection(),
        onnx_model_path = "models/faceNet.onnx", 
        anchors = "faces",
        force_cpu = True,
    )
    engine = Engine(webcam_id=0, show=True, custom_objects=[facenet, FPSmetric()])

    # save first face crop as anchor, otherwise don't use
    while not facenet.detect_save_faces(engine.process_webcam(return_frame=True), output_dir="faces"):
        continue

    engine.run()