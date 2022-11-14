from utils import FPSmetric
from engine import Engine
# from selfieSegmentation import MPSegmentation
# from faceDetection import MPFaceDetection
# from faceNet.faceNet import FaceNet
from uNetSegmentation import unetSegmentation

if __name__ == '__main__':
    fpsMetric = FPSmetric()
    segmentationModule = unetSegmentation(threshold=0.5, bg_images_path='', bg_blur_ratio=(45, 45), force_cpu=True, bg_color=(255,255,255))
    selfieSegmentation = Engine(webcam_id=0, show=True, custom_objects=[segmentationModule, fpsMetric])
    selfieSegmentation.run()


    # segmentationModule = MPSegmentation(threshold=0.3, bg_images_path='', bg_blur_ratio=(45, 45))
    # facenet = FaceNet(
    #     detector = MPFaceDetection(),
    #     onnx_model_path = "models/faceNet.onnx", 
    #     anchors = "faces",
    #     threshold=0.3,
    #     force_cpu = True,
    # )
    # engine = Engine(webcam_id=0, show=True, custom_objects=[segmentationModule, facenet, FPSmetric()])

    # # save first face crop as anchor, otherwise don't use
    # while not facenet.detect_save_faces(engine.process_webcam(return_frame=True), output_dir="faces"):
    #     continue

    # engine.run()