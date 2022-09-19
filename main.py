from utils import FPSmetric
from faceDetection import MPFaceDetection
from engine import Engine

if __name__ == '__main__':
    fpsMetric = FPSmetric()
    mpFaceDetector = MPFaceDetection() 
    selfieSegmentation = Engine(webcam_id=0, show=True, custom_objects=[mpFaceDetector, fpsMetric])
    selfieSegmentation.run()