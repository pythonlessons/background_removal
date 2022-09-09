from utils import FPSmetric
from selfieSegmentation import MPSegmentation
from engine import Engine

if __name__ == '__main__':
    fpsMetric = FPSmetric()
    segmentationModule = MPSegmentation(threshold=0.3, bg_images_path='', bg_blur_ratio=(45, 45))
    selfieSegmentation = Engine(webcam_id=0, show=True, custom_objects=[segmentationModule, fpsMetric])
    selfieSegmentation.run()