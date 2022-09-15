from utils import FPSmetric
from selfieSegmentation import MPSegmentation
from faceDetection import MPFaceDetection
from engine import Engine
from animegan import AnimeGAN

if __name__ == '__main__':
    fpsMetric = FPSmetric()
    animegan = AnimeGAN("models/Shinkai_53.onnx")
    #segmentationModule = MPSegmentation(threshold=0.3, bg_images_path='', bg_blur_ratio=(45, 45))
    #selfieSegmentation = Engine(webcam_id=0, show=True, custom_objects=[segmentationModule, fpsMetric])
    # selfieSegmentation = Engine(image_path="C:/Users/rokas/Desktop/PyLessons Videos/Remove-background/01_Selfie_segmentation/01_Selfie_segmentation/RM_Background.png", 
    #                             show=True, custom_objects=[segmentationModule, animegan], output_extension="anime")

    mpFaceDetector = MPFaceDetection() 
    selfieSegmentation = Engine(webcam_id=0, show=True, custom_objects=[mpFaceDetector, fpsMetric])
    selfieSegmentation.run()