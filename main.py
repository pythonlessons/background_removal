from utils import FPSmetric

from selfieSegmentation import MPSegmentations
from faceDetection import MPFaceDetection
from engine import Engine
from animeGAN import AnimeGAN


if __name__ == '__main__':
    fpsMetric = FPSmetric()
    segmentationModule = MPSegmentations(threshold=0.5, bg_images_path='')
    faceDetector = MPFaceDetection()
    animeGan = AnimeGAN('models\Shinkai_53.onnx')

    # selfieSegmentation = Engine(video_path='Selfie_video.mkv', show=True, custom_objects=[segmentationModule])
    selfieSegmentation = Engine(webcam_id=0, show=True, flip_view=True, custom_objects=[animeGan, fpsMetric])
    selfieSegmentation.run()