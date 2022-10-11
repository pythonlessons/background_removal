# Background removal with Python

All this repository is for learning purposes. I cover here how simple is to remove background from selfie view just like Zoom, Google Meets, Skype, and MS Teams.

## Installation on Windows:
- Clone this repository. (don't forget to Star it)
- Install virtual environment: ```python -m venv venv```
- Activate virtual environment: ```venv\Scripts\activate```
- Install all the requirements: ```pip install -r requirements.txt```
- (Optional if have Nvidia GPU): install onnxruntime with GPU support: ```pip install onnxruntime-gpu```

## How to run basic background removal:
At this point, when you are looking at this project, I might be already updated this project with more features, but if you want only to run a quick test on your own webcam replace the ```main.py``` code with the following:
```Python
# main.py
from utils import FPSmetric
from selfieSegmentation import MPSegmentations
from engine import Engine

if __name__ == '__main__':
    fpsMetric = FPSmetric()
    segmentationModule = MPSegmentations(threshold=0.3, bg_images_path='', bg_blur_ratio=(45, 45))
    selfieSegmentation = Engine(webcam_id=0, show=True, custom_objects=[segmentationModule, fpsMetric])
    selfieSegmentation.run()
```
You can run it by typing ```python main.py``` in a terminal.

## Run basic MediaPipe face detection:
```Python
# main.py
from utils import FPSmetric
from faceDetection import MPFaceDetection
from engine import Engine

if __name__ == '__main__':
    fpsMetric = FPSmetric()
    mpFaceDetector = MPFaceDetection() 
    selfieSegmentation = Engine(webcam_id=0, show=True, custom_objects=[mpFaceDetector, fpsMetric])
    selfieSegmentation.run()
```
You can run it by typing ```python main.py``` in a terminal.

## Test "Pencil" sketch with Python on saved image:
```Python
# main.py
from pencilSketch import PencilSketch
from engine import Engine

if __name__ == '__main__': 
    pencilSketch = PencilSketch(blur_simga=5)
    selfieSegmentation = Engine(image_path='data/porche.jpg', show=True, custom_objects=[pencilSketch])
    selfieSegmentation.run()
```
You can run it by typing ```python main.py``` in a terminal.

## Test facial recognition example on webcam stream
```Python
# main.py
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
```
You can run it by typing ```python main.py``` in a terminal.

## Detailed Tutorials:
- [Selfie background remove or blur with Python](https://pylessons.com/remove-background)
- [Real Time CPU face detection tutorial](https://pylessons.com/face-detection)
- [Pencil sketch image with Python](https://pylessons.com/pencil-sketch)
