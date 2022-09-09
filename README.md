# Background removal with Python

All this repository is for learning purposes. I cover here how simple is to remove background from selfie view just like Zoom, Google Meets, Skype, and MS Teams.

## Installation on Windows:
- Clone this repository. (don't forget to Star it)
- Install virtual environment: ```python -m venv venv```
- Activate virtual environment: ```venv\Scripts\activate```
- Install all the requirements: ```pip install -r requirements.txt```
- (Optional if have Nvidia GPU): install onnxruntime with GPU support: ```pip install onnxruntime-gpu```

## How to run basic background removal:
At this point, when you are looking at this project, I might be already updated this project with more features, but if you want only to run a quick test on your own webcam replace ```the main.py``` code with the following:
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

## Detailed Tutorials:
- First One
