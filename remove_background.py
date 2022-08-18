from tqdm import tqdm 
from utils import FPSmetric, process_image, post_precess
import cv2
import mediapipe as mp
import numpy as np
import typing
import stow

class SegmentationModule:
    """
    """
    def __init__(
        self,
        bg_color: typing.Tuple[int, int, int] = (192, 192, 192), # gray
        bg_image: typing.Optional[np.ndarray] = None,
        threshold: float = 0.5,
        model_selection: bool = 1,
        bg_images_path: str = None,
        ) -> None:
        """
        """
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection)

        self.bg_color = bg_color
        self.bg_image = bg_image
        self.threshold = threshold

        if bg_images_path:
            self.bg_images = [cv2.imread(image.path) for image in stow.ls(bg_images_path)]
            self.bg_image = self.bg_images[0]

    def change_image(self, prevOrNext: bool = True):
        if not self.bg_images:
            return False

        if prevOrNext:
            self.bg_images = self.bg_images[1:] + [self.bg_images[0]]
        else:
            self.bg_images = [self.bg_images[-1]] + self.bg_images[:-1]
        self.bg_image = self.bg_images[0]
        return True

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        """
        results = self.selfie_segmentation.process(frame)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > self.threshold

        if self.bg_image is None:
            self.bg_image = np.zeros(frame.shape, dtype=np.uint8)
            self.bg_image[:] = self.bg_color
    
        frame = np.where(condition, frame, cv2.resize(self.bg_image, frame.shape[:2][::-1]))
 
        return frame


class SelfieSegmentation:
    """
    """
    def __init__(
        self, 
        video_path: str = "", 
        image_path: str = "",
        webcam_id: int = 0,
        show: bool = False,
        flip_view: bool = False,
        custom_objects = [],
        ) -> None:
        """
        """
        self.video_path = video_path
        self.image_path = image_path
        self.webcam_id = webcam_id
        self.show = show
        self.flip_view = flip_view
        self.custom_objects = custom_objects

    def process_image(self):
        """
        """
        frame = cv2.imread(self.image_path)

        if self.flip_view:
            frame = cv2.flip(frame, 1)

        for custom_object in self.custom_objects:
            frame = custom_object(frame)

        extension = stow.extension(self.image_path)
        output_path = self.image_path.replace(f".{extension}", f"_out.{extension}")
        cv2.imwrite(output_path, frame)

    def process_webcam(self):
        """
        """
        cap = cv2.VideoCapture(self.webcam_id)
        while cap.isOpened():  
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            if self.flip_view:
                frame = cv2.flip(frame, 1)

            for custom_object in self.custom_objects:
                frame = custom_object(frame)

            if self.show:
                cv2.imshow('Remove Background', frame)
                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    break
                elif k & 0xFF == ord('a'):
                    for custom_object in self.custom_objects:
                        if type(custom_object).__name__ == 'SegmentationModule':
                            custom_object.change_image(True)
                elif k & 0xFF == ord('d'):
                    for custom_object in self.custom_objects:
                        if type(custom_object).__name__ == 'SegmentationModule':
                            custom_object.change_image(False)

        cap.release()
        cv2.destroyAllWindows()

    def process_video(self):
        """
        """
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(self.video_path)

        # Check if camera opened successfully
        if not cap.isOpened():
            raise Exception(f"Error opening video stream or file {self.video_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = self.video_path.replace(f".{stow.extension(self.video_path)}", "_out.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

        # Read all frames from video
        for _ in tqdm(range(frames)):
            # Capture frame-by-frame
            success, frame = cap.read()
            if not success:
                break

            if self.flip_view:
                frame = cv2.flip(frame, 1)

            for custom_object in self.custom_objects:
                frame = custom_object(frame)

            out.write(frame)

            if self.show:
                cv2.imshow('Remove Background', frame)
                k = cv2.waitKey(1)
                if k & 0xFF == ord('q'):
                    print('q')
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def run(self):
        """
        """
        if self.video_path:
            self.process_video()
        elif self.image_path:
            self.process_image()
        else:
            self.process_webcam()


if __name__ == '__main__':
    fpsMetric = FPSmetric()
    segmentationModule = SegmentationModule(threshold=0.5, bg_image_path='backgrounds')
    # selfieSegmentation = SelfieSegmentation(video_path='Selfie_video.mkv', show=True, custom_objects=[segmentationModule])
    selfieSegmentation = SelfieSegmentation(webcam_id=0, show=True, flip_view=True, custom_objects=[segmentationModule, fpsMetric])
    selfieSegmentation.run()