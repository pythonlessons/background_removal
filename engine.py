import cv2
import stow
import typing
import numpy as np
from tqdm import tqdm 

from selfieSegmentation import MPSegmentations

class Engine:
    """Object to process webcam stream, video source or images
    All the processing can be customized and enchanced with custom_objects
    """
    def __init__(
        self, 
        image_path: str = "",
        video_path: str = "", 
        webcam_id: int = 0,
        show: bool = False,
        flip_view: bool = False,
        custom_objects: typing.Iterable = [],
        ) -> None:
        """Initialize Engine object for further processing

        Args:
            image_path: (str) - path to image to process
            video_path: (str) - path to video to process
            webcam_id: (int) - ID of the webcam to process
            show: (bool) - argument whether to display or not processing
            flip_view: (bool) - argument whether to flip view horizontally or not
            custom_objects: (typing.Iterable) - custom objects to call every iteration (must have call function)
        """
        self.video_path = video_path
        self.image_path = image_path
        self.webcam_id = webcam_id
        self.show = show
        self.flip_view = flip_view
        self.custom_objects = custom_objects

    def flip(self, frame: np.ndarray) -> np.ndarray:
        """Flip given frame horizontally
        Args:
            frame: (np.ndarray) - frame to be fliped horizontally

        Returns:
            frame: (np.ndarray) - fliped frame if self.flip_view = True
        """
        if self.flip_view:
            return cv2.flip(frame, 1)

        return frame

    def custom_processing(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with custom objects (custom object must have call function for each iteration)
        Args:
            frame: (np.ndarray) - custom processed frame

        Returns:
            frame: (np.ndarray) - custom processed frame
        """
        if self.custom_objects:
            for custom_object in self.custom_objects:
                frame = custom_object(frame)

        return frame

    def display(self, frame: np.ndarray, webcam: bool = False) -> bool:
        """Display current frame if self.show = True

        Args:
            frame: (np.ndarray) - frame to be displayed
            webcam: (bool) - Add aditional function for webcam. Keyboard 'a' for next or 'd' for previous

        Returns:
            (bool) - Teturn True if no keyboard "Quit" interruption
        """
        if self.show:
            cv2.imshow('Remove Background', frame)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return False

            if webcam:
                if k & 0xFF == ord('a'):
                    for custom_object in self.custom_objects:
                        # change background to next with keyboar 'a' button
                        if isinstance(custom_object, MPSegmentations):
                            custom_object.change_image(True)
                elif k & 0xFF == ord('d'):
                    for custom_object in self.custom_objects:
                        # change background to previous with keyboar 'd' button
                        if isinstance(custom_object, MPSegmentations):
                            custom_object.change_image(False)

        return True

    def process_image(self) -> np.ndarray:
        """Function do to processing with given image in image_path

        Returns:
            frame: (np.ndarray) - final processed image
        """
        if not stow.exists(self.image_path):
            raise Exception(f"Given image path doesn't exists {self.image_path}")

        frame = self.custom_processing(self.flip(cv2.imread(self.image_path)))

        extension = stow.extension(self.image_path)
        output_path = self.image_path.replace(f".{extension}", f"_out.{extension}")
        cv2.imwrite(output_path, frame)

        return frame

    def process_webcam(self) -> None:
        """Process webcam stream for given webcam_id
        """
        # Create a VideoCapture object for given webcam_id
        cap = cv2.VideoCapture(self.webcam_id)
        while cap.isOpened():  
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = self.custom_processing(self.flip(frame))

            if not self.display(frame, webcam=True):
                break

        else:
            raise Exception(f"Webcam with ID ({self.webcam_id}) can't be opened")

        cap.release()

    def process_video(self) -> None:
        """Process video for given video_path and creates processed video in same path
        """
        if not stow.exists(self.video_path):
            raise Exception(f"Given video path doesn't exists {self.video_path}")

        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(self.video_path)

        # Check if camera opened successfully
        if not cap.isOpened():
            raise Exception(f"Error opening video stream or file {self.video_path}")

        # Capture video details
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer in the same location as original video
        output_path = self.video_path.replace(f".{stow.extension(self.video_path)}", "_out.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

        # Read all frames from video
        for _ in tqdm(range(frames)):
            # Capture frame-by-frame
            success, frame = cap.read()
            if not success:
                break

            frame = self.custom_processing(self.flip(frame))

            out.write(frame)

            if not self.display(frame):
                break

        cap.release()
        out.release()

    def run(self):
        """Main object function to start processing image, video or webcam
        """
        if self.video_path:
            self.process_video()
        elif self.image_path:
            self.process_image()
        else:
            self.process_webcam()