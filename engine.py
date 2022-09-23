import cv2
import stow
import typing
import numpy as np
from tqdm import tqdm 

from selfieSegmentation import MPSegmentation

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
        output_extension: str = 'out',
        start_video_frame: int = 0,
        end_video_frame: int = 0,
        break_on_end: bool = False,
        ) -> None:
        """Initialize Engine object for further processing

        Args:
            image_path: (str) - path to image to process
            video_path: (str) - path to video to process
            webcam_id: (int) - ID of the webcam to process
            show: (bool) - argument whether to display or not processing
            flip_view: (bool) - argument whether to flip view horizontally or not
            custom_objects: (typing.Iterable) - custom objects to call every iteration (must have call function)
            output_extension: (str) - additional text to add to processed image or video when saving output
            start_video_frame: (int) - video frame from which to start applying custom_objects to video
            end_video_frame: (int) - last video frame to which apply custom_objects to video
        """
        self.video_path = video_path
        self.image_path = image_path
        self.webcam_id = webcam_id
        self.show = show
        self.flip_view = flip_view
        self.custom_objects = custom_objects
        self.output_extension = output_extension
        self.start_video_frame = start_video_frame
        self.end_video_frame = end_video_frame
        self.break_on_end = break_on_end

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
            frame: (np.ndarray) - frame to apply custom processing to

        Returns:
            frame: (np.ndarray) - custom processed frame
        """
        if self.custom_objects:
            for custom_object in self.custom_objects:
                frame = custom_object(frame)

        return frame

    def display(self, frame: np.ndarray, webcam: bool = False, waitTime: int = 1) -> bool:
        """Display current frame if self.show = True
        When displaying webcam you can control the background images

        Args:
            frame: (np.ndarray) - frame to be displayed
            webcam: (bool) - Add aditional function for webcam. Keyboard 'a' for next or 'd' for previous

        Returns:
            (bool) - Teturn True if no keyboard "Quit" interruption
        """
        if self.show:
            cv2.imshow('Remove Background', frame)
            k = cv2.waitKey(waitTime)
            if k & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return False

            if webcam:
                if k & 0xFF == ord('a'):
                    for custom_object in self.custom_objects:
                        # change background to next with keyboar 'a' button
                        if isinstance(custom_object, MPSegmentation):
                            custom_object.change_image(True)
                elif k & 0xFF == ord('d'):
                    for custom_object in self.custom_objects:
                        # change background to previous with keyboar 'd' button
                        if isinstance(custom_object, MPSegmentation):
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
        output_path = self.image_path.replace(f".{extension}", f"_{self.output_extension}.{extension}")
        cv2.imwrite(output_path, frame)

        self.display(frame, waitTime=0)

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

    def check_video_frames_range(self, fnum):
        """Not to waste resources this function processes only specified range of video frames

        Args:
            fnum: (int) - current video frame number

        Returns:
            status: (bool) - Return True if skip processing otherwise False
        """
        if self.start_video_frame and fnum < self.start_video_frame:
            return True

        if self.end_video_frame and fnum > self.end_video_frame:
            return True
        
        return False

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
        output_path = self.video_path.replace(f".{stow.extension(self.video_path)}", f"_{self.output_extension}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))

        # Read all frames from video
        for fnum in tqdm(range(frames)):
            # Capture frame-by-frame
            success, frame = cap.read()
            if not success:
                break

            if self.check_video_frames_range(fnum):
                out.write(frame)
                if self.break_on_end and fnum >= self.end_video_frame:
                    break
                continue

            frame = self.custom_processing(self.flip(frame))

            out.write(frame)

            if not self.display(frame):
                break

        cap.release()
        out.release()

    def run(self):
        """Main object function to start processing image, video or webcam input
        """
        if self.video_path:
            self.process_video()
        elif self.image_path:
            self.process_image()
        else:
            self.process_webcam()