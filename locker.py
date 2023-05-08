import time
import os
import dbus
from engine import Engine

engine = Engine()

class Locker:
    def __init__(self):
        self.noRecognizedFacesFrom = None
        self.isRecognizedFacesFrom = None
        self.deviceLocked = False
        self.screenSaverSetActive = None
        self.emptyFaceCropTime = None
        session_bus = dbus.SessionBus()
        proxy_obj = session_bus.get_object('org.gnome.ScreenSaver', '/org/gnome/ScreenSaver')
        interface = dbus.Interface(proxy_obj, 'org.gnome.ScreenSaver')
        self.screenSaverSetActive = interface.get_dbus_method("SetActive")
    
    def onFaceNetPipeline(self, face_crops) -> None:
        if self.isFaceCropsEmpty(face_crops):
            self.noRecognizedFacesFrom = None
            self.isRecognizedFacesFrom = None
            if self.emptyFaceCropTime is None:
                self.emptyFaceCropTime = time.time()

            secondsSinceEmptyFaceCrop = time.time() - self.emptyFaceCropTime
            if secondsSinceEmptyFaceCrop > 10 and not self.deviceLocked:
                self.lockDevice2()
                self.emptyFaceCropTime = None

            return

        self.emptyFaceCropTime = None

        if self.doesFaceCropContainKnownFaces(face_crops):
            self.noRecognizedFacesFrom = None

            if self.isRecognizedFacesFrom is None:
                self.isRecognizedFacesFrom = time.time()

            secondsSinceKnownFace = time.time() - self.isRecognizedFacesFrom
            if secondsSinceKnownFace > 2 and self.deviceLocked:
                self.unlockDevice()

            return

        if self.noRecognizedFacesFrom is None:
            self.noRecognizedFacesFrom = time.time()
            self.isRecognizedFacesFrom = None
            return

        secondsSinceNoKnownFace = time.time() - self.noRecognizedFacesFrom
        if secondsSinceNoKnownFace > 2 and not self.deviceLocked:
            self.lockDevice()

    def isFaceCropsEmpty(self, face_crops) -> bool:
        return not bool(face_crops)

    def doesFaceCropContainKnownFaces(self, face_crops) -> bool:
        for key in face_crops:
            face_crop = face_crops[key]
            if face_crop['name'] != 'Unknown':
                return True

        return False
    
    def lockDevice(self):
        print("Locking user screen")
        self.deviceLocked = True
        #pause
        #capture
        #unpause
        #engine.capture_webcam(save_dir="unauthorized/")
        #os.system('gnome-screensaver-command -l')

    def lockDevice2(self):
        print("Locking user screen because of inactivity")
        self.deviceLocked = True
        #os.system('gnome-screensaver-command -l')

    def unlockDevice(self):
        print("Unlocking user screen")
        self.deviceLocked = False
        #self.screenSaverSetActive(False)
