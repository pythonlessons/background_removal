import time
import os
import dbus

class Locker:
    noRecognizedFacesFrom = None
    deviceLocked = False
    screenSaverSetActive = None

    def __init__(self):
        session_bus = dbus.SessionBus()
        proxy_obj = session_bus.get_object('org.gnome.ScreenSaver', '/org/gnome/ScreenSaver')
        interface = dbus.Interface(proxy_obj, 'org.gnome.ScreenSaver')
        self.screenSaverSetActive = interface.get_dbus_method("SetActive")
    
    def onFaceNetPipeline(self, face_crops) -> None:
        if self.isFaceCropsEmpty(face_crops):
            self.noRecognizedFacesFrom = None
            return

        if self.doesFaceCropContainKnownFaces(face_crops):
            self.noRecognizedFacesFrom = None

            if self.deviceLocked:
                self.unlockDevice()
            return

        if self.noRecognizedFacesFrom == None:
            self.noRecognizedFacesFrom = time.time()
            return

        secondsSinceNoKnownFace = time.time() - self.noRecognizedFacesFrom
        if secondsSinceNoKnownFace > 2 and self.deviceLocked == False:
            self.lockDevice()
            return

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
        self.deviceLocked = True;
        os.system('gnome-screensaver-command -l')

    def unlockDevice(self):
        print("Unlocking user screen")
        self.deviceLocked = False;
        self.screenSaverSetActive(False)


