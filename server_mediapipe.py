from ctypes import ArgumentError
import socket
from collections.abc import Callable
import cv2
from cv2.typing import MatLike
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
from typing import Any
from spell_detection import identifySpell

def detect(
        cam:cv2.VideoCapture, 
        preprocess,
        predict:Callable[[MatLike],Any],
        # listeners: hand key points, raw_img
        listeners:list[Callable[[list[list[float]],MatLike],None]],
):
    if len(listeners) == 0:
        raise ArgumentError("No listeners")

    while cam.isOpened():
        success, raw_img = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        raw_img = cv2.flip(raw_img, 1)
        img = preprocess(raw_img)
        result = predict(img)

        # Note: if multiple hands are detected, listeners can't handle multiple in one iteration
        if len(result.hand_landmarks):
            for hand_landmarks in result.hand_landmarks:
                pts = [
                    [l.x, l.y, l.z] for l in hand_landmarks
                ]
                for listener in listeners:
                    listener(pts, raw_img)
        else:
            for listener in listeners:
                listener([], raw_img)

        # if escape is pressed, break the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

W=640
H=480
FPS=30
def main():
    '''Run if main module'''
    while True:
        cam = cv2.VideoCapture(0)
        # -- Workaround for wsl2
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cam.set(cv2.CAP_PROP_FPS, FPS)
        # -- end of workaround
        # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # cam.set(3, 1920)
        # cam.set(4, 1080)
        detect(
            cam,
            preprocess=preprocess,
            predict=predictWithGestureRecognizer,
            listeners=[
                handRenderer,
                identifySpell,
            ],
        )
        cam.release()
        cv2.destroyAllWindows()
        input("Press enter to run again")

VisionRunningMode = mp.tasks.vision.RunningMode
def preprocess(img:MatLike):
    return mp.Image(
        image_format=mp.ImageFormat.SRGB, 
        # data=cv2.flip(img,1),
        data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # data=img,
    )

def predictWithGestureRecognizer(mp_image:mp.Image):
    return recognizer.recognize(mp_image)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
HandLandMarkerResult = mp.tasks.vision.HandLandmarkerResult

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip = "172.26.0.1"
port = 4242
ip_port = (ip,port)
def sendToClient(pts:list[list[float]],_):
    client_socket.sendto(str(pts).encode("ascii"), ip_port)
 
def handRenderer(pts:list[list[float]], raw_img):
    if len(pts):
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=pt[0], y=pt[1], z=pt[2]) for pt in pts
        ])

        mp_drawing.draw_landmarks(
            raw_img,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    cv2.imshow("Webcam", raw_img)

options = vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(
        # model_asset_path='gesture_recognizer.task',
        # hand_landmarker
        model_asset_path='gesture_recognizer.task',
        delegate=mp.tasks.BaseOptions.Delegate.GPU,
    ),
    min_hand_detection_confidence=0.4,
    min_tracking_confidence=0.5,
    running_mode=VisionRunningMode.IMAGE,
    # canned_gestures_classifier_options=["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"],
    num_hands=1,
)
recognizer = vision.GestureRecognizer.create_from_options(options)
if __name__ == '__main__':
    main()
