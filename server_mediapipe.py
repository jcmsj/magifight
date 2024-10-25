import socket
from typing import Any, Callable

import cv2
from cv2.typing import MatLike

from saver import xyn_to_bitmap
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "172.26.0.1"
port = 4242
ip_port = (ip,port)
# Using ascii since i'm only sending numbers
client_socket.sendto("Hello from Python!".encode(encoding="ascii"), ip_port)
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
def detect(
        cam:cv2.VideoCapture, 
        winname:str, 
        onResult:Callable,
        predict:Callable,
        preprocess,
):
    while cam.isOpened():
        success, raw_img = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        raw_img = cv2.flip(raw_img, 1)
        img = preprocess(raw_img)
        result = predict(img)
        onResult(result,raw_img)

        # draw a 3pixel circle for each pt
        for pt in index_tip_pts:
            # draw an orange circle with a radius of 5 pixels and thickness of 8 pixels
            cv2.circle(raw_img, (int(pt[0]*raw_img.shape[1]), int(pt[1]*raw_img.shape[0])), 3, (0, 165, 255), 3)

        cv2.imshow(winname, raw_img)
        # if escape is pressed, break the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

W=640
H=480
FPS=30
def main():
    '''Run if main module'''
    while True:
        cam = cv2.VideoCapture(0)
        # -- Workaround for wsl2
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cam.set(cv2.CAP_PROP_FPS, FPS)
        # -- end of workaround
        # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # cam.set(3, 1920)
        # cam.set(4, 1080)
        detect(
            cam,
            "Webcam",
            preprocess=preprocess,
            predict=predictWithGestureRecognizer,
            onResult=onLandmarkResult,
        )
        input("Press enter to run again")


VisionRunningMode = mp.tasks.vision.RunningMode
# STEP 2: Create an HandLandmarker object.
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path='hand_landmarker.task',
        delegate=mp.tasks.BaseOptions.Delegate.GPU,
    ),
        num_hands=1
    )
landmarker = vision.HandLandmarker.create_from_options(options)

def preprocess(img:MatLike):
    return mp.Image(
        image_format=mp.ImageFormat.SRGB, 
        # data=cv2.flip(img,1),
        data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # data=img,
    )
def predictWithHandlandmarker(mp_image:mp.Image):
    return landmarker.detect(mp_image)

def predictWithGestureRecognizer(mp_image:mp.Image):
    return recognizer.recognize(mp_image)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
HandLandMarkerResult = mp.tasks.vision.HandLandmarkerResult

index_tip_pts = []
import numpy as np
import torch
from spell_classification import SpellClassifier,predict, im2Tensor
model = SpellClassifier(3)
model.load_state_dict(torch.load('model.ckpt',weights_only=True))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
from PIL import Image
def pred(pts:list[list[float]]):
    grid = xyn_to_bitmap(pts)
    # Convert grid to PIL grayscale
    grid = Image.fromarray(grid).convert('L')
    # save to temp.bmp
    grid.save("temp.bmp")
    grid = im2Tensor(grid)
    grid = grid.unsqueeze(0)
    return predict(model, device, grid)

import time
debounce = time.time()
last_spell = ''
text = ['Wingardium Leviosa', 'Protego', 'Stupefy']
def onLandmarkResult(result, raw_img):
    global debounce, last_spell,text
    for hand_landmarks in result.hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        mp_drawing.draw_landmarks(
            raw_img,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        # NormalizedLandmark(x=0.3741799592971802, y=0.5949898958206177, z=1.317286688617969e-07, visibility=0.0, presence=0.0), 
        # convert to a matrix of [[x,y,z]]
        pts = [
            [landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks
        ]
        client_socket.sendto(str(pts).encode("ascii"), ip_port)

        # if the rest of the fingers are closed, then the user is pointing
        # if all pts 12,16,20 have y less than 9,13,17 respectively, then add pt 8 to the list
        # alt: using INDEX_FINGER_PIP(pt 6), check if the rest of the fingertips are below it
        if pts[8]:
            # pts[12][1] > pts[7][1] and
            if  pts[16][1] > pts[7][1] and pts[20][1] > pts[7][1]:
                # if pt 8 is below pt 6, then the user ain't pointing
                if pts[8][1] < pts[7][1]:
                    index_tip_pts.append(pts[8])
                elif len(index_tip_pts) > 0 and time.time() - debounce > 1:
                    pred_cls = pred(index_tip_pts)
                    last_spell = text[pred_cls.item()]
                    print("Casted spell: ", last_spell)
                    index_tip_pts.clear()
                    debounce = time.time()

    cv2.putText(raw_img, last_spell, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # if len(result.hand_landmarks)== 0:
    #     index_tip_pts.clear()
        # print(len(hand_landmarks), hand_landmarks)
        

options = vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(
        # model_asset_path='gesture_recognizer.task',
        # hand_landmarker
        model_asset_path='gesture_recognizer.task',
        # delegate=mp.tasks.BaseOptions.Delegate.GPU,
    ),
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.6,
    running_mode=VisionRunningMode.IMAGE,
    # canned_gestures_classifier_options=["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"],
    num_hands=1,
)
recognizer = vision.GestureRecognizer.create_from_options(options)
if __name__ == '__main__':
    main()
