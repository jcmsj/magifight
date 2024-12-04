import time
import cv2
import torch
from saver import xyn_to_bitmap,xyn_to_matrix, save
from PIL import Image
from spell_classification import predict, grayscale_transform
import socket
from model import HarryNet
debounce = time.time()
last_spell = ''
do_save = False
# win:0
# protego:1
# stupefy:2
# engorgio:3
# reducio:4
# unknown:5
text = ['Wingardium Leviosa', 'Protego', 'Stupefy', 'Engorgio','Reducio', 'Unknown']
index_tip_pts = []
num_classes = len(text) -1
UNKNOWN_CLS = num_classes
model = HarryNet(num_classes)
model.load_state_dict(torch.load('harrynet.ckpt',weights_only=True))
# model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
# num_features = model.fc.in_features
# model.fc = nn.Sequential(
#     nn.Linear(num_features, num_classes), nn.Softmax(dim=1)
# )
# model.load_state_dict(torch.load('resnet34_1.ckpt'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

s:socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "127.0.0.1"
ip_port = (ip,4243)
def pred(pts:list[list[float]]):
    grid = xyn_to_bitmap(pts)
    # Convert grid to PIL grayscale
    grid = Image.fromarray(grid).convert('L')
    grid = grayscale_transform(grid)
    grid = grid.unsqueeze(0) # type: ignore
    return predict(model, device, grid)

def identifySpell(pts:list[list[float]],_):
    global debounce, last_spell,text,s,ip_port
    
    # if the rest of the fingers are closed, then the user is pointing
    # if all pts 12,16,20 have y less than 9,13,17 respectively, then add pt 8 to the list
    # alt: using INDEX_FINGER_DIP(point 7), check if the rest of the fingertips are below it
    if len(pts):
        # pts[12][1] > pts[7][1] and
        if  pts[16][1] > pts[7][1] and pts[20][1] > pts[7][1]:
            # if pt 8 is below pt 7, then the user ain't pointing
            if pts[8][1] < pts[7][1]:
                index_tip_pts.append(pts[8])
            elif len(index_tip_pts) > 0 and time.time() - debounce > 1:
                pred_cls,confidence = pred(index_tip_pts)
                pred_cls = int(pred_cls)
                last_spell = text[pred_cls]
                print(f"Casted spell(conf={confidence}): ", last_spell)
                if s is not None and ip_port is not None:
                    s.sendto(str(pred_cls).encode("utf-8"), ip_port)
                if do_save and pred_cls != UNKNOWN_CLS:
                    save(xyn_to_matrix(index_tip_pts), pred_cls, "./detect/")
                    
                index_tip_pts.clear()
                debounce = time.time()

def gui_identifySpell(pts:list[list[float]],raw_img):
    identifySpell(pts,raw_img)
    spellGestureRenderer(raw_img)
    cv2.putText(raw_img, last_spell, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam', raw_img)

def spellGestureRenderer(raw_img):
    # draw a 3pixel radius circle for each pt
    for pt in index_tip_pts:
        cv2.circle(raw_img, (int(pt[0]*raw_img.shape[1]), int(pt[1]*raw_img.shape[0])), 3, (0, 165, 255), 3)

    