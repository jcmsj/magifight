import socket
from typing import Any, Callable

import cv2
from cv2.typing import MatLike

from spell_classification import SpellClassifier
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "172.26.0.1"
port = 4242
ip_port = (ip,port)
import torch

def main():
    '''Run if main module'''
    model = SpellClassifier(3)
    model.load_state_dict(torch.load('model.ckpt'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

if __name__ == '__main__':
    main()
