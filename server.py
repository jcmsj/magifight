import socket
from spell_detection import identifySpell,gui_identifySpell,ip_port as s_ip_port
from server_mediapipe import detect, handRenderer,predictWithGestureRecognizer,preprocess
W=640
H=480
FPS=30

def camera_thread(socket:socket.socket,ip_port, headless):
    import cv2
    cam = cv2.VideoCapture(0)
    global s_ip_port
    s_ip_port = ip_port
    
    def sendPtsToClient(pts:list[list[float]],_):
        socket.sendto(str(pts).encode("ascii"), ip_port)
    headless_listeners = [
        identifySpell,
        sendPtsToClient,
    ]

    gui_listeners = [
        gui_identifySpell,
        handRenderer,
        sendPtsToClient,
    ]
    detect(
        cam,
        preprocess=preprocess,
        predict=predictWithGestureRecognizer,
        listeners=headless_listeners if headless else gui_listeners,
    )

def main():
    '''Run if main module'''
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--ip', type=str, help='ip address', default='127.0.0.1')
    parser.add_argument('--port', type=int, help='port number', default=4242)
    # headless
    parser.add_argument('--headless', action='store_true', help='Does not open a cv2 window', default=False)
    args = parser.parse_args()
    ip_port = (args.ip, args.port)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if (args.headless):
        print("Headless mode")
    print("Listening on", ip_port)
    camera_thread(s, ip_port, args.headless)

if __name__ == '__main__':
    main()
