import cv2
import numpy as np
import mediapipe as mp

def make_connections(img,x, y,color = (0, 0, 255),thickness=2):
    connections=list(mp.solutions.hands.HAND_CONNECTIONS)
    for connection in connections:
        start_point = (x[connection[0]], y[connection[0]])
        end_point = (x[connection[1]], y[connection[1]])
        img = cv2.line(img, start_point, end_point, color, thickness)
    return img

def top_view(frame,x,z,diff):
    for x_ ,z_ in zip(x,z):
        y_=int(frame.shape[0]*5/6)+np.array(z_)
        frame=make_connections(frame,np.array(x_)-diff,y_)
    return frame
