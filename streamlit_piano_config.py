import numpy as np
import cv2
import streamlit as st

from hand_detection2 import Hands_detection


def circle_fingertips(img,x,y):
        radius=4
        color=(0,0,255)
        thickness=-1
        fingertips=[4,8,12]
        if len(x)>0:
            for x_,y_ in zip(x,y):
                for tip in fingertips:
                    cv2.circle(img,(x_[tip],y_[tip]),radius,color,thickness)
        if len(x)==2:
            pts=[[[x[0][4],y[0][4]]],[[x[1][4],y[1][4]]],[[x[1][8],y[1][8]]],[[x[0][8],y[0][8]]]]
            img=cv2.polylines(img,[np.array(pts,dtype=np.int32)],True,color,2)
        return img

def get_coordinates(model_path,shape,distance_threshold):
    FRAME_WINDOW = st.image([])
    hd=Hands_detection(model_path)
    cap1=cv2.VideoCapture(0)
    while True:
        _,frame=cap1.read()
        frame = cv2.resize(frame, (shape[0],shape[1]))
        frame = cv2.flip(frame, 1)
        hd.detect(frame)
        frame=circle_fingertips(frame,hd.x,hd.y)
        FRAME_WINDOW.image(frame, channels="BGR")
        cv2.waitKey(1)
        if len(hd.x)==2:
            [thumb_x0,thumb_y0] = hd.x[0][4],hd.y[0][4]
            [thumb_x1,thumb_y1] = hd.x[1][4],hd.y[1][4]
            [forefinger_x0,forefinger_y0] = hd.x[0][8],hd.y[0][8]
            [forefinger_x1,forefinger_y1] = hd.x[1][8],hd.y[1][8]
            [middlefinger_x0,middlefinger_y0] = hd.x[0][12],hd.y[0][12]
            [middlefinger_x1,middlefinger_y1] = hd.x[1][12],hd.y[1][12]
            distance0 = cv2.norm(np.array((forefinger_x0,forefinger_y0)), np.array((middlefinger_x0,middlefinger_y0)), cv2.NORM_L2)
            distance1 = cv2.norm(np.array((forefinger_x1,forefinger_y1)), np.array((middlefinger_x1,middlefinger_y1)), cv2.NORM_L2)
            if distance1 < distance_threshold and distance0 < distance_threshold:
                pts=[[[thumb_x0,thumb_y0]],[[thumb_x1,thumb_y1]],[[forefinger_x1,forefinger_y1]],[[forefinger_x0,forefinger_y0]]]
                cap1.release()
                cv2.destroyAllWindows()
                FRAME_WINDOW.empty()
                return pts

def piano_configuration(model_path,shape,distance_threshold):
    pts=get_coordinates(model_path,shape,distance_threshold)
    return pts