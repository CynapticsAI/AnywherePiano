import cv2
import numpy as np


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def tap_detection(previous_x,previous_y,x,y,threshold):
    keys_to_check=[4,8,12,16,20]
    tapped_keys=[]
    for key in keys_to_check:
        # dis=calculate_distance((x[key],y[key]),(previous_x[key],previous_y[key]))
        dis=y[key]-previous_y[key]
        if dis>threshold:
            tapped_keys.append(key)
    return tapped_keys


def check_keys(x,y,white,black,white_piano_notes,black_piano_notes,tapped_keys):
    keys_to_check=tapped_keys
    pressed_notes=[]
    pressed_keys={"White":[],"Black":[]}
    for key_to_check in keys_to_check:
        x1,y1=x[key_to_check],y[key_to_check]
        flag=False
        for i,key in enumerate(black):
            distance = cv2.pointPolygonTest(np.array(key), (x1,y1), measureDist=False)
            if distance>0:
                # print("Black ",i)
                pressed_keys['Black'].append(i)
                pressed_notes.append(black_piano_notes[i])
                flag=True
                break
        if flag:
            continue
        for i,key in enumerate(white):
            distance = cv2.pointPolygonTest(np.array(key), (x1,y1), measureDist=False)
            if distance>0:
                # print("White ",i)
                pressed_keys['White'].append(i)
                pressed_notes.append(white_piano_notes[i])
                break
        pressed_notes=list(set(pressed_notes))
        pressed_keys['White']=list(set(pressed_keys['White']))
        pressed_keys['Black']=list(set(pressed_keys['Black']))
    return pressed_keys,pressed_notes