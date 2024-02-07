import numpy as np
import cv2
import time
import argparse
import ast

from keyboard import Piano
from piano_config import piano_configuration
from hand_detection2 import Hands_detection
from check_key import check_keys,tap_detection
from piano_sound import play_piano_sound
from top_view import top_view


class VirPiano():
    def __init__(self,model_path='model/hand_landmarker.task',num_octaves=2,list_of_octaves=[3,4],height_and_width_black=[[5,8],[5,8]],shape=(800,600,3),tap_threshold=20,piano_config_threshold=30,piano_config=1):
        self.model_path=model_path
        self.hand_detection=Hands_detection(self.model_path)
        self.shape=shape
        self.image=np.zeros(self.shape, np.uint8)
        self.num_octaves=num_octaves
        self.list_of_octaves=list_of_octaves
        self.height_and_width_black=height_and_width_black
        self.tap_threshold=tap_threshold
        self.piano_config_threshold=piano_config_threshold
        self.pts=np.array([[[100,350]],[[700,350]],[[700,550]],[[100,550]]])
        self.piano_keyboard=Piano(self.pts,self.num_octaves,self.height_and_width_black)
        self.piano_config=piano_config
        self.x=[]
        self.y=[]
        self.z=[]
        self.previous_x=[]
        self.previous_y=[]
        self.white_piano_notes,self.black_piano_notes=self.get_piano_notes()
        self.start()

    def get_piano_notes(self):
        white_piano_notes=['A0','B0']
        black_piano_notes=['Bb0']
        # white_piano_notes=['A0','B0','A1','B1','C1','D1','E1','F1','G1','A2','B2','C2','D2','E2','F2','G2']
        # black_piano_notes=['Bb0','Bb1','Db1','Eb1','Gb1','Ab1','Bb2','Db2','Eb2','Gb2','Ab2']
        for i in range(self.list_of_octaves[0],self.list_of_octaves[1]+1):
            for note in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                white_piano_notes.append(f"{note}{i}")
            for note in ['Bb', 'Db', 'Eb', 'Gb', 'Ab']:
                black_piano_notes.append(f"{note}{i}")
        return white_piano_notes,black_piano_notes

    def circle_fingertips(self,img):
        radius=4
        color=(0,0,255)
        thickness=-1
        fingertips=[4,8,12,16,20]
        if len(self.x)>0:
            for x,y in zip(self.x,self.y):
                for tip in fingertips:
                    cv2.circle(img,(x[tip],y[tip]),radius,color,thickness)
        return img

    def start(self):
        cap=cv2.VideoCapture(0)
        previousTime=0
        while True:
            _,frame=cap.read()
            frame = cv2.resize(frame, (self.shape[0],self.shape[1]))
            frame = cv2.flip(frame, 1)

            while self.piano_config==1:
                self.pts=piano_configuration(self.model_path,self.shape,self.piano_config_threshold)
                self.piano_keyboard=Piano(self.pts,self.num_octaves,self.height_and_width_black)
                self.piano_config=0
                cap=cv2.VideoCapture(0)

            self.image=frame.copy()
            self.hand_detection.detect(frame)
            self.x=self.hand_detection.x
            self.y=self.hand_detection.y
            self.z=self.hand_detection.z

            self.image=self.piano_keyboard.make_keyboard(self.image)

            pressed_keys={"White":[],"Black":[]}
            pressed_notes=[]
            if len(self.previous_x)==len(self.x):
                for (x_,y_),(previous_x_,previous_y_) in zip(zip(self.x,self.y),zip(self.previous_x,self.previous_y)):
                    tapped_keys=tap_detection(previous_x_,previous_y_,x_,y_,self.tap_threshold)
                    keys,notes=check_keys(x_,y_,self.piano_keyboard.white,self.piano_keyboard.black,self.white_piano_notes,self.black_piano_notes,tapped_keys)
                    for note in notes:
                        pressed_notes.append(note)
                    for w in keys['White']:
                        pressed_keys['White'].append(w)
                    for b in keys['Black']:
                        pressed_keys['Black'].append(b)

            self.image=self.piano_keyboard.change_color(self.image,pressed_keys)

            top_view_shape=(500,250)
            pts1=np.reshape(np.array(self.pts,dtype=np.float32),(-1,2))
            pts2=np.float32([[0,0],[top_view_shape[0],0],[top_view_shape[0],top_view_shape[1]],[0,top_view_shape[1]]])
            M = cv2.getPerspectiveTransform(pts1,pts2)
            top_view_image = cv2.warpPerspective(self.image,M,top_view_shape)
            top_view_image=cv2.flip(top_view_image,0)
            top_view_image=top_view(top_view_image,self.x,self.z,self.pts[3][0][0])
            cv2.imshow('top view',top_view_image)
            self.image=self.circle_fingertips(self.image)


            self.previous_x=self.x
            self.previous_y=self.y

            if len(pressed_notes)>0:
                play_piano_sound(pressed_notes)

            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime
            cv2.putText(self.image, str(int(fps)) + "FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (88, 205, 54), 3)

            cv2.namedWindow("Hand detection", cv2.WINDOW_NORMAL)
            cv2.imshow('Hand detection', self.image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='model/hand_landmarker.task')
    parser.add_argument('--num_octaves', default=2, type=int)
    parser.add_argument('--list_of_octaves', default=[3,4], type=ast.literal_eval)
    parser.add_argument('--height_and_width_black', default=[[5,8],[5,8]], type=ast.literal_eval)
    parser.add_argument('--shape', default=(800,600,3), type=ast.literal_eval)
    parser.add_argument('--tap_threshold', default=20, type=int)
    parser.add_argument('--piano_config_threshold', default=30, type=int)
    parser.add_argument('--piano_config', default=1,type=int)

    args = parser.parse_args()


    vp=VirPiano(model_path=args.model_path,num_octaves=args.num_octaves,list_of_octaves=args.list_of_octaves,height_and_width_black=args.height_and_width_black,shape=args.shape,tap_threshold=args.tap_threshold,piano_config_threshold=args.piano_config_threshold,piano_config=args.piano_config)

    #cli for running the file is
    # python main.py --model_path "model/hand_landmarker.task" --num_octaves 2 --list_of_octaves "[3,4]" --height_and_width_black "[[5,8],[5,8]]" --shape "(800,600,3)" --tap_threshold 20 --piano_config_threshold 30 --piano_config 1
    # or
    # python main.py