import cv2
import numpy as np

class Piano():
    def __init__(self,pts,num_octaves,height_and_width):
        self.pts=pts
        self.num_octaves=num_octaves
        self.white=[]
        self.black=[]
        self.height_of_black=height_and_width[0]
        self.width_of_black=height_and_width[1]
        self.get_keyboard_keys()

    def section_formula(self,x1,y1,x2,y2,m,n):
        x=(m*x2+n*x1)/(m+n)
        y=(m*y2+n*y1)/(m+n)
        return int(x),int(y)

    def add_octaves(self,pts,list,n):
        [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]=pts
        for i in range(n):
            temp=np.zeros((4,1,2), dtype=np.int32)
            x,y=self.section_formula(x0,y0,x1,y1,i,n-i)
            temp[0][0]=[x,y]
            x,y=self.section_formula(x0,y0,x1,y1,i+1,n-i-1)
            temp[1][0]=[x,y]
            x,y=self.section_formula(x3,y3,x2,y2,i+1,n-i-1)
            temp[2][0]=[x,y]
            x,y=self.section_formula(x3,y3,x2,y2,i,n-i)
            temp[3][0]=[x,y]
            list.append(temp)

    def add_white_keys(self,pts):
        [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]=pts
        for i in range(7):
            temp=np.zeros((4,1,2), dtype=np.int32)
            x,y=self.section_formula(x0,y0,x1,y1,i,7-i)
            temp[0][0]=[x,y]
            x,y=self.section_formula(x0,y0,x1,y1,i+1,7-i-1)
            temp[1][0]=[x,y]
            x,y=self.section_formula(x3,y3,x2,y2,i+1,7-i-1)
            temp[2][0]=[x,y]
            x,y=self.section_formula(x3,y3,x2,y2,i,7-i)
            temp[3][0]=[x,y]
            self.white.append(temp)

    def add_black_keys(self,pts):
        [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]=pts
        last_x_,last_y_=(x3,y3)
        last_x_up,last_y_up=(x0,y0)
        for i in range(1,7):
            x_,y_=self.section_formula(x3,y3,x2,y2,i,7-i)
            x_up,y_up=self.section_formula(x0,y0,x1,y1,i,7-i)
            if i==3:
                last_x_up,last_y_up=x_up,y_up
                last_x_,last_y_=x_,y_
                continue
            temp=np.zeros((4,1,2), dtype=np.int32)
            xy_coords=np.zeros((4,1,2), dtype=np.int32)
            # black to be 5/8 times (width) of white key so n=5 and d=8
            n=self.width_of_black[0]
            d=self.width_of_black[1]
            x,y=self.section_formula(last_x_,last_y_,x_,y_,2*d-n,n)
            temp[3][0]=[x,y]
            x,y=self.section_formula(last_x_up,last_y_up,x_up,y_up,2*d-n,n)
            xy_coords[0][0]=[x,y]
            x,y=self.section_formula(last_x_,last_y_,x_,y_,2*d+n,-n)
            temp[2][0]=[x,y]
            x,y=self.section_formula(last_x_up,last_y_up,x_up,y_up,2*d+n,-n)
            xy_coords[1][0]=[x,y]
            # black to be 5/8 times (height) of white key so n_=5 and d_=8
            n_=self.height_of_black[0]
            d_=self.height_of_black[1]
            x,y=self.section_formula(temp[2][0][0],temp[2][0][1],xy_coords[1][0][0],xy_coords[1][0][1],n_,d_-n_)
            temp[1][0]=[x,y]
            x,y=self.section_formula(temp[3][0][0],temp[3][0][1],xy_coords[0][0][0],xy_coords[0][0][1],n_,d_-n_)
            temp[0][0]=[x,y]
            last_x_up,last_y_up=x_up,y_up
            last_x_,last_y_=x_,y_
            self.black.append(temp)

    def add_minor_keys(self,pts):
        [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]=pts
        for i in range(2):
            temp=np.zeros((4,1,2), dtype=np.int32)
            x,y=self.section_formula(x0,y0,x1,y1,i,2-i)
            temp[0][0]=[x,y]
            x,y=self.section_formula(x0,y0,x1,y1,i+1,2-i-1)
            temp[1][0]=[x,y]
            x,y=self.section_formula(x3,y3,x2,y2,i+1,2-i-1)
            temp[2][0]=[x,y]
            x,y=self.section_formula(x3,y3,x2,y2,i,2-i)
            temp[3][0]=[x,y]
            self.white.append(temp)
        temp=np.zeros((4,1,2), dtype=np.int32)
        xy_coords=np.zeros((4,1,2), dtype=np.int32)
        x_,y_=self.section_formula(x3,y3,x2,y2,1,1)
        x_up,y_up=self.section_formula(x0,y0,x1,y1,1,1)
        # black to be 5/8 times (width) white key so n=5 and d=8
        n=self.width_of_black[0]
        d=self.width_of_black[1]
        x,y=self.section_formula(x3,y3,x_,y_,2*d-n,n)
        temp[3][0]=[x,y]
        x,y=self.section_formula(x0,y0,x_up,y_up,2*d-n,n)
        xy_coords[0][0]=[x,y]
        x,y=self.section_formula(x3,y3,x_,y_,2*d+n,-n)
        temp[2][0]=[x,y]
        x,y=self.section_formula(x0,y0,x_up,y_up,2*d+n,-n)
        xy_coords[1][0]=[x,y]
        # black to be 5/8 times (height) of white key so n_=5 and d_=8
        n_=self.height_of_black[0]
        d_=self.height_of_black[1]
        x,y=self.section_formula(temp[2][0][0],temp[2][0][1],xy_coords[1][0][0],xy_coords[1][0][1],n_,d_-n_)
        temp[1][0]=[x,y]
        x,y=self.section_formula(temp[3][0][0],temp[3][0][1],xy_coords[0][0][0],xy_coords[0][0][1],n_,d_-n_)
        temp[0][0]=[x,y]
        self.black.append(temp)

    def get_keyboard_keys(self):
        [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]=self.pts
        n=self.num_octaves
        x_up,y_up=self.section_formula(x0,y0,x1,y1,2,7*n)
        x_,y_=self.section_formula(x3,y3,x2,y2,2,7*n)
        pts=np.array([[[x0,y0]],[[x_up,y_up]],[[x_,y_]],[[x3,y3]]])
        self.add_minor_keys(pts)
        list_of_octaves=[]
        pts=np.array([[[x_up,y_up]],[[x1,y1]],[[x2,y2]],[[x_,y_]]])
        self.add_octaves(pts,list_of_octaves,n)
        for i in range(n):
            self.add_white_keys(list_of_octaves[i])
            self.add_black_keys(list_of_octaves[i])

    def make_keyboard(self,img):
        img=cv2.fillPoly(img,self.white,(255,255,255))
        img=cv2.polylines(img,self.white,True,(255,0,255),2)
        img=cv2.fillPoly(img,self.black,(0,0,0))
        return img
    
    def change_color(self,img,pressed_keys):
        white_keys=[self.white[i] for i in pressed_keys['White']]
        black_keys=[self.black[i] for i in pressed_keys['Black']]
        img=cv2.fillPoly(img,white_keys,(0,255,0))
        img=cv2.polylines(img,self.white,True,(255,0,255),2)
        img=cv2.fillPoly(img,self.black,(0,0,0))
        img=cv2.fillPoly(img,black_keys,(128,128,128))
        return img