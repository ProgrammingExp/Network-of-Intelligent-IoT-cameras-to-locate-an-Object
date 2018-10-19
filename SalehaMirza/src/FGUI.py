import numpy as np
import pickle
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image

import cv2

def feature():
    face_cascade = cv2.CascadeClassifier('haarscade/haarcascade_profileface.xml')
    eye_cascade = cv2.CascadeClassifier('haarscade/haarcascade_eye.xml')
    lip_cascade = cv2.CascadeClassifier('haarscade/haarcascade_smile.xml')
    print("lip ",lip_cascade)
    print(face_cascade)
    print(eye_cascade)
    #print(smile_cascade)
    img = cv2.imread('opencv_frame_0.png')
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #cv2.imshow('image',resized)
    #cv2.waitKey(0) #Before moving on, wait for a keyboard click.
    count=1
    rowcount=3
    grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('image',grey)
    #cv2.waitKey(0)
    
    faces = face_cascade.detectMultiScale(grey,1.3,5)
    print("faces",faces)
    for (x,y,w,h) in faces:
        cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grey= grey[y:y+h, x:x+w]
        roi_color = resized[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_grey)
    for (ex,ey,ew,eh) in eyes:
        crop_img = roi_color[ey: ey + eh, ex: ex + ew]
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        s="{0}.jpg"
        s1=s.format(count)
        cv2.imwrite(s1,crop_img)
        image = Image.open(s1)
        photo = ImageTk.PhotoImage(image)
        label = Label(image=photo)
        label.image = photo
        label.grid(row=7,column=rowcount)
        rowcount=rowcount+1
        print(rowcount)
        #cv2.imshow(s1)
        count=count+1
    lips = lip_cascade.detectMultiScale(roi_grey)
    for (lx,ly,lw,lh) in lips:
        crop_img1 = roi_color[ly: ly + lh, lx: lx + lw]
        cv2.rectangle(roi_color,(lx,ly),(lx+lw,ly+lh),(0,255,0),2)
        ss="{0}.jpg"
        s11=ss.format(count)
        cv2.imwrite(s11,crop_img1)
        count=count+1
        limage = Image.open(s11)
        lphoto = ImageTk.PhotoImage(limage)
        label = Label(image=lphoto)
        label.limage = lphoto
        label.grid(row=7,column=rowcount)
        rowcount=rowcount+1
        print(rowcount)
    #cv2.imshow('img',resized)
    return  


def browse_button():
    global c
    filename = filedialog.askopenfilename()
    print(filename)
    c=filename
    return filename



def face_rec():
    global Dimg
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
    lip_cascade = cv2.CascadeClassifier('haarscade/Nariz.xml')
    #print("lip ",lip_cascade)
    #print(face_cascade)
    #print(eye_cascade)
    #print(smile_cascade)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face-trainner.yml")
    labels = {"person_name": 1}
    with open("face-labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        img_counter = 0
        if img_counter < 1:
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            gimage=img_name
            img_counter += 1
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        #print("faces 2 ",faces)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
            roi_color = frame[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)
            if conf>=4 and conf <= 85:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            img_item = "7.png"
            cv2.imwrite(img_item, roi_color)
            Dimg=img_item
            color = (255, 0, 0) #BGR 0-255 
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        cv2.imshow('frame',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    image = Image.open("opencv_frame_0.png")
    photo = ImageTk.PhotoImage(image)
    label = Label(image=photo,width=500,height=500)
    label.image = photo
    label.grid(row=7,column=1)
    Faceimage = Image.open("7.png")
    Facephoto = ImageTk.PhotoImage(Faceimage)
    label = Label(image=Facephoto,height=500)
    label.Faceimage = Facephoto
    label.grid(row=7,column=2)
    return img_item



window =Tk()

l1=Label(window, text="Select to Start")
l1.grid(row=0,column=0)



LiveBtn = Button(window,text="Live",width=15,command=face_rec)
LiveBtn.grid(row=0,column=1)

l2=Label(window, text="Select Image ")
l2.grid(row=4,column=0)

#BrowseBtn = Button(window,text="Browse", width=15,command=browse_button)
#BrowseBtn.grid(row=4,column=1)


FeatureBtn = Button(window,text="Feature", width=15,command=feature)
FeatureBtn.grid(row=4,column=1)



l3=Label(window, text="Image Detected ")
l3.grid(row=6,column=1)
 
l4=Label(window, text="Face Detected ")
l4.grid(row=6,column=2)


l4=Label(window, text="Features Detected ")
l4.grid(row=6,column=3)

window.mainloop()