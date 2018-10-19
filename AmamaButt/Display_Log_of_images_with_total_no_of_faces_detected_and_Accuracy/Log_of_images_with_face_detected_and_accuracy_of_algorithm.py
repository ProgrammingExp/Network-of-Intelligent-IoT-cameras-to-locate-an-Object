import numpy as np
import pickle
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image,ImageDraw
import face_recognition
import cv2
import os,random

def browse():
    global c,imgcount
    filename = filedialog.askdirectory()
    print(filename)
    c=filename
    path, dirs, files = next(os.walk(filename))
    file_count = len(files)
    imgcount=file_count
    print("Total Images:",file_count)
     
    files = os.listdir(filename)
    for f in files:
        print (f)
    return filename 
global text,r,accuracy ,show

def face_rec():
    text= ' '
    show=' '
    accuracy=' '
    count=1
    r=0
    files = os.listdir(c)
    for f in files:
        # Load the jpg file into a numpy array
        image = face_recognition.load_image_file(f)
    
        
        # Find all the faces in the image using the default HOG-based model.
        # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
        # See also: find_faces_in_picture_cnn.py
        face_locations = face_recognition.face_locations(image)
        #if face_locations is None:
        #    r=r
       # else:
    
        display_text ="Found {} face(s) in {}".format((len(face_locations)),f)+ "\n" 
        text += display_text
        print("Found {} face(s) in {}".format((len(face_locations)),f))
        if len(face_locations)!=0:
          r=r+1
        
        for face_location in face_locations:

            # Print the location of each face in this image
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            # You can access the actual face itself like this:
            face_image = image[top:bottom, left:right]
            s="{}.jpg"
            s1=s.format(count)
            count=count+1
            path = o
            cv2.imwrite(os.path.join(path , s1), face_image)
        print("Total Images Processed :",r)
        label = Label(text= text)
        label.grid(row=4,column=1)
        accuracy=(r/imgcount)*100
        acc=round(accuracy,2)
        show=str(acc)+"%"
        label3 = Label(text= "Total Images in which faces are detected: ")
        label3.grid(row=5,column=1)
        label3 = Label(text= r)
        label3.grid(row=5,column=2)
        label1 = Label(text= show)
        label1.grid(row=6,column=2)
    return
    
def output():
        
    global o
    foldername = filedialog.askdirectory()
    o=foldername
    return foldername
    

window =Tk()

l1=Label(window, text="Select to Start")
l1.grid(row=0,column=1)

l2=Label(window, text="Choose Images from folder")
l2.grid(row=1,column=1)
StaticBtn = Button(window,text="Browse Folder",width=15,command=browse)
StaticBtn.grid(row=1,column=2)

#l1=Text(window, text="Choose File")
#l1.grid(row=0,column=2)

l3=Label(window, text="Select path to save detected images")
l3.grid(row=2,column=1)

LiveBtn = Button(window,text="Browse Folder",width=15,command=output)
LiveBtn.grid(row=2,column=2)

Btn = Button(window,text="Proceed",width=15,command=face_rec)
Btn.grid(row=3,column=1)


l4=Label(window, text="Accuracy")
l4.grid(row=6,column=1)

 
window.mainloop()