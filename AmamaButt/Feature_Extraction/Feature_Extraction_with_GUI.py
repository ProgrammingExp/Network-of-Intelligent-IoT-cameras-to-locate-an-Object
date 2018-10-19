import numpy as np
import pickle
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image,ImageDraw
import face_recognition
import cv2

def feature():
    face_cascade = cv2.CascadeClassifier('haarscade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarscade/haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('haarscade/Nariz.xml')
    lip_cascade = cv2.CascadeClassifier('haarscade/Mouth.xml')


    #Identify the image of interest to import. Ensure that when you import a file path
    #that you do not use / in front otherwise it will return empty.
    img = cv2.imread('obama.jpg')

    # Resize the image to save space and be more manageable.
    # We do this by calculating the ratio of the new image to the old image
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))

    # Perform the resizing and show
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

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
    filename = filedialog.askopenfilename()
    global Dimg,Dimg1
    face_cascade = cv2.CascadeClassifier('haarscade/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarscade/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('haarscade/Mouth.xml')
    obama_image = face_recognition.load_image_file("obama.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file("biden.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [
        obama_face_encoding,
        biden_face_encoding
    ]
    known_face_names = [
        "Obama",
        "Unknown"
    ]

    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file(filename)
    
    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))


    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()
    
    frame=unknown_image



    #Identify the image of interest to import. Ensure that when you import a file path
    #that you do not use / in front otherwise it will return empty.
    img = frame

    # Resize the image to save space and be more manageable.
    # We do this by calculating the ratio of the new image to the old image
    r = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * r))

    # Perform the resizing and show
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #Display the image
    
    cv2.imwrite("pic.jpg",resized)
    count=1
    #Process the image - convert to BRG to grey
    grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(grey, 1.3,5) 
    for (x,y,w,h) in faces:
        cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
        roi_grey = grey[y:y+h, x:x+w]
        roi_color = resized[y:y+h, x:x+w]
   
        s="{0}.jpg"
        s1=s.format(count)
        count=count+1
        cv2.imwrite(s1,roi_color)
    image = Image.open("pic.jpg")
    photo = ImageTk.PhotoImage(image)
    label = Label(image=photo,width=500,height=500)
    label.image = photo
    label.grid(row=7,column=1)
    Faceimage = Image.open("1.jpg")
    Facephoto = ImageTk.PhotoImage(Faceimage)
    label = Label(image=Facephoto,height=500)
    label.Faceimage = Facephoto
    label.grid(row=7,column=2)
    return



window =Tk()

l1=Label(window, text="Select to Start")
l1.grid(row=0,column=0)


StaticBtn = Button(window,text="Static",width=15,command=face_rec)
StaticBtn.grid(row=0,column=1)

LiveBtn = Button(window,text="Live",width=15)
LiveBtn.grid(row=0,column=2)

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