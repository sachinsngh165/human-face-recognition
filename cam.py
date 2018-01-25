import cv2
from main import recognize_face
# import numpy as np

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# print facec
font = cv2.FONT_HERSHEY_SIMPLEX



while True:
    _, fr = rgb.read()
    
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # print gray.shape
    faces = facec.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:

        if not any([w,x, y, h]) :
            continue 

        fc = gray[x:x+w, y:y+h]
        out = recognize_face(fc)

        cv2.putText(gray, out, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    

    cv2.imshow('rgb', gray)

    if cv2.waitKey(100) == 27:
        break

cv2.destroyAllWindows()
