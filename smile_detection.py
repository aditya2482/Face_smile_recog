# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:04:17 2020

@author: Aditya
"""
'''
hardcascade is a algorithm
'''
#find teeth and wide curvature of the smile

import cv2
# face classifier
face_detector = cv2.CascadeClassifier('file:///C:/Users/Aditya/Desktop/python/MachineLearning/extras/haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('file:///C:/Users/Aditya/Desktop/python/MachineLearning/extras/haarcascade_smile.xml')
#grab webcam feed
webcam = cv2.VideoCapture(0)
while True:
    #grab webcam feed
    sucessfull_frame_read,frame=webcam.read() #SINGLE frame read each time
    
    
    #changing to grayscale
    frame_gray = cv2.Color(frame,cv2.COLOR_BGR2GRAY)
    
    #detect faces 
    faces = face_detector.detectMultiScale(frame_gray)
    smiles = smile_detector.detectMultiScale(frame_gray,scaleFactor=1.7,minNeighbour=20)  #scalefactor is a method of optimixation it will blurr the data (face) and hence canbe accurate
    #minneighbour will set the amount of neighbour of the rectangle
    
    #just array of point
    print(faces) #print coordinate of faces list 
    #for (x,y,w,h) in faces(for face detection)
    for (x,y,w,h) in smiles:    #xandy
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,123),2) 
    
    cv2.imshow("why so serious",frame) #display the window
    
    
    
    cv2.waitKey(10) #display
    

#clean up
webcam.release()
cv2.destroyAllWindows()
print("completed")
