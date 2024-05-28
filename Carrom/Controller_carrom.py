import cv2
import numpy as np
import Hand_Tracking_Module as htm
import time
import pyautogui


wCam , hCam = 648 , 488 #height and width of cam
frameR = 100 
cime=0
ptime=0
clocX,clocY=0,0
plocX,plocY=0,0
smoothening=2

cap = cv2.VideoCapture(0)
cap.set(3,wCam)#set width o  f cam
cap.set(4,hCam)#set hieghtsize of cam

detector=htm.handDetector(detectionCon=0.75)
wScr , hScr =pyautogui.size()
print(wScr,hScr)


while True:
        # FIND LANDMARKS
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img,draw=False)
        
        # Get tip of index and middle finger
        if len(lmlist)!=0:
            x1 , y1 = lmlist[8][1:]
            x2 , y2 =lmlist[12][1:]
            
            # print(x1,y1,x2,y2)
        # Fingers up
            fingers=detector.fingersUp()
            # print(fingers)
            
        # Index finger is moving
            if fingers[1]==1 and fingers[2]==0:
                
                cv2.rectangle(img,(frameR , frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
                x3= np.interp(x1, (frameR,wCam-frameR),(0,wScr))
                y3= np.interp(y1, (frameR,hCam-frameR),(0,hScr))
                
                # Smothening
                clocX=plocX+(x3-plocX) / smoothening
                clocY=plocY+(y3-plocY) /smoothening
                
                        
                pyautogui.moveTo(wScr-x3,y3)
                cv2.circle(img,(x1,y1),15,(255,0,0),cv2.FILLED)
                plocX,plocY=clocX,clocY
        # Click Mode
            if fingers[1]==1 and fingers[2]==1:
                
                length, img , _ = detector.findDistance(8, 12 , img)
                print(length)
                if length<40:
                    pyautogui.mouseDown()
                    time.sleep(2)                   
                    pyautogui.moveTo(wScr-x1,y1)
                    
                pyautogui.mouseUp()
                    
                
# Perform a mouse up 
                    
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime

        cv2.putText(img,f'FPS: {int(fps)}',(400,70), cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)             
        cv2.imshow('Image', img)
        cv2.waitKey(1)
    