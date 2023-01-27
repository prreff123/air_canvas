import numpy as np
import cv2
import mediapipe as mp
from collections import deque

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index=0
green_index=0
red_index=0
yellow_index=0

kernel = np.ones((5,5),np.uint8)

colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorindex = 0

# Paint
paintwindow = np.zeros((471,636,3)) + 255
paintwindow = cv2.rectangle(paintwindow,(40,1),(140,65),(0,0,0),2)
paintwindow = cv2.rectangle(paintwindow,(160,1),(255,65),(255,0,0),2)
paintwindow = cv2.rectangle(paintwindow,(275,1),(370,65),(0,255,0),2)
paintwindow = cv2.rectangle(paintwindow,(390,1),(485,65),(0,0,255),2)
paintwindow = cv2.rectangle(paintwindow,(505,1),(600,65),(0,255,255),2)

cv2.putText(paintwindow, "CLEAR" , (49,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintwindow, "BLUE" , (185,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintwindow, "GREEN" , (290,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintwindow, "RED" , (420,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintwindow, "YELLOW" , (520,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# initialize mediapipe
mphands = mp.solutions.hands
hands = mphands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpdraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # frame
    frame = cv2.rectangle(frame,(40,1),(140,65),(0,0,0),2)
    frame = cv2.rectangle(frame,(160,1),(255,65),(255,0,0),2)
    frame = cv2.rectangle(frame,(275,1),(370,65),(0,255,0),2)
    frame = cv2.rectangle(frame,(390,1),(485,65),(0,0,255),2)
    frame = cv2.rectangle(frame,(505,1),(600,65),(0,255,255),2)

    cv2.putText(frame, "CLEAR" , (49,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "BLUE" , (185,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "GREEN" , (290,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "RED" , (420,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "YELLOW" , (520,33), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)

    result = hands.process(frame)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            mpdraw.draw_landmarks(frame, handslms ,mphands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame,center, 3,(0,255,0), -1)
        print(center[1]-thumb[1])
        
        if (thumb[1]-center[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1         
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]    
                rpoints = [deque(maxlen=512)]    
                ypoints = [deque(maxlen=512)]    

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintwindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                colorindex = 0    
            elif 275 <= center[0] <= 370:
                colorindex = 1
            elif 390 <= center[0] <= 485:
                colorindex = 2
            elif 505 <= center[0] <= 600:
                colorindex = 3

        else:
            if colorindex == 0:
                bpoints[blue_index].appendleft(center)                    
            elif colorindex == 1:
                gpoints[green_index].appendleft(center)
            elif colorindex == 2:
                rpoints[red_index].appendleft(center)
            elif colorindex == 3:
                ypoints[yellow_index].appendleft(center)            
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1            
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1
    
    points = [bpoints,gpoints,rpoints,ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k-1], points[i][j][k], colors[i], 2)
                cv2.line(paintwindow, points[i][j][k-1], points[i][j][k], colors[i], 2)      

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintwindow)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()    