import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
# this is to access the camera of device or any cam module like webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Example resolution
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    result = hands.process(imgRGB)
    #print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handLMS in result.multi_hand_landmarks:
            for id, lm in enumerate(handLMS.landmark):
                #print(id, lm)                           # upto this we will get the co-ordinate values
                h, w, c = img.shape
                cx, cy = int(lm.x*w) , int(lm.y*h)
                print(id,cx,cy)

                if id == 0: # if remove the condition it will apply for all finger tips
                    cv2.circle(img,(cx,cy),10,(255,10,255),cv2.FILLED)

            mpDraw.draw_landmarks(img, handLMS , mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)


    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()