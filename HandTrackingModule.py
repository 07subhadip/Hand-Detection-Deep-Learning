import cv2
import mediapipe as mp
import time


# this is to access the camera of device or any cam module like webcam
class handDetector():
    def __init__(self, mode = False,maxHands = 2,detectionCon = int(0.5), trackCon = int(0.5)):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon )
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        #print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for handLMS in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS , self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self,img,handNo = 0,draw = True):
            lmList = []
            if self.result.multi_hand_landmarks:
                myHand = self.result.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    #print(id, lm)                           # upto this we will get the co-ordinate values
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w) , int(lm.y*h)
                    # print(id,cx,cy)
                    lmList.append([id,cx,cy])

                    # if id == 0: # if remove the condition it will apply for all finger tips
                    if draw:
                        cv2.circle(img,(cx,cy),10,(255,10,255),cv2.FILLED)

            return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Example resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = handDetector()
    while True:
        success , img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist)!=0:
            print(lmlist[0])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)


        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()