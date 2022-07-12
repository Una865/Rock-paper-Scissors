from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np

def findDistance(l1, l2):
    x1 = l1[0]
    y1 = l1[1]
    x2 = l2[0]
    y2 = l2[1]

    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def Detection():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            bbox1 = hand1["bbox"]
            centerPoint1 = hand1['center']
            handType1 = hand1["type"]
            l1 = findDistance(lmList1[8], lmList1[2])  # <80
            l2 = findDistance(lmList1[16], lmList1[2])  # <150
            org = (00, 100)
            fontScale = 3
            color = (0, 0, 0)
            thickness = 2

            if l1 < 80:
                img = cv2.putText(img, 'Rock', org, cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                                  color, thickness, cv2.LINE_AA, False)
            elif l2 < 150:
                img = cv2.putText(img, 'Scissors', org, cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                                  color, thickness, cv2.LINE_AA, False)
            else:
                img = cv2.putText(img, 'Paper', org, cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                                  color, thickness, cv2.LINE_AA, False)


        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()



Detection()

