import os
import numpy as np
import cv2

os.chdir("C:/Users/abdul/OneDrive/Pictures/New folder")
def points(s, a):
    if s == a or a > a+5 or a < a-5:
        print("match")
    else:
        print("not match")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners=100,
                       qualityLevel=0.3,
                       minDistance=7,
                       blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize=(15,15),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

cap = cv2.VideoCapture(0)
_, frame = cap.read()
old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
old_points = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_params)
old_points = old_points[0:10]
mask = np.zeros_like(frame)

while True:
    _, frame = cap.read()
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    new_points, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, old_points, None, **lk_params)
    s = 0
    for i, (new, old) in enumerate(zip(new_points, old_points)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 5)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        print(a)
    img = cv2.add(frame, mask)
    old_frame = new_frame
    old_points = new_points
    cv2.imshow("frame", img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
