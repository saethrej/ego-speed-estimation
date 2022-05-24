import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

cap = cv.VideoCapture('../data/comma2k19/Chunk_6/99c94dc769b5d96e|2018-07-09--11-25-27/11/video_comcro.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.1,
                       minDistance = 1,
                       blockSize = 15 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
print(old_gray.shape)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

hsv = np.zeros_like(old_frame)
hsv[..., 1] = 255
counter = 0
while(1 and counter < 15):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate flow_per_pixel
    flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # calculate optical flow for imp points
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    cv.imshow('frame', bgr)
    cv.imshow('fram2', frame_gray)

    k = cv.waitKey(30) & 0xff
    if counter == 10:
        cv.imwrite('frame_gray.png', frame_gray)
        cv.imwrite('opticalflow.png', bgr)


    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    counter += 1
#cv.destroyAllWindows()


'''
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 2,
                       qualityLevel = 0.1,
                       minDistance = 1,
                       blockSize = 15 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p1 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print(p10)
print(type(p10))
#p0 = []
#for i in range(1):
#    for j in range(1):
#        p0.append([[float(j), float(i)]])
px = []
px.append([[241., 28.]])
px.append([[123., 123.]])
p0 = np.asarray(px)
print(p0)
print(type(p0))
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
counter = 0
while(1 and counter < 60):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #print(p1)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    counter += 1
cv.destroyAllWindows()

'''