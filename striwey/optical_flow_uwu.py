import numpy as np
import cv2
import time


def draw_flow(img, flow, step=20):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

#Capture the video and generate a output with cv2
videoPath = 'dom2.mp4'
input = cv2.VideoCapture(videoPath)
output = cv2.VideoWriter('./Striwey/Output/output2.1_' + videoPath, cv2.VideoWriter_fourcc(*'mp4v'), 50, (int(input.get(3)),int(input.get(4))))

suc, prev = input.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:

    suc, img = input.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray

    #Show the process and write frames in output
    imgToShow = draw_flow(gray, flow)
    cv2.imshow('flow', imgToShow)
    output.write(imgToShow)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

output.release()
input.release()
cv2.destroyAllWindows()

"""
# calculating fps
# start time to calculate FPS
    start = time.time()

# End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end-start)

print(f"{fps:.2f} FPS")
"""