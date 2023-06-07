import numpy as np
import cv2


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
def cropping_person_with_ball(self, results, frame):

        detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        for i, (xyxy, confidence, class_id, tracker_id) in enumerate(detections):
            if class_id == 0:
                x1, y1, x2, y2 = xyxy.astype(int)
        
        black = cv2.rectangle(np.copy(frame), (0, 0), (width, height), (0, 0, 0), -1)
        try:
            frame = frame[y1:y2, x1:x2]
        except: pass

        
        frame = frame[int(xyxys[0][0][0]):int(xyxys[0][0][2]), int(xyxys[0][0][1]):int(xyxys[0][0][3])]
        for ball in ballOnImage:
            for xyxy in xyxys:
                if ball[0][0][0] >= xyxy[0][0][0] and ball[0][0][1] >= xyxy[0][0][1] and ball[0][0][2] <= xyxy[0][0][2] and ball[0][0][2] <= xyxy[0][0][2]:
                    frame = frame[xyxy[0][0][0]:xyxy[0][0][1], xyxy[0][0][2]:xyxy[0][0][3]]
                    pass
        
        black[y1:y2, x1:x2] = frame
        return black
"""