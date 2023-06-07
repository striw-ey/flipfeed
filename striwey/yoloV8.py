import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    

    def load_model(self):
       
        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def detecting_person_and_ball(self, results, frame):
        person, ball, val = False

        detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        for i, (xyxy, confidence, class_id, tracker_id) in enumerate(detections):
            if class_id == 0:
                person = True
            if class_id == 1:
                ball = True
        
        if person and ball:
            val = True

        return val
    

    def draw_flow(self, img, flow, step=20):

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
    
    
    def __call__(self):
        
        
        videoPath = 'dom2.mp4'
        cap = cv2.VideoCapture('./TrainingVideos/' + videoPath)
        assert cap.isOpened()

        width  = int(cap.get(3))
        height = int(cap.get(4))
        #output = cv2.VideoWriter('./Striwey/Output/output_' + videoPath, cv2.VideoWriter_fourcc(*'mp4v'), 50, (width, height))

        suc, prev = cap.read()
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        while True:
          
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            #Yolo v8
            results = self.predict(frame)
            valBool = self.detecting_person_and_ball(results, frame)

            if valBool:
                #opticalFlow
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                prevgray = gray
                frame = self.draw_flow(gray, flow)
            else:
                print('No se encuentra la persona y el balón')
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
            #output.write(frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        #output.release()
        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture_index=0)
detector()