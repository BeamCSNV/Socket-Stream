from ultralytics import YOLO
import cv2


# load yolov8 model
#model = YOLO('yolov8n.pt')  # load an official detection model
#model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
model = YOLO('best_collab.pt')

# load video
url = "rtsp://admin:admin@192.168.0.100:8554/live"
cap = cv2.VideoCapture(url)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, imgsz=640,stream=True, conf=0.5)
        
        #print(results[0])
        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


