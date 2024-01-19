from ultralytics import YOLO
import cv2
import face_recognition
import os
import zmq
import pickle
import threading
import psycopg2
import time
import numpy as np
import string
import random

# load yolov8 model
model = YOLO(r'C:\Users\HP\Desktop\Socket Stream\Model\best_lastest.pt')

# กำหนดตำแหน่งไดเรกทอรีของไฟล์ที่ทำงาน
directory = os.path.dirname(__file__)

# เชื่อมต่อกับฐานข้อมูล PostgreSQL
db = psycopg2.connect('dbname=DB-Face user=postgres password=Beam12345 host=127.0.0.1 port=5432')

# ฟังก์ชันสำหรับบันทึกใบหน้าลงในฐานข้อมูล
def save_face_to_database(encoding):
    # สร้างคำสั่ง SQL สำหรับการเพิ่มข้อมูลใบหน้าลงในฐานข้อมูล
    insert_query = "INSERT INTO vectors (vec_low, vec_high) VALUES (CUBE(array[{}]), CUBE(array[{}]))".format(
        ','.join(str(s) for s in encoding[0:64]),
        ','.join(str(s) for s in encoding[64:128])
    )

    # เริ่มการเพิ่มข้อมูลในฐานข้อมูล
    with db.cursor() as cursor:
        cursor.execute(insert_query)
    db.commit()
    
# สำหรับสร้างชื่อสุ่ม
def generate_random_name(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

# ฟังก์ชันสำหรับทำงานใน background thread
def background_thread():
    # กำหนดตัวแปรสำหรับใช้ในการตรวจจับใบหน้า
    boxes = ""
    weights = os.path.join(directory, r"C:\Users\HP\Desktop\Socket Stream\Model\face_detection_yunet_2023mar.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0), 0.8, 0.3, 5000, 5, 6)
    detected_faces = set()

    # กำหนดตัวแปร name เป็น "unknown"
    name = "unknown"

    while True:
        success, frame = capture.read()
        if not success:
            break

        # detect objects
        results = model.track(frame, imgsz=800,stream=True, conf=0.5)

        # iterate through the results (each result corresponds to an object detected)
        for result in results:
            frame = result.plot()

        # แปลงรูปภาพเป็น RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ดึงขนาดของภาพ
        height, width, _ = frame.shape
        face_detector.setInputSize((width, height))

        # ตรวจจับใบหน้าในภาพ
        _, faces = face_detector.detect(frame)
        faces = faces if faces is not None else []

        # Loop สำหรับการประมวลผลทุกใบหน้า
        for face in faces:
            box = list(map(int, face[:4]))
            x, y, w, h = box
            confidence = face[14]
            boxes = [(y, x + w, y + h, x)]
            face_location = np.array([(x, y), (x + w, y + h)])

            encodings = face_recognition.face_encodings(frame, boxes)
            threshold = 0.5

            if len(encodings) > 0 and confidence > 0.6:
                query = "SELECT id, file, image_id FROM vectors WHERE sqrt(power(CUBE(array[{}]) <-> vec_low, 2) + power(CUBE(array[{}]) <-> vec_high, 2)) <= {} ".format(
                    ','.join(str(s) for s in encodings[0][0:64]),
                    ','.join(str(s) for s in encodings[0][64:128]),
                    threshold,
                ) + \
                    "ORDER BY sqrt(power(CUBE(array[{}]) <-> vec_low, 2) + power(CUBE(array[{}]) <-> vec_high, 2)) ASC".format(
                        ','.join(str(s) for s in encodings[0][0:64]),
                        ','.join(str(s) for s in encodings[0][64:128])
                    )

                with db.cursor() as cursor:
                    cursor.execute(query)
                    face_record = cursor.fetchall()

                select_query = "SELECT id, file, image_id FROM vectors ORDER BY id DESC LIMIT 1"

                with db.cursor() as cursor:
                    cursor.execute(select_query)
                    latest_face_record = cursor.fetchone()

                if latest_face_record:
                    print("Latest Face Data:")
                    print("ID:", latest_face_record[0])
                    print("File:", latest_face_record[1])
                    print("Image ID:", latest_face_record[2])
                else:
                    print("No face data found in the database")

                if len(face_record) > 0:
                    i = face_record[0]
                    if len(i) > 3:
                        name = str(i[3])
                        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
                    else:
                        print("Invalid face data")
                    name = "Unknown"
                else:
                    if name not in detected_faces:
                        name = generate_random_name()
                        save_face_to_database(encodings[0])
                        detected_faces.add(name)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

        # ในที่นี้เราไม่ได้ใช้ฟังก์ชัน send_frame ดังนั้นต้องแก้ไขตรงนี้
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()

        # ส่งข้อมูล frame ผ่านทาง ZeroMQ
        sender.send(pickle.dumps(frame_data))

        time.sleep(0.1)

# ตั้งค่า ZeroMQ context และ socket สำหรับส่งข้อมูล
context = zmq.Context()
sender = context.socket(zmq.PUSH)
sender.bind("tcp://127.0.0.1:5555")  # กำหนด IP และ Port ตามที่คุณต้องการ

# เปิดกล้อง VideoCapture ด้วย URL ของกล้อง RTSP
capture = cv2.VideoCapture(0)

# ตรวจสอบว่ากล้องถูกเปิดหรือไม่
if not capture.isOpened():
    exit()

# เริ่ม background thread
thread = threading.Thread(target=background_thread)
thread.start()

while True:
    # อ่านภาพจากกล้อง
    result, frame = capture.read()
    if result is False:
        cv2.waitKey(0)
        break

    ret, buffer = cv2.imencode('.jpg', frame)
    frame_data = buffer.tobytes()

    # ส่งข้อมูล frame ผ่านทาง ZeroMQ
    sender.send(pickle.dumps(frame_data))

    time.sleep(0.1)  # ให้รอสักครู่เพื่อควบคุมอัตราการส่ง

# ปิดทุกอย่างเมื่อเสร็จสิ้น
sender.close()
context.term()
capture.release()
cv2.destroyAllWindows()