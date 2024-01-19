from ultralytics import YOLO
import cv2
import dlib
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import pickle
import os
import random
import requests

# กำหนด Line Token
LINE_NOTIFY_TOKEN = "6n6NUX9mdUjHysH3IqGEdyuN5Wmh6BPZwaZNlzuJr44"

# โหลด pre-trained YOLO model
model = YOLO(r'C:\Users\HP\Desktop\Detect-SQL\Model\best_lastest.pt')

# เลือก source สำหรับการทดสอบ (ในที่นี้ใช้กล้องที่เชื่อมต่อกับเครื่อง)
source = '0'

# กำหนดค่าเริ่มต้นของชื่อในการตรวจจับใบหน้า
currentname = "ไม่รู้จัก"

# โหลด encodings และตัวตรวจหน้าจากไฟล์ pickle
encodingsP = r"C:\Users\HP\Desktop\Detect-SQL\Model\encodings.pickle"
print("[INFO] กำลังโหลด encodings + ตัวตรวจหน้า...")

data = pickle.loads(open(encodingsP, "rb").read())

# กำหนดที่เก็บ dataset ของใบหน้า (ถ้ายังไม่มีจะสร้างใหม่)
dataset_path = r"C:\Users\HP\Desktop\Detect-SQL\dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# ใช้ dlib library เพื่อตรวจจับใบหน้า
face_detector = dlib.get_frontal_face_detector()

# ฟังก์ชันสำหรับส่งข้อความผ่าน Line Notify
def send_line_notify(message, image_path=None):
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": "Bearer " + LINE_NOTIFY_TOKEN}
    payload = {"message": message}
    files = {}

    if image_path:
        files = {"imageFile": open(image_path, "rb")}

    requests.post(url, headers=headers, params=payload, files=files)
    
# ใช้ YOLO model ในการตรวจจับวัตถุในวิดีโอและสตรีม
results = model(source, stream=True, conf=0.5)

# กำหนดตัวแปรสำหรับกำหนดรหัสใบหน้า
face_id = 1

# วนลูปผลลัพธ์จาก YOLO model
for result in results:
    frame = result.plot()

    # ใช้ face_recognition library เพื่อหาตำแหน่งและ encodings ของใบหน้า
    boxes = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, boxes)
    names = []

    # วนลูปใน encodings ที่ได้จากใบหน้าที่ตรวจจับได้
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "ไม่รู้จัก"

        # ถ้ามีการตรงกันใน encodings ที่เก็บไว้
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # นับจำนวนการตรงกันและเลือกชื่อที่มีจำนวนมากที่สุด
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

            # ถ้าชื่อไม่เปลี่ยนให้ไม่พิมพ์
            if currentname != name:
                currentname = name
                print(currentname)
        else:
            # สร้างรหัสใบหน้าใหม่
            num = f"ID{face_id:02d}"
            data["names"].append(num)
            data["encodings"].append(encoding)
            print(data["names"])

            # สร้างโฟลเดอร์เก็บรูปใบหน้า
            face_folder_path = os.path.join(dataset_path, f"{num}")
            if not os.path.exists(face_folder_path):
                os.makedirs(face_folder_path)

            # บันทึกรูปใบหน้า
            face_filename = os.path.join(face_folder_path, f"{num}.jpg")
            cv2.imwrite(face_filename, frame[boxes[0][0]:boxes[0][2], boxes[0][3]:boxes[0][1]])

            # บันทึกข้อมูลใหม่ลงในไฟล์ encodings.pickle
            train_data = {"encodings": [encoding], "names": [num]}
            with open(encodingsP, "wb") as f:
                f.write(pickle.dumps(train_data))

            face_id += 1

            # ส่งการแจ้งเตือนผ่าน Line Notify
            send_line_notify("ตรวจพบใบหน้าใหม่!", image_path=face_filename)

        names.append(name)

    # วาดกรอบและแสดงชื่อในกรอบของใบหน้า
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

    # แสดงวิดีโอที่ได้ผลลัพธ์
    cv2.imshow(f'{result.path}', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดหน้าต่างทุกอย่างเมื่อทำการทดสอบเสร็จสิ้น
cv2.destroyAllWindows()