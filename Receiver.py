import zmq
import cv2
import pickle
import numpy as np

# เชื่อมต่อกับ ZeroMQ context
context = zmq.Context()

# สร้าง socket แบบ PULL
receiver = context.socket(zmq.PULL)
receiver.connect("tcp://127.0.0.1:5555")  # ใส่ IP และ Port ตามที่คุณต้องการ

while True:
    # รับข้อมูลจาก sender
    frame_data = receiver.recv()

    # แปลงข้อมูลที่รับมาเป็นรูปภาพ
    frame = cv2.imdecode(np.frombuffer(pickle.loads(frame_data), dtype=np.uint8), cv2.IMREAD_COLOR)

    # แสดงรูปภาพ
    cv2.imshow("Received Frame", frame)

    # หยุดการทำงานเมื่อกด 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดหน้าต่างทุกครั้งที่จบการทำงาน
cv2.destroyAllWindows()

# ปิด socket
receiver.close()
context.term()
