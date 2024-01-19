import psycopg2

# กำหนดการเชื่อมต่อกับฐานข้อมูล
db = psycopg2.connect('dbname=DB-Face user=postgres password=Beam12345 host=127.0.0.1 port=5432')

# ฟังก์ชันสำหรับดึงข้อมูลใบหน้าจากฐานข้อมูล
def fetch_all_faces():
    # สร้างคำสั่ง SQL สำหรับการดึงข้อมูลทั้งหมด
    select_query = "SELECT * FROM vectors"

    # เริ่มการดึงข้อมูล
    cursor = db.cursor()
    cursor.execute(select_query)
    
    # ดึงข้อมูลทั้งหมด
    face_records = cursor.fetchall()
    
    # ปิด cursor
    cursor.close()
    
    return face_records

# ดึงข้อมูลใบหน้าทั้งหมด
faces = fetch_all_faces()

# แสดงข้อมูลใบหน้า
for face in faces:
    print("ID:", face[0])
    print("vec_low:", face[1])
    print("vec_high:", face[2])
    print("-------------------------------")

# ปิดการเชื่อมต่อฐานข้อมูล
db.close()
