import psycopg2

# เชื่อมต่อกับฐานข้อมูล PostgreSQL
db = psycopg2.connect('dbname=DB-Face user=postgres password=Beam12345 host=127.0.0.1 port=5432')

# สร้างคำสั่ง SQL DELETE ที่ไม่ระบุเงื่อนไข WHERE เพื่อลบทุก ID
delete_all_query = "DELETE FROM vectors;"

# เริ่มการลบข้อมูลทั้งหมดในฐานข้อมูล
cursor = db.cursor()
cursor.execute(delete_all_query)
db.commit()
cursor.close()

# ปิดการเชื่อมต่อฐานข้อมูล
db.close()
