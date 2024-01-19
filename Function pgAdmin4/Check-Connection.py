import psycopg2

try:
    connection = psycopg2.connect(
        user="postgres",
        password="Beam12345",
        host="127.0.0.1",
        port="5432",
        database="DB-Face",
    )
    print("Connected to PostgreSQL")
    connection.close()
except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)
