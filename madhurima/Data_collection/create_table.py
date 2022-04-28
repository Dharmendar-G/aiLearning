import psycopg2                         # import the postgres library

# connect to the database
conn = psycopg2.connect(database="postgres",
                        user='postgres', password='Sreemontini@2013',
                        host='127.0.0.1', port='5432')
# create a cursor object
# cursor object is used to interact with the database
cur = conn.cursor()
# create table with same headers as csv file
sql ='''CREATE TABLE CVE_DETAILS(
   CVSS_SCORE VARCHAR ( 50 ) NOT NULL,
   NUMBER_OF_VULNERABILITIES FLOAT,
   PERCENTAGE FLOAT
)'''
cur.execute(sql)
print("Table created successfully........")
conn.commit()
# Closing the connection
conn.close()