import sqlite3
conn=sqlite3.connect('database.db')
c=conn.cursor()

c.execute('select * from user')

data=c.fetchall()
print(data)

