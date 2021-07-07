import pyodbc 

conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=56013d871977;'
                      'Database=Practice_DB;'
                      'Trusted_Connection=yes;')

cursor = conn.cursor()
cursor.execute('SELECT * FROM Practice_DB.Products')