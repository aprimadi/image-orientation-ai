from __future__ import print_function
import os
import MySQLdb

if __name__ == '__main__':
    conn = MySQLdb.Connection(
        host='localhost',
        user='root',
        port=3306,
        db='image_classifier',
    )
    conn.query("""SELECT * FROM images""")
    result = conn.store_result()
    for i in range(result.num_rows()):
        row = result.fetch_row()
        image_id = row[0][0]

        path = "data-sanitized/%07d.png" % image_id
        if not os.path.exists(path):
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM images WHERE image_id = %d" % image_id)
    conn.commit()
