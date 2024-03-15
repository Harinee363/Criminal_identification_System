import pymysql

def insertData(data):
    rowId = 0

    try:
        # Using a context manager for handling the connection and cursor
        with pymysql.connect(host='localhost', user='root', password='eenirah', database='criminaldb') as db, db.cursor() as cursor:
            print("Database connected")

            query = "INSERT INTO criminaldata VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

            values = (
                data["Name"], data["Father's Name"], data["Mother's Name"], data["Gender"],
                data["DOB(yyyy-mm-dd)"], data["Blood Group"], data["Identification Mark"],
                data["Nationality"], data["Religion"], data["Crimes Done"]
            )

            cursor.execute(query, values)
            db.commit()
            rowId = cursor.lastrowid
            print("Data stored on row %d" % rowId)

    except Exception as e:
        print(f"Data insertion failed: {e}")

    print("Connection closed")
    return rowId

def retrieveData(name):
    id = None
    crim_data = None

    try:
        # Using a context manager for handling the connection and cursor
        with pymysql.connect(host='localhost', user='root', password='eenirah', database='criminaldb') as db, db.cursor() as cursor:
            print("Database connected")

            # Modified the query to handle case-insensitive search and remove leading/trailing whitespaces
            query = "SELECT * FROM criminaldata WHERE TRIM(LOWER(name)) = TRIM(LOWER(%s))"
            cursor.execute(query, (name,))

            result = cursor.fetchone()

            if result:
                id = result[0]
                crim_data = {
                    "Name": result[1],
                    "Father's Name": result[2],
                    "Mother's Name": result[3],
                    "Gender": result[4],
                    "DOB(yyyy-mm-dd)": result[5],
                    "Blood Group": result[6],
                    "Identification Mark": result[7],
                    "Nationality": result[8],
                    "Religion": result[9],
                    "Crimes Done": result[10]
                }

                print("Data retrieved")
            else:
                print(f"No data found for the specified name: {name}")

    except Exception as e:
        print(f"Error: Unable to fetch data: {e}")

    print("Connection closed")

    return (id, crim_data)
