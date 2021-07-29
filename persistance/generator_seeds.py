import sqlite3


def open_sql_connection():
    db_filename = 'generator_seeds.db'
    return sqlite3.connect(db_filename)


def sql_table_seeds(conn):
    cursorObj = conn.cursor()

    cursorObj.execute(
        "CREATE TABLE generator_seeds("
        "ID INTEGER PRIMARY KEY AUTOINCREMENT, "
        "SEED INTEGER,"
        "UNIQUE (SEED)"
        ")"
    )

    conn.commit()


def insert_into_seeds(conn, seed: int):
    cursorObj = conn.cursor()

    cursorObj.execute(
        "INSERT INTO generator_seeds(seed) VALUES(?)",
        (seed,))

    conn.commit()


def fetch_id(conn, id: int):
    cursorObj = conn.cursor()

    cursorObj.execute('SELECT SEED FROM generator_seeds WHERE ID == ?', (id, ))

    return cursorObj.fetchall()


def sql_fetch(con):
    cursorObj = con.cursor()

    cursorObj.execute('SELECT * FROM generator_seeds')

    return cursorObj.fetchall()


def close_sql_connection(conn):
    conn.close()


if __name__ == '__main__':
    connection = open_sql_connection()

    # ONLY FIRST TIME, THEN COMMENT
    sql_table_seeds(connection)

    insert_into_seeds(connection, 8106)

    rows = sql_fetch(connection)
    for row in rows:
        print(row)

    print(fetch_id(connection, 1))

