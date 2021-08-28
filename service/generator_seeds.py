import sqlite3


class GeneratorSeedsDb:
    def __init__(self, path: str):
        self.db_filename = path + '/generator_seeds.db'

    def open(self):
        connection = sqlite3.connect(self.db_filename)
        GeneratorSeedsDb.create_sql_table_seeds(connection)
        return connection

    @staticmethod
    def create_sql_table_seeds(connection):

        if connection is None:
            return

        connection.cursor().execute(
            "CREATE TABLE generator_seeds("
            "ID INTEGER PRIMARY KEY AUTOINCREMENT, "
            "SEED INTEGER,"
            "UNIQUE (SEED)"
            ")"
        )

        connection.commit()

    @staticmethod
    def insert_seeds(connection, seeds: [int]):

        if connection is None:
            return

        for seed in seeds:
            connection.cursor().execute(
                "INSERT INTO generator_seeds(seed) VALUES(?)",
                (seed,))

        connection.commit()

    @staticmethod
    def fetch_id(connection, id: int):

        cursor = connection.cursor()

        cursor.execute('SELECT SEED FROM generator_seeds WHERE ID == ?', (id,))

        return cursor.fetchall()

    @staticmethod
    def fetch_all(connection):

        cursor = connection.cursor()

        cursor.execute('SELECT * FROM generator_seeds')

        return cursor.fetchall()


# if __name__ == '__main__':
#
#     db = GeneratorSeedsDb()
#     db.open_sql_connection()
#
#     # ONLY FIRST TIME, THEN COMMENT
#     # db.create_sql_table_seeds()
#
#     db.insert_seeds([8107])
#
#     rows = db.fetch_all()
#     for row in rows:
#         print(row)
#
#     print(db.fetch_id(1))
