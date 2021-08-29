import sqlite3


class GeneratorSeedsDb:
    def __init__(self, path: str):
        self.db_filename = path + '/generator_seeds.db'
        self.create_sql_table_seeds()

    def create_sql_table_seeds(self):

        connection = sqlite3.connect(self.db_filename)

        connection.cursor().execute(
            "CREATE TABLE generator_seeds("
            "ID INTEGER PRIMARY KEY AUTOINCREMENT, "
            "SEED INTEGER,"
            "UNIQUE (SEED)"
            ")"
        )

        connection.commit()

    def insert_seeds(self, seeds: [int]):

        connection = sqlite3.connect(self.db_filename)

        for seed in seeds:
            connection.cursor().execute(
                "INSERT INTO generator_seeds(seed) VALUES(?)",
                (seed,))

        connection.commit()

    def fetch_id(self, id: int):

        connection = sqlite3.connect(self.db_filename)

        cursor = connection.cursor()

        cursor.execute('SELECT SEED FROM generator_seeds WHERE ID == ?', (id,))

        return cursor.fetchall()

    def fetch_all(self):

        connection = sqlite3.connect(self.db_filename)

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
