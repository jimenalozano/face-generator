import sqlite3


class GeneratorSeedsDb:
    def __init__(self, path: str):
        self.db_filename = path + '/generator_seeds.db'
        self.connection = None

    def open_sql_connection(self):
        self.connection = sqlite3.connect(self.db_filename)

    def create_sql_table_seeds(self):
        cursorObj = self.connection.cursor()

        cursorObj.execute(
            "CREATE TABLE generator_seeds("
            "ID INTEGER PRIMARY KEY AUTOINCREMENT, "
            "SEED INTEGER,"
            "UNIQUE (SEED)"
            ")"
        )

        self.connection.commit()

    def insert_seeds(self, seeds: [int]):
        cursorObj = self.connection.cursor()

        for seed in seeds:
            cursorObj.execute(
                "INSERT INTO generator_seeds(seed) VALUES(?)",
                (seed,))

        self.connection.commit()

    def fetch_id(self, id: int):
        cursorObj = self.connection.cursor()

        cursorObj.execute('SELECT SEED FROM generator_seeds WHERE ID == ?', (id,))

        return cursorObj.fetchall()

    def fetch_all(self):
        cursorObj = self.connection.cursor()

        cursorObj.execute('SELECT * FROM generator_seeds')

        return cursorObj.fetchall()

    def close_sql_connection(self):
        self.connection.close()


if __name__ == '__main__':

    db = GeneratorSeedsDb()
    db.open_sql_connection()

    # ONLY FIRST TIME, THEN COMMENT
    # db.create_sql_table_seeds()

    db.insert_seeds([8107])

    rows = db.fetch_all()
    for row in rows:
        print(row)

    print(db.fetch_id(1))
