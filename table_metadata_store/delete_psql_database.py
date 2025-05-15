import psycopg2
import json

# This code is used to delete PostgreSQL tables and databases.

# Load credentials from db_config.json
with open("table_metadata_store/db_config.json", "r") as f:
    db_config = json.load(f)

# Function to connect to PostgreSQL and delete a table within the database
def delete_table(table_name, db_config):
    try:
        conn = psycopg2.connect(
            host=db_config["DB_HOST"],
            database=db_config["DB_NAME"],
            user=db_config["DB_USER"],
            password=db_config["DB_PASSWORD"],
            port=db_config.get("DB_PORT", 5432)
        )
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute(f"DROP TABLE {table_name} CASCADE;")
        print(f"Table '{table_name}' has been deleted.")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error deleting table '{table_name}': {e}")

# Function to connect to PostgreSQL, terminate connections, and delete the database
def delete_database(db_name, db_config):
    try:
        conn = psycopg2.connect(
            host=db_config["DB_HOST"],
            database="postgres", 
            user=db_config["DB_USER"],
            password=db_config["DB_PASSWORD"],
            port=db_config.get("DB_PORT", 5432)
        )
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute(f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{db_name}'
              AND pid <> pg_backend_pid();
        """)

        cur.execute(f"DROP DATABASE IF EXISTS {db_name};")
        print(f"Database '{db_name}' has been deleted.")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error deleting database '{db_name}': {e}")

# Right now, this function destroy the entire database of the db_config.json
if __name__ == "__main__":
    db_name = db_config["DB_NAME"]
    delete_database(db_name, db_config)