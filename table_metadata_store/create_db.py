import psycopg2
import json
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Load credentials from db_config.json
with open("table_metadata_store/db_config.json", "r") as f:
    db_config = json.load(f)

# Create PostgreSQL Database
def create_database():
    try:
        conn = psycopg2.connect(
            host=db_config["DB_HOST"],
            user=db_config["DB_USER"],
            password=db_config["DB_PASSWORD"],
            port=db_config.get("DB_PORT", 5432),
            dbname="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE {db_config['DB_NAME']};")
        print(f"Database '{db_config['DB_NAME']}' created successfully.")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {e}")

if __name__ == "__main__":
    create_database()
