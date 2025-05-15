import json
import psycopg2

# Load credentials from db_config.json
with open("table_metadata_store/db_config.json", "r") as f:
    db_config = json.load(f)

# PostgreSQL connection parameters
db_params = {
    "host": db_config["DB_HOST"],
    "database": db_config["DB_NAME"],
    "user": db_config["DB_USER"],
    "password": db_config["DB_PASSWORD"],
    "port": db_config.get("DB_PORT", 5432)
}

# Load train_tables.json
with open("bird/data/train/train_tables.json", "r") as f:
    train_tables = json.load(f)

# Connect to PostgreSQL
def connect_to_postgres():
    try:
        conn = psycopg2.connect(**db_params)
        print("Connected to PostgreSQL database.")
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def insert_metadata_to_postgres(schema_json, db_config):
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    print("Connected to PostgreSQL database.")

    # Create table_metadata table if it doesn't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS table_metadata (
            id SERIAL PRIMARY KEY,
            table_name TEXT,
            column_name TEXT,
            column_type TEXT,
            is_primary_key BOOLEAN DEFAULT FALSE,
            is_foreign_key BOOLEAN DEFAULT FALSE,
            foreign_key_table TEXT,
            foreign_key_column TEXT,
            foreign_key_index INTEGER,
            db_id TEXT
        );
    """)

    # Iterate through all tables in the schema_json
    for schema_entry in schema_json:
        db_id = schema_entry["db_id"]
        table_names = schema_entry["table_names"]
        column_names = schema_entry["column_names"]
        column_types = schema_entry["column_types"]
        primary_keys = schema_entry["primary_keys"]
        foreign_keys = schema_entry["foreign_keys"]

        # Flatten primary_keys to handle nested lists
        flat_primary_keys = [pk if isinstance(pk, int) else pk[0] for pk in primary_keys]

        # Insert metadata for each table
        for table_idx, table_name in enumerate(table_names):
            for col_idx, (col_table_idx, col_name) in enumerate(column_names):
                if col_table_idx != table_idx:
                    continue

                # Determine if the column is a primary key
                is_primary_key = col_idx in flat_primary_keys

                # Determine if the column is a foreign key
                is_foreign_key = False
                foreign_key_table = None
                foreign_key_column = None
                foreign_key_index = None

                for from_idx, to_idx in foreign_keys:
                    if from_idx == col_idx:
                        is_foreign_key = True
                        foreign_key_table_idx, foreign_key_column = column_names[to_idx]
                        foreign_key_table = table_names[foreign_key_table_idx]
                        foreign_key_index = to_idx
                        break

                # Insert the metadata for the column
                cur.execute("""
                    INSERT INTO table_metadata (
                        table_name, column_name, column_type, is_primary_key,
                        is_foreign_key, foreign_key_table, foreign_key_column,
                        foreign_key_index, db_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """, (
                    table_name, col_name, column_types[col_idx], is_primary_key,
                    is_foreign_key, foreign_key_table, foreign_key_column,
                    foreign_key_index, db_id
                ))

    conn.commit()
    cur.close()
    conn.close()
    print("PostgreSQL connection closed.")

if __name__ == "__main__":
    conn = connect_to_postgres()
    if conn:
        insert_metadata_to_postgres(train_tables, db_params)
        conn.close()
        print("PostgreSQL connection closed.")