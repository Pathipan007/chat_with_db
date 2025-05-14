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

    # Ensure the metadata table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS table_metadata (
            table_name TEXT PRIMARY KEY,
            columns TEXT[],
            column_types TEXT[],
            primary_keys JSONB,
            foreign_keys JSONB,
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

        # Group columns by table index
        table_columns = {i: [] for i in range(len(table_names))}
        table_col_types = {i: [] for i in range(len(table_names))}

        for i, (table_idx, col_name) in enumerate(column_names):
            if table_idx == -1:
                continue
            table_columns[table_idx].append(col_name)
            table_col_types[table_idx].append(column_types[i])

        # Flatten primary_keys to handle nested lists
        flat_primary_keys = [pk if isinstance(pk, int) else pk[0] for pk in primary_keys]

        # Insert metadata for each table
        for table_idx, table_name in enumerate(table_names):
            columns = table_columns[table_idx]
            col_types = table_col_types[table_idx]

            # Find primary keys that belong to this table
            pk_indices = [
                i for i in flat_primary_keys
                if column_names[i][0] == table_idx
            ]

            # Make primary keys more readable
            pk_readable = [
                {
                    "column": column_names[pk_idx][1],
                    "index": pk_idx
                }
                for pk_idx in pk_indices
            ]

            # Make foreign keys more readable
            fk_readable = []
            for from_idx, to_idx in foreign_keys:
                from_table_idx, from_col_name = column_names[from_idx]
                to_table_idx, to_col_name = column_names[to_idx]

                # Filter out foreign keys that do not associate with the current table
                if from_table_idx != table_idx and to_table_idx != table_idx:
                    continue

                fk_readable.append({
                    "from_table": table_names[from_table_idx],
                    "from_column": from_col_name,
                    "from_index": from_idx,
                    "to_table": table_names[to_table_idx],
                    "to_column": to_col_name,
                    "to_index": to_idx
                })

            # Insert metadata into table_metadata
            cur.execute("""
                INSERT INTO table_metadata (db_id, table_name, columns, column_types, primary_keys, foreign_keys)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (table_name) DO NOTHING;
            """, (db_id, table_name, columns, col_types, json.dumps(pk_readable), json.dumps(fk_readable)))

            print(f"Inserted metadata for table: {table_name}")

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