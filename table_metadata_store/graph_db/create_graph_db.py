from neo4j import GraphDatabase
import json

"""
This script loads table metadata into a Neo4j Graph Database.
The nodes and relationships are created based on the schema information provided in the train_tables.json file.
The script connects to the Neo4j database using credentials from db_config.json and iterates through the schemas,
creating nodes for tables and columns, and establishing relationships such as HAS_COLUMN, HAS_PRIMARY_KEY, and FOREIGN_KEY_TO.
"""

with open("table_metadata_store/graph_db/db_config.json") as f:
    db_config = json.load(f)
with open("bird/data/train/train_tables.json") as f:
    schemas = json.load(f)

URI = "bolt://localhost:7687"
AUTH = (db_config["username"], db_config["password"])

driver = GraphDatabase.driver(URI, auth=AUTH)

def load_schema(tx, db_id, table_names, column_names, column_types, primary_keys, foreign_keys):
    # Create Table nodes
    for idx, table_name in enumerate(table_names):
        full_table_name = f"{db_id}.{table_name}"
        tx.run(
            "MERGE (t:Table {db_id: $db_id, name: $name, full_name: $full_name})",
            db_id=db_id, name=table_name, full_name=full_table_name
        )

    # Create Column nodes and HAS_COLUMN/HAS_PRIMARY_KEY edges
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:  # skip *
            continue
        table_name = table_names[table_idx]
        full_table_name = f"{db_id}.{table_name}"
        full_col_name = f"{db_id}.{table_name}.{col_name}"
        data_type = column_types[col_idx]
        is_pk = col_idx in [pk if isinstance(pk, int) else pk[0] for pk in primary_keys]
        if is_pk:
            tx.run(
                """
                MERGE (c:Column {db_id: $db_id, table_name: $table_name, name: $col_name, data_type: $data_type, full_name: $full_col_name})
                WITH c
                MATCH (t:Table {full_name: $full_table_name})
                MERGE (t)-[:HAS_PRIMARY_KEY]->(c)
                """,
                db_id=db_id, table_name=table_name, col_name=col_name, data_type=data_type,
                full_col_name=full_col_name, full_table_name=full_table_name
            )
        else:
            tx.run(
                """
                MERGE (c:Column {db_id: $db_id, table_name: $table_name, name: $col_name, data_type: $data_type, full_name: $full_col_name})
                WITH c
                MATCH (t:Table {full_name: $full_table_name})
                MERGE (t)-[:HAS_COLUMN]->(c)
                """,
                db_id=db_id, table_name=table_name, col_name=col_name, data_type=data_type,
                full_col_name=full_col_name, full_table_name=full_table_name
            )

    # Create FOREIGN_KEY edges
    for fk in foreign_keys:
        if not fk: continue
        from_idx = fk[0]
        to_idx = fk[1] if len(fk) > 1 else None
        if to_idx is None: continue
        from_table_idx, from_col_name = column_names[from_idx]
        to_table_idx, to_col_name = column_names[to_idx]
        from_full_col = f"{db_id}.{table_names[from_table_idx]}.{from_col_name}"
        to_full_col = f"{db_id}.{table_names[to_table_idx]}.{to_col_name}"
        tx.run(
            """
            MATCH (from:Column {full_name: $from_full_col})
            MATCH (to:Column {full_name: $to_full_col})
            MERGE (from)-[:FOREIGN_KEY_TO]->(to)
            """,
            from_full_col=from_full_col, to_full_col=to_full_col
        )


# Iterate through the schemas in the train_tables.json file and load them into the graph database
with driver.session() as session:
    for schema in schemas:
        session.execute_write(
            load_schema,
            schema["db_id"],
            schema["table_names"],
            schema["column_names"],
            schema["column_types"],
            schema["primary_keys"],
            schema["foreign_keys"]
        )

driver.close()