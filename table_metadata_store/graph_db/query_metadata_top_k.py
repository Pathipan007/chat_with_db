from neo4j import GraphDatabase
from collections import defaultdict
import json
from steiner_tree import KouMarkowskyAlgorithm
"""
This module retrieves metadata for tables in a Neo4j Graph Database.
It implements the Steiner Tree approximation algorithm to find the minimum spanning tree
that connects a set of terminal nodes (tables).
"""
class TopKSteinerMetadata:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.steiner = KouMarkowskyAlgorithm(uri, auth)

    def close(self):
        self.driver.close()
        self.steiner.close()

    # TODO: Need to check the true format of top_k
    @staticmethod
    def extract_info_top_k(top_k):
        """
        Expects top_k as a list of dicts with at least 'db_id' and 'table_name' keys.
        Returns a dict: {db_id: [table1, table2, ...], ...}
        """
        db_id_map = defaultdict(list)
        for item in top_k:
            db_id = item['db_id']
            table_name = item['table_name']
            db_id_map[db_id].append(f"{db_id}.{table_name}")
        return db_id_map

    def get_table_metadata(self, db_id, table_names):
        """
        Returns metadata for given tables: columns, data types, PK/FK info.
        """
        with self.driver.session() as session:
            query = """
            MATCH (t:Table {db_id: $db_id})-[:HAS_COLUMN|HAS_PRIMARY_KEY]->(c:Column)
            WHERE t.full_name IN $table_names
            OPTIONAL MATCH (c)-[fk:FOREIGN_KEY_TO]->(c2:Column)
            RETURN t.full_name AS table_name, c.name AS column_name, c.data_type AS data_type,
                   exists((t)-[:HAS_PRIMARY_KEY]->(c)) AS is_pk,
                   c2.full_name AS fk_target
            """
            result = list(session.run(query, db_id=db_id, table_names=table_names))
            metadata = defaultdict(list)
            relevant_columns = set(f"{rec['table_name']}.{rec['column_name']}" for rec in result)
            for record in result:
                fk_target = record["fk_target"]
                if fk_target and fk_target not in relevant_columns:
                    fk_target = None
                col_info = {
                    "column_name": record["column_name"],
                    "data_type": record["data_type"],
                    "is_pk": record["is_pk"],
                    "fk_target": fk_target
                }
                metadata[record["table_name"]].append(col_info)
            return dict(metadata)

    def get_steiner_connection_metadata(self, db_id, steiner_tables, involved_columns):
        """
        Returns metadata for steiner tables, only for involved columns.
        """
        # Use involved_columns directly in your Cypher query
        with self.driver.session() as session:
            query = """
            MATCH (t:Table {db_id: $db_id})-[:HAS_COLUMN|HAS_PRIMARY_KEY]->(c:Column)
            WHERE t.full_name IN $table_names AND c.full_name IN $col_names
            OPTIONAL MATCH (c)-[fk:FOREIGN_KEY_TO]->(c2:Column)
            RETURN t.full_name AS table_name, c.name AS column_name, c.data_type AS data_type,
                   exists((t)-[:HAS_PRIMARY_KEY]->(c)) AS is_pk,
                   c2.full_name AS fk_target
            """
            result = list(session.run(
                query,
                db_id=db_id,
                table_names=steiner_tables,
                col_names=list(involved_columns)
            ))
        metadata = defaultdict(dict)
        for record in result:
            col_key = (record["column_name"], record["data_type"], record["is_pk"])
            col_info = metadata[record["table_name"]].get(col_key, {
                "column_name": record["column_name"],
                "data_type": record["data_type"],
                "is_pk": record["is_pk"],
                "fk_target": []
            })
            fk_target = record["fk_target"]
            if fk_target and fk_target in involved_columns:
                col_info["fk_target"].append(fk_target)
            metadata[record["table_name"]][col_key] = col_info

        # Convert metadata to a list of dictionaries
        final_metadata = {}
        for table, cols in metadata.items():
            final_metadata[table] = []
            for col in cols.values():
                if not col["fk_target"]:
                    col["fk_target"] = None
                elif len(col["fk_target"]) == 1:
                    col["fk_target"] = col["fk_target"][0]
                # else keep as list
                final_metadata[table].append(col)
        return final_metadata

    def run(self, top_k):
        """
        Main entry: takes top_k, returns metadata for terminal and steiner tables for each db_id.
        """
        db_id_map = self.extract_info_top_k(top_k)
        all_results = {}
        for db_id, tables in db_id_map.items():
            steiner_result = self.steiner.steiner_tree(tables, db_id)
            involved_columns = set(steiner_result.get("involved_columns", []))
            # Use sets for membership checks
            tables_set = set(tables)
            table_nodes_set = set()
            for node in steiner_result["nodes"]:
                if node.count(".") == 1:
                    table_nodes_set.add(node)
                elif node.count(".") == 2:
                    table_name = KouMarkowskyAlgorithm.extract_table_name(node)
                    if table_name:
                        table_nodes_set.add(table_name)
            terminal_tables = list(table_nodes_set & tables_set)
            for t in tables_set:
                if t not in terminal_tables:
                    terminal_tables.append(t)
            steiner_tables = list(table_nodes_set - tables_set)
            # Get metadata
            terminal_metadata = self.get_table_metadata(db_id, terminal_tables)
            steiner_metadata = self.get_steiner_connection_metadata(
                db_id, steiner_tables, involved_columns
            )
            all_results[db_id] = {
                "terminal_tables": terminal_metadata,
                "steiner_tables": steiner_metadata
            }
        return all_results

def print_metadata(result):
    """
    Print the metadata in a readable format.
    """
    for db_id, db_data in result.items():
        print(f"\n=== Database: {db_id} ===")
        print("Terminal Tables:")
        for table, columns in db_data["terminal_tables"].items():
            print(f"  - {table}")
            for col in columns:
                pk = "PK" if col["is_pk"] else ""
                fk = f"(FK to {col['fk_target']})" if col["fk_target"] else ""
                print(f"      {col['column_name']} : {col['data_type']} {pk} {fk}")
        print("Steiner Tables (only PK/FK columns):")
        for table, columns in db_data["steiner_tables"].items():
            print(f"  - {table}")
            for col in columns:
                pk = "PK" if col["is_pk"] else ""
                fk = f"(FK to {col['fk_target']})" if col["fk_target"] else ""
                print(f"      {col['column_name']} : {col['data_type']} {pk} {fk}")
        print("-" * 40)

if __name__ == "__main__":
    with open("table_metadata_store/graph_db/db_config.json") as f:
        db_config = json.load(f)
    fetcher = TopKSteinerMetadata("bolt://localhost:7687", (db_config["username"], db_config["password"]))
    top_k = [
        {"db_id": "movie_platform", "table_name": "movies"},
        {"db_id": "movie_platform", "table_name": "lists"},
        {"db_id": "works_cycles", "table_name": "ProductSubcategory"},
        {"db_id": "works_cycles", "table_name": "ProductCategory"},
        {"db_id": "works_cycles", "table_name": "BusinessEntityContact"},
        {"db_id": "works_cycles", "table_name": "SalesOrderDetail"},
        # Add more as needed
    ]
    result = fetcher.run(top_k)
    print(json.dumps(result, indent=2))
    print_metadata(result)
    fetcher.close()
