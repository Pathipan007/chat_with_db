from neo4j import GraphDatabase
import json

"""
This script is used to delete all nodes and relationships in a Neo4j Graph Database.
Please use this script with caution, as it will remove all data in the database.
"""

# Load credentials from db_config.json
with open("table_metadata_store/graph_db/db_config.json") as f:
    db_config = json.load(f)

URI = "bolt://localhost:7687"
AUTH = (db_config["username"], db_config["password"])

driver = GraphDatabase.driver(URI, auth=AUTH)

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")
    print("All nodes and relationships have been deleted.")

driver.close()
