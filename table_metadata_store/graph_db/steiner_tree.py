from neo4j import GraphDatabase
import heapq
from collections import defaultdict
from functools import lru_cache

"""
This module implements the Steiner Tree approximation algorithm for Neo4j Graphs.
"""

class KouMarkowskyAlgorithm:
    """
    This class implements the Kou-Markowsky-Berman algorithm for finding a Steiner tree in a graph.
    It uses Dijkstra's algorithm to find the shortest path from multiple source nodes to all other nodes in the graph.
    The class is initialized with the URI and authentication credentials for the Neo4j database.
    """
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    """
    This function finds the Steiner tree for a given set of terminals in the graph.
    It uses the Kou-Markowsky-Berman algorithm to find the minimum spanning tree that connects all terminals.
    The function returns a dictionary containing the nodes and edges of the Steiner tree.
    """

    def steiner_tree(self, terminals, db_id):
        graph, edge_types = self.driver.session().execute_read(KouMarkowskyAlgorithm._get_trimmed_subgraph, db_id)
        terminal_set = set(terminals)
        all_tree_nodes = []
        all_tree_edges = set()
        all_involved_columns = set()
        all_edge_types = dict(edge_types)  # Copy

        components = self._get_connected_components(graph)
        for component in components:
            component_terminals = terminal_set & component
            if not component_terminals:
                continue  # No terminals in this component
            # Pick a starting terminal in this component
            start = next(iter(component_terminals))
            used_terminals = {start}
            tree_nodes = [start]
            tree_edges = set()
            involved_columns = set()
            while used_terminals != component_terminals:
                frontier, parents = self._multi_source_dijkstra(graph, set(tree_nodes))
                min_dist, best_terminal = float('inf'), None
                for terminal in component_terminals - used_terminals:
                    if frontier[terminal] < min_dist:
                        min_dist = frontier[terminal]
                        best_terminal = terminal
                if best_terminal is None:
                    break
                # Reconstruct path
                path = []
                current = best_terminal
                while parents[current] is not None:
                    prev = parents[current]
                    edge = frozenset([prev, current])
                    tree_edges.add(edge)
                    rel_type = edge_types.get(edge)
                    if rel_type in ("HAS_PRIMARY_KEY", "FOREIGN_KEY_TO"):
                        involved_columns.update([prev, current])
                    if current not in tree_nodes:
                        path.append(current)
                    current = prev
                if current not in tree_nodes:
                    path.append(current)
                for node in reversed(path):
                    tree_nodes.append(node)
                used_terminals.add(best_terminal)
            all_tree_nodes.extend(tree_nodes)
            all_tree_edges.update(tree_edges)
            all_involved_columns.update(involved_columns)
        return {
            "nodes": all_tree_nodes,
            "edges": [list(edge) for edge in all_tree_edges],
            "involved_columns": list(all_involved_columns),
            "edge_types": all_edge_types
        }

    """
    This function calls the algorithm to find Steiner tree for a given set of terminals in the graph.
    The function returns the terminal tables and Steiner tables in the order they were found.
    """
    def find_steiner_tree(self, db_id, terminals, verbose=False):
        result = self.steiner_tree(terminals, db_id)
        table_set = set()
        for node in result["nodes"]:
            if node.count(".") == 1:
                table_set.add(node)
            elif node.count(".") == 2:
                table_name = self.extract_table_name(node)
                if table_name:
                    table_set.add(table_name)

        terminal_tables = [t for t in table_set if t in terminals]
        steiner_tables = [t for t in table_set if t not in terminals]
        if verbose:
            print("\nTerminal Table Nodes (unordered):", terminal_tables)
            print("Steiner Table Nodes (unordered):", steiner_tables)
        return terminal_tables, steiner_tables

    @staticmethod
    def _multi_source_dijkstra(graph, terminals):
        """
        This function implements Dijkstra's algorithm for multiple sources.
        It finds the shortest path from multiple source nodes to all other nodes in the graph.
        It returns the distances and the previous nodes in the path.
        """
        dist = defaultdict(lambda: float('inf'))
        prev = {}
        heap = []
        for terminal in terminals:
            dist[terminal] = 0
            heapq.heappush(heap, (0, terminal))
            prev[terminal] = None

        while heap:
            current_dist, current_node = heapq.heappop(heap)
            if current_dist > dist[current_node]:
                continue
            for neighbor in graph[current_node]:
                distance = current_dist + 1
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    prev[neighbor] = current_node
                    heapq.heappush(heap, (distance, neighbor))
        return dist, prev

    @staticmethod
    def _get_subgraph(tx, db_id):
        """
        This function retrieves the subgraph from the Neo4j database.
        The subgraph is a graph of the specified database."""
        query = """
            MATCH (a)-[r:HAS_COLUMN|HAS_PRIMARY_KEY|FOREIGN_KEY_TO]-(b)
            WHERE a.db_id = $db_id AND b.db_id = $db_id
            RETURN a.full_name AS src, b.full_name AS dst, type(r) AS rel_type
            """
        graph = defaultdict(list)
        edge_types = {}
        for record in tx.run(query, db_id=db_id):
            src, dst, rel_type = record["src"], record["dst"], record["rel_type"]
            graph[src].append(dst)
            graph[dst].append(src)
            edge_types[frozenset([src, dst])] = rel_type
        return graph, edge_types

    @staticmethod
    def _get_trimmed_subgraph(tx, db_id):
        """
        Retrieves the subgraph for the given db_id, excluding column nodes of degree 1 (leaf columns).
        """
        query = """
            MATCH (n)
            WHERE n.db_id = $db_id
              AND NOT (
                n:Column AND size([(n)--()|1]) = 1
              )
            WITH collect(n) AS nodes
            UNWIND nodes AS n
            MATCH (n)-[r]-(m)
            WHERE m IN nodes
            RETURN n.full_name AS src, m.full_name AS dst, type(r) AS rel_type
        """
        graph = defaultdict(list)
        edge_types = {}
        for record in tx.run(query, db_id=db_id):
            src, dst, rel_type = record["src"], record["dst"], record["rel_type"]
            graph[src].append(dst)
            graph[dst].append(src)
            edge_types[frozenset([src, dst])] = rel_type
        return graph, edge_types
    
    @staticmethod
    @lru_cache(maxsize=128)
    def extract_table_name(node):
        """
        This function extracts the table name from a node.
        The node is expected to be in the format "db_id.table_name.column_name".
        It returns the table name in the format "db_id.table_name".
        """
        if node.count(".") == 2:
            return ".".join(node.split(".")[:2])
        return None

    @staticmethod
    def _get_connected_components(graph):
        """
        This function retrieves the connected components of the graph.
        The algorithm uses a depth-first search to find all connected components in the graph.
        It returns a list of sets, where each set contains the nodes in a connected component.
        """
        visited = set()
        components = []
        for node in graph:
            if node not in visited:
                stack = [node]
                component = set()
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        component.add(curr)
                        stack.extend([nbr for nbr in graph[curr] if nbr not in visited])
                components.append(component)
        return components


