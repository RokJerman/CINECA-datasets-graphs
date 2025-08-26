import os
import networkx as nx
import traceback

MODEL_DIR = 'model_storage'
RESULTS_DIR = 'results_storage'

class GraphStorage:
    """
    A class to store graph objects (as GraphML).
    """

    def __init__(self, input_queue=None, graphs_dir='graph_storage'):
        self.input_queue = input_queue
        self.graphs_dir = graphs_dir
        os.makedirs(self.graphs_dir, exist_ok=True)

    def save_graph(self, name, nodes, edges):
        """Builds a graph from nodes/edges and saves it to a GraphML file."""
        safe_filename = f"{name.replace(':', '-')}_graph.graphml"
        filepath = os.path.join(self.graphs_dir, safe_filename)

        g = nx.DiGraph()

        if nodes is not None and len(nodes) > 0:
            # The 'nodes' variable is now a list of tuples like
            # [(0, {'x': 1.2, 'y':-3.4}), (1, ...)]
            # add_nodes_from() correctly interprets this as adding nodes 0, 1, ...
            # with their x and y coordinates as attributes.
            g.add_nodes_from(nodes)

            if edges and len(edges) > 0:
                # nodes are named 0, 1, 2...
                for edge, weight in edges.items():
                    # Ensure edge nodes are integers
                    u, v = int(edge[0]), int(edge[1])
                    g.add_edge(u, v, weight=weight)

        with open(filepath, 'wb') as f:
            nx.write_graphml(g, f)

        print(f"[SAVED] Graph for '{name}' to {filepath}")

    def run(self):
        processed_count = 0
        while True:
            try:
                package = self.input_queue.get()
                if package is None:
                    break

                name = package['name']
                nodes = package['nodes']
                edges = package['edges']

                if nodes is not None and edges is not None:
                    self.save_graph(name, nodes, edges)

                processed_count += 1

            except Exception as e:
                print(f"Error in GraphStorage run loop: {e}")
                continue
        print(f"GraphStorage run completed. Total items processed: {processed_count}")