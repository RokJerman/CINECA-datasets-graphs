import os
import networkx as nx
import traceback


class GraphStorage:
    """
    A class to store both anomaly scores (as CSV) and graph objects (as GraphML).
    """

    def __init__(self, input_queue=None, scores_dir='scores_storage', graphs_dir='graph_storage'):
        self.input_queue = input_queue
        self.scores_dir = scores_dir
        self.graphs_dir = graphs_dir
        os.makedirs(self.scores_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)

    def save_scores(self, name, scores):
        """Saves a pandas Series of anomaly scores to a CSV file."""
        safe_filename = f"{name.replace(':', '-')}_scores.csv"
        filepath = os.path.join(self.scores_dir, safe_filename)
        scores.to_csv(filepath, header=['anomaly_score'])
        print(f"[SAVED] Anomaly scores for '{name}' to {filepath}")

    def save_graph(self, name, nodes, edges):
        """Builds a graph from nodes/edges and saves it to a GraphML file."""
        safe_filename = f"{name.replace(':', '-')}_graph.graphml"
        filepath = os.path.join(self.graphs_dir, safe_filename)

        g = nx.Graph()
        if nodes is not None and len(nodes) > 0:
            for node_coord in nodes:
                g.add_node(tuple(node_coord))

            if edges and len(edges) > 0:
                for edge, weight in edges.items():
                    g.add_edge(edge[0], edge[1], weight=weight)

        with open(filepath, 'wb') as f:
            nx.write_graphml(g, f)

        print(f"[SAVED] Graph for '{name}' to {filepath}")

    def run(self):
        print(f"Starting Storage... Scores -> '{self.scores_dir}', Graphs -> '{self.graphs_dir}'")
        processed_count = 0
        while True:
            try:
                package = self.input_queue.get()
                if package is None:
                    break

                name = package['name']
                scores = package['scores']
                nodes = package['nodes']
                edges = package['edges']

                if scores is not None:
                    self.save_scores(name, scores)

                if nodes is not None and edges is not None:
                    self.save_graph(name, nodes, edges)

                processed_count += 1

            except Exception as e:
                print(f"Error in GraphStorage run loop: {e}")
                continue
        print(f"GraphStorage run completed. Total items processed: {processed_count}")