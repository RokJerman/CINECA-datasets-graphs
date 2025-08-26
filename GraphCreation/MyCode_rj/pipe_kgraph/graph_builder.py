import threading
from queue import Queue
import pandas as pd
import glob
import os
import numpy as np

from read_and_emit import StateFileReader
from persist import GraphStorage
from kgraph_full import run_full_kgraph_training, extract_subsequences

# --- Configuration ---
# Folder containing your .parquet files that read_and_emit will use
tar_folder = 'C:/Users/rokje/PycharmProjects/pythonProject/venv/Scripts/data'
# size of the initial training chunk.
bootstrap_chunk_size = 3000
# The number of different subsequence lengths for the initial training.
m = 35
# size of received chunk after training
chunk_size = 2500
# Minimum and maximum subsequence lengths
min_l = 100
max_l = 1500
# size of sample used out of each subsequence to be used in training
sample_size = 75
# The columns from the Parquet file to analyze.
selected_signals = [
    'gpu0_mem_temp_avg', 'gpu1_mem_temp_avg',
    'gpu3_mem_temp_avg', 'gpu4_mem_temp_avg'
]
# Number of desired clusters
num_clusters = 2

class GraphBuilder:
    def __init__(self, buffer, output_queue):
        self.buffer = buffer
        self.output_queue = output_queue
        self.data_chunk = []
        self.model = None  # This will hold the one-time trained master model
        self.is_bootstrapping = True  # Flag to control the current stage
        self.bootstrap_buffer = []  # Buffer for the large initial training data

    def run(self):
        while True:
            state_data = self.buffer.get()
            if state_data is None:
                print("GraphBuilder received termination signal.")
                if not self.is_bootstrapping and self.data_chunk:
                    self.build_and_dispatch_graph()
                self.output_queue.put(None)
                break

            if self.is_bootstrapping:
                # --- STAGE 1: BOOTSTRAPPING (Collect data for training) ---
                self.bootstrap_buffer.append(state_data)
                if len(self.bootstrap_buffer) >= bootstrap_chunk_size:
                    # Use build_graph method for the one-time training.
                    self.build_graph(self.bootstrap_buffer)
            else:
                # --- STAGE 2: LIVE PROCESSING (Analyze small chunks) ---
                self.data_chunk.append(state_data)
                if len(self.data_chunk) >= chunk_size:
                    # Use build_and_dispatch_graph for ongoing analysis.
                    self.build_and_dispatch_graph()

        print("GraphBuilder processing loop finished.")

    def build_and_dispatch_graph(self):
        """
        This method is used for fast analysis of small chunks.
        """
        # df_chunk = pd.DataFrame([row[1] for row in self.data_chunk])
        df_chunk = pd.DataFrame([pkg['data'][1] for pkg in self.data_chunk])

        # Extract components from the stored master model
        nodes = self.model['nodes']
        pca = self.model['pca_model']
        l = self.model['l']

        # --- Fast Analysis (No re-training) ---
        dataset = [df_chunk[col].dropna().to_numpy() for col in selected_signals]
        node_paths = []
        for series in dataset:
            if len(series) < l: continue
            subsequences = extract_subsequences(series, l)
            if subsequences.shape[0] == 0: continue

            s_proj = pca.transform(subsequences)
            distances_sq = np.sum((s_proj[:, np.newaxis, :] - nodes[np.newaxis, :, :]) ** 2, axis=-1)
            node_paths.append(np.argmin(distances_sq, axis=1))

        # Count transitions for this new chunk
        transitions = {}
        for path in node_paths:
            for i in range(len(path) - 1):
                from_node, to_node = int(path[i]), int(path[i + 1])
                if from_node != to_node:
                    key = (from_node, to_node)
                    transitions[key] = transitions.get(key, 0) + 1

        # Prepare nodes in the format for persist.py
        nodes_for_storage = [(i, {'x': pos[0], 'y': pos[1]}) for i, pos in enumerate(nodes)]

        # Get timestamp from the first package in the chunk
        timestamp = str(self.data_chunk[0]['data'][1]['timestamp'])
        # Get the source filename from the first package
        source_filename = self.data_chunk[0]['source_file']
        # Extract the number from the filename
        file_number = os.path.splitext(source_filename)[0]

        # Create the new, more descriptive name
        graph_name = f"{file_number}_{timestamp}"

        output_package = {'name': "Node"+graph_name, 'nodes': nodes_for_storage, 'edges': transitions}
        self.output_queue.put(output_package)

        self.data_chunk = []  # Reset for the next chunk

    def build_graph(self, chunk):
        """
        This method is used once for the initial training on the large bootstrap chunk.
        """
        print(f"\n--- Bootstrap complete. Starting full k-Graph training on {len(chunk)} rows... ---")
        list_of_readings = [pkg['data'][1] for pkg in chunk]
        df_train = pd.DataFrame(list_of_readings)

        # Call the complete training function from our k-Graph script
        self.model = run_full_kgraph_training(
            df=df_train,
            time_series_columns=selected_signals,
            m=m,
            min_l=min_l,
            max_l=max_l,
            num_clusters=num_clusters,
            sample_size_per_series = sample_size
        )

        if self.model:
            # Switch to live processing mode
            self.is_bootstrapping = False
            self.bootstrap_buffer = []  # Free up memory
            print("\n--- Master model trained. Switching to LIVE ANALYSIS mode. ---\n")
        else:
            print("--- MODEL TRAINING FAILED. PIPELINE HALTED. ---")
            self.buffer.put(None)


if __name__ == '__main__':
    data_folder_path = tar_folder
    glob_pattern = os.path.join(data_folder_path, '*.parquet')
    all_parquet_files = glob.glob(glob_pattern)

    if not all_parquet_files:
        print(f"Error: No .parquet files found in the folder: {data_folder_path}")
    else:
        print(f"Found {len(all_parquet_files)} Parquet files to process.")

        data_queue = Queue()
        graph_queue = Queue()

        # Initialize the components
        reader = StateFileReader(buffer=data_queue, state_files=all_parquet_files)
        builder = GraphBuilder(buffer=data_queue, output_queue=graph_queue)
        storage = GraphStorage(input_queue=graph_queue)

        # Run the pipeline in separate threads
        reader_thread = threading.Thread(target=reader.read_and_emit, name="ReaderThread")
        builder_thread = threading.Thread(target=builder.run, name="BuilderThread")
        storage_thread = threading.Thread(target=storage.run, name="StorageThread")

        print("Starting pipeline threads...")
        reader_thread.start()
        builder_thread.start()
        storage_thread.start()

        reader_thread.join()
        builder_thread.join()
        storage_thread.join()

        print("\n--- Pipeline Finished ---")