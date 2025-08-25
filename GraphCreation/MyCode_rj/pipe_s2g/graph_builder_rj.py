import threading
from queue import Queue
import networkx as nx
import pandas as pd
import numpy as np
import glob
import os

from read_and_emit import StateFileReader
from persist import GraphStorage
from Series2Graph import create_graph_for_series

# Configuration
chunk_size = 168 #336;168
length = 24 #24;12
r= 50
smoothing_window = 4 # For smaller time windows use less smoothing
TARGET_SCORE_LENGTH = 1000
column = 'total_power_max'
# Copy the path to your folder containing parquet files
tar_folder = 'C:/Users/rokje/PycharmProjects/pythonProject/venv/Scripts/data'

class GraphBuilder:
    """
    It takes data from the input queue, builds a graph, and
    puts the graph on the output queue.
    """
    # Constructor
    def __init__(self, buffer, output_queue):
        self.buffer = buffer
        self.output_queue = output_queue
        self.data_chunk = [] # We store the desired number of rows before making the graph

    def run(self):
        """
        The main processing loop.
        """
        while True:
            package = self.buffer.get()  # <-- Changed

            if package is None:
                print("GraphBuilder received termination signal.")
                if self.output_queue:
                    self.output_queue.put(None)
                break

            state_data = package['data']
            source_file = package['source_file'] # Node

            self.data_chunk.append((state_data, source_file))

            if len(self.data_chunk) >= chunk_size:
                # Unpack all three results from the build_graph function
                scores, nodes, edges = self.build_graph(self.data_chunk)

                if scores is not None and nodes is not None:
                    #--------------------------------------------------------------------------
                    """
                    linear interpolation
                    
                    original_indexes = np.linspace(0, 1, num=len(original_scores))
                        This takes our original anomaly scores (for example we had 250 of them) and creates a list of
                        250 evenly spaced "x-coordinates" between 0.0 and 1.0. This is our original dot grid.

                    target_indexes = np.linspace(0, 1, num=TARGET_SCORE_LENGTH)
                        This creates our new, standardized grid. It makes a list of 1,000 evenly spaced "x-coordinates"
                        between 0.0 and 1.0.

                    resampled_scores = np.interp(...)
                        It looks at the new 1,000-point grid and, for each new point, it finds its position relative
                        to the original 250 points and calculates its value based on the straight line between them.
                    """
                    if not scores.dropna().empty:
                        original_scores = scores.dropna().values
                        # Create an array representing the original x-axis (0.0 to 1.0)
                        original_indexes = np.linspace(0, 1, num=len(original_scores))
                        # Create an array representing the new, fixed-length x-axis
                        target_indexes = np.linspace(0, 1, num=TARGET_SCORE_LENGTH)
                        # Use linear interpolation to map the original scores to the new axis
                        resampled_scores = np.interp(target_indexes, original_indexes, original_scores)
                        final_scores = pd.Series(resampled_scores)
                    else:
                        # If the original scores were empty, just pass them along.
                        final_scores = scores
                    # --------------------------------------------------------------------------
                    first_row_data, source_filename = self.data_chunk[0]
                    first_row_timestamp = str(first_row_data[1]['timestamp'])
                    combined_name = f"{source_filename}_{first_row_timestamp}"

                    # Package all results into a single dictionary
                    output_package = {
                        'name': "Node" + combined_name,
                        'scores': final_scores, # --------------------------------------------------------------------------
                        'nodes': nodes,
                        'edges': edges
                    }
                    self.output_queue.put(output_package)

                self.data_chunk = []

        print("GraphBuilder processing loop finished.")

    def build_graph(self, chunk):
        "Builds a graph out of the desired data_chunk size"

        list_of_readings = [item[0][1] for item in chunk]
        df = pd.DataFrame(list_of_readings)

        # 1. First, check if the column even exists in the DataFrame
        if column not in df.columns:
            return None, None, None

        # 2. Create the series by dropping any rows with missing values
        series_data = df[column].dropna().values

        # 3. Check for sufficient length
        if len(series_data) < length:
            print(
                f"[WARNING] Not enough valid data points ({len(series_data)}) for subsequence length ({length}). Skipping.")
            return None, None, None

        # 4. ONLY if there is enough data, check for zero variance.
        if np.std(series_data) == 0:
            print(f"[WARNING] Data has zero variance. No graph will be built.")
            return None, None, None

        anomaly_scores, edges, nodes = create_graph_for_series(
            series=series_data,
            length=length,
            r=r,
            smoothing_window=smoothing_window
        )

        return anomaly_scores, nodes, edges
if __name__ == '__main__':

    # 1. Define the PATH TO THE FOLDER containing your Parquet files
    data_folder_path = tar_folder

    # 2. Use glob to find all files ending with .parquet in that folder
    # os.path.join ensures it works on any operating system
    glob_pattern = os.path.join(data_folder_path, '*.parquet')
    all_parquet_files = glob.glob(glob_pattern)

    if not all_parquet_files:
        print(f"Error: No .parquet files found in the folder: {data_folder_path}")
    else:
        print(f"Found {len(all_parquet_files)} Parquet files to process.")

        # 3. Create the queues that connect the pipeline components
        data_queue = Queue()
        graph_queue = Queue()

        # 4. Initialize the components
        # Pass the ENTIRE LIST of file paths to the StateFileReader
        reader = StateFileReader(buffer=data_queue, state_files=all_parquet_files)
        builder = GraphBuilder(buffer=data_queue, output_queue=graph_queue)
        storage = GraphStorage(input_queue=graph_queue)

        # 5. Run the pipeline in separate threads (this part stays the same)
        reader_thread = threading.Thread(target=reader.read_and_emit)
        builder_thread = threading.Thread(target=builder.run)
        storage_thread = threading.Thread(target=storage.run)

        # Start the threads
        reader_thread.start()
        builder_thread.start()
        storage_thread.start()

        # Wait for all threads to complete
        reader_thread.join()
        builder_thread.join()
        storage_thread.join()

