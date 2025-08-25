import json
from collections import defaultdict
from queue import Queue
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from copy import deepcopy
import os


class StateFileReader:
    """
    A class to read data from a list of Parquet files and put it into a buffer (queue).
    """

    def __init__(self, buffer, state_files):
        """
        Initializes the reader.

        Args:
            buffer (Queue): The queue to put the data into.
            state_files (list or str): A list of file paths or a single file path.
        """
        # Ensure state_files is always a list for consistent processing
        if isinstance(state_files, str):
            self.state_files = [state_files]
        else:
            self.state_files = state_files
        self.buffer = buffer

    def read_and_emit(self):
        """
        Reads each Parquet file in the list, processes its rows, and puts them
        into the buffer. Sends a termination signal (None) after all files are done.
        """
        # Loop through each file path provided during initialization
        for file_path in self.state_files:
            source_filename = os.path.basename(file_path).replace('.parquet', '')
            if not os.path.exists(file_path):
                print(f"[WARNING] File not found, skipping: {file_path}")
                continue

            print(f"[INFO] Now processing file: {os.path.basename(file_path)}")
            pq_file = pq.ParquetFile(file_path)

            state = {}
            current_t = None

            # The core logic for reading a single file
            for batch in pq_file.iter_batches(batch_size=100):
                batch_df = batch.to_pandas()
                for _, row in batch_df.iterrows():
                    if current_t is None:
                        current_t = row["timestamp"]
                    elif current_t != row["timestamp"]:
                        if state:  # Ensure state is not empty before putting
                            output_package = {'data': deepcopy(state), 'source_file': source_filename}
                            self.buffer.put(output_package)
                        state = {}
                        current_t = row["timestamp"]
                    state[1] = deepcopy(row.to_dict())

            # Put the very last state from the file into the buffer
            if state:
                output_package = {'data': state, 'source_file': source_filename}
                self.buffer.put(output_package)

        print("\n[INFO] All files have been processed.")
        self.buffer.put(None)
