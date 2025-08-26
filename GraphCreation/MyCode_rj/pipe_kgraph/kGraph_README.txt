Real-Time K-Graph Graph Creation Pipeline

1. Overview
This pipeline is designed for real-time analysis of time-series data stored
in Parquet files. It operates in two main stages:

Initial Training (Bootstrap): The pipeline first reads a large, initial chunk
 of data to train a master "k-Graph" model. This model learns the fundamental
 patterns and states from the time-series signals.

Live Analysis: After the master model is trained, the pipeline switches to a
 live analysis mode. It processes subsequent smaller chunks of data, using 
 the trained model to identify state transitions and represent them as
 directed graphs.

The entire process is multi-threaded to ensure that data reading, graph
building, and file storage can happen concurrently without blocking each other.

2. How It Works
The pipeline consists of three main components that run in parallel:

StateFileReader (read_and_emit.py): This component reads .parquet files from
 a specified input folder. It batches the data by timestamp and passes it to
 the GraphBuilder via a queue. It also tags each data package with its source
 filename.

GraphBuilder (graph_builder.py): This is the core of the pipeline.
 It receives data from the reader. Initially, it collects data until it
 has enough for the bootstrap training. Once trained, it uses the resulting
 model to quickly analyze incoming data chunks, generating graph structures
 that represent system behavior. These graphs are then sent to the
 GraphStorage component.

GraphStorage (persist.py): This component receives the fully-formed graph
 data from the builder and saves each one as a .graphml file. The filename
 is a combination of the original source file number and the timestamp of
 the data chunk, making it easy to trace back.

3. Requirements
Ensure you have the following Python libraries installed:

-pandas
-numpy
-scikit-learn
-networkx
-pyarrow

4. Setup & Configuration
Before running the pipeline, you need to configure a few parameters inside 
graph_builder.py:

tar_folder: Set this variable to the absolute or relative path of the
 directory containing your input .parquet files.

Example: tar_folder = 'C:/path/to/your/data'

selected_signals: This is a list of the column names from your Parquet
 files that you want to analyze.

Example: selected_signals = ['gpu0_mem_temp_avg', 'gpu1_mem_temp_avg']

Analysis Parameters (Optional): You can fine-tune the k-Graph algorithm
 by adjusting variables like bootstrap_chunk_size, chunk_size, num_clusters,
 min_l, and max_l to better suit your dataset.

Output Directory (Optional): The GraphStorage class in persist.py saves
 files to a folder named graph_storage by default. You can change this by
 modifying the graphs_dir parameter when GraphStorage is initialized in
 graph_builder.py.

5. How to Run
Place all your numbered .parquet files (e.g., 1.parquet, 2.parquet...) in
 the folder specified by tar_folder.

Run the main script:
python graph_builder.py

The pipeline will start, displaying progress messages in the console for
each stage (reading files, training, live analysis, and saving graphs).

6. Output
The pipeline will create a directory (default: graph_storage) and fill it
 with .graphml files. Each file represents a directed graph of the system's
 state transitions for a specific chunk of time.

The naming format for the output files is:
<file_number>_<timestamp>_graph.graphml

file_number: The number from the source .parquet file (e.g., 123.parquet).

timestamp: The starting timestamp of the first row of the data chunk that
 was analyzed.

7. File Descriptions

graph_builder.py: The main executable script. It orchestrates the entire
 pipeline, manages the threads, and contains the core configuration.

read_and_emit.py: Handles reading and parsing the input Parquet files.

kgraph_full.py: Contains the complex mathematical and algorithmic logic
 for the k-Graph training and analysis.

persist.py: Responsible for taking the final graph data and writing it
 to disk as .graphml files.

