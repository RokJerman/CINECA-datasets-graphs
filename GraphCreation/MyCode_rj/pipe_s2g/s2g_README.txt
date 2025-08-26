Real-Time Series2Graph Anomaly Graph Creation Pipeline

1. Overview
This pipeline is made to perform real-time anomaly detection on time-series 
data from Parquet files. It uses the "Series2Graph" method to transform
segments of a time series into a graph structure. By analyzing the properties
of this graph, it calculates anomaly scores for the underlying data.

The entire process is multi-threaded, allowing for concurrent data ingestion, graph construction, and storage of results, making it efficient for continuous monitoring tasks.

2. How It Works
The pipeline is composed of three primary components that operate in parallel:

StateFileReader (read_and_emit.py): This component is responsible for 
 reading .parquet files from a specified directory. It batches the
 data by timestamp and enriches each batch with its source filename
 These data packages are then passed to the GraphBuilder via a queue.

GraphBuilder (graph_builder_rj.py): This is the analytical core of the
 pipeline. It collects data packages from the reader until a chunk_size
 is reached. It then calls the Series2Graph algorithm on this chunk to
 generate a graph and a corresponding series of anomaly scores. The results
 are packaged and sent to the GraphStorage component.

GraphStorage (persist.py): This final component receives the processed
 data from the builder. It saves the anomaly scores to a .csv file and
 the graph structure to a .graphml file. The filenames are systematically
 generated from the source file number and the timestamp of the data chunk.

3. Requirements

Ensure you have the following Python libraries installed:

-pandas
-numpy
-scikit-learn
-networkx
-pyarrow

4. Setup & Configuration
All configuration is done at the top of the graph_builder_rj.py script:

tar_folder: Set this variable to the absolute path of the directory
 containing your input .parquet files.

Example: tar_folder = 'C:/path/to/your/data'

column: Specify the column name from your Parquet files that contains the
 time-series data you wish to analyze.

Example: column = 'total_power_max'

Algorithm Parameters: You can tune the Series2Graph algorithm's sensitivity
and granularity by adjusting these parameters:

chunk_size: The number of rows to process in each batch.

length: The length of the subsequence used for graph construction.

r: The number of rays used for creating graph nodes.

smoothing_window: The window size for smoothing the final anomaly scores.

Output Directories (Optional): In persist.py, the output directories are
 set to scores_storage and graph_storage by default. You can change
 these values in the GraphStorage class constructor if needed.

5. How to Run
Place all your numbered .parquet files (e.g., 1.parquet, 2.parquet...) in
 the folder specified by the tar_folder variable.

Navigate to the directory where you saved the pipeline scripts.

Execute the main script:
graph_builder_rj.py

The pipeline will start running, and you will see log messages in the
 console indicating the progress of file processing, analysis, and storage.

6. Output
The pipeline will create two directories:

scores_storage: Contains .csv files with the calculated anomaly scores
 for each processed chunk.

graph_storage: Contains .graphml files representing the graph structure
 for each processed chunk.

The naming format for the output files is:
Node<file_number>_<timestamp>_scores.csv
Node<file_number>_<timestamp>_graph.graphml

file_number: The number from the source .parquet file (e.g., 20 from 20.parquet).

timestamp: The starting timestamp of the first row of the data chunk that
 was analyzed.

7. File Descriptions

graph_builder_rj.py: The main script that orchestrates the pipeline, manages
 the threads, and holds all configuration parameters.

read_and_emit.py: Handles reading and batching data from the input
 Parquet files.

Series2Graph.py: Contains the core algorithmic logic for converting a
 time series into a graph and calculating anomaly scores.

persist.py: Manages the saving of the results (scores and graphs) to the
 disk.

