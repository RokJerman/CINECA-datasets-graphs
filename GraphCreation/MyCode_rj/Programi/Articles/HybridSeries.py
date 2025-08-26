import networkx as nx
import pandas as pd
import numpy as np
import itertools
import os

from sklearn.decomposition import PCA
# function from the SciPy library to calculate the distance between embedding vectors.
from scipy.spatial.distance import cosine

G = nx.Graph()
#=======================================================================================================================
# CONFIGURATION
#=======================================================================================================================

dataset_path = 'Dataset'
daily_subsequence_length = 96
weekly_window_size = 672
all_weekly_embeddings = {}
column = "total_power_max"

#=======================================================================================================================
# PHYSICAL RACK NODE CREATION (ADD MANUALLY)
#=======================================================================================================================
# Adding each rack as a rack type node

G.add_node(33, type='rack', x=20, y=10)
G.add_node(34, type='rack', x=19, y=10)
G.add_node(35, type='rack', x=18, y=10)

G.add_node(18, type='rack', x=20, y=6)
G.add_node(19, type='rack', x=19, y=6)
G.add_node(20, type='rack', x=18, y=6)

G.add_node(0, type='rack', x=21, y=2)
G.add_node(1, type='rack', x=20, y=2)
G.add_node(2, type='rack', x=19, y=2)
G.add_node(3, type='rack', x=18, y=2)

#=======================================================================================================================
# PHYSICAL SERVER NODE CREATION
#=======================================================================================================================

# Adding each parquet file as a server type node
# Loop through each folder in the dataset_path directory
for rack_id_str in os.listdir(dataset_path):
    rack_path = os.path.join(dataset_path, rack_id_str)

    # Make sure it's actually a folder
    if os.path.isdir(rack_path):
        # Get a list of all parquet files inside the rack folder
        parquet_files = [f for f in os.listdir(rack_path) if f.endswith('.parquet')]

        # Sort the files numerically based on their name
        parquet_files.sort(key=lambda name: int(name.split('.')[0]))

        # Loop through the sorted files to create a server node for each
        for z_index, filename in enumerate(parquet_files):
            rack_id = int(rack_id_str)
            server_id = f"{rack_id}-{z_index}"

            G.add_node(server_id, type='server', rack_id=rack_id, z_index=z_index)

#=======================================================================================================================
# PHYSICAL EDGE CREATION
#=======================================================================================================================

#-----------------------------------------------------------------------------------------------------------------------
# SERVER VERTCAL NEIGHBOUR EDGE CREATION
#-----------------------------------------------------------------------------------------------------------------------

# Adding edges to server nodes that are above/below one another on the same rack
for server_id, attributes in G.nodes(data=True):
    if attributes.get('type') == 'server':
        rack_id = attributes.get('rack_id')
        z_index = attributes.get('z_index')
        G.add_edge(server_id, rack_id, type='PART_OF')
        if z_index > 0:
            server_below_id = f"{rack_id}-{z_index - 1}"
            G.add_edge(server_id, server_below_id, type='ABOVE')

#-----------------------------------------------------------------------------------------------------------------------
# RACK ADJACENT TO RACK EDGE CREATION
#-----------------------------------------------------------------------------------------------------------------------

rack_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == "rack"]
for rack1, rack2 in itertools.combinations(rack_nodes, 2):
    x1, y1 = G.nodes[rack1]['x'], G.nodes[rack1]['y']
    x2, y2 = G.nodes[rack2]['x'], G.nodes[rack2]['y']
    if y1 == y2 and abs(x1 - x2) == 1:
        G.add_edge(rack1, rack2, type='ADJACENT_TO')

#-----------------------------------------------------------------------------------------------------------------------
# SAME LEVEL SERVER EDGE CREATION
#-----------------------------------------------------------------------------------------------------------------------
servers_by_rack = {}
for server_id, attr in G.nodes(data=True):
    if attr.get('type') == 'server':
        rack_id = attr.get('rack_id')
        z_index = attr.get('z_index')
        if rack_id not in servers_by_rack:
            servers_by_rack[rack_id] = {}
        servers_by_rack[rack_id][z_index] = server_id

for rack1, rack2, attr in G.edges(data=True):
    if attr.get('type') == 'ADJACENT_TO':
        servers_in_rack1 = servers_by_rack.get(rack1, {})
        servers_in_rack2 = servers_by_rack.get(rack2, {})
        common_z_indexes = set(servers_in_rack1.keys()) & set(servers_in_rack2.keys())
        for z in common_z_indexes:
            G.add_edge(servers_in_rack1[z], servers_in_rack2[z], type='SAME_LEVEL_AS')

#=======================================================================================================================
# TIME SERIES DATA
#=======================================================================================================================

def generate_embedding_from_series(series_data, subsequence_length):
    if len(series_data) < subsequence_length:
        return None

    length = subsequence_length
    convolution_size = length // 3
    if convolution_size == 0: return None

    initial_P = []
    first_subseq = series_data[:length]
    for j in range(length - convolution_size + 1):
        window_sum = np.sum(first_subseq[j:j + convolution_size])
        initial_P.append(window_sum)
    initial_P = np.array(initial_P)

    Proj = [initial_P]
    for i in range(1, len(series_data) - length + 1):
        new_window_sum = np.sum(series_data[i + length - convolution_size : i + length])
        new_P_vector = np.empty_like(Proj[-1])
        new_P_vector[:-1] = Proj[-1][1:]
        new_P_vector[-1] = new_window_sum
        Proj.append(new_P_vector)
    Proj = np.array(Proj)

    if Proj.shape[0] < 3 or Proj.shape[1] < 3:
        return None

    pca = PCA(n_components=3)
    Proj_reduced = pca.fit_transform(Proj)

    max_val, min_val = np.max(series_data), np.min(series_data)
    input_value = (max_val - min_val) * convolution_size
    ref_input_vector = np.full((1, Proj.shape[1]), input_value)
    v_ref = pca.transform(ref_input_vector)[0]

    def _calculate_scalar_angle(v1, v2):
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        dot_product = np.dot(v1_norm, v2_norm)
        return np.arccos(np.clip(dot_product, -1.0, 1.0))

    u_x, u_y, u_z = np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])
    phi_x, phi_y, phi_z = _calculate_scalar_angle(u_x, v_ref), _calculate_scalar_angle(u_y, v_ref), _calculate_scalar_angle(u_z, v_ref)

    def _get_rotation_matrix_x(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def _get_rotation_matrix_y(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    def _get_rotation_matrix_z(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    R_ux, R_uy, R_uz = _get_rotation_matrix_x(phi_x), _get_rotation_matrix_y(phi_y), _get_rotation_matrix_z(phi_z)
    current_SProj_3d = Proj_reduced @ R_ux.T @ R_uy.T @ R_uz.T
    SProj = current_SProj_3d[:, 1:]

    # np.mean(SProj, axis=0)
    # The average of all the ry values for that week.
    # The average of all the rz values for that week.
    # final_embedding is an array containing [average_ry, average_rz]
    final_embedding = np.mean(SProj, axis=0)
    return final_embedding

#=======================================================================================================================
# MAIN LOOP TO PROCESS ALL SERVERS
#=======================================================================================================================
print("Starting embedding generation..")
"""
It loops through each rack folder and each server's Parquet file within it. For each server, it breaks down its entire
power usage time series into weekly chunks. It then calls 'generate_embedding_from_series' on each 
of these weekly chunks to convert it into a single vector that represents that week's behavior. Finally, it
stores all these vectors in a dictionary, organized by week number and server ID.
"""

# Start a loop that iterates through every file and folder inside the 'Dataset' directory.
# 'os.listdir' gets a list of all item names, and 'rack_id_str' will be the file name e.g., '0', '1', '18'.
for rack_id_str in os.listdir(dataset_path):

    # Create a path to the item.
    # 'os.path.join' combines path parts e.g., 'Dataset' + '0' -> 'Dataset/0'.
    rack_path = os.path.join(dataset_path, rack_id_str)

    # Check if the current item is a directory
    if os.path.isdir(rack_path):

        # Get a list of all files inside the current rack folder that end with '.parquet'.
        parquet_files = [f for f in os.listdir(rack_path) if f.endswith('.parquet')]

        # It sorts the list of filenames numerically, not alphabetically.
        # key=lambda name:        Specifies a custom sorting rule.
        # name.split('.')[0]      Takes a filename like '10.parquet', splits it at '.', and takes the first part ('10').
        # int(...)                Converts the datatype to int.
        parquet_files.sort(key=lambda name: int(name.split('.')[0]))

        # Start a new loop that goes through the sorted list of Parquet files for the current rack.
        # We use index ('enumerate') for z_index
        for z_index, filename in enumerate(parquet_files):

            # Convert the rack ID from a string (e.g., '0') to an integer (0).
            rack_id = int(rack_id_str)

            # Create a unique ID for the server
            server_id = f"{rack_id}-{z_index}"
            print(f"Processing server: {server_id}")

            # Create the full path to the specific server's Parquet file.
            file_path = os.path.join(rack_path, filename)

            # Use the pandas library to read the Parquet file into a dataframe
            df = pd.read_parquet(file_path)

            # Prepare the time-series data for analysis.
            #   df["total_power_max"]     Selects just the 'total_power_max' column from the DataFrame.
            #   .dropna()                 Removes any rows with missing (NaN) values.
            #   .values                   Converts the data from a pandas Series into a NumPy array for faster execution.
            series = df[column].dropna().values

            # Start the innermost loop. This slides a window across the server's full time series.
            # 'range(start, stop, step)'        it starts at index 0 and jumps forward by 'weekly_window_size' each time.
            for i in range(0, len(series), weekly_window_size):

                # Calculate the week number.
                week_index = i // weekly_window_size

                # Extract the data for the current week using array slicing.
                weekly_data_chunk = series[i: i + weekly_window_size]

                # Call our function to turn this week's data into a single embedding vector.
                embedding = generate_embedding_from_series(weekly_data_chunk, daily_subsequence_length)

                if embedding is not None:

                    # Check if we have seen this week_index before.
                    if week_index not in all_weekly_embeddings:
                        # If not, create a new empty dictionary for it.
                        all_weekly_embeddings[week_index] = {}

                    # Store the calculated embedding in our nested dictionary.
                    # The structure is: {week_index: {server_id: embedding_vector}}
                    all_weekly_embeddings[week_index][server_id] = embedding

#=======================================================================================================================
# WEEKLY SIMILARITY GRAPH CREATION
#=======================================================================================================================
"""
Builds a series of weekly "similarity graphs" based on the server embeddings you previously generated. For each week,
it creates a new graph where an edge is drawn between any two servers if their behavior was more than 95% similar.
The final output is a dictionary of these weekly graphs
"""

print("\nBuilding weekly similarity graphs...")

# This will store our final graphs, one for each week.
weekly_graphs = {}
similarity_threshold = 0.95  # Connect servers if they are 95% similar or more.

# Loop through each week that we have embeddings for
for week_index, embeddings_dict in all_weekly_embeddings.items():

    # Create a new, empty graph for this week's similarity data.
    G_week = nx.Graph()

    # Get a list of all servers that have an embedding for this week.
    server_ids = list(embeddings_dict.keys())

    # Compare every server with every other server for this week.
    for server1, server2 in itertools.combinations(server_ids, 2):
        embedding1 = embeddings_dict[server1]
        embedding2 = embeddings_dict[server2]

        # Calculate cosine similarity. The function calculates distance (0=identical, 2=opposite),
        # so we subtract from 1 to get similarity (1=identical, -1=opposite).
        cosine_similarity = 1 - cosine(embedding1, embedding2)

        # If the servers are very similar, add a 'SIMILAR_TO' edge
        if cosine_similarity >= similarity_threshold:
            G_week.add_node(server1)  # Ensure nodes exist before adding edge
            G_week.add_node(server2)
            G_week.add_edge(server1, server2, type='SIMILAR_TO', weight=cosine_similarity)

    weekly_graphs[week_index] = G_week
    print(
        f"  - Week {week_index}: Created graph with {G_week.number_of_nodes()} nodes and {G_week.number_of_edges()} similarity edges.")

#=======================================================================================================================
# PRINT SPECIFIC WEEK
#=======================================================================================================================
target_week = 90

if target_week in all_weekly_embeddings:
    print(f"\n--- Displaying all server embeddings for Week {target_week} ---")

    # Get the inner dictionary containing all server data for that specific week
    embeddings_for_target_week = all_weekly_embeddings[target_week]

    # Loop through each server and its embedding in that week's dictionary
    # .items() gives us both the key (server_id) and the value (embedding)
    for server_id, embedding in embeddings_for_target_week.items():
        # Print the server ID and its calculated embedding vector
        print(f"  Server: {server_id}, Embedding: {embedding}")

else:
    print(f"\nWeek {target_week} was not found in the dataset.")