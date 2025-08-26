import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

# Clustering
from sklearn.cluster import KMeans

# Load the Parquet file locally
df = pd.read_parquet("20.parquet")

# -----------------------------------------------------------------------------------------------------------------------
# --------------------------------------(A) GRAPH EMBEDDING--------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------
# SUBSEQUENCE EMBEDDING
# -----------------------------------------------------------------------------------------------------------------------
"""
Subsequence Embedding: For each time series, we collect all the subsequences of a given length ℓ into an array
called Proj(T,λ). We then concatenate all the computed Proj(T,λ) into Proj for all the time series in the dataset. 
We then sample Proj (user-defined parameter smpl) and keep only a limited number of subsequences stored in Proj smpl.
We use the latter to train a Principal Component Analysis (PCA). We then use the trained PCA and a rotation 
step to project all the subsequences into a two-dimensional space that only preserves the shapes of the subsequences.
 The result is denoted as SProj. We denote the PCA and rotation steps Reduce(Proj,pca), where pca is the trained PCA.
"""

M = 5  # Number of different subsequence lengths L
min_l = 10000  # Minimum subsequence length (PCA projection doesn’t work for < 5 points)
max_l = 80000  # represents the maximum subsequence length

time_series_columns = [
    'gpu0_mem_temp_avg',
    'gpu1_mem_temp_avg',
    'gpu3_mem_temp_avg',
    'gpu4_mem_temp_avg'
]

# Load each column into a list of time series arrays
# We create a list where each item is a NumPy array for one GPU's memory temperature
dataset = []
for column in time_series_columns:
    dataset.append(df[column].dropna().to_numpy())
print(f"Loaded {len(dataset)} time series.")
"""
dataset structure:
[
    array([36., 35., 35., 35.,...]),  # From 'gpu0_mem_temp_avg'
    array([37., 37., 37., 37.,...]),  # From 'gpu1_mem_temp_avg'
    array([34., 34., 34., 34.,...]),  # From 'gpu3_mem_temp_avg'
    array([38., 38., 38., 38.,...])   # From 'gpu4_mem_temp_avg'
]
"""

# Set the seed for random number generation so that the same L values are picked every time
np.random.seed(42)

# Generate M random L values (lengths of subsequences), between min_l and max_l (including),
# no duplicates, and sort them for easier reading.
# np.arrange() This function creates a sequence of all possible integer lengths, starting from min_l and going up to max_l
#       size = M : picks M different numbers  replace=False: each chosen number is unique
# np.random.choice() This function randomly selects values from the sequence created by np.arange
L_values = sorted(np.random.choice(np.arange(min_l, max_l + 1), size=M, replace=False))
print(f"Selected L values: {L_values}")


# Algorithm 2
def extract_subsequences(time_series, l):
    """
    Extracts all overlapping subsequences of length l from a time series.
    Algorithm 2, Lines 1-3.

    Example: time_series is [1, 2, 3, 4, 5] and length l is 3
            function returns  a numpy array ([1, 2, 3],
                                            [2, 3, 4],
                                            [3, 4, 5])
    """

    # .lib is a sub-module within the NumPy library.
    # .stride_tricks is a sub-module within np.lib.
    #       Strides are a fundamental concept in how NumPy arrays are stored and accessed in memory.
    #       They define how many bytes you need to skip in memory to get from one element to the next along a particular dimension.
    # .sliding_window_view()
    #       It returns a new array object that is a view into the original data. This view has an additional dimension,
    #       where each "slice" along this new dimension represents a window of the original data.
    return np.lib.stride_tricks.sliding_window_view(time_series,
                                                    window_shape=l)  # window shape is lenght of each window


def reduce_dimensionality_pca(proj_subsequences, sample_size):
    """
    Reduces subsequences to a two-dimensional space using PCA.
    Algorithm 2, Lines 4-6:
    """
    # Line 4: randomly select Proj elements in Proj (Projsmpl)

    # np.random.choice: This is a NumPy function that randomly chooses items from a given range.
    # proj_subsequences.shape[0]: This provides the range to choose from. If you have 50,000 subsequences,
    #       this gives the function the numbers 0 through 49,999 to pick from.
    # sample size tells the function how many random indexes to pick
    # replace=False ensures that every index it picks is unique
    sample_indexes = np.random.choice(proj_subsequences.shape[0], sample_size, replace=False)
    projsmpl = proj_subsequences[sample_indexes]

    # Line 5: train PCA
    pca = PCA(n_components=2)
    pca.fit(projsmpl)

    # Line 6: SProj = Reduce(Proj, pca)
    s_proj = pca.transform(proj_subsequences)
    s_proj_smpl = pca.transform(projsmpl)  # Also get SProj_smpl for NodeCr

    return s_proj, s_proj_smpl, pca  # Return PCA model as well for later use with individual time series


# -----------------------------------------------------------------------------------------------------------------------
# NODE CREATION
# -----------------------------------------------------------------------------------------------------------------------
"""
 Create a node for each of the densest parts
 of the above two-dimensional space. In practice, we
 perform a radial scan of SProjsmpl. For each radius, we
 collect the intersection with the trajectories of SProjsmpl,
 and we apply kernel density estimation on the intersected
 points: each local maximum of the density estimation
 curve is assigned to a node. These nodes can be seen as
 a summarization of all the major length patterns ℓ that
 occurred in D. For this step, we only consider the sampled
 collection of subsequences SProjsmpl.
"""


def find_peaks(data, min_height):
    """Finds local maxima (peaks) in a 1D array."""
    peaks = []
    # Iterate from the second to the second-to-last element
    for i in range(1, len(data) - 1):
        # Check if the current point is a peak and above the minimum height
        if data[i] > data[i - 1] and data[i] > data[i + 1] and data[i] >= min_height:
            peaks.append(i)
    return np.array(peaks)


def create_nodes(s_proj_smpl, bandwidth=0.1, num_eval_points=360):
    """
    Creates nodes by performing Kernel Density Estimation (KDE) on the
    angular distribution of the 2D projected sample.
    """

    # ---Step 1: Handle Circular Data for KDE
    """The KDE algorithm thinks your data lives on a long, straight number line. It doesn't know that -3.14
    and +3.14 are actually neighbors on a circle.
    we must use a standard trick: triplicate the data by shifting it by a full circle."""

    # np.arctan2() is a NumPy function that computes the arctangent of y/x, and takes two separate arguments, y and x
    # s_proj_smpl[:, 1]: This selects the second column (index 1) of your data, which contains all the y-coordinates.
    # s_proj_smpl[:, 0]: This selects the first column (index 0), which contains all the x-coordinates.
    angles = np.arctan2(s_proj_smpl[:, 1], s_proj_smpl[:, 0])  # An array of angles (in radians)

    """ Let's say your original angles array is very simple:
        angles = np.array([-3.0, 0, 3.0])
        (Note: np.pi is approx. 3.14, and 2 * np.pi is approx. 6.28)

        The Left Copy (angles - 2 * np.pi) becomes:
        [-3.0 - 6.28, 0 - 6.28, 3.0 - 6.28] which is [-9.28, -6.28, -3.28]

        The Original Data (angles) is:
        [-3.0, 0, 3.0]

        The Right Copy (angles + 2 * np.pi) becomes:
        [-3.0 + 6.28, 0 + 6.28, 3.0 + 6.28] which is [3.28, 6.28, 9.28]

        After np.concatenate, the three arrays are joined into one long 1D array:
        [-9.28, -6.28, -3.28, -3.0, 0.0, 3.0, 3.28, 6.28, 9.28]

        After .reshape(-1, 1), the final circular_angles array is formatted into a column:
        [[-9.28],
        [-6.28],
        [-3.28],
        [-3.0 ],
        [ 0.0 ],
        [ 3.0 ],
        [ 3.28],
        [ 6.28],
        [ 9.28]]"""
    circular_angles = np.concatenate([
        # Part 1: The Left-Shifted Copy
        angles - 2 * np.pi,

        # Part 2: The Original Data
        angles,

        # Part 3: The Right-Shifted Copy
        angles + 2 * np.pi

        # The np.concatenate function takes the three arrays above and joins them
        # into one single, long 1D array.
    ]).reshape(-1, 1)  # The .reshape() method takes our 1D array to define the new shape.
    # The 1 in (-1, 1) tells NumPy that the new array must have exactly one column.
    # The -1 is a placeholder that tells NumPy to automatically calculate the correct number
    # of rows needed to make the new shape work with all the original data.
    # Before .reshape(-1, 1), you have a 1D array like this: [10, 20, 30, 40]
    # After .reshape(-1, 1), it becomes a 2D array with one column: [[10],[20],[30],[40]]
    # We need to reshape it because of sklearn

    # ---Step 2: Fit the KDE Model
    # We now fit (train) the KDE model on this augmented circular data.
    # The 'gaussian' kernel tells the algorithm to use a smooth bell curve as the shape for each bump.
    # The bandwidth controls the size or width of each of those bell curve bumps.
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(circular_angles)

    # ---Step 3: Evaluate the Density Estimation Curve
    # -We evaluate the density at many points around the original circle to get our smooth curve.
    # -np.linspace function generates a 1D array of evenly spaced numbers. Here, it creates num_eval_points
    #       numbers starting from -np.pi and ending at +np.pi.
    # -After .reshape(-1, 1), it becomes a 2D array with each row containing one column
    eval_points = np.linspace(-np.pi, np.pi, num_eval_points).reshape(-1, 1)

    # kde.score_samples(eval_points) This part asks your trained kde model to calculate the density at each of
    #       your 360 evenly-spaced eval_points.
    # Log-densities are a computer-friendly way to represent probability values. A log-density is the natural logarithm
    #       of a real density value.
    # np.exp() takes the log-densities and converts tehm back into, human-friendly density values.
    # density_curve variable holds a NumPy array with 360 numbers, where each number is the real density
    #       at that point on the circle
    density_curve = np.exp(kde.score_samples(eval_points))

    # ---Step 4: Find Local Maxima (Peaks)
    # Find the peaks in our density curve.
    min_peak_height = np.mean(density_curve)  # A simple threshold: peak must be above average density
    peaks = find_peaks(density_curve,
                       min_height=min_peak_height)  # Returns a list of peak positions (indexes) as a NumPy array.

    if len(peaks) == 0:
        return np.array([])

    # --- Step 5: Create Nodes from Peaks
    # For each peak, the node is the centroid of the points in that slice.
    # A centroid here is the average position or the geometric center of a collection of points in that slice.
    nodes = []
    # Define a small slice width around each peak to grab the points
    slice_width = (2 * np.pi / num_eval_points)  # 6.28 / 360

    for peak_index in peaks:
        peak_angle = eval_points[peak_index][0]  # Extracts the angle of this peak point
        angle_start = peak_angle - (slice_width / 2)
        angle_end = peak_angle + (slice_width / 2)

        # Create a mask to select points that fall within this angular slice
        # A mask is an array of True and False values that is used as a filter to select data from another array.
        # In our scenario we only want the points within angle_start and angle_end
        mask = (angles >= angle_start) & (angles < angle_end)  # angles has 1000 rows
        points_in_slice = s_proj_smpl[mask]  # s_proj_smpl has 1000 rows

        if points_in_slice.shape[0] > 0:
            # axis=0 tells the np.mean() function to compute the average down each column independently.
            # It calculates the mean of all the values in Column 0 (all the x-coordinates).
            # It then calculates the mean of all the values in Column 1 (all the y-coordinates).
            # The result, node_centroid, is a new array with just two values: [average_x, average_y].
            node_centroid = np.mean(points_in_slice, axis=0)
            nodes.append(node_centroid)

    return np.array(nodes)


# -----------------------------------------------------------------------------------------------------------------------
# EDGE CREATION
# -----------------------------------------------------------------------------------------------------------------------
"""
 Edge Creation: Retrieve all transitions between pairs of
 subsequences represented by two different nodes: each
 transition corresponds to a pair of subsequences, where
 one occurs immediately after the other in a time series
 T of the dataset D. We represent transitions with an
 edge between the corresponding nodes.
 """


def create_edges(s_proj, nodes_l, series_lengths):
    """
        Creates edges by mapping each subsequence to its closest node and counting
        the transitions between consecutive subsequences.
    """
    if nodes_l.shape[0] < 2:
        return []  # Cannot create edges with fewer than 2 nodes

    # --- Step 1: Map Every Subsequence to its Closest Node
    # This is a way to calculate the Euclidean distance between
    # every point in s_proj and every node in nodes_l. It's a vectorized
    # operation, making it much faster than a standard for-loop.

    # Calculating distance between points in s_proj and nodes
    # - s_proj[:, np.newaxis, :]: This reshapes the s_proj array. If its original shape was (1000, 2), np.newaxis adds
    #       a new dimension in the middle, making its shape (1000, 1, 2). This prepares it for broadcasting.
    # - nodes_l[np.newaxis, :, :]: This does the same for the nodes_l array. If its original shape was (7, 2),
    #       it becomes (1, 7, 2).
    # - We perform the subtraction because it is the first step in calculating the Euclidean distance
    # - (np.sum(..., axis=-1)) It adds the squared x and y differences together: x_difference_squared + y_difference_squared.
    #       Example if we had 2 points and 3 nodes:
    #
    #       [[[49, 1], [4, 9], [81, 16]], [[4, 9], [49, 49], [0, 0]]]
    #
    #       [[  50,  # <-- 49 + 1       [ 13,  # <-- 4 + 9
    #           13,  # <-- 4 + 9          98,  # <-- 49 + 49
    #           97], # <-- 81 + 16         0]]  # <-- 0 + 0
    #
    #       [[50, 13, 97], [13, 98, 0]]
    distances_sq = np.sum((s_proj[:, np.newaxis, :] - nodes_l[np.newaxis, :, :]) ** 2, axis=-1)

    """
    distances_sq is a 2D numpy array, that contains the squared Euclidean distance between every subsequence and every node.
        - Each row corresponds to a single subsequence from your s_proj array.
        - Each column corresponds to the squared distance between a node and a point in that row.
        - The number at distances_sq[i, j] is the squared distance between the i-th subsequence and the j-th node.
        - s_proj and distances_sq are parallel

    """
    """BROADCASTING
        Broadcasting is how NumPy performs calculations on arrays that have different shapes without making extra copies
        of the data.

        s_proj: (9000,2)
        nodes_l: (7,2)

        Reshaped s_proj: (90000, 1, 2)
        Reshaped nodes_l: (1, 7, 2)

        Last Dimension: Both are 2. They are equal. OK.
        Middle Dimension: One is 1, one is 7. Since one is 1, NumPy can stretch it. OK.
        First Dimension: One is 90000, one is 1. Since one is 1, NumPy can stretch it. OK.

        By adding the new axes, you make the shapes compatible so broadcasting can perform the calculation correctly.
    """
    # For each row (point), find the column index (the node)
    # that has the minimum distance. This creates the full path of nodes.
    # - np.argmin returns index of lowest value in array
    # - axis=1 because you want to find the minimum distance for each row independently. The axis=1 command
    #       tells the function to search through the columns of each row.
    # The final result will be a 1D array of indexes of nodes closest to each point e.g.([1,4,2,0,5,3,4,4,6...])
    node_path = np.argmin(distances_sq, axis=1)

    # --- Step 2: Count Transitions
    edge_counts = {}
    start_index = 0

    # We must process each series's path separately to avoid creating an
    # artificial edge between the end of one GPU's series and the start of the next.
    for num_subsequences_in_series in series_lengths:
        # The end index for the current series' path
        end_index = start_index + num_subsequences_in_series

        # Get the path of nodes for just this one time series
        single_series_path = node_path[start_index:end_index]

        # Loop through each pair of nodes, for example: [1,2,4,1,5,6] is out single_series path. We will count how many
        #       transitions from one node to another occur. Like [1,2],[2,4],[4,1]...
        for i in range(len(single_series_path) - 1):
            # Define the edge as a tuple (source_node, target_node)
            source_node = single_series_path[i]
            target_node = single_series_path[i + 1]
            edge = (source_node, target_node)

            # Add the edge to our dictionary and increment its count
            # .get(edge, 0) method safely gets the current count, or returns 0 if the edge hasn't been stored before.
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

        # Update the start_index for the next time series in the dataset
        start_index = end_index

    # --- Step 3: Format the Edges
    # Convert the dictionary of counts into the final list format.
    # The format is (source_node_index, target_node_index, weight/count)
    final_edges = []
    for (u, v), count in edge_counts.items():
        final_edges.append((u, v, count))

    return final_edges


# -----------------------------------------------------------------------------------------------------------------------
# --------------------------------------(B) GRAPH CLUSTERING-------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
"""
Graph Clustering phase uses the M graphs created in the previous step to generate M different sets of cluster labels.
This is a two-step process performed for each of the M graphs:

- Feature Extraction: For each of the original time series, a feature vector is calculated based on its unique path
  through the graph. These features include counts of which nodes and edges were crossed.

- k-Means Clustering: The k-Means algorithm is applied to the matrix of these feature vectors, producing one
  set of cluster labels for that specific graph.
"""

import networkx as nx


def extract_features_for_series(time_series, l, pca_model, nodes_l, edges_l):
    """
    Extracts node, edge, and degree features for a single time series
    based on its path through a given graph.
    """
    # --- Find the path of this series through the graph
    subsequences = extract_subsequences(time_series, l)
    s_proj = pca_model.transform(subsequences)

    # Explanation on line 306
    distances_sq = np.sum((s_proj[:, np.newaxis, :] - nodes_l[np.newaxis, :, :]) ** 2, axis=-1)
    # Explanation on line 338
    node_path = np.argmin(distances_sq, axis=1)

    # np.zeros(n) Create a 1D numpy array of n length filled with values 0
    node_counts = np.zeros(len(nodes_l))
    edge_counts = np.zeros(len(edges_l))

    # Create an empty dictionary to store the mapping
    edge_to_index = {}
    # Loop through each edge
    for i in range(len(edges_l)):
        # Get the full edge tuple (u, v, w) at the current index
        edge_with_weight = edges_l[i]

        # Create the key, which is just the source node (u) and target node (v)
        u = edge_with_weight[0]
        v = edge_with_weight[1]
        edge_key = (u, v)

        # Assign the index 'i' as the value for this key
        edge_to_index[edge_key] = i

    # Count node occurrences
    # Note: the node_path inside the extract_features_for_series function contains the path for only one single time series.
    # We count how many times each node appears in the node path.
    for node_idx in node_path:
        node_counts[node_idx] += 1

    # Count edge occurrences
    # The edges_l variable contains the total weight (the total transition count) from all four series combined.
    #       We only want the number of edges from this series
    for i in range(len(node_path) - 1):
        edge_tuple = (node_path[i], node_path[i + 1])
        if edge_tuple in edge_to_index:
            edge_idx = edge_to_index[edge_tuple]
            edge_counts[edge_idx] += 1

    # Degree features

    # path_graph = nx.DiGraph(): This line creates a new, empty directed graph object using the networkx library.
    path_graph = nx.DiGraph()

    # Creates a list of all the consecutive transitions (edges) from the node_path.
    path_edges = list(zip(node_path[:-1], node_path[1:]))

    # path_graph.add_edges_from(path_edges): This method takes the list of edges created in the previous step and adds
    #       them all to the empty graph object, creating a graph of just that one path.
    path_graph.add_edges_from(path_edges)

    # Counts the number of degrees (connections) of each node
    degree_features = np.zeros(len(nodes_l))
    for i in range(len(nodes_l)):
        if i in path_graph:
            degree_features[i] = path_graph.degree(i)

    # Combine all features into a single vector
    feature_vector = np.concatenate([node_counts, edge_counts, degree_features])

    # Normalize the feature vector
    mean = feature_vector.mean()
    std = feature_vector.std()
    if std == 0:
        return np.zeros_like(feature_vector)  # Avoid division by zero

    normalized_features = (feature_vector - mean) / std

    return normalized_features


# -----------------------------------------------------------------------------------------------------------------------
# --------------------------------------(D) INTERPRETABILITY AND EXPLAINIBILITY -----------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

from sklearn.metrics import adjusted_rand_score  # We import ARI (Adjusted Random Index)


def select_best_graph(graph_models, all_cluster_labels, final_labels, dataset):
    """
    Selects the most relevant graph based on consistency (ARI) and an
    interpretability factor.
    """
    print("\n--- Selecting Best Graph for Interpretability ---")
    best_score = -1
    best_model_info = None
    best_node_paths = None

    # Loop through each of the 5 graph models to calculate its score
    for index, model in enumerate(graph_models):
        curr_graph_labels = all_cluster_labels[index]

        consistency = adjusted_rand_score(final_labels, curr_graph_labels)

        # This measures how "exclusive" the nodes of this graph are to the final clusters
        num_clusters = len(np.unique(final_labels))
        max_exclusivity_per_cluster = []

        # We need to recalculate the node paths for this specific graph
        node_paths = []
        for series in dataset:
            subsequences = extract_subsequences(series, model['l'])
            s_proj = model['pca_model'].transform(subsequences)
            distances_sq = np.sum((s_proj[:, np.newaxis, :] - model['nodes'][np.newaxis, :, :]) ** 2, axis=-1)
            node_paths.append(np.argmin(distances_sq, axis=1))

        for cluster_id in range(num_clusters):
            indexes_for_this_cluster = np.where(final_labels == cluster_id)[0]
            max_exclusivity_for_this_cluster = 0

            for node_id in range(len(model['nodes'])):
                members_using_node = 0
                for series_idx in indexes_for_this_cluster:
                    if node_id in node_paths[series_idx]:
                        members_using_node += 1

                total_using_node = 0
                for path in node_paths:
                    if node_id in path:
                        total_using_node += 1

                if total_using_node > 0:
                    exclusivity = members_using_node / total_using_node
                    if exclusivity > max_exclusivity_for_this_cluster:
                        max_exclusivity_for_this_cluster = exclusivity

            max_exclusivity_per_cluster.append(max_exclusivity_for_this_cluster)

        interpretability_factor = np.mean(max_exclusivity_per_cluster)

        score = consistency * interpretability_factor
        print(
            f"  Graph L= {model['l']}: Consistency(ARI)= {consistency:.2f}, Interpretability= {interpretability_factor:.2f}, Final Score= {score:.2f}")

        if score > best_score:
            best_score = score
            best_model_info = model
            best_node_paths = node_paths

    print(f"\nBest graph found for L = {best_model_info['l']} with a score of {best_score:.2f}")
    return best_model_info, best_node_paths

def compute_interpretability(best_model, final_labels, dataset, node_paths):
    """
    Finds and displays the most representative and exclusive node (pattern) for each cluster.
    """
    l = best_model['l']
    pca_model = best_model['pca_model']
    nodes = best_model['nodes']
    num_clusters = len(np.unique(final_labels))

    # --- Step 1: Gather subsequences using the pre-calculated paths ---
    # This part gathers the raw data needed for the plots at the very end.
    all_subsequences_by_node = {i: [] for i in range(len(nodes))}

    for series_idx, path in enumerate(node_paths):
        series = dataset[series_idx]
        subsequences = extract_subsequences(series, l)

        for sub_idx, node_id in enumerate(path):
            all_subsequences_by_node[node_id].append(subsequences[sub_idx])

    # --- Step 2: Find the best node for each cluster ---
    for cluster_id in range(num_clusters):
        # Get the indexes of the GPUs in the current cluster
        indexes_for_curr_cluster = np.where(final_labels == cluster_id)[0]
        num_series_in_cluster = len(indexes_for_curr_cluster)

        best_node_for_cluster = -1
        best_score = -1
        best_representativity = -1
        best_exclusivity = -1

        print(f"\nAnalyzing Cluster {cluster_id}:")

        # Check every node to see how well it represents this cluster
        for node_id in range(len(nodes)):
            members_using_node = 0
            for series_idx in indexes_for_curr_cluster:
                if node_id in node_paths[series_idx]:
                    members_using_node += 1

            total_using_node = 0
            for path in node_paths:
                if node_id in path:
                    total_using_node += 1

            if total_using_node == 0 or num_series_in_cluster == 0:
                continue
            # --- Calculate Representativity and Exclusivity ---
            representativity = members_using_node / num_series_in_cluster
            exclusivity = members_using_node / total_using_node

            score = representativity * exclusivity

            if score > best_score:
                best_score = score
                best_node_for_cluster = node_id
                best_representativity = representativity
                best_exclusivity = exclusivity

        if best_node_for_cluster != -1:
            print(f"  - Most interpretable pattern is Node {best_node_for_cluster}.")
            print(f"    - Representativity: {best_representativity:.2f}")
            print(f"    - Exclusivity: {best_exclusivity:.2f}")

            # --- Step 3: Visualize the Characteristic Pattern ---
            # This check is now safer to prevent any crashes.
            if best_node_for_cluster in all_subsequences_by_node and all_subsequences_by_node[best_node_for_cluster]:
                centroid_subsequence = np.mean(all_subsequences_by_node[best_node_for_cluster], axis=0)

                plt.figure(figsize=(8, 3))
                plt.title(f"Pattern for Cluster {cluster_id} (from Node {best_node_for_cluster})")
                plt.plot(centroid_subsequence)
                plt.grid(True)
                plt.show()


# -----------------------------------------------------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------------------------------------------------
graph_models = []  # list containing five dictionaries, one for each of the five graphs generated.

for l in L_values:
    print(f"\nProcessing L = {l}")

    # Step 1: Get a sample to train PCA (Memory Efficient)
    # Instead of creating all subsequences first, we will get a small sample from each time series
    # and combine only those samples.

    sample_subsequences_list = []
    # We want a total sample of 1000, so we take 250 from each of the 4 GPUs.
    sample_size_of_series = 250

    for series in dataset:
        # len(series): total number of data points
        possible_num_subsequences = len(series) - l + 1
        # Randomly choose indexes from this one series
        # min(sample_size_of_series, num_subs_in_series) prevents crash if possible_num_subsequences < 250
        # np.random.choice(1st arg:population to choose from |2nd arg: number of indexes picked |3rd arg: make sure all indexes are unique)
        random_indexes = np.random.choice(possible_num_subsequences,
                                          min(sample_size_of_series, possible_num_subsequences), replace=False)

        # Grabs 250 slices of length L with the random indexes
        for index in random_indexes:
            sample_subsequences_list.append(series[index: index + l])

    # Create a small array from our collected samples to train the PCA
    projsmpl = np.array(sample_subsequences_list)

    # Train PCA on the small, combined sample
    pca = PCA(n_components=2)
    pca.fit(projsmpl)

    # Step 2: Transform all subsequences in batches
    # Now that PCA is trained, we process each series's subsequences one by one.

    s_proj_list = []
    series_lengths = []  # Necessary for edge creation
    for series in dataset:
        # Extract all subsequences for just ONE series
        subsequences_one_series = extract_subsequences(series, l)
        # Transform them into 2D space. This uses much less memory.
        s_proj_one_series = pca.transform(subsequences_one_series)
        s_proj_list.append(s_proj_one_series)
        # Necessary for edge creation
        series_lengths.append(len(s_proj_one_series))

    # Concatenate the small 2D arrays. This is memory-efficient.
    # axis = 0, means it concatenates lists into rows
    s_proj = np.concatenate(s_proj_list, axis=0)
    # We also need to transform our sample for node creation
    s_proj_smpl = pca.transform(projsmpl)

    pca_model = pca  # Save the trained model

    """
    s_proj_list = [
    np.array([[2.1, 2.0], [1.9, 2.1], [2.0, 1.9]]),  # The paper for GPU 0
    np.array([[2.0, 2.1], [1.9, 1.9], [2.1, 2.0]]),  # The paper for GPU 1 
    np.array([[2.1, 2.0], [1.9, 2.1], [2.0, 1.9]]),  # The paper for GPU 3 
    np.array([[2.0, 2.1], [1.9, 1.9], [2.1, 2.0]])   # The paper for GPU 4]

    s_proj (after concat) = np.array([
    [2.1, 2.0],
    [1.9, 2.1],
    [2.0, 1.9],
    [2.0, 2.1],
    [1.9, 1.9],
    [2.1, 2.0],
    [2.1, 2.0],
    [1.9, 2.1],
    [2.0, 1.9],
    [2.0, 2.1],
    [1.9, 1.9],
    [2.1, 2.0] ])
    Each row now has two collumns because we transformed it into a 2 dimentional point
    """

    print(f"  PCA transformation complete. SProj shape: {s_proj.shape}")

    # --- Step 3: Node Creation------------------------------------------------------------------------------------------
    # Call the function to create nodes using the 2D projection of our sample.
    # nodes_l is a NumPy array that holds the 2D coordinates for all the nodes found for a specific subsequence length l
    nodes_l = create_nodes(s_proj_smpl)

    # Check if any nodes were found
    if nodes_l.shape[0] == 0:
        print(f"  No node clusters found for L = {l}.")
        continue

    print(f"  Created {nodes_l.shape[0]} nodes.")
    # -------------------------------------------------------------------------------------------------------------------
    # --- Step 4: Edge Creation
    edges_l = create_edges(s_proj, nodes_l, series_lengths)

    print(f"  Created {len(edges_l)} edges.")
    # ------------------------------------------------------------------------------------------------------------------
    # --- Clustering
    # Store the models for this 'l'
    graph_models.append({
        'l': l,
        'pca_model': pca_model,
        'nodes': nodes_l,
        'edges': edges_l
    })
    # ------------------------------------------------------------------------------------------------------------------
    # Visualize the 2D projection of the sampled subsequences
    """plt.figure(figsize=(8, 6))
    plt.scatter(s_proj_smpl[:, 0], s_proj_smpl[:, 1], s=5, alpha=0.6)
    plt.title(f'2D PCA-reduced Subsequences for L = {l}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()"""

print("\nFinished processing all L values.")

# --- GRAPH CLUSTERING -------------------------------------------------------------------------------------------------
print("\n--- Graph Clustering ---")
all_cluster_labels = []
num_clusters = 3  # The desired number of clusters

for model in graph_models:
    l = model['l']
    pca_model = model['pca_model']
    nodes_l = model['nodes']
    edges_l = model['edges']

    # Build the feature matrix for this graph
    feature_matrix = []
    for time_series in dataset:
        # Calculate the specific node, edge, and degree features for the current series
        features = extract_features_for_series(time_series, l, pca_model, nodes_l, edges_l)
        # Add the feature profile for this series to your list.
        feature_matrix.append(features)

    # Convert into numpy matrix because that is the required input format for the scikit-learn k-Means algorithm.
    feature_matrix = np.array(feature_matrix)
    """
                (Node 0 Count) (Node 1 Count) ... (Edge 0 Count) ... (Node 0 Degree) ...
        Row 0: [    1.2,          -0.5,     ...      0.8,       ...        1.5,      ...]  # Features for GPU 0
        Row 1: [    1.3,          -0.6,     ...      0.9,       ...        1.6,      ...]  # Features for GPU 1
        Row 2: [   -0.9,           1.8,     ...     -1.1,       ...       -0.8,      ...]  # Features for GPU 3
        Row 3: [   -0.8,           1.9,     ...     -1.2,       ...       -0.7,      ...]  # Features for GPU 4
    """

    # Apply k-Means Clustering
    # - n_clusters tells it how many clusters to find
    # - random_state ensures the results are the same each time you run the code.
    # - n_init='auto' is a default setting for how the algorithm is initialized.
    #       auto setting automatically runs the k-Means algorithm multiple times with different random starting
    #       points to ensure you get a good, stable result.
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')

    # Assigns each series to a cluster
    # .fit(): trains the kmeans model to this data
    # .predict(): This is the assignment step. After the centroids have been found, the method goes back through each
    #       of our four feature profiles and assigns it to whichever centroid it is closest to.
    labels = kmeans.fit_predict(feature_matrix)

    # Add the resulting array of labels (e.g., [0, 1, 0, 1]) to your master list.
    # This list stores the clustering results from every graph you process.
    all_cluster_labels.append(labels)

    print(f"  Cluster labels for L={l}: {labels}")

# --- CONSENSUS CLUSTERING ---------------------------------------------------------------------------------------------
from sklearn.cluster import SpectralClustering

print("\n--- Consensus Clustering ---")

# The number of time series in your dataset
num_series = len(dataset)

# --- Step 1: Build the Consensus Matrix ---
# Initialize a 4x4 matrix of zeros (since we have 4 time series)
consensus_matrix = np.zeros((num_series, num_series))

# Loop through each of the 5 sets of cluster labels you generated
# This block of code builds the consensus matrix by "voting" on how similar each pair of GPUs is. It checks every pair
#       of GPUs against every one of your five clustering results and adds a point to their similarity score if they were grouped together
for labels in all_cluster_labels:
    # Use nested loops to check every pair of GPUs
    for i in range(num_series):
        for j in range(num_series):
            # If two GPUs have the same label in this result, it strengthens their connection
            if labels[i] == labels[j]:
                # Increment the cell for this pair in the matrix
                consensus_matrix[i, j] += 1

# Normalize the matrix by dividing by the number of clustering results (M)
# Before this line, the number in each cell of consensus_matrix is a raw count, from 0 to 5, representing how many times a pair of GPUs were clustered together.
# After this line, the number in each cell will be a value between 0.0 and 1.0, representing the percentage of time that pair was clustered together.
consensus_matrix /= len(all_cluster_labels)
print("\nConsensus Matrix:")
print(consensus_matrix)

# --- Step 2: Apply Spectral Clustering ---
# This algorithm finds the communities in the similarity matrix
# n_clusters should be the same as the k for k-Means.

# affinity = 'precomputed'
# It tells the algorithm that the matrix you are about to give it is not raw data, but a pre-computed similarity
#       matrix (our consensus_matrix).
spectral_clusterer = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)

# Get the final, single set of cluster labels
final_labels = spectral_clusterer.fit_predict(consensus_matrix)

print(f"\nFinal k-Graph Cluster Labels: {final_labels}")

# UserWarning: Graph is not fully connected...
# This warning occurs because your consensus matrix is an identity matrix. In graph terms, this represents four
#       separate points (your GPUs) with no connections between them at all.

best_graph, best_paths = select_best_graph(graph_models, all_cluster_labels, final_labels, dataset)
compute_interpretability(best_graph, final_labels, dataset, best_paths)