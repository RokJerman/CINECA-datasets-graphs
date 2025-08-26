import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score



#=======================================================================================================================
# HELPER FUNCTIONS
#=======================================================================================================================

def extract_subsequences(time_series, l):
    return np.lib.stride_tricks.sliding_window_view(time_series, window_shape=l)


def find_peaks(data, min_height):
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1] and data[i] >= min_height:
            peaks.append(i)
    return np.array(peaks)


def create_nodes(s_proj_smpl, bandwidth=0.1, num_eval_points=360):
    angles = np.arctan2(s_proj_smpl[:, 1], s_proj_smpl[:, 0])
    circular_angles = np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi]).reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(circular_angles)
    eval_points = np.linspace(-np.pi, np.pi, num_eval_points).reshape(-1, 1)
    density_curve = np.exp(kde.score_samples(eval_points))
    min_peak_height = np.mean(density_curve)
    peaks = find_peaks(density_curve, min_height=min_peak_height)
    if len(peaks) == 0:
        return np.array([])
    nodes = []
    slice_width = (2 * np.pi / num_eval_points)
    for peak_index in peaks:
        peak_angle = eval_points[peak_index][0]
        angle_start = peak_angle - (slice_width / 2)
        angle_end = peak_angle + (slice_width / 2)
        mask = (angles >= angle_start) & (angles < angle_end)
        points_in_slice = s_proj_smpl[mask]
        if points_in_slice.shape[0] > 0:
            node_centroid = np.mean(points_in_slice, axis=0)
            nodes.append(node_centroid)
    return np.array(nodes)


def create_edges(s_proj, nodes_l, series_lengths):
    if nodes_l.shape[0] < 2:
        return []
    distances_sq = np.sum((s_proj[:, np.newaxis, :] - nodes_l[np.newaxis, :, :]) ** 2, axis=-1)
    node_path = np.argmin(distances_sq, axis=1)
    edge_counts = {}
    start_index = 0
    for num_subsequences_in_series in series_lengths:
        end_index = start_index + num_subsequences_in_series
        single_series_path = node_path[start_index:end_index]
        for i in range(len(single_series_path) - 1):
            source_node = single_series_path[i]
            target_node = single_series_path[i + 1]
            edge = (source_node, target_node)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        start_index = end_index
    final_edges = []
    for (u, v), count in edge_counts.items():
        final_edges.append((u, v, count))
    return final_edges


def extract_features_for_series(time_series, l, pca_model, nodes_l, edges_l):
    subsequences = extract_subsequences(time_series, l)
    if subsequences.shape[0] == 0:
        # Create a zero vector of the expected shape if there are no subsequences
        num_nodes = len(nodes_l)
        num_edges = len(edges_l)
        return np.zeros(num_nodes * 2 + num_edges)

    s_proj = pca_model.transform(subsequences)
    distances_sq = np.sum((s_proj[:, np.newaxis, :] - nodes_l[np.newaxis, :, :]) ** 2, axis=-1)
    node_path = np.argmin(distances_sq, axis=1)
    node_counts = np.zeros(len(nodes_l))
    edge_counts = np.zeros(len(edges_l))
    edge_to_index = {tuple(edge[:2]): i for i, edge in enumerate(edges_l)}
    for node_idx in node_path:
        node_counts[node_idx] += 1
    for i in range(len(node_path) - 1):
        edge_tuple = (node_path[i], node_path[i + 1])
        if edge_tuple in edge_to_index:
            edge_idx = edge_to_index[edge_tuple]
            edge_counts[edge_idx] += 1
    path_graph = nx.DiGraph(list(zip(node_path[:-1], node_path[1:])))
    degree_features = np.zeros(len(nodes_l))
    for i in range(len(nodes_l)):
        if i in path_graph:
            degree_features[i] = path_graph.degree(i)
    feature_vector = np.concatenate([node_counts, edge_counts, degree_features])
    mean = feature_vector.mean()
    std = feature_vector.std()
    if std == 0:
        return np.zeros_like(feature_vector)
    return (feature_vector - mean) / std


def select_best_graph(graph_models, all_cluster_labels, final_labels, dataset):
    best_score = -1
    best_model_info = None
    best_node_paths = None
    for index, model in enumerate(graph_models):
        curr_graph_labels = all_cluster_labels[index]
        consistency = adjusted_rand_score(final_labels, curr_graph_labels)
        num_clusters = len(np.unique(final_labels))
        max_exclusivity_per_cluster = []
        node_paths = []
        for series in dataset:
            if len(series) < model['l']:
                node_paths.append(np.array([]))
                continue
            subsequences = extract_subsequences(series, model['l'])
            s_proj = model['pca_model'].transform(subsequences)
            distances_sq = np.sum((s_proj[:, np.newaxis, :] - model['nodes'][np.newaxis, :, :]) ** 2, axis=-1)
            node_paths.append(np.argmin(distances_sq, axis=1))
        for cluster_id in range(num_clusters):
            indexes_for_this_cluster = np.where(final_labels == cluster_id)[0]
            max_exclusivity_for_this_cluster = 0
            for node_id in range(len(model['nodes'])):
                members_using_node = sum(
                    1 for series_idx in indexes_for_this_cluster if node_id in node_paths[series_idx])
                total_using_node = sum(1 for path in node_paths if node_id in path)
                if total_using_node > 0:
                    exclusivity = members_using_node / total_using_node
                    if exclusivity > max_exclusivity_for_this_cluster:
                        max_exclusivity_for_this_cluster = exclusivity
            max_exclusivity_per_cluster.append(max_exclusivity_for_this_cluster)
        interpretability_factor = np.mean(max_exclusivity_per_cluster)
        score = consistency * interpretability_factor
        if score > best_score:
            best_score = score
            best_model_info = model
            best_node_paths = node_paths
    print(f"\nBest graph found for L = {best_model_info['l']} with a score of {best_score:.2f}")
    return best_model_info, best_node_paths


def compute_interpretability(best_model, final_labels, dataset, node_paths):
    l = best_model['l']
    nodes = best_model['nodes']
    num_clusters = len(np.unique(final_labels))
    all_subsequences_by_node = {i: [] for i in range(len(nodes))}
    for series_idx, path in enumerate(node_paths):
        if len(dataset[series_idx]) < l: continue
        subsequences = extract_subsequences(dataset[series_idx], l)
        for sub_idx, node_id in enumerate(path):
            all_subsequences_by_node[node_id].append(subsequences[sub_idx])
    for cluster_id in range(num_clusters):
        indexes_for_curr_cluster = np.where(final_labels == cluster_id)[0]
        num_series_in_cluster = len(indexes_for_curr_cluster)
        best_node_for_cluster = -1
        best_score = -1
        print(f"\nAnalyzing Cluster {cluster_id}:")
        for node_id in range(len(nodes)):
            members_using_node = sum(1 for series_idx in indexes_for_curr_cluster if node_id in node_paths[series_idx])
            total_using_node = sum(1 for path in node_paths if node_id in path)
            if total_using_node == 0 or num_series_in_cluster == 0: continue
            representativity = members_using_node / num_series_in_cluster
            exclusivity = members_using_node / total_using_node
            score = representativity * exclusivity
            if score > best_score:
                best_score = score
                best_node_for_cluster = node_id
        if best_node_for_cluster != -1:
            print(f"  - Most interpretable pattern is Node {best_node_for_cluster}.")
            if best_node_for_cluster in all_subsequences_by_node and all_subsequences_by_node[best_node_for_cluster]:
                centroid_subsequence = np.mean(all_subsequences_by_node[best_node_for_cluster], axis=0)


#=======================================================================================================================
# MAIN TRAINING FUNCTION
#=======================================================================================================================
def run_full_kgraph_training(df, time_series_columns, m, min_l, max_l, num_clusters, sample_size_per_series):
    dataset = [df[column].dropna().to_numpy() for column in time_series_columns]
    np.random.seed(42)
    L_values = sorted(np.random.choice(np.arange(min_l, max_l + 1), size=m, replace=False))
    print(f"[TRAINING] Testing {m} different L values: {L_values}")

    graph_models = []
    for l in L_values:
        print(f"[TRAINING] Processing L = {l}")
        sample_subsequences_list = []
        for series in dataset:
            num_subs = len(series) - l + 1
            if num_subs <= 0: continue
            random_indexes = np.random.choice(num_subs, min(sample_size_per_series, num_subs), replace=False)
            for index in random_indexes:
                sample_subsequences_list.append(series[index: index + l])
        if not sample_subsequences_list:
            print(f"  Not enough data to create subsequences for L={l}. Skipping.")
            continue
        projsmpl = np.array(sample_subsequences_list)
        if projsmpl.ndim < 2:
            print(f"  Not enough samples to train PCA for L={l}. Skipping.")
            continue
        pca = PCA(n_components=2)
        pca.fit(projsmpl)
        s_proj_list = []
        series_lengths = []
        for series in dataset:
            if len(series) < l:
                series_lengths.append(0)
                continue
            subsequences = extract_subsequences(series, l)
            if subsequences.shape[0] == 0:
                series_lengths.append(0)
                continue
            s_proj_list.append(pca.transform(subsequences))
            series_lengths.append(len(subsequences))
        if not s_proj_list: continue
        s_proj = np.concatenate(s_proj_list, axis=0)
        s_proj_smpl = pca.transform(projsmpl)
        nodes_l = create_nodes(s_proj_smpl)
        if nodes_l.shape[0] == 0:
            print(f"  No nodes found for L={l}. Skipping.")
            continue
        edges_l = create_edges(s_proj, nodes_l, series_lengths)
        graph_models.append({'l': l, 'pca_model': pca, 'nodes': nodes_l, 'edges': edges_l})

    if not graph_models:
        print("[TRAINING] Could not build any valid graph models.")
        return None

    print("\n--- [TRAINING] Graph Clustering ---")

    all_cluster_labels = []
    for model in graph_models:
        feature_matrix = np.array(
            [extract_features_for_series(ts, model['l'], model['pca_model'], model['nodes'], model['edges']) for ts in
             dataset])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(feature_matrix)
        print(f"L: {model['l']} {labels}")
        all_cluster_labels.append(labels)


    print("\n--- [TRAINING] Consensus Clustering ---")
    num_series = len(dataset)
    consensus_matrix = np.zeros((num_series, num_series))
    for labels in all_cluster_labels:
        for i in range(num_series):
            for j in range(num_series):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1
    consensus_matrix /= len(all_cluster_labels)
    spectral_clusterer = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
    final_labels = spectral_clusterer.fit_predict(consensus_matrix)
    print(f"Final k-Graph Cluster Labels: {final_labels}")

    best_graph_info, best_node_paths = select_best_graph(graph_models, all_cluster_labels, final_labels, dataset)

    # Run the visualization part, which will now save plots instead of showing them
    compute_interpretability(best_graph_info, final_labels, dataset, best_node_paths)

    G = nx.DiGraph()
    for i, pos in enumerate(best_graph_info['nodes']):
        G.add_node(i, x=pos[0], y=pos[1])
    for u, v, weight in best_graph_info['edges']:
        G.add_edge(u, v, weight=weight)
    print(f"[TRAINING] Best model selected (L={best_graph_info['l']}). Training complete.")

    return {'graph': G, 'pca_model': best_graph_info['pca_model'], 'l': best_graph_info['l'],
            'nodes': best_graph_info['nodes']}