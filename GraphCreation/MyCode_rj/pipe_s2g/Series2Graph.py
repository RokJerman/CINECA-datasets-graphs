import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def create_graph_for_series(series, length, r, smoothing_window):
    def calculate_initial_convolution_vector(series, length):
        convolution_size = length // 3
        first_subseq = series[:length]
        P = []
        for j in range(length - convolution_size + 1):
            window_sum = 0
            for k in range(j, j + convolution_size):
                window_sum += first_subseq[k]
            P.append(window_sum)
        return np.array(P), convolution_size

    initial_P, convolution = calculate_initial_convolution_vector(series, length)

    def compute_projection_matrix(series, length, convolution_size, initial_P_vector):
        total_length = len(series)
        Proj = [initial_P_vector]
        for i in range(1, total_length - length + 1):
            new_window_sum = np.sum(series[i + length - convolution_size: i + length])
            new_P_vector = np.empty_like(Proj[-1])
            new_P_vector[:-1] = Proj[-1][1:]
            new_P_vector[-1] = new_window_sum
            Proj.append(new_P_vector)
        return np.array(Proj)

    Proj = compute_projection_matrix(series, length, convolution, initial_P)

    num_components = 3

    # SAFETY CHECK: Before running PCA, we must ensure that the number of samples
    # (Proj.shape[0]) is at least as large as the number of components we are asking for.
    if Proj.shape[0] < num_components:
        print(
            f"[WARNING] Skipping chunk because it has too few samples ({Proj.shape[0]}) for PCA to run (requires at least {num_components}).")
        return None, None, None

    pca = PCA(num_components)
    Proj_reduced = pca.fit_transform(Proj)

    max_val = np.max(series)
    min_val = np.min(series)

    input_value = (max_val - min_val) * convolution
    ref_input_vector = np.full((1, Proj.shape[1]), input_value)

    v_ref = pca.transform(ref_input_vector)[0]

    u_x_vec = np.array([1.0, 0.0, 0.0])
    u_y_vec = np.array([0.0, 1.0, 0.0])
    u_z_vec = np.array([0.0, 0.0, 1.0])

    def _calculate_scalar_angle(v1, v2):
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        dot_product = np.dot(v1_norm, v2_norm)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return angle_rad

    phi_x = _calculate_scalar_angle(u_x_vec, v_ref)
    phi_y = _calculate_scalar_angle(u_y_vec, v_ref)
    phi_z = _calculate_scalar_angle(u_z_vec, v_ref)

    def _get_rotation_matrix_x(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def _get_rotation_matrix_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def _get_rotation_matrix_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    R_ux = _get_rotation_matrix_x(phi_x)
    R_uy = _get_rotation_matrix_y(phi_y)
    R_uz = _get_rotation_matrix_z(phi_z)

    current_SProj_3d = Proj_reduced @ R_ux.T @ R_uy.T @ R_uz.T

    SProj = current_SProj_3d[:, 1:]

    def cross_product_2d(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    def intersect_segment_ray(segment_start_point, segment_end_point, ray_origin, ray_direction):
        segment_start_point = np.array(segment_start_point)
        segment_end_point = np.array(segment_end_point)
        ray_origin = np.array(ray_origin)
        ray_direction = np.array(ray_direction)
        segment_vector = segment_end_point - segment_start_point
        determinant = cross_product_2d(segment_vector, -ray_direction)
        if determinant == 0:
            return None
        t = cross_product_2d(ray_origin - segment_start_point, -ray_direction) / determinant
        u = cross_product_2d(ray_origin - segment_start_point, segment_vector) / determinant
        if 0 <= t <= 1 and u >= 0:
            intersection_point = segment_start_point + t * segment_vector
            return intersection_point
        else:
            return None

    def gaussian_kernel_exponent(x, xi, std_dev, bandwidth, mean):
        numerator = (x - xi - bandwidth * mean) ** 2
        denominator = 2 * bandwidth * (std_dev ** 2)
        if denominator == 0:
            return 0.0
        return np.exp(-numerator / denominator)

    def kernel_density_estimate(x, intersections, bandwidth):
        n = len(intersections)
        std_dev = np.std(intersections)
        mean = np.mean(intersections)
        denominator_function = n * bandwidth * np.sqrt(2 * np.pi * std_dev ** 2)
        if denominator_function == 0:
            if x in intersections:
                return 1
            else:
                return 0
        sum_of_kernels = 0
        for xi in intersections:
            sum_of_kernels += gaussian_kernel_exponent(x, xi, std_dev, bandwidth, mean)
        density = (1 / denominator_function) * sum_of_kernels
        return density

    psi_angles_list = []
    for i in range(r + 1):
        psi_angles_list.append(i * (2 * np.pi / r))

    nodes = []
    ray_origin_point = np.array([0.0, 0.0])

    for psi_angle in psi_angles_list:
        intersections = []
        ray_direction_vector = np.array([np.cos(psi_angle), np.sin(psi_angle)])
        for i in range(len(SProj) - 1):
            SProj_i = SProj[i]
            SProj_i_plus_1 = SProj[i + 1]
            intersection_point = intersect_segment_ray(
                SProj_i,
                SProj_i_plus_1,
                ray_origin_point,
                ray_direction_vector
            )
            if intersection_point is not None:
                radial_distance = np.linalg.norm(intersection_point)
                intersections.append(radial_distance)
        if len(intersections) < 2:
            # If there aren't enough points, we can't form a node for this
            # angle, so we skip to the next one. This prevents the warnings.
            continue
        std_dev = np.std(intersections)
        h_scott = 0.0
        if len(intersections) > 0:
            if len(intersections) > 1 and std_dev > 0:
                h_scott = std_dev * (len(intersections) ** (-1 / 5))
            else:
                if len(intersections) > 0:
                    h_scott = 1
            densities = []
            for rad_distance in intersections:
                densities.append(kernel_density_estimate(rad_distance, intersections, h_scott))
            if densities:
                max_density_index = np.argmax(densities)
                densest_radial_distance = intersections[max_density_index]
                node_coord = np.array([densest_radial_distance * np.cos(psi_angle),
                                       densest_radial_distance * np.sin(psi_angle)])
                nodes.append(node_coord)
    N = np.array(nodes)

    psi_angles_list = []
    for i in range(r + 1):
        psi_angles_list.append(i * (2 * np.pi / r))

    node_map_by_angle = {}
    for i in range(len(psi_angles_list)):
        angle = psi_angles_list[i]
        if i < len(N):
            node_map_by_angle[round(angle, 10)] = N[i]

    NodeSeq = []
    ray_origin_point = np.array([0.0, 0.0])

    for i in range(len(SProj)):
        SProj_i = SProj[i]
        if i == len(SProj) - 1:
            continue
        SProj_i_plus_1 = SProj[i + 1]
        psi_i = np.arctan2(SProj_i[1], SProj_i[0])
        psi_i_plus_1 = np.arctan2(SProj_i_plus_1[1], SProj_i_plus_1[0])
        psi_i_norm = (psi_i + 2 * np.pi) % (2 * np.pi)
        psi_i_plus_1_norm = (psi_i_plus_1 + 2 * np.pi) % (2 * np.pi)
        if psi_i_norm <= psi_i_plus_1_norm:
            angular_range_start = psi_i_norm
            angular_range_end = psi_i_plus_1_norm
            is_wrapped = False
        else:
            angular_range_start = psi_i_norm
            angular_range_end = psi_i_plus_1_norm
            is_wrapped = True
        closest_node = None
        min_dist = float('inf')
        for psi_angle in psi_angles_list:
            psi_angle_norm = (psi_angle + 2 * np.pi) % (2 * np.pi)
            if (not is_wrapped and angular_range_start <= psi_angle_norm <= angular_range_end) or \
            (is_wrapped and (psi_angle_norm >= angular_range_start or psi_angle_norm <= angular_range_end)):
                ray_dir = np.array([np.cos(psi_angle_norm), np.sin(psi_angle_norm)])
                intersection = intersect_segment_ray(SProj_i, SProj_i_plus_1, ray_origin_point, ray_dir)
                if intersection is not None:
                    if round(psi_angle_norm, 10) in node_map_by_angle:
                        current_node = node_map_by_angle[round(psi_angle_norm, 10)]
                        dist_to_node = np.linalg.norm(intersection - current_node)
                        if dist_to_node < min_dist:
                            min_dist = dist_to_node
                            closest_node = current_node
                if closest_node is not None:
                    NodeSeq.append(closest_node)

    Edges_dict = {}
    for i in range(len(NodeSeq) - 1):
        node_from = tuple(NodeSeq[i])
        node_to = tuple(NodeSeq[i+1])
        edge = (node_from, node_to)
        Edges_dict[edge] = Edges_dict.get(edge,0) + 1

    node_degrees = {}
    for node_tuple in N:
        node_degrees[tuple(node_tuple)] = 0

    for edge, weight in Edges_dict.items():
        node_from, node_to = edge
        if node_from in node_degrees:
            node_degrees[node_from] += weight

    def calculate_normality_score(subpath_of_edges, all_edges, node_degrees):
        score = 0
        for edge in subpath_of_edges:
            node_from, node_to = edge
            weight = all_edges.get(edge, 0)
            degree = node_degrees.get(node_from, 1)
            score += weight * (degree - 1)
        return score / len(subpath_of_edges)

    normality_scores = []
    for i in range(len(NodeSeq) - length):
        current_path_edges = []
        for j in range(length - 1):
            edge = (tuple(NodeSeq[i+j]), tuple(NodeSeq[i+j+1]))
            current_path_edges.append(edge)
        window_score = calculate_normality_score(current_path_edges, Edges_dict, node_degrees)
        normality_scores.append(window_score)

    print(f"Calculated {len(normality_scores)} normality scores.")

    scores_series = pd.Series(normality_scores)
    smoothed_normality_scores = scores_series.rolling(window=smoothing_window).mean()

    anomaly_scores = 1 / (smoothed_normality_scores)
    return anomaly_scores, Edges_dict, N


#df = pd.read_parquet("20.parquet")
#series = df["total_power_max"].dropna().values
# anomaly_scores, edges, nodes = create_graph_for_series(series,672, 50, 100)

"""plt.figure(figsize=(15, 6))
plt.plot(anomaly_scores, color='red')
plt.title("Anomaly Score Graph", fontsize=16)
plt.xlabel('Subsequence Window Index / Time', fontsize=12)
plt.ylabel('Anomaly Score', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()"""
