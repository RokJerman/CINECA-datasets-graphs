import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Series2Graph

#-----------------------------------------------------------------------------------------------------------------------
# SUBSEQUENCE EMBEDDING
#-----------------------------------------------------------------------------------------------------------------------
# Show all columns when printing
pd.set_option('display.max_columns', None)

# Load the Parquet file
df = pd.read_parquet("20.parquet")

"""# Show the number of rows
print(f"Loaded 20.parquet with {len(df)} time windows")

# Print only the 'gpu4_core_temp_avg' column
print("\nValues from 'gpu4_core_temp_avg':")
print(df["gpu4_core_temp_avg"])"""

# Subsequence length is l = 672 (1 week)
# Local convolution will therefore be 224 (lambda)
# Number of all time windows: 86648
# I picked the "p0_io_power_avg" for a univariate system following the paper's instructions

# Extract the univariate series
#series = df["p0_io_power_avg"].dropna().values
series = df["total_power_max"].dropna().values

# Subsequence length is ℓ = 672 (1 week)
length = 672
# three-day window length = 288
# two-day window   length = 192
# one-day window   length = 96

"""
Example on a smaller scale for better understanding:

series = [10, 12, 11, 15, 13, 16, 14, 18, 17, 20, ...]
length L = 6
convolution_size (λ) = length // 3 = 2

First Convolution vector loop:
first_subseq = [10, 12, 11, 15, 13, 16]
Number of iterations: (0, L - λ) in our case (0, 6 - 2) = 5 iterations, from 0 to 4
i = 0: window_sum = first_subseq[0] + first_subseq[1] = 10 + 12 | P = [22]
i = 1: window_sum = first_subseq[1] + first_subseq[2] = 12 + 11 | P = [22, 23]
i = 2: window_sum = first_subseq[2] + first_subseq[3] = 11 + 15 | P = [22, 23, 26]
i = 3: window_sum = first_subseq[3] + first_subseq[4] = 15 + 13 | P = [22, 23, 26, 28]
i = 4: window_sum = first_subseq[4] + first_subseq[5] = 13 + 16 | P = [22, 23, 26, 28, 29]
P = our first convolution vector and it is the first row in our Proj matrix.

Projection matrix loop:

Proj (after 3rd subsequence):
[[22, 23, 26, 28, 29],
 [23, 26, 28, 29, 30],
 [26, 28, 29, 30, 32]]
"""

def calculate_initial_convolution_vector(series, length):
    """
    Calculates the convolution vector for the very first subsequence.
    This function should only be called once.
    """
    convolution_size = length // 3  # Local convolution window size (λ)
    first_subseq = series[:length]  # Extracts the first L-length subsequence

    P = []
    for j in range(length - convolution_size + 1): # +1 because Python doesnt include the last value in the loop
        window_sum = 0
        for k in range(j, j + convolution_size):
            window_sum += first_subseq[k]
        P.append(window_sum)
    # We are using numpy array because its faster than the original Python one (fixed length)
    return np.array(P), convolution_size

# Calculate the first convolution vector and convolution size ONLY ONCE
initial_P, convolution = calculate_initial_convolution_vector(series, length)

def compute_projection_matrix(series, length, convolution_size, initial_P_vector):
    """
    Computes the full projection matrix, starting with the pre-calculated
    initial_P_vector and then efficiently updating for subsequent subsequences.
    """
    total_length = len(series)

    # Proj list with the already computed initial vector
    Proj = [initial_P_vector]

    # We start with index 1
    for i in range(1, total_length - length + 1): # We subtract one length because we already calculated the initial one
        # Get the new window of 'convolution_size' that slides in
        # First element index: i + length - convolution_size
        # Last element index: i + length
        new_window_sum = np.sum(series[i + length - convolution_size: i + length])

        # Shift previous vector and insert new value at the end
        # np.empty_like creates a new array with the same shape and data type as the given array. Proj[-1] returns
        #   the latest vector from the list Proj
        new_P_vector = np.empty_like(Proj[-1])
        new_P_vector[:-1] = Proj[-1][1:] # vzamemo zadnji vektor iz Proj (Proj[-1]) and drop the first element (([1:]) starts from index 1)
        new_P_vector[-1] = new_window_sum # It places the new_window_sum into the last position of the vector

        Proj.append(new_P_vector)

    return np.array(Proj)

# Call the optimized function, passing the pre-calculated initial vector
Proj = compute_projection_matrix(series, length, convolution, initial_P)

#PCA INTEGRATION -------------------------------------------------------------------------------------------------------
# We reduce the vector to three dimensions

from sklearn.decomposition import PCA

# Number of dimensions
num_components = 3

# Apply PCA to the projection matrix
pca = PCA(num_components) # We initialize pca with how many dimensions we want to keep
# .fit() "teaches" the pca what the input data is like
# .transform transforms the matrix into a "num_components" dimentional matrix
Proj_reduced = pca.fit_transform(Proj)
#Proj_reduced will keep the same number of rows, and each row will now have a vector containing 3 elements

#PCA ROTATION ----------------------------------------------------------------------------------------------------------

max_val = np.max(series) # Gets the maximum value from our original univariate series
min_val = np.min(series) # Gets the minimum value from our original univariate series

input_value = (max_val - min_val) * convolution # (max(T) - min(T)) * lambda
# np.full() creates a new array of a specified shape and fills all its elements with a given value
# (1, Proj.shape[1])
#   Tells the program to create an array with 1 row with Proj.shape[1] number of columns
#   Proj.shape[1] represents the number of columns in our Proj matrix, which is equivalent to (L−convolution)
ref_input_vector = np.full((1, Proj.shape[1]), input_value)

# Line 10:
# Transforms a specially constructed input vector (representing a constant signal's convoluted form) into
# the 3-dimensional PCA space
# pca.transform always returns a 2D array, but our [0] tells the program to only use the first(and only) row which then
#   converts it into a 1D array. v_ref = [1 , 2.4, ...]
v_ref = pca.transform(ref_input_vector)[0]

# Line 11: phi_x, phi_y, phi_z <- getAngle((u_x, u_y, u_z), v_ref);
# coordinate arrays, we will need them to calc. the angle between v_ref and every axis.
u_x_vec = np.array([1.0, 0.0, 0.0]) # It points purely in the positive X direction
u_y_vec = np.array([0.0, 1.0, 0.0]) # It points purely in the positive y direction
u_z_vec = np.array([0.0, 0.0, 1.0]) # It points purely in the positive z direction

def _calculate_scalar_angle(v1, v2):
    """Calculates the scalar angle (in radians) between two 3D vectors."""
    #np.linalg.norm(vector) returns the distance of the vector
    v1_norm = v1 / np.linalg.norm(v1) # Normalize v1 (convert the vector's length into 1 while also retaining its original direction)
    v2_norm = v2 / np.linalg.norm(v2) # Normalize v2

    dot_product = np.dot(v1_norm, v2_norm) # Calculate dot product (we get the cosine value between v1_norm and v2_norm)
    # np.clip() is a NumPy function that "clips" (limits) values in an array. it ensures that the dot_product value stays
    #   strictly within the range of [-1.0, 1.0].
    # np.arccos(cos) returns the opposite of cosine (in radians)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0)) # Calculates our final scalar angle
    return angle_rad

phi_x = _calculate_scalar_angle(u_x_vec, v_ref) # Angle between x-axis and v_ref
phi_y = _calculate_scalar_angle(u_y_vec, v_ref) # Angle between y-axis and v_ref
phi_z = _calculate_scalar_angle(u_z_vec, v_ref) # Angle between z-axis and v_ref

# Line 12: R_ux, R_uy, R_uz <- GetRotationMatrices(phi_x, phi_y, phi_z);
# Define standard 3D rotation matrix for each axis
def _get_rotation_matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _get_rotation_matrix_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _get_rotation_matrix_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# R_ux, R_uy, R_uz are designed to rotate the entire 3D PCA space so that the v_ref vector becomes aligned with the X-axis.
# These three angles are then used to construct three separate rotation matrices. When these matrices are applied
# they effectively rotate the coordinate system such that the v_ref vector, which represents the "time dimension"
# of our data, is now aligned with the X-axis. This ensures that the remaining Y and Z axes in the new rotated space
# (which the paper calls r_y and r_z) will primarily capture the shape-related characteristics of our subsequences.

R_ux = _get_rotation_matrix_x(phi_x) # Rotation matrix around x-axis
R_uy = _get_rotation_matrix_y(phi_y) # Rotation matrix around y-axis
R_uz = _get_rotation_matrix_z(phi_z) # Rotation matrix around z-axis

# Line 13: SProj <- R_ux.R_uy.R_uz.Proj_r^T
# Apply the sequence of rotations to the PCA-reduced data.
# Proj_reduced contains row vectors, so we multiply by the transpose of rotation matrices.
# In Python, specifically with NumPy arrays, the @ symbol is the matrix multiplication operator
current_SProj_3d = Proj_reduced @ R_ux.T @ R_uy.T @ R_uz.T

# "In the rest of the paper, SProj(T,l,λ) will refer to the 2-dimensions matrix keeping only the ry and rz components."
SProj = current_SProj_3d[:, 1:]
#-----------------------------------------------------------------------------------------------------------------------
# 3D VECTOR VISUALIZATION
#-----------------------------------------------------------------------------------------------------------------------

"""# Store the full 3D SProj for plotting BEFORE it's reduced to 2D
SProj_full_3d_for_plot = current_SProj_3d

# As per the paper: "In the rest of the paper, SProj(T,l,λ) will refer to the 2-dimensions matrix
# keeping only the ry and rz components." (Section 4.1, paragraph 7)
# This line performs the final reduction to 2D for subsequent steps of the algorithm.
SProj = current_SProj_3d[:, 1:]

# Subsample the 3D data for plotting if it's too dense
# Plotting every 10th point for better performance and visibility
sample_3d = SProj_full_3d_for_plot[::10]

# Create a 3D plot
fig_3d = plt.figure(figsize=(10, 7))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# Plot a continuous 3D line
ax_3d.plot(sample_3d[:, 0], sample_3d[:, 1], sample_3d[:, 2], color='blue', linewidth=0.7)

# Set plot titles and labels
ax_3d.set_title("3D Line Plot of Rotated Projection")

# Adjust layout and display the plot
plt.tight_layout()
plt.show()"""
#-----------------------------------------------------------------------------------------------------------------------
# 2D LINE GRAPH VISUALIZATION
#-----------------------------------------------------------------------------------------------------------------------
"""# Calculate the Euclidean distance for each point from the origin (0,0).
# This single value per time step represents the shape of the data.

# np.linalg.norm() calculates the Euclidean norm of a vector.
# the norm of a vector is its length or magnitude. For a 2D point (x, y), the norm is simply its straight-line
#       distance from the origin (0,0).
# axis = 1 goes through the SProj matrix row by row. Treat each individual row as a vector and calculate its norm.
shape_scores = np.linalg.norm(SProj, axis=1)

# --- Plotting the single line ---
plt.figure(figsize=(15, 5))

# Plot the main line graph
plt.plot(shape_scores, color='blue')

plt.title('Series2Graph 2D Line Graph', fontsize=16)
plt.xlabel('Subsequence Indexes - Represents Time', fontsize=12)
plt.ylabel('Shape Score', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.75)
plt.tight_layout()
plt.show()"""
#-----------------------------------------------------------------------------------------------------------------------
# SPROJ SUMMARY STATISTICS
#-----------------------------------------------------------------------------------------------------------------------
"""# Convert the numpy array of scores into a pandas Series
scores_series = pd.Series(shape_scores)
# Get the full set of summary statistics using the .describe() method
# .describe() is a built-in pandas method that automatically calculates a standard set of descriptive statistics for
#       all the data in the Series.
summary_stats = scores_series.describe()
print("--- Summary Statistics for the 2D Line Graph ---")
print(summary_stats)
print()"""

#-----------------------------------------------------------------------------------------------------------------------
# NODE CREATION
#-----------------------------------------------------------------------------------------------------------------------

# SProj has shape (num_subsequences, 2)

# Parameters for Node Extraction
r = 50  # Number of angles (sampling the space) - paper states r=50


# Algorithm 2: Node Extraction
# input: 2-dimensional point sequence SProj, rate r, bandwidth h
# output: Node Set N

def cross_product_2d(v1, v2):
    """Calculates the 2D cross product magnitude for two 2D vectors."""
    return v1[0] * v2[1] - v1[1] * v2[0]

def intersect_segment_ray(segment_start_point, segment_end_point, ray_origin, ray_direction):
    """
    Finds the intersection point of a line segment and a ray.
    Returns the intersection point (2D NumPy array) if it lies on the segment and ray, otherwise None.
    """
    segment_start_point = np.array(segment_start_point)
    segment_end_point = np.array(segment_end_point)
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)

    # Calculate the vector representing the segment
    # If segment_start_point is [x1, y1] and segment_end_point is [x2, y2], then: segment_vector = [x2 - x1, y2 - y1]
    segment_vector = segment_end_point - segment_start_point

    # Calculate the determinant (2D cross product) of the matrix formed by segment_vector and -ray_direction
    # This determines if the lines are parallel (det == 0) or intersect
    determinant = cross_product_2d(segment_vector, -ray_direction)

    if determinant == 0: # Lines are parallel, no intersection point
        return None

    # Calculate the parameter 't' for the segment and 'u' for the ray

    # Segment Point = segment_start_point + t * segment_vector
    # t = (Segment Point - segment_start_point) / segment_vector <-- Does not work
    # Ray Point = ray_origin + u * ray_direction

    # 't' indicates where along the segment the intersection occurs
    # 'u' indicates where along the ray the intersection occurs
    # Both can only range from 0 - 1. 0 meaning the inter. is in the beginning, 1 in the end and 0.5 in the middle
    t = cross_product_2d(ray_origin - segment_start_point, -ray_direction) / determinant # Formula
    u = cross_product_2d(ray_origin - segment_start_point, segment_vector) / determinant

    # Check if the intersection point lies within the bounds of the segment and along the ray
    if 0 <= t <= 1 and u >= 0:
        # Calculate the actual intersection point using the parameter 't'
        intersection_point = segment_start_point + t * segment_vector
        return intersection_point
    else:
        # No valid intersection point found within the specified bounds
        return None


# Helper function for the kernel_density_est funciton (the exponent part only)
def gaussian_kernel_exponent(x, xi, std_dev, bandwidth, mean):
    """
    Calculates the exponent part of the Gaussian kernel.
    """
    numerator = (x - xi - bandwidth * mean) ** 2
    denominator = 2 * bandwidth * (std_dev ** 2)

    if denominator == 0:
        return 0.0

    # np.exp() returns the mathematical constant e
    return np.exp(-numerator / denominator)

def kernel_density_estimate(x, intersections, bandwidth):
    """
    Computes Kernel Density Estimate (KDE)
    """
    # 'rad_distance' is 'x' in the paper's formula (the point to estimate density for).
    # 'intersections' is 'I_psi' in the paper's formula (the set of intersection radial distances).
    # 'bandwidth' is 'h' in the paper's formula.

    # Calculate n, std_dev, mu(I_psi) as per paper's formula elements.
    n = len(intersections)
    std_dev = np.std(intersections)
    mean = np.mean(intersections)

    # Calculate the global normalization factor as per paper's formula: 1 / (n * h * sqrt(2*pi*std_dev^2))
    # Note: std_dev^2 is variance. np.std is standard deviation.
    # The term sqrt(2*pi*std_dev^2) is sqrt(2*pi) * std_dev
    denominator_function = n * bandwidth * np.sqrt(2 * np.pi * std_dev ** 2)

    if denominator_function == 0:
        # In such a case, the KDE is a Dirac delta function; density is only at the data_point(s).
        if x in intersections:
            return 1
        else:
            return 0

    sum_of_kernels = 0
    for xi in intersections:
        sum_of_kernels += gaussian_kernel_exponent(x, xi, std_dev, bandwidth, mean)

    density = (1 / denominator_function) * sum_of_kernels
    return density


# Ψ (psi) angles are a set of angles that are uniformly distributed around a circle
# With r = 50 we will have 51 psi_angles
psi_angles_list = []

for i in range(r + 1):
    psi_angles_list.append(i * (2 * np.pi / r))

nodes = []

# Define the origin for the rays (Omega)
ray_origin_point = np.array([0.0, 0.0])

for psi_angle in psi_angles_list:
    intersections = []

    # Calculate the vector for the current angle (ray_direction)
    # np.cos() and .sin() in NumPy is a mathematical function that calculates the cosine or sinus of a given angle. Returns Radian.
    ray_direction_vector = np.array([np.cos(psi_angle), np.sin(psi_angle)])

    for i in range(len(SProj) - 1):  # Algorithm 2, Line 5
        SProj_i = SProj[i]
        SProj_i_plus_1 = SProj[i + 1]

        intersection_point = intersect_segment_ray(
            SProj_i,
            SProj_i_plus_1,
            ray_origin_point,
            ray_direction_vector
        )
        if intersection_point is not None:
            # how far away the intersection point is from the origin (center) of our 2D SProj space
            radial_distance = np.linalg.norm(intersection_point) # returns length of vector
            intersections.append(radial_distance) # add distance to array

    # Extract Nodes 10-11

    # .std calculates the standard deviation of the numerical values contained within the intersections list.
    std_dev = np.std(intersections) # standard deviation between
    h_scott = 0.0  # Initialize h_scott

    if len(intersections) > 0:  # Only proceed if there are intersection points for this angle
        """ DYNAMIC WAY OF DETERMINING BANDWIDTH (WEIRD GRAPH RESULTS)"""
        # Condition 1: Apply Scott's Rule if there's sufficient and varied data.
        # Scott's rule can only be reliably applied if:
        # 1. There are at least two distinct data points (len(intersections) > 1).
        # 2. The standard deviation is greater than zero (std_dev > 0), meaning the points are not all identical.
        if len(intersections) > 1 and std_dev > 0:
            # Calculate Scott's Rule bandwidth (h_scott = std_dev * |I_psi|^-1/5)
            h_scott = std_dev * (len(intersections) ** (-1 / 5))
        else:
            # Condition 2: Handle edge cases where Scott's Rule cannot be directly applied.
            # This covers scenarios with very few points or points that are all identical.
            if len(intersections) > 0:
                # If there's at least one point but Scott's rule can't be applied (e.g., std_dev=0)
                # or only one point, use a default bandwidth
                h_scott = 1

        densities = []
        # rad_distance represents the distance from the origin coord(0,0) to an intersection point
        for rad_distance in intersections:
            densities.append(kernel_density_estimate(rad_distance, intersections, h_scott))

        if densities:  # Ensure densities list is not empty
            # np.argmax() function returns the index of the maximum value in an array or list.
            max_density_index = np.argmax(densities)  # Find index of the point with max density
            densest_radial_distance = intersections[max_density_index]  # Get the radial distance of this point

            # Convert the densest radial distance back to a 2D node coordinate (x, y)
            # The node coordinate is along the current ray (psi_angle) at densest_radial_distance
            node_coord = np.array([densest_radial_distance * np.cos(psi_angle),
                                   densest_radial_distance * np.sin(psi_angle)])

            # Add node coordinates to list
            nodes.append(node_coord)

# Final conversion of the nodes list to a NumPy array after the loop completes.
N = np.array(nodes)


#-----------------------------------------------------------------------------------------------------------------------
# NODE VISUALIZATION
#-----------------------------------------------------------------------------------------------------------------------
"""print(f"Total nodes extracted: {len(N)}")

# --- VISUALIZATION OF RESULTS ---
plt.figure(figsize=(10, 8))
plt.scatter(SProj[:, 0], SProj[:, 1], s=5, alpha=0.5, label='SProj Points')
if len(N) > 0:
    plt.scatter(N[:, 0], N[:, 1], s=100, c='red', marker='o', edgecolors='black', label='Extracted Nodes')
plt.title('SProj Data with Extracted Nodes')
plt.xlabel('Dimension 1 (ry)')
plt.ylabel('Dimension 2 (rz)')
plt.legend()
plt.grid(True)
plt.show()"""

#-----------------------------------------------------------------------------------------------------------------------
# EDGE CREATION
#-----------------------------------------------------------------------------------------------------------------------

# Line 1:
# This defines the list of angles, same as in Node Creation.
psi_angles_list = []

for i in range(r + 1):
    psi_angles_list.append(i * (2 * np.pi / r))

# node_map_by_angle dictionary will store a mapping from a normalized angle to the 2D coordinates of the node
# Key: Angle in radians, Value: Node on that ray angle
node_map_by_angle = {}
for i in range(len(psi_angles_list)):
    angle = psi_angles_list[i]
    if i < len(N): # Ensure N has a node for this angle at this index
        # Rounding is important for float key comparison.
        node_map_by_angle[round(angle, 10)] = N[i]

# Line 2:
# Initialize an empty list to store the sequence of nodes
NodeSeq = []

# Define the origin point for angle calculations and ray intersections (Omega)
ray_origin_point = np.array([0.0, 0.0])

# Line 3: foreach i in [0, |SProj| - 1] do
for i in range(len(SProj)):  # Iterate through each point in SProj

    SProj_i = SProj[i]  # Current point in the SProj sequence
    if i == len(SProj) - 1:
        # This means SProj_i is the very last point, so there's no SProj_i_plus_1
        # So, we skip this last point as no segment can be formed from it.
        continue
    SProj_i_plus_1 = SProj[i + 1]

    # Line 4:
    # Need to calculate the Polar angles
    # It's the angle measured counter-clockwise from the positive x-axis to the vector pointing from the origin to the SProj point.

    # SProj_i[1]: This accesses the y-coordinate of the SProj_i point. rz_value
    # SProj_i[0]: This accesses the x-coordinate of the SProj_i point. ry_value

    # np.arctan2() is a NumPy function that computes the arctangent of y/x, and it takes two separate arguments, y and x
    psi_i = np.arctan2(SProj_i[1], SProj_i[0])  # y-coordinate first, then x-coordinate
    # Line 5:
    psi_i_plus_1 = np.arctan2(SProj_i_plus_1[1], SProj_i_plus_1[0])  # y-coordinate first, then x-coordinate

    """
    Why we need it:
        Normalization means we "shift" all angles into a consistent, positive [0, 360) degree range.
        We do this using (angle + 360) % 360 (or (angle + 2 * np.pi) % (2 * np.pi) for radians).
    """
    # Normalize angles to [0, 2*pi) and determine angular range
    psi_i_norm = (psi_i + 2 * np.pi) % (2 * np.pi)
    psi_i_plus_1_norm = (psi_i_plus_1 + 2 * np.pi) % (2 * np.pi)

    if psi_i_norm <= psi_i_plus_1_norm:
        # Example: psi_i_norm is 120 degrees, psi_i_plus_1_norm is 200 degrees.
        # This represents a segment that sweeps from 120 degrees to 200 degrees.
        # The sweep occurs in a continuous, numerically increasing direction without crossing the 0/360 boundary.
        angular_range_start = psi_i_norm
        # In this example, angular_range_start becomes 120 degrees. This is the numerical start of our angular slice.

        angular_range_end = psi_i_plus_1_norm
        # In this example, angular_range_end becomes 200 degrees. This is the numerical end of our angular slice.

        is_wrapped = False
        # This flag is False because the interval [120, 200] does not cross the 0/360 boundary.
    else:
        # Example: psi_i_norm is 330 degrees, psi_i_plus_1_norm is 20 degrees.
        # The segment sweeps counter-clockwise from 330 degrees, through 0/360 degrees, and stops at 20 degrees.
        angular_range_start = psi_i_norm
        # In the example above, angular_range_start becomes 330 degrees. This is the start of the first part of the wrapped range.

        angular_range_end = psi_i_plus_1_norm
        # In the example above, angular_range_end becomes 20 degrees. This is the end of the second part of the wrapped range.

        is_wrapped = True
        # This flag is True because the interval [330, 20] crosses the 0/360 boundary.

    closest_node = None
    min_dist = float('inf') # Sets the value to positive infinity

    # Line 6:
    # Iterate through all global psi_angles from Node Creation to find relevant rays
    # A ray is considered "relevant" if its psi_angle falls within the slice
    # (defined by angular_range_start, angular_range_end) of the current SProj segment (SProj_i to SProj_i_plus_1).
    for psi_angle in psi_angles_list:
        # psi_angle is a radian value for example 0.2465

        """
        Let's say psi_angle = -0.7854 radians (which is -45 degrees). This is a value np.arctan2() might return.
            psi_angle + 2 * np.pi
            -0.7854 + 6.28318 = 5.49778     (In degrees: -45 + 360 = 315 degrees)
            5.49778 % 6.28318 = 5.49778     (In degrees: 315 % 360 = 315 degrees)    
        """
        psi_angle_norm = (psi_angle + 2 * np.pi) % (2 * np.pi) # Normalizing the psi_angle
        # + 2 * np.pi: This part ensures that if psi_angle were ever a negative value it would shit to positive

        # Check if global_psi_angle falls within the segment's angular range (slice)
        if (not is_wrapped and angular_range_start <= psi_angle_norm <= angular_range_end) or \
        (is_wrapped and (psi_angle_norm >= angular_range_start or psi_angle_norm <= angular_range_end)):

        #Lines 7 and 8 in Algorithm 3 are conceptual steps in the paper's pseudo-code for defining the ray and
        # passing it to Intersect. Our Python implementation directly provides the intersect_segment_ray function
        # with the essential components of a ray (origin and direction vector).

            # A unit vector is a special type of vector that has a magnitude (length) of exactly 1.

            # This line creates a 2-dimensional unit vector that points in the direction of the current psi_angle_norm.
            ray_dir = np.array([np.cos(psi_angle_norm), np.sin(psi_angle_norm)])
            # Line 9:
            # This variable will store the resulting 2D coordinates of the intersection point
            intersection = intersect_segment_ray(SProj_i, SProj_i_plus_1, ray_origin_point, ray_dir)

            # Line 10: n_int <- argmin_n in N_psi (|x_int - n|);
            # We are looking for the closest node to current intersection
            # the "best" node from N_psi to represent that part of the trajectory is the one closest to where the
            # ray actually hits the segment
            if intersection is not None:  # Only proceed if an intersection point was found for this ray and segment.
                # Check if this normalized psi_angle corresponds to an actual node in N (node_map_by_angle).
                if round(psi_angle_norm, 10) in node_map_by_angle:
                    # Get the specific node from N that was associated with this angle during Node Creation.
                    current_node = node_map_by_angle[round(psi_angle_norm, 10)]

                    # Calculate the Euclidean distance from the intersection point to the current_node
                    dist_to_node = np.linalg.norm(intersection - current_node)

                    # Argmin step: if this node is closer than any found so far, update
                    if dist_to_node < min_dist:
                        min_dist = dist_to_node
                        closest_node = current_node

            # Line 11: add node in NodeSeq;
            if closest_node is not None:
                NodeSeq.append(closest_node)

# Line 13: E <- {(NodeSeq_i, NodeSeq_i+1)}_{i in [0, |fullPath|]};
Edges_dict = {}

for i in range(len(NodeSeq) - 1):
    node_from = tuple(NodeSeq[i])
    node_to = tuple(NodeSeq[i+1])

    # The 'edge' is a pair of node coordinate tuples ((x1, y1), (x2, y2))
    # This unique tuple will serve as the key in the 'Edges_dict'
    edge = (node_from, node_to)

    # dict.get(key, default_value): This form returns the value for key if key is in the dictionary, otherwise it returns default_value.
    Edges_dict[edge] = Edges_dict.get(edge,0) + 1

#-----------------------------------------------------------------------------------------------------------------------
# SUBSEQUENCE SCORING & ANOMALY DETECTION GRAPH
#-----------------------------------------------------------------------------------------------------------------------

# Definition 9
# We precalculate the degree of each node because we need it for the score formula
node_degrees = {}
# Initialize all nodes with 0

# causes an error: node_degrees[node_tuple] = 0
for node_tuple in N:
    node_degrees[tuple(node_tuple)] = 0

# Calculate the degree for each node based on the edges starting from it
for edge, weight in Edges_dict.items():
    node_from, node_to = edge
    # The degree is the sum of weights of outgoing edges
    if node_from in node_degrees:
        node_degrees[node_from] += weight

def calculate_normality_score(subpath_of_edges, all_edges, node_degrees):
    # Instead of looking at just one transition at a time (from Node A to Node B), the final scoring algorithm looks
    #       at a whole sequence of consecutive transitions at once.
    # all_edges = Edges.dict()
    score = 0

    for edge in subpath_of_edges:
        node_from, node_to = edge
        weight = all_edges.get(edge, 0)
        degree = node_degrees.get(node_from, 1) # Default to 1 to handle dead-end nodes and prevent the score from becoming negative.

        score += weight * (degree - 1)

    return score / len(subpath_of_edges)

# Score all subsequences by sliding a window over the NodeSeq
normality_scores = []
# Loop through the entire sequence of nodes

for i in range(len(NodeSeq) - length):
    # Get the path of edges for the current window of nodes
    current_path_edges = []
    for j in range(length - 1):
        # Create an edge tuple from consecutive nodes in the NodeSeq
        edge = (tuple(NodeSeq[i+j]), tuple(NodeSeq[i+j+1]))
        current_path_edges.append(edge)

    # Calculate the score for this specific window's path
    window_score = calculate_normality_score(current_path_edges, Edges_dict, node_degrees)
    normality_scores.append(window_score)

print(f"Calculated {len(normality_scores)} normality scores.")


# Apply Moving Average Filter
# This is the final smoothing step
scores_series = pd.Series(normality_scores) # Transform into a pandas series so we can use .rolling
smoothing_window = 300 # lower means less smoothing

# data.rolling(num) creates overlapping groups of "num" data points.
# data.rolling(num).mean() then calculates the average for each of those "num"-point groups.
smoothed_normality_scores = scores_series.rolling(window=smoothing_window).mean()


# Plot the Final Anomaly Score Graph
anomaly_scores = 1 / (smoothed_normality_scores)

plt.figure(figsize=(15, 6))
plt.plot(anomaly_scores, color='red')
plt.title("Anomaly Score Graph", fontsize=16)
plt.xlabel('Subsequence Window Index / Time', fontsize=12)
plt.ylabel('Anomaly Score', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
#-----------------------------------------------------------------------------------------------------------------------
# SUMMARY STATISTICS
#-----------------------------------------------------------------------------------------------------------------------
print("\n--- Summary Statistics ---")

print(f"\n--Basic Properties--")
print(f"Total Number of Nodes: {len(N)}")
print(f"Total Number of Unique Edges: {len(Edges_dict)}")

# Edge Weight Analysis
# The weights tell us how frequently each transition occurs
if len(Edges_dict) > 0:
    edge_weights = list(Edges_dict.values())
    edge_weights_series = pd.Series(edge_weights)

    print("\n--Edge Weight Analysis--")
    print(edge_weights_series.describe())
else:
    print("\nNo edges found.")
print("\n mean: average number of transitions")

# Node Degree Analysis
# The degrees tell us how connected the nodes are.
if len(N) > 0:
    degrees = list(node_degrees.values())
    degrees_series = pd.Series(degrees)

    print("\n--Node Degree Analysis--")
    print(degrees_series.describe())
else:
    print("\nNo nodes found.")
print("\n mean: average number of degrees")

print("\n----------------------------------------")
#-----------------------------------------------------------------------------------------------------------------------
# FULL GRAPH VISUALIZATION
#-----------------------------------------------------------------------------------------------------------------------

"""plt.figure(figsize=(12, 10))
plt.scatter(SProj[:, 0], SProj[:, 1],
            s=5,
            alpha=0.15,
            label='SProj Points')

plt.scatter(N[:, 0], N[:, 1],
            s=150,
            c='orange',
            marker='o',
            edgecolors='black',
            zorder=2,
            label='Extracted Nodes')  # Increased node size, added zorder

# Plot the edges
for edge, weight in Edges_dict.items():
    node_from_coords = np.array(edge[0])  # Convert tuple back to numpy array for plotting
    node_to_coords = np.array(edge[1])  # Convert tuple back to numpy array for plotting

    # Plot a line segment for each edge
    plt.plot(
        [node_from_coords[0], node_to_coords[0]],  # x-coordinates
        [node_from_coords[1], node_to_coords[1]],  # y-coordinates
        color='red',
        linewidth=0.5 + (weight / max(Edges_dict.values()) * 1.5),  # Line width proportional to weight
        alpha=0.7,  # Transparency for edges
        zorder=1  # Ensure edges are drawn behind nodes but above SProj points
    )

plt.title('SProj Data with Extracted Nodes and Edges')
plt.xlabel('Dimension 1 (ry)')
plt.ylabel('Dimension 2 (rz)')
plt.legend()
plt.grid(True)
plt.show()"""
