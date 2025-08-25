# Imports
import os
import pandas as pd
import numpy as np

FEATURE_FILE = "features_raw.parquet"  # Cached file to avoid re-extracting features every run

# -----------------------------------------------------------------------------------------------------------------------
# FEATURE EXTRACTION
# -----------------------------------------------------------------------------------------------------------------------

def extract_or_load_features():
    """
        Loads cached features if available. Otherwise:
        - Loads time series signal data
        - Performs windowing
        - Extracts statistical features using tsfresh
        - Imputes missing values
        - Saves extracted features to a cache file
    """

    # Preverjanje če imamo že cached feature file, da ne rabimo ponovno Extractat features------------------------------
    if os.path.exists(FEATURE_FILE):
        print("Loading cached features...")
        return pd.read_parquet(FEATURE_FILE), None
        # pandas.read_parquet -"Load a parquet object from the file path, returning a DataFrame."

    # Nimamo cached feature extraction datoteke
    print("No cached features found. Running extraction...")
    # ------------------------------------------------------------------------------------------------------------------
    # Load raw sensor data df-DataFrame(tabela)
    df = pd.read_parquet("20.parquet")
    # pandas.read_parquet -"Load a parquet object from the file path, returning a DataFrame."

    # Izberemo katere podatke senzorjev zajamemo------------------------------------------------------------------------
    selected_signals = [
        "gpu0_core_temp_avg", "gpu1_core_temp_avg",
        "p0_power_avg", "p1_power_avg"
    ]

    # 'value' column separate for server status checking
    value_series = df['value'].ffill()
    window_statuses = []

    # V DataFrame-u zapolnemo manjkajoče vrednosti -NaN(Not a number)
    # .fillna - Nastavi novo vrednost v DataFrame-u, kjer ta manjka
    # ffil (forwards fill) zapolni NaN z vrednostjo pred NaN | Primer: [10, NaN, NaN, 20] --> [10, 10, 10, 20]
    # bfill (backwards fill) zapolni NaN z vrednostjo po Nan | Primer: [10, NaN, NaN, 20] --> [10, 20, 20, 20]
    df = df[selected_signals].ffill()

    # Windowing --------------------------------------------------------------------------------------------------------

    # The goal is to transform your single long DataFrame into a format where tsfresh
    # can understand each 'window' as a separate time series instance.

    window_size = 48  # 96 = 15 minutnih intervalov oziroma 24h

    # 1. Add an 'id' column to distinguish different time series instances.
    #    For windowing, each window will be a unique 'id'.
    #    We'll also keep the original 'timestamp' as tsfresh needs a time index.

    # Seznam za "windowed" data
    windowed_data = []
    # Število oken (število elementov v DataFrame-u // velikost okna)
    num_windows = len(df) // window_size

    # Zanka ki naredi DataFrame za vsako okno
    #   Vstavi: id, timestamp
    for i in range(num_windows):
        start_index = i * window_size
        end_index = start_index + window_size

        # Izrežemo del glavnega DataFrame-a df
        # .iloc(integer-location based indexing) - vzame izrez med dvema indexoma
        # .copy uporabi kopijo iz originalnega df -ja da se izognemu spreminjanju podatkov v og df (SettingWithCopyWarning)
        window_df = df.iloc[start_index:end_index].copy()

        # naredimo unikatni ID in timestamp
        window_id = f"window_{i}"

        window_df["id"] = window_id
        window_df["timestamp"] = df.index[
                                 start_index:end_index]  # Shrani timestamp med dvema indexoma vrstic npr. med 0 in 3(not included) : [2020-03-09 12:00:00, 2020-03-09 12:15:00, 2020-03-09 12:30:00]

        windowed_data.append(window_df)

        # SERVER STATUS CHECKING
        # Check the status of the value column for the current window
        window_values = value_series.iloc[start_index:end_index]
        # If any value in this window is not 0 we mark it as an error window (True)
        is_error_window = (window_values != 0).any()
        window_statuses.append(is_error_window)

    # Concatenate all window DataFrames into a single DataFrame suitable for tsfresh
    # tsfresh pričakuje DataFrame z stolpci: 'id', 'timestamp'

    # pd.concat združi vse zapise iz seznama "windowed_data" v en dolg DataFrame
    df_long = pd.concat(windowed_data)

    # Tsfresh-----------------------------------------------------------------------------------------------------------
    from tsfresh import extract_features

    print("Extracting features using tsfresh...")
    # Pass the windowed DataFrame to tsfresh
    # tsfresh will automatically group by the 'id' column and extract features for each window

    # extract_features bo za vsako "časovno okno" (identificirano z unikatnim id) in za vsak "signal" znotraj
    # tega okna (kot so gpu0_core_temp_avg, p0_power_avg itd.) izračunal veliko različnih
    # statističnih značilnosti (npr. povprečje, standardni odklon, minimum, maksimum...).
    # Rezultat bo nov DataFrame z imenom "extracted_features".
    extracted_features = extract_features(
        # extract_features
        df_long,  # Naš DataFrame z združenimi "windowed" podatki
        column_id='id',  # Da lahko vsako okno podatkov obravnava kot ločeno časovno serijo. Stolpec
        column_sort='timestamp',  # Izračun časovno odvisnih značilnosti: Številne pomembne značilnosti timestamp-ov
        # (kot so trend, avtokorelacija, število prehodov čez ničlo, spremembe v variabilnosti)
        # so odvisne od zaporedja podatkov. Brez pravilnega časovnega indeksa tsfresh
        # ne bi mogel natančno izračunati teh značilnosti
    )
    # Znebimo se dodatnih Nan, ki jih včasih naredi tsfresh
    # včasih naredi NaN za lastnosti, ki ne morajo biti izračunani za podan series

    # Step 1: Replace all infinity values with NaN
    cleaned_features = extracted_features.replace([np.inf, -np.inf], np.nan)
    print("Replaced infinity values with NaN.")

    # Step 2: Drop any column where ALL values are NaN.
    features_with_data = cleaned_features.dropna(axis='columns', how='all')
    print(
        f"Removed {extracted_features.shape[1] - features_with_data.shape[1]} useless (all-NaN or inf) feature columns.")

    # Step 3: Use ffill/bfill method to impute the remaining NaNs.
    imputed_features = features_with_data.ffill().bfill()
    print("Imputation complete. No NaN or inf values remain.")

    # Shranimo že "extracted features" datoteko, da je nerabimo vsakič na novo naredit
    # .to_parquet() je pandas metoda, ki shrani DataFrame v obliko .parquet datoteke
    imputed_features.to_parquet(FEATURE_FILE)

    return imputed_features, window_statuses


if __name__ == "__main__":
    features_df, window_statuses = extract_or_load_features()  # Kličemo zgornjo funkcijo in shranimo predelane feature-je v spremenljijvko features_df
    print(f"Shape of extracted features: {features_df.shape}")
    # ----------------------------------------------------------------------------------------------------------------------
    # FEATURE SELECTION
    # ----------------------------------------------------------------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Features often have different scales, which can affect distance-based algorithms like PCA and clustering.
    # Thats why its common practice to scale them first.

    """
        .fit()
        For each individual feature column, it calculates the mean and the standard deviation.
        The scaler object internally stores these calculated values.

        .transform()
        Using the mean and standard deviation it learned in the fit step, the StandardScaler then transforms
        each value in features_df.

        Xscaled = (X−μ)/σ
        μ - sredina feature column-a
        σ - standardni odklon feature column-a
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    n_features_to_select = 10 # 20 is good for 96 window size, 5 is okay for 48 window size ------------------------------Num of selected features
    pca = PCA(n_features_to_select)

    pca.fit(scaled_features)
    # pca.components_ retrieves the results from the trained model. It contains the principal components that were
    #       calculated in the .fit() step.
    # The values inside this attribute are called loadings, and they represent the weight or influence of each original
    #       feature.
    loadings = pca.components_

    # Identify the most important feature for each component.
    # We use a set to ensure we only select unique features.
    selected_feature_indexes = set()
    for i in range(n_features_to_select):
        component_loadings = loadings[i]

        # Get the indexes of features sorted by their absolute loading value
        # np.abs() we take the absolute value because a loading of -0.8 is just as strong as +0.8
        # np.argsort() returns the original indexes of the values in ascending order. Example:
        #       Original Index  	0	    1	    2	    3
        #       Value	           0.19	   0.75	   0.62	   0.08
        #       Returns [3, 0, 2, 1]
        # [::-1] reverses the order so now it would look like [1,2,0,3]
        sorted_feature_indexes = np.argsort(np.abs(component_loadings))[::-1]

        # Find the first feature in the sorted list that has not been selected yet
        for feature_index in sorted_feature_indexes:
            if feature_index not in selected_feature_indexes:
                selected_feature_indexes.add(feature_index)
                break

    # Convert to a list for pandas library
    selected_feature_indexes = list(selected_feature_indexes)

    # Create a new DataFrame containing only the selected "principal features".
    scaled_pfa_features = scaled_features[:, selected_feature_indexes]

    print(f"Original number of features: {features_df.shape[1]}")
    print(f"Features reduced to {scaled_pfa_features.shape[1]} dimensions using PFA.")
    # ----------------------------------------------------------------------------------------------------------------------
    # CLUSTERING (UNSUPERVISED MODE)
    # ----------------------------------------------------------------------------------------------------------------------
    # Hierarchical Clustering (AgglomerativeClustering in sklearn) is the default in the paper.
    #   -It starts with each data point as its own individual cluster.
    #   -Then, it iteratively merges the closest pairs of clusters until all data points are in one of the clusters
    #    and the set number of clusters is reached

    #   -The "closeness" between clusters is determined by a linkage criterion (e.g., Ward, average, complete, single),
    #    which defines how the distance between two clusters is calculated.
    #   -The "distance" between individual data points is determined by a metric (in our case: Euclidean distance)

    n_clusters = 3

    # Initialize and fit the Hierarchical Clustering model
    # 'metric' defines how distance between samples is measured. We will use "Euclidean"
    # 'linkage' defines how distance between clusters is measured. We will use "Ward"
    from sklearn.cluster import AgglomerativeClustering

    clustering_model = AgglomerativeClustering(n_clusters, metric='euclidean', linkage='ward')
    # .fit This is the learning phase for the clustering algorithm. the fit method builds a dendrogram
    #   (the hierarchy of clusters) by iteratively merging (or splitting) data points based on the specified metric and linkage
    # .predict After the fit method has learned the cluster structure, the predict method then assigns
    #   each data point in selected_features to its corresponding cluster.
    clusters = clustering_model.fit_predict(scaled_pfa_features)  # fit_predict returns the cluster labels for each sample

    print(f"Clustering complete")
    # ----------------------------------------------------------------------------------------------------------------------
    # CLUSTER/SERVER STATUS ANALYSIS
    # ----------------------------------------------------------------------------------------------------------------------
    print("\n--- Cluster Status Analysis ---")
    error_per_cluster = {}

    if window_statuses is not None:
        for i in range(len(clusters)):
            cluster = clusters[i]
            if(window_statuses[i]): # If there is and error in this window
                error_per_cluster[cluster] = error_per_cluster.get(cluster, 0) + 1
                error_per_cluster.get(cluster,0)

        for (key, value) in error_per_cluster.items():
            print("\n")
            print(f"Cluster {key} contains {value} Error windows")
        print("\n")

    # ----------------------------------------------------------------------------------------------------------------------
    # TRANSITION COUNT
    # ----------------------------------------------------------------------------------------------------------------------
    trans_counts = {}

    for i in range(len(clusters) - 1):
        current_element = clusters[i]
        next_element = clusters[i+1]
        if current_element != next_element:
            key = (current_element, next_element)
            trans_counts[key] = trans_counts.get(key, 0) + 1

    print("--- Transition Analysis ---\n")

    # A set to keep track of pairs we've already printed
    processed_pairs = set()

    for key, value in trans_counts.items():

        from_cluster, to_cluster = key

        # If we have already processed this pair, skip it.
        if key in processed_pairs:
            continue

        # Find the reverse transition key
        reverse_key = (to_cluster, from_cluster)

        # Get the count of the reverse transition. Defaults to 0 if it doesn't exist.
        reverse_value = trans_counts.get(reverse_key, 0)

        print(f"cluster {from_cluster} to cluster {to_cluster}, {value} times")
        print(f"cluster {to_cluster} to cluster {from_cluster}, {reverse_value} times")
        print()
        processed_pairs.add(key)
        processed_pairs.add(reverse_key)

    # ----------------------------------------------------------------------------------------------------------------------
    # CLUSTER VISUALIZATION
    # ----------------------------------------------------------------------------------------------------------------------
    print("Visualizing Clusters")

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    """
    pd.Series() -This creates a new pandas Series object. A Series in pandas is a one-dimensional labeled array.

    clusters: These are the values that will fill this new Series.
    These are the numerical cluster labels (0, 1, 2, etc.) that your clustering algorithm assigned.
    """

    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
    ]

    cluster_assignments = pd.Series(clusters)

    # Perform t-SNE for 2D visualization
    tsne = TSNE(n_components=2, random_state=42)  # Initialize t-SNE for 2D visualization with a fixed random state.
    tsne_results = tsne.fit_transform(scaled_pfa_features)  # Apply t-SNE to reduce PCA features to 2 dimensions for drawing the graph.

    plt.figure(figsize=(10, 8))  # Določi velikost grafa (prikaza)
    # Iterate through unique clusters to draw each one with a different color
    for i in range(n_clusters):  # Loop through each identified cluster (from 0 to n_clusters-1).
        # Select data points belonging to the current cluster
        cluster_points = tsne_results[
            cluster_assignments == i]  # Filter the t-SNE results to get only points belonging to the current cluster.
        plt.scatter(  # Nariše piko
            cluster_points[:, 0],  # X-coordinates for the scatter plot.
            cluster_points[:, 1],  # Y-coordinates for the scatter plot.
            color=colors[i],  # Use the color from our list
            label=f'Cluster {i}',  # Add a label for the legend
            alpha=0.7  # Naredi pike transparente
        )
    # EDGE CREATE-------------------------------------------------------------------------------------------------------

    # CLUSTER CENTROIDS
    centroids = {}
    for i in range(n_clusters):
        # Filter points for the current cluster using masks
        # It compares every value in the cluster_assignments list to the current cluster ID from the loop
        cluster_points = tsne_results[cluster_assignments == i]

        # Check if the cluster has any points before calculating the mean
        if len(cluster_points) > 0:
            # Calculate the centroid (the average x and y position)
            centroid = np.mean(cluster_points, axis=0)
            centroids[i] = centroid  # Optionally store centroids in a dict

            # Plot the large centroid dot
            plt.scatter(
                centroid[0],  # Centroid x-coordinate
                centroid[1],  # Centroid y-coordinate
                color=colors[i],
                s=250,  # size of the marker
                edgecolor='black',
                linewidth=1,
                zorder=10  # zorder ensures centroids are drawn on top
            )
    # EDGES
    # Find max count for scaling the line thickness
    max_count = max(trans_counts.values()) if trans_counts else 1
    # Draw edge
    for (start_cluster, end_cluster), count in trans_counts.items():
        if start_cluster in centroids and end_cluster in centroids:
            # Calculate dynamic linewidth
            linewidth = 0.5 + (count / max_count) * 5.5
            plt.annotate(
                "",
                xy=centroids[end_cluster],
                xytext=centroids[start_cluster],
                arrowprops=dict(
                    arrowstyle="-|>,head_length=0.8,head_width=0.4",
                    color="black",
                    alpha=0.9,
                    linewidth=linewidth,
                    # Increase 'rad' for a more apparent curve
                    connectionstyle="arc3,rad=0.3"
                )
            )

    plt.title('t-SNE using PFA')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()