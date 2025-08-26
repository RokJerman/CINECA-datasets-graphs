import tkinter as tk
from tkinter import filedialog
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# --- Default folder paths ---
DEFAULT_GRAPH_DIR = 'graph_storage'
DEFAULT_SCORES_DIR = 'scores_storage'


def display_graph():
    """
    Opens a file dialog to select a GraphML file, then loads it with NetworkX
    and displays it using Matplotlib.
    """
    filepath = filedialog.askopenfilename(
        initialdir=DEFAULT_GRAPH_DIR,
        title="Select a GraphML file",
        filetypes=(("GraphML files", "*.graphml"), ("All files", "*.*"))
    )
    if not filepath:
        print("No file selected.")
        return

    try:
        G = nx.read_graphml(filepath)
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, iterations=50)
        nx.draw(
            G,
            pos,
            with_labels=False,
            node_size=50,
            node_color='skyblue',
            edge_color='gray',
            width=0.5
        )
        filename = os.path.basename(filepath)
        plt.title(f"Graph Visualization: {filename}", fontsize=16)
        plt.show()
    except Exception as e:
        tk.messagebox.showerror("Error", f"Could not display graph.\nError: {e}")


def display_scores():
    """
    Opens a file dialog to select a CSV file, then loads it with Pandas
    and plots the anomaly scores using Matplotlib with a FIXED y-axis.
    """
    filepath = filedialog.askopenfilename(
        initialdir=DEFAULT_SCORES_DIR,
        title="Select an Anomaly Scores CSV file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if not filepath:
        print("No file selected.")
        return

    try:
        df = pd.read_csv(filepath)
        if 'anomaly_score' not in df.columns:
            tk.messagebox.showerror("Error", "The CSV file must contain an 'anomaly_score' column.")
            return

        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df['anomaly_score'], color='red', linewidth=1.5)

        # --- FIX IS HERE ---
        # Set a fixed range for the y-axis for consistent comparison.
        # You can adjust the upper limit (e.g., 20, 50) based on your data.
        plt.ylim(0, 0.01)
        # --- END FIX ---

        filename = os.path.basename(filepath)
        plt.title(f"Anomaly Scores: {filename}", fontsize=16)
        plt.xlabel("Time (Resampled Index)", fontsize=12)
        plt.ylabel("Anomaly Score", fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        tk.messagebox.showerror("Error", f"Could not plot scores.\nError: {e}")


def main():
    """
    Sets up and runs the main Tkinter GUI window.
    """
    root = tk.Tk()
    root.title("Graph and Anomaly Visualizer")
    root.geometry("400x200")
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(expand=True, fill=tk.BOTH)

    graph_button = tk.Button(
        main_frame,
        text="Load and Display GraphML",
        command=display_graph,
        font=("Helvetica", 12),
        bg="#E0F7FA",
        fg="black",
        relief=tk.RAISED,
        borderwidth=2
    )
    graph_button.pack(pady=10, fill=tk.X)

    scores_button = tk.Button(
        main_frame,
        text="Load and Plot Anomaly Scores",
        command=display_scores,
        font=("Helvetica", 12),
        bg="#FFEBEE",
        fg="black",
        relief=tk.RAISED,
        borderwidth=2
    )
    scores_button.pack(pady=10, fill=tk.X)

    root.mainloop()


if __name__ == "__main__":
    main()

