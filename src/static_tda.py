import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def extract_structure_for_mapper(thread_df: pd.DataFrame) -> np.ndarray:
    """
    Extract structured point cloud for one thread.
    For mapper we need:
      1. normalized time in thread
      2. is_source and is_reply
      3. normalized reply depth
      4. normalized order position of its parent tweet
      5. normalized author activity within thread
      6. normalized children count
    """

    # 1. normalized time in thread
    t = thread_df["created_at_ts"].to_numpy(dtype=float)
    t0 = t.min()
    dt = t - t0
    scale_t = dt.max() if dt.max() > 0 else 1.0
    t_norm = dt / scale_t

    # 2. is_source and is_reply
    is_source = thread_df["is_source"].astype(int).to_numpy()
    is_reply = thread_df["in_reply_to_status_id"].notna().astype(int).to_numpy()

    # 3. normalized reply depth, we will use a graph to compute this
    g = nx.DiGraph()

    tweet_ids = set(thread_df["tweet_id"].dropna().astype(str))

    # nodes from tweets
    for _, row in thread_df.iterrows():
        tweet_id = str(row["tweet_id"])
        g.add_node(tweet_id)

    # edges from replies
    for _, row in thread_df.iterrows():
        child = str(row["tweet_id"])
        parent = row["in_reply_to_status_id"]

        if pd.notna(parent):
            parent = str(parent)
            if parent in tweet_ids:
                g.add_edge(parent, child)


    source_rows = thread_df[thread_df["is_source"] == True]
    source_tweet_id = str(source_rows.iloc[0]["tweet_id"]) if len(source_rows) > 0 else str(thread_df.iloc[0]["tweet_id"])

    # add shortest-path depth from source tweet to every tweet in the reply graph.
    depths = {node: 0 for node in g.nodes()}
    if source_tweet_id in g:
        lengths = nx.single_source_shortest_path_length(g, source_tweet_id)
        depths.update(lengths)

    max_depth = max(depths.values()) if len(depths) > 0 else 1
    max_depth = max(max_depth, 1)
    reply_depth = np.array([depths.get(str(tid), 0) / max_depth for tid in thread_df["tweet_id"]], dtype=float)

    # 4. normalized order position of its parent tweet
    tweet_to_order = {str(tid): i for i, tid in enumerate(thread_df["tweet_id"].tolist())}
    denom = max(len(thread_df) - 1, 1)

    parent_order_map = {}
    for _, row in thread_df.iterrows():
        tweet_id = str(row["tweet_id"])
        parent = row["in_reply_to_status_id"]

        if pd.notna(parent) and str(parent) in tweet_to_order:
            parent_order_map[tweet_id] = tweet_to_order[str(parent)] / denom
        else:
            parent_order_map[tweet_id] = 0.0

    parent_order = np.array([parent_order_map[str(tid)] for tid in thread_df["tweet_id"]], dtype=float)

    # 5. normalized author activity within thread
    user_counts_series = thread_df["user_id"].astype(str).value_counts()
    max_user_count = max(int(user_counts_series.max()), 1)
    author_activity = np.array(
        [user_counts_series.get(str(uid), 0) / max_user_count for uid in thread_df["user_id"]],
        dtype=float
    )

    # 6. normalized children count
    children_count_map = {str(tid): 0 for tid in thread_df["tweet_id"]}

    for _, row in thread_df.iterrows():
        parent = row["in_reply_to_status_id"]
        if pd.notna(parent):
            parent = str(parent)
            if parent in children_count_map:
                children_count_map[parent] += 1

    max_children = max(children_count_map.values()) if len(children_count_map) > 0 else 1
    max_children = max(max_children, 1)
    children_count = np.array(
        [children_count_map.get(str(tid), 0) / max_children for tid in thread_df["tweet_id"]],
        dtype=float
    )

    X = np.column_stack([
        t_norm,
        is_source,
        is_reply,
        reply_depth,
        parent_order,
        author_activity,
        children_count,
    ])

    return X, g
    

def betti_from_graph(G: nx.Graph) -> dict:
    """
    H0 = connected components
    H1 = cycle rank = E - V + C
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G) if n_nodes > 0 else 0
    h0 = n_components
    h1 = n_edges - n_nodes + n_components if n_nodes > 0 else 0

    return {
        "n_nodes_mapper": n_nodes,
        "n_edges_mapper": n_edges,
        "h0_mapper": h0,
        "h1_mapper": h1,
    }


def plot_graph(G: nx.Graph):
    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8)
    plt.title("Mapper graph")
    plt.axis("off")
    plt.show()
