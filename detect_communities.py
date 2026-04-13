#! /usr/bin/python3

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pprint

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from multiprocessing import Pool
from k_means_constrained import KMeansConstrained

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import torch_geometric.transforms as T
from torch_geometric.datasets import MovieLens100K
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.nn import GCNConv, GAE, VGAE

from create_pyg_dataset import dict_to_pyg_dataset

prefix_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{prefix_dir}')
sys.path.append(f'{prefix_dir}/..')
sys.path.append(f'{prefix_dir}/../..')

import maxsetsbp
from largebicliques import run_largebicliques
from removedominatorsbp import removedominators, readem, saveem, get_em, dmfromem, hasbeenremoved
from readup import readup, readup_and_usermap_permmap, dumpup
import greedythenlattice


from colorama import Fore, Back, Style, init

# ---------------------------
# 1. Load MovieLens100K and convert to homogeneous graph
# ---------------------------
#
# def load_movielens_homogeneous1(root="data/MovieLens100K"):
#     # Heterogeneous graph with "user" and "movie" node types
#     hetero = MovieLens100K(root=root)[0]
#     hetero = T.ToUndirected()(hetero)   # optional but nice
#
#     # Convert to homogeneous Data: x, edge_index, node_type
#     homo = hetero.to_homogeneous()
#     # Map node type string -> integer id (consistent with homo.node_type)
#     node_types = hetero.node_types           # e.g. ['user', 'movie']
#     node_type_to_id = {t: i for i, t in enumerate(node_types)}
#
#     print("Homogeneous graph:")
#     print(f"  num_nodes       = {homo.num_nodes}")
#     print(f"  num_edges       = {homo.num_edges}")
#     print(f"  num_features    = {homo.num_node_features}")
#     print(f"  node_type_to_id = {node_type_to_id}")
#     print(f"  node_type shape = {homo.node_type.shape}")
#     return homo, node_type_to_id

def load_up(upfilename: str):
    up = readup(upfilename)
    hetero, _, _ = dict_to_pyg_dataset(up)
    # Convert to homogeneous Data: x, edge_index, node_type
    # homo = hetero.to_homogeneous()
    # homo = hetero.to_homogeneous()
    homo = hetero

    # Map node type string -> integer id (consistent with homo.node_type)
    node_types = hetero.node_types  # e.g. ['user', 'movie']
    node_type_to_id = dict()
    for i, t in enumerate(node_types):
        if t == 0:
            node_type_to_id['user'] = 0
        else:
            node_type_to_id['perm'] = 1

    print("Homogeneous graph:")
    print(f"  num_nodes       = {homo.num_nodes}")
    print(f"  num_users       = {homo.num_users}")
    # print(f"  num_perms       = {homo.num_perms}")
    print(f"  num_edges       = {homo.num_edges}")
    print(f"  num_features    = {homo.num_node_features}")
    print(f"  node_type_to_id = {node_type_to_id}")
    print(f"  node_type shape = {homo.node_types.shape}")

    return homo, node_type_to_id


# ---------------------------
# 2. GCN Encoder + GAE (unsupervised) WITH EDGE WEIGHTS
# ---------------------------

def train_gae(
        data,
        hidden_channels=64,
        embedding_dim=32,
        epochs=2000,
        lr=1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    encoder = GCNEncoder(
        in_channels=data.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=embedding_dim,
    )
    model = GAE(encoder).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # assume edge weights are in data.edge_weight (or set to None)
    # edge_weight = getattr(data, "edge_weight", None)
    edge_index = data.edge_index.long()  # shape: [2, num_edges]
    edge_weight = data.edge_weight.float()

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        # pass edge_weight through encode -> encoder.forward
        z = model.encode(data.x, edge_index, edge_weight=edge_weight)

        # UNSUPERVISED reconstruction loss on adjacency (still unweighted)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        opt.step()

        if (ep + 1) % 50 == 0:
            print(f"Epoch {ep + 1}/{epochs}, loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, edge_index, edge_weight=edge_weight)

    return z.cpu().numpy()


# class GCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, 2 * hidden_channels)
#         self.conv3 = GCNConv(2 * hidden_channels, hidden_channels)
#         self.conv4 = GCNConv(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index, edge_weight=None):
#         x = self.conv1(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv3(x, edge_index, edge_weight=edge_weight)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv4(x, edge_index, edge_weight=edge_weight)
#         return x


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2 * hidden_channels)
        self.conv3 = GCNConv(2 * hidden_channels, hidden_channels)

        # Two heads instead of one:
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        mu = self.conv_mu(x, edge_index, edge_weight=edge_weight)
        logstd = self.conv_logstd(x, edge_index, edge_weight=edge_weight)
        return mu, logstd  # EXACTLY TWO outputs


def train_gae_many_large_homogeneous(
        model_filename: str,
        graphs,  # list[Data]
        hidden_channels=64,
        embedding_dim=32,
        epochs=50,
        lr=1e-3,
        batch_size=1024,
        num_neighbors=(15, 10, 10),  # must match number of GCN layers
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Check consistent input feature dimension across graphs
    in_channels = graphs[0].num_node_features
    for i, g in enumerate(graphs):
        if g.num_node_features != in_channels:
            raise ValueError(
                f"Graph {i} has num_node_features={g.num_node_features}, expected {in_channels}. "
                "All graphs must share the same node feature dimension."
            )

    # ---- Create ONE shared model
    encoder = GCNEncoder(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         out_channels=embedding_dim)
    model = VGAE(encoder).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Load if checkpoint exists
    if os.path.exists(model_filename):
        ckpt = torch.load(model_filename, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

    # ---- Train across graphs
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_steps = 0

        for data in graphs:
            # NeighborLoader expects CPU data; it yields sampled subgraphs
            loader = NeighborLoader(
                data,
                input_nodes=None,  # sample seed nodes from all nodes
                num_neighbors=list(num_neighbors),
                batch_size=batch_size,
                shuffle=True,
            )

            for batch in loader:
                batch = batch.to(device)
                edge_weight = getattr(batch, "edge_weight", None)
                num_nodes = getattr(batch, "num_users", None)

                opt.zero_grad()
                z = model.encode(batch.x, batch.edge_index, edge_weight=edge_weight)
                loss = model.recon_loss(z, batch.edge_index)
                loss = loss + (1 / num_nodes) * model.kl_loss()
                loss.backward()
                opt.step()

                total_loss += float(loss)
                total_steps += 1

        if (ep + 1) % 5 == 0:
            print(f"Epoch {ep + 1}/{epochs} | avg loss = {total_loss / max(total_steps, 1):.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "embedding_dim": embedding_dim,
    }, model_filename)

    return model


@torch.no_grad()
def infer_embeddings_all_nodes_neighbor(
        model: GAE,
        data,
        embedding_dim: int,
        device,
        batch_size=1024,
        num_neighbors=(15, 10, 10),
):
    model.eval()

    loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        shuffle=False,
    )

    z_all = torch.empty((data.num_nodes, embedding_dim), dtype=torch.float32)

    for batch in loader:
        batch = batch.to(device)
        edge_weight = getattr(batch, "edge_weight", None)

        z = model.encode(batch.x, batch.edge_index, edge_weight=edge_weight)

        # Only commit embeddings for the SEED nodes in this sampled subgraph.
        seed_count = batch.batch_size
        seed_global_ids = batch.n_id[:seed_count].cpu()
        z_seed = z[:seed_count].detach().cpu()

        z_all[seed_global_ids] = z_seed

    return z_all


def train_and_embed_many_graphs(
        model_filename: str,
        graphs,
        hidden_channels=64,
        embedding_dim=32,
        epochs=50,
        lr=1e-3,
        batch_size=1024,
        num_neighbors=(15, 10, 10, 10, 10),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_gae_many_large_homogeneous(
        model_filename=model_filename,
        graphs=graphs,
        hidden_channels=hidden_channels,
        embedding_dim=embedding_dim,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
    ).to(device)

    embeddings_per_graph = []
    for g in graphs:
        z = infer_embeddings_all_nodes_neighbor(
            model=model,
            data=g,
            embedding_dim=embedding_dim,
            device=device,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
        )
        embeddings_per_graph.append(z)  # Tensor [num_nodes_g, embedding_dim] on CPU

    return embeddings_per_graph


def train_gae_large(model_filename,
                    data,
                    hidden_channels=64,
                    embedding_dim=32,
                    epochs=200,
                    lr=1e-2,
                    batch_size=1024,
                    num_neighbors=[15, 10, 10],  # for 2 GCN layers after the first one
                    ):
    """
    Train GAE on a large graph using neighbor sampling.
    - data: PyG Data object (kept on CPU)
    - Returns: z_all (embeddings for all nodes) as a NumPy array
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create NeighborLoader that samples subgraphs
    # CHANGE: new loader for large graphs
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,  # one entry per GCN layer (beyond first)
        batch_size=batch_size,
        shuffle=True,
    )

    encoder = GCNEncoder(
        in_channels=data.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=embedding_dim,
    )
    model = GAE(encoder).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if not os.path.exists(model_filename):
        for ep in range(epochs):
            model.train()
            total_loss = 0.0

            # CHANGE: iterate over subgraph batches instead of full data
            for batch in train_loader:
                # MOVE ONLY THIS BATCH TO GPU
                batch = batch.to(device)
                edge_weight = getattr(batch, "edge_weight", None)

                opt.zero_grad()

                # CHANGE: encode on the batch graph
                z = model.encode(batch.x, batch.edge_index, edge_weight=edge_weight)

                # CHANGE: reconstruction loss on batch subgraph
                loss = model.recon_loss(z, batch.edge_index)

                loss.backward()
                opt.step()

                total_loss += loss.item()

            if (ep + 1) % 10 == 0:
                print(f"Epoch {ep + 1}/{epochs}, loss = {total_loss:.4f}")

        save_checkpoint(model_filename, model, optimizer=opt, epoch=epochs,
                        extra={"input_channels": data.num_node_features, "hidden_channels": hidden_channels,
                               "out_channels": embedding_dim})

    else:
        model_ctor = lambda: GAE(GCNEncoder(in_channels=data.num_node_features, hidden_channels=hidden_channels,
                                            out_channels=embedding_dim)).to(device)

        model, device = load_model_for_inference(model_filename, model_ctor)
        # model = torch.load(model_filename)

    # ---------------------------
    # After training, get embeddings for ALL nodes
    # ---------------------------
    model.eval()
    with torch.no_grad():
        z_all = infer_embeddings_all_nodes(model, data, device, num_neighbors, batch_size)

    return z_all


def infer_embeddings_all_nodes(model, data, device, num_neighbors, batch_size):
    """
    Compute embeddings for all nodes using neighbor sampling.
    Returns a torch.Tensor of shape [num_nodes, out_channels] on CPU.
    """
    from torch_geometric.loader import NeighborLoader

    # Loader that covers all nodes as seeds
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
    )

    num_nodes = data.num_nodes
    # Allocate final embedding matrix on CPU
    # out_channels = model.encoder.conv3.out_channels (or infer from a first batch)
    # We'll infer from a first batch:
    out_channels = None

    # Placeholder; we'll fill per batch
    z_all = None

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        edge_weight = getattr(batch, "edge_weight", None)

        # use model.encode; GAE will call encoder internally
        z_batch = model.encode(batch.x, batch.edge_index, edge_weight=edge_weight)

        # First time: allocate z_all with the right shape
        if z_all is None:
            out_channels = z_batch.size(1)
            z_all = torch.empty((num_nodes, out_channels), dtype=z_batch.dtype, device="cpu")

        # batch.n_id gives original node indices for this subgraph
        # Move z_batch to CPU and place in correct rows
        z_all[batch.n_id.cpu()] = z_batch.cpu()

    return z_all


# ---------------------------
# 3. k-means on USERS ONLY
# ---------------------------

def cluster_users_kmeans(
        embeddings: torch.Tensor,
        node_type: torch.Tensor,
        # node_type_to_id: dict,
        num_clusters: int = 20,
):
    node_type_np = node_type.cpu().numpy()
    # user_type_id = node_type_to_id["user"]
    # user_type_id = node_type_to_id["movie"]

    # user_mask = (node_type_np == user_type_id)
    user_mask = torch.ones(embeddings.size(0), dtype=torch.bool)
    # perm_mask = ~user_mask

    # user_emb = embeddings[user_mask]  # [n_users, d]
    user_emb = embeddings[user_mask]  # [n_users, d]
    # perm_emb = embeddings[perm_mask]  # [m_perms, d]

    print(f"Running k-means on users only: {user_emb.shape[0]} users")

    # kmeans = KMeans(
    #     n_clusters=num_clusters,
    #     n_init=20,
    #     random_state=42,
    # )
    # min_size = embeddings.size(0) // (2 * num_clusters)
    # max_size = embeddings.size(0) // (0.5*num_clusters)

    n = embeddings.size(0)
    # Target cluster size
    target = n // num_clusters

    # Allow some flexibility (±20%)
    min_size = max(1, int(0.8 * target))
    max_size = max(1, int(1.2 * target))

    # Enforce strict constraints
    max_size = min(max_size, n)
    min_size = min(min_size, max_size)

    if num_clusters * min_size > n:
        min_size = max(1, n // num_clusters)

    if num_clusters * max_size < n:
        # ceil(n / k)
        max_size = (n + num_clusters - 1) // num_clusters

    # max_size = (embeddings.size(0))
    kmeans = KMeansConstrained(
        n_clusters=num_clusters,
        size_min=min_size,
        size_max=max_size,
        n_jobs=-1,
        random_state=42
    )
    # kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
    user_labels = kmeans.fit_predict(user_emb)  # [n_users]
    # perm_labels = kmeans.fit_predict(perm_emb)  # [m_perms]

    # Build labels for all nodes: users get their cluster, perms get -1
    labels_all = np.full(embeddings.shape[0], fill_value=-1, dtype=int)
    labels_all[user_mask] = user_labels
    # labels_all[perm_mask] = perm_labels

    # return labels_all, user_mask, perm_mask
    return labels_all, user_mask


# ---------------------------
# 4. t-SNE visualization
# ---------------------------

def tsne_vis_users_clusters(
        embeddings: np.ndarray,
        labels_all: np.ndarray,
        user_mask: np.ndarray,
        # perm_mask: np.ndarray,
        plt_filename: str
):
    print("Running t-SNE")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    pos = tsne.fit_transform(embeddings)
    x, y = pos[:, 0], pos[:, 1]

    plt.figure(figsize=(7, 6))

    # Plot movies first (grey, unclustered)
    # plt.scatter(
    #     x[movie_mask],
    #     y[movie_mask],
    #     s=5,
    #     alpha=0.3,
    #     c="lightgray",
    #     label="movies (unclustered)",
    #     edgecolors="none",
    # )

    # Plot users colored by cluster
    user_labels = labels_all[user_mask]
    for c in sorted(set(user_labels)):
        mask_c = np.zeros_like(labels_all, dtype=bool)
        # mask_c = np.ones_like(labels_all, dtype=bool)
        mask_c[user_mask] = (user_labels == c)
        plt.scatter(
            x[mask_c],
            y[mask_c],
            s=8,
            alpha=0.8,
            label=f"user cluster {c}",
            edgecolors="none",
        )

    # perm_labels = labels_all[perm_mask]
    # for c in sorted(set(perm_labels)):
    #     mask_c = np.zeros_like(labels_all, dtype=bool)
    #     # mask_c = np.ones_like(labels_all, dtype=bool)
    #     mask_c[perm_mask] = (perm_labels == c)
    #     plt.scatter(
    #         x[mask_c],
    #         y[mask_c],
    #         s=8,
    #         marker='^',
    #         alpha=0.8,
    #         label=f"permission cluster {c}",
    #         cmap='Paired',
    #         edgecolors="none",
    #     )

    # plt.title("t-SNE of GAE embeddings\nUsers clustered by k-means, permissions just shown")
    plt.axis("off")
    # plt.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(plt_filename)
    # plt.show()


def build_user_perm_dict(data, up, labels_all, user_mask, usermap: dict, permmap: dict, em: dict):
    inv_usermap = {v: k for k, v in usermap.items()}
    inv_permmap = {v: k for k, v in permmap.items()}
    # Load heterogeneous MovieLens100K graph

    # Edge type: ('user1', 'weight', 'user2')
    edge_index = data.edge_index[:2, :]

    user_labels = labels_all[user_mask]  # cluster for each user
    user_indices = np.where(user_mask)[0]  # actual node indices of users

    clusters = sorted(set(user_labels))

    # Build dict with U* and P* prefixes
    user_to_perms_by_cluster = dict()
    usermaps_by_cluster = dict()
    permmaps_by_cluster = dict()
    print("\n=== Users grouped by cluster ===")
    for cluster in clusters:
        user_to_perms = dict()

        print(f"\nCluster {cluster}:")
        user1_nodes = edge_index[0].tolist()
        user2_nodes = edge_index[1].tolist()

        for u, v in zip(user1_nodes, user2_nodes):
            # if both u and m are assigned to the same cluster
            if user_labels[u] == cluster and user_labels[v] == cluster:
                if u in up and v in up:
                    # perms_u_to_remove = {e[1] for e in em if e[0] == u}
                    # perms_v_to_remove = {e[1] for e in em if e[0] == v}
                    perms_u = up[u]
                    perms_v = up[v]
                    u_key = int(f"{u}")
                    v_key = int(f"{v}")
                    user_to_perms[u_key] = up[u]
                    user_to_perms[v_key] = up[v]
                    # if u_key not in user_to_perms:
                    #     user_to_perms[u_key] = set()
                    # user_to_perms[u_key].update(perms_u)
                    # if v_key not in user_to_perms:
                    #     user_to_perms[v_key] = set()
                    # user_to_perms[v_key].update(perms_v)
                else:
                    # print('ERROR: one of the users not in UP')
                    # print(f'Is user {u} found? {u in up}')
                    # print(f'Is user {v} found? {v in up}')
                    pass
            else:
                # If user u and v assigned to different clusters, then we handle such cases below with LABEL: USERS_FIX
                # print(f'Users {u} and {v} assigned to different clusters')
                # print(f'User {u} is assigned to {user_labels[u]}')
                # print(f'User {v} is assigned to {user_labels[v]}')
                pass

        if len(user_to_perms) > 0:
            user_to_perms_by_cluster[cluster] = {u1: sorted(list(perms)) for u1, perms in user_to_perms.items()}
            usermaps_by_cluster[cluster] = {u1: inv_usermap[u1] for u1 in user_to_perms_by_cluster[cluster]}
            permmaps_by_cluster[cluster] = {p1: inv_permmap[p1] for u1 in user_to_perms_by_cluster[cluster] for p1 in
                                            up[u1]}

    # LABEL: USERS_FIX - find users and their permissions not currently assigned to any clusters right now
    users_clustered = set()
    for cluster in clusters:
        for u in up:
            if cluster in user_to_perms_by_cluster and u in user_to_perms_by_cluster[cluster]:
                users_clustered.add(u)
    users_not_clustered = set(up.keys()).difference(users_clustered)

    for u in users_not_clustered:
        user_to_perms = dict()
        for cluster in clusters:
            # if user u should be assigned to this cluster
            if user_labels[u] == cluster:
                # perms_u = up[u]
                u_key = int(f"{u}")
                user_to_perms[u_key] = up[u]

                # if u_key not in user_to_perms:
                #     user_to_perms[u_key] = perms_u

                if cluster not in user_to_perms_by_cluster:
                    user_to_perms_by_cluster[cluster] = dict()
                if cluster not in usermaps_by_cluster:
                    usermaps_by_cluster[cluster] = dict()
                if cluster not in permmaps_by_cluster:
                    permmaps_by_cluster[cluster] = dict()
                user_to_perms_by_cluster[cluster].update({u1: sorted(list(perms)) for u1, perms in
                                                          user_to_perms.items()})
                usermaps_by_cluster[cluster].update({u1: inv_usermap[u1] for u1 in user_to_perms_by_cluster[cluster]})
                permmaps_by_cluster[cluster].update({p1: inv_permmap[p1] for u1 in user_to_perms_by_cluster[cluster]
                                                     for p1 in up[u1]})
    return user_to_perms_by_cluster, usermaps_by_cluster, permmaps_by_cluster


def print_users_by_cluster(labels_all, user_mask, node_type_to_id, node_names=None):
    user_labels = labels_all[user_mask]
    user_indices = np.where(user_mask)[0]

    clusters = sorted(set(user_labels))

    print("\n=== Users grouped by cluster ===")
    for c in clusters:
        print(f"\nCluster {c}:")
        indices_in_c = user_indices[user_labels == c]
        for idx in indices_in_c:
            if node_names is None:
                print(f"  user_node_{idx}")
            else:
                print(f"  {node_names[idx]}")


def save_checkpoint(path, model, optimizer=None, epoch=None, extra: dict | None = None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_model_for_inference(path, model_ctor, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = model_ctor().to(device)

    # If you saved full checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # If you saved just state_dict
        model.load_state_dict(ckpt)

    model.eval()
    return model, device


# Example:
# save_checkpoint("gae_ckpt.pt", model, optimizer=opt, epoch=epoch, extra={"in_dim": in_dim})


def get_files_in_directory_pathlib(directory_path):
    # Convert the input path string to a Path object
    p = Path(directory_path)

    # 1. Check if the path is an existing directory
    if p.is_dir():
        print(f"Listing files in directory: {p}")
        # 2. Get all files in the directory (non-recursive)
        files_list = [entry for entry in p.iterdir() if entry.is_file()
                      and '-upmap.txt' not in entry.name and not 'em.txt' in entry.name]
        return files_list
    elif p.is_file():
        return [p]
    else:
        print(f"Error: '{directory_path}' is not a valid directory or does not exist.")
        return []


def work(work_args):
    c, user_perm_dict_by_cluster, user_perm_dict_by_cluster_mapped, inv_usermap_orig, inv_permmap_orig, cluster_filename_prefix, args = work_args
    print('Processing cluster: ', c)
    if len(user_perm_dict_by_cluster[c]) > 0:
        user_perm_dict_by_cluster_mapped = dict()
        for u1 in user_perm_dict_by_cluster[c]:
            if inv_usermap_orig[u1] not in user_perm_dict_by_cluster_mapped:
                user_perm_dict_by_cluster_mapped[inv_usermap_orig[u1]] = set()
            for p1 in user_perm_dict_by_cluster[c][u1]:
                user_perm_dict_by_cluster_mapped[inv_usermap_orig[u1]].add(inv_permmap_orig[p1])
        cluster_up, cluster_usermap, cluster_permmap = readup_and_usermap_permmap(
            f'{cluster_filename_prefix}_{c}.txt')
        inv_cluster_usermap = {v: k for k, v in cluster_usermap.items()}
        inv_cluster_permmap = {v: k for k, v in cluster_permmap.items()}

        num_cluster_edges = sum([len(cluster_up[u]) for u in cluster_up])
        print('num_cluster_edges', num_cluster_edges)
        if num_cluster_edges > args.graph_thresh:
            print('--------------------------------------------------------------')
            print(f'Running large bicliques on {cluster_filename_prefix}_{c}.txt')
            print('--------------------------------------------------------------')
            run_largebicliques(upfilename=f'{cluster_filename_prefix}_{c}.txt', bcsize_THRESHOLD=args.bcsize,
                               nbc_THRESHOLD=args.nbc, remove_dominators=args.run_remove_dominators_on_cluster)
            print('--------------------------------------------------------------')
            print(f'Done running large bicliques on {cluster_filename_prefix}_{c}.txt')
            print('--------------------------------------------------------------')
        else:
            print('--------------------------------------------------------------')
            print(f'Running maxsetsbp on remaining edges for {cluster_filename_prefix}_{c}.txt')
            print('--------------------------------------------------------------')
            num_roles, cluster_roles = maxsetsbp.run(upfilename=f'{cluster_filename_prefix}_{c}.txt',
                                      remove_dominators=args.run_remove_dominators_on_cluster)
            print('--------------------------------------------------------------')
            print(f'Done running maxsetsbp on remaining edges for {cluster_filename_prefix}_{c}.txt')
            print('--------------------------------------------------------------')




# ---------------------------
# 5. Main
# ---------------------------

def main(args):
    if args.input_filepath:
        files = get_files_in_directory_pathlib(args.input_filepath)
    elif args.files:
        files = args.files
    else:
        print('No input files provided')
        return
    graphs = [load_up(f)[0] for f in files]
    dims = sorted(set(g.x.size(1) for g in graphs))
    assert dims == [4], f"Inconsistent feature dims: {dims}"

    model_filename = args.model_file

    embeddings_per_graph = train_and_embed_many_graphs(model_filename,
                                                       graphs,
                                                       hidden_channels=256,
                                                       embedding_dim=64,
                                                       epochs=1000,
                                                       lr=3e-4,
                                                       batch_size=1024,
                                                       # num_neighbors=2
                                                       )

    # GET EMBEDDINGS AND RUN CLUSTERING ALGORITHM TO FIND COMMUNITIES
    ctr = 0
    for input_filepath in args.files:
        graph = graphs[ctr]

        up_orig, usermap_orig, permmap_orig = readup_and_usermap_permmap(input_filepath)
        inv_usermap_orig = {v: k for k, v in usermap_orig.items()}
        inv_permmap_orig = {v: k for k, v in permmap_orig.items()}

        em_filepath = input_filepath + '-em.txt'

        if args.run_remove_dominators:
            em = readem(em_filepath)
        else:
            em = dict()
        edges_marked_removed = set()
        for e, m in em.items():
            edges_marked_removed.add(e)

        up = dict()
        inv_usermap = dict()
        inv_permmap = dict()
        for u in up_orig:
            for p in up_orig[u]:
                if (u, p) not in edges_marked_removed:
                    if u not in up:
                        up[u] = set()
                    up[u].add(p)
                    inv_usermap[u] = inv_usermap_orig[u]
                    inv_permmap[p] = inv_permmap_orig[p]
        usermap = {v: k for k, v in inv_usermap.items()}
        permmap = {v: k for k, v in inv_permmap.items()}

        input_filepath_splits = input_filepath.split("/")[-1].split(".")
        if '.' in input_filepath:
            inputfilename = '.'.join(input_filepath_splits)
        else:
            inputfilename = '.'.join(input_filepath_splits[:-1])

        cluster_filename_prefix = f'{inputfilename}_cluster'
        print('Cluster filename prefix: ', cluster_filename_prefix)

        embeddings = embeddings_per_graph[ctr]
        # 1) Train GAE on full (users + movies) graph
        # embeddings = train_gae(
        #     data,
        #     hidden_channels=64,
        #     embedding_dim=32,
        #     epochs=10000,
        #     lr=3e-4,
        # )
        # embeddings = train_gae_large(model_filename,
        #                              data,
        #                              hidden_channels=64,
        #                              embedding_dim=32,
        #                              epochs=10000,
        #                              lr=3e-3,
        #                              batch_size=1000,
        #                              # num_neighbors=2
        #                              )

        # 2) Run k-means on USERS ONLY
        max_num_clusters = int(args.num_clusters)
        threshold = 0.5
        user_perm_dict_by_cluster = dict()
        for num in range(max_num_clusters, max_num_clusters + 1):
            num_clusters = num
            labels_all, user_mask = cluster_users_kmeans(
                embeddings,
                graph.node_types,
                # node_type_to_id,
                num_clusters=num_clusters,
            )
            # print("User clusters:", sorted(set(labels_all[user_mask])))
            # print_users_by_cluster(labels_all, user_mask, node_type_to_id)

            # 3) t-SNE: users colored by cluster, perms
            plt_filename = f'plots/{cluster_filename_prefix}_{num}.png'
            tsne_vis_users_clusters(
                embeddings,
                labels_all,
                user_mask,
                plt_filename=plt_filename
            )

        # return
        user_perm_dict_by_cluster, usermaps_by_cluster, permmaps_by_cluster = build_user_perm_dict(
            graph, up, labels_all, user_mask, usermap, permmap, em)

        num_edges_by_cluster = dict()
        for c in user_perm_dict_by_cluster:
            if c not in num_edges_by_cluster:
                num_edges_by_cluster[c] = 0
            for u1 in user_perm_dict_by_cluster[c]:
                num_edges_by_cluster[c] += len(user_perm_dict_by_cluster[c][u1])

        sorted_num_edges_by_cluster = sorted(num_edges_by_cluster.items(), key=lambda x: x[1])
        print('Sorted num_edges_by_cluster: ', sorted_num_edges_by_cluster)

        start_time = datetime.now()
        # write cluster files
        for c in user_perm_dict_by_cluster:
            if len(user_perm_dict_by_cluster[c]) > 0:
                user_perm_dict_by_cluster_mapped = dict()
                for u1 in user_perm_dict_by_cluster[c]:
                    if inv_usermap_orig[u1] not in user_perm_dict_by_cluster_mapped:
                        user_perm_dict_by_cluster_mapped[inv_usermap_orig[u1]] = set()
                    # print(f'c: {c}, u: {u1}')
                    for p1 in user_perm_dict_by_cluster[c][u1]:
                        user_perm_dict_by_cluster_mapped[inv_usermap_orig[u1]].add(inv_permmap_orig[p1])
                dumpup(user_perm_dict_by_cluster_mapped, f'{cluster_filename_prefix}_{c}.txt', include_prefixes=False)
                # print('Users in clusters:')
                # pprint(user_perm_dict_by_cluster_mapped)

        # read the cluster UP files and run maxsetsbp
        # num_edges = 0
        full_em = dict()
        seq_offset = 0

        seqs = set()
        for e in em:
            full_em[e] = em[e]
            seqs.add(em[e][2])
        max_seq = max(seqs) if len(seqs) > 0 else 0
        seq_offset += max_seq + 1 if max_seq > 0 else 0

#        for c in user_perm_dict_by_cluster:
#            print('Processing cluster: ', c)
#            if len(user_perm_dict_by_cluster[c]) > 0:
#                user_perm_dict_by_cluster_mapped = dict()
#                for u1 in user_perm_dict_by_cluster[c]:
#                    if inv_usermap_orig[u1] not in user_perm_dict_by_cluster_mapped:
#                        user_perm_dict_by_cluster_mapped[inv_usermap_orig[u1]] = set()
#                    for p1 in user_perm_dict_by_cluster[c][u1]:
#                        user_perm_dict_by_cluster_mapped[inv_usermap_orig[u1]].add(inv_permmap_orig[p1])
#                cluster_up, cluster_usermap, cluster_permmap = readup_and_usermap_permmap(
#                    f'{cluster_filename_prefix}_{c}.txt')
#                inv_cluster_usermap = {v: k for k, v in cluster_usermap.items()}
#                inv_cluster_permmap = {v: k for k, v in cluster_permmap.items()}
#
#                num_cluster_edges = sum([len(cluster_up[u]) for u in cluster_up])
#                print('num_cluster_edges', num_cluster_edges)
#                if num_cluster_edges > args.graph_thresh:
#                    print('--------------------------------------------------------------')
#                    print(f'Running large bicliques on {cluster_filename_prefix}_{c}.txt')
#                    print('--------------------------------------------------------------')
#                    run_largebicliques(upfilename=f'{cluster_filename_prefix}_{c}.txt', bcsize_THRESHOLD=args.bcsize,
#                                       nbc_THRESHOLD=args.nbc, remove_dominators=args.run_remove_dominators_on_cluster)

#                    print('--------------------------------------------------------------')
#                    print(f'Done running large bicliques on {cluster_filename_prefix}_{c}.txt')
#                    print('--------------------------------------------------------------')
#                else:
#                    print('--------------------------------------------------------------')
#                    print(f'Running maxsetsbp on remaining edges for {cluster_filename_prefix}_{c}.txt')
#                    print('--------------------------------------------------------------')
#                    num_roles, cluster_roles = maxsetsbp.run(upfilename=f'{cluster_filename_prefix}_{c}.txt',
#                                              remove_dominators=args.run_remove_dominators_on_cluster)
#                    print('--------------------------------------------------------------')
#                    print(f'Done running maxsetsbp on remaining edges for {cluster_filename_prefix}_{c}.txt')
#                    print('--------------------------------------------------------------')


        work_args = [
                (c, user_perm_dict_by_cluster, user_perm_dict_by_cluster_mapped, inv_usermap_orig, inv_permmap_orig, cluster_filename_prefix, args)
                     for c in user_perm_dict_by_cluster.keys()
                     ]

        #with Pool(processes=4) as pool:
        #    pool.map(work, work_args)  # BLOCKS until done

        for c in user_perm_dict_by_cluster:
            work_args =  (c, user_perm_dict_by_cluster, user_perm_dict_by_cluster_mapped, inv_usermap_orig, inv_permmap_orig, cluster_filename_prefix, args)
            work(work_args)
            cluster_up, cluster_usermap, cluster_permmap = readup_and_usermap_permmap(f'{cluster_filename_prefix}_{c}.txt')
            inv_cluster_usermap = {v: k for k, v in cluster_usermap.items()}
            inv_cluster_permmap = {v: k for k, v in cluster_permmap.items()}

            cluster_em = get_em(upfilename=f'{cluster_filename_prefix}_{c}.txt')
            for e, t in cluster_em.items():
                seq = t[2]
                f = (t[0], t[1])
                new_seq = seq + seq_offset
                if e[1] not in inv_cluster_permmap:
                    print(e[1], 'not in inv_cluster_permmap')
                if inv_cluster_permmap[e[1]] not in permmap_orig:
                    print(inv_cluster_permmap[e[1]], 'not in permmap_orig')
                full_e = (usermap_orig[inv_cluster_usermap[e[0]]], permmap_orig[inv_cluster_permmap[e[1]]])
                if f == (-1, -1):
                    full_em[full_e] = (-1, -1, new_seq)
                else:
                    full_em[full_e] = (usermap_orig[inv_cluster_usermap[f[0]]], permmap_orig[inv_cluster_permmap[f[1]]], new_seq)
                max_seq = max(max_seq, new_seq)
        seq_offset += max_seq + 1
        saveem(full_em, filename=f'{input_filepath}-full-em.txt')

        print(Fore.RED + 'Full em size: ', len(full_em))
        print(Style.RESET_ALL)

        # Create roles as permissions
        rolesasperms = list()
        dm = dmfromem(full_em)
        G = nx.Graph()
        for e in dm:
            if e == tuple((-1, -1)):
                for f in dm[e]:
                    G.add_node(f)
            else:
                for f in dm[e]:
                    G.add_edge(e, f)

        # One role per connected component in G
        for c in nx.connected_components(G):
            r = set()  # our role as a set of permissions
            for t in c:
                r.add(t[1])
            rolesasperms.append(r)

        print('done! len(rolesasperms):', len(rolesasperms))
        sys.stdout.flush()

        print('--------------------------------------------------------------')
        print(f'Running lattice shrink')
        print('--------------------------------------------------------------')
        greedythenlattice.latticeshrink(rolesasperms)
        print('--------------------------------------------------------------')
        print(f'Done running lattice shrink')
        print('--------------------------------------------------------------')

        roles_after_lattice = []
        for role in rolesasperms:
            role_as_edges = set()
            for u in up_orig:
                perm_intersect = role & up_orig[u]
                for p in perm_intersect:
                    role_as_edges.add((u, p))
            roles_after_lattice.append(role_as_edges)


        print('Save roles after lattice shrink')
        serializable_roles = [
            [list(e) for e in r]  # set → list, tuple → list
            for r in roles_after_lattice
        ]
        # print(roles_after_lattice)
        with open(f"{input_filepath}-roles.txt", "w") as f:
            json.dump(serializable_roles, f)

        edges_in_total_roles = set()
        num_edges = 0
        for role_edges in roles_after_lattice:
            edges_in_total_roles.update(role_edges)
            num_edges += len(role_edges)

        print(f'All roles: {len(roles_after_lattice)}')

        up_edges = {(u, p) for u in up for p in up_orig[u]}

        missing_edges = up_edges.difference(edges_in_total_roles)
        print('Missing edges:', len(missing_edges))

        print(f'Total # roles: {len(roles_after_lattice)}')
        print(f'Total # edges in roles: {num_edges}')
        end_time = datetime.now()
        delta_time = end_time - start_time

        print(f'Total time taken: {str(timedelta(seconds=delta_time.total_seconds()))}')
        #continue
        # delete em and upmap files
        print('Delete em files:', args.delete_em_files)
        if args.delete_em_files:
            for c in user_perm_dict_by_cluster:
                cluster_filepath = f'{cluster_filename_prefix}_{c}.txt'
                cluster_em_filepath = f'{cluster_filepath}-em.txt'
                cluster_upmap_filepath = f'{cluster_filepath}-upmap.txt'
                if os.path.exists(cluster_filepath):
                    os.remove(cluster_filepath)
                if os.path.exists(cluster_em_filepath):
                    os.remove(cluster_em_filepath)
                if os.path.exists(cluster_upmap_filepath):
                    os.remove(cluster_upmap_filepath)

        ctr += 1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect Communities in UP files')
    # parser.add_argument('input_file', help='Path to the input UP file')
    parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        required=False,
        help="Input UP file paths"
    )
    parser.add_argument(
        "--input_filepath",
        required=False,
        help="Input UP file paths"
    )
    parser.add_argument('--model_file', type=str, required=True, help='Path to the model file')
    parser.add_argument('--num_clusters', type=int, default=1, required=True, help='Number of clusters')
    parser.add_argument('--nbc', type=int, default=200, required=True, help='Number of large bicliques')
    parser.add_argument('--bcsize', type=int, default=200, required=True, help='Size threshold for large bicliques')
    parser.add_argument('--graph_thresh', type=int, default=5000, required=True,
                        help='minimum # of edges needed in the graph to run large bicliques')
    parser.add_argument('--run_remove_dominators', type=str2bool, default=True, required=False,
                        help='Remove dominators before clustering?')
    parser.add_argument('--run_remove_dominators_on_cluster', type=str2bool, default=True, required=False,
                        help='Remove dominators before clustering?')
    parser.add_argument('--delete_em_files', type=str2bool, default=True, required=False,
                        help='Remove em files after clustering?')


    args = parser.parse_args()
    main(args)
