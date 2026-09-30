# -*- coding: utf-8 -*-
"""
Graph Neural Network for predicting a scalar from a transition matrix.

Input .npz file must contain:
    matrixlist
    Exponential_Decay_parameters

Each transition matrix is treated as a directed weighted graph:
    T[i,j] = edge weight from node i -> node j

Node features are all equal to 1.

Uses PyTorch Geometric NNConv so that the edge weights themselves
are passed through a learned edge network.

Usage:
    python transition_gnn.py -i data.npz

Example:
    python transition_gnn.py -i SaveFiles/data.npz --epochs 500
"""

import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="GNN mapping transition matrices to a scalar"
)

parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="Input .npz file containing matrixlist and Exponential_Decay_parameters"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    help="Number of training epochs"
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size"
)

parser.add_argument(
    "--hidden-dim",
    type=int,
    default=64,
    help="Hidden dimension of the GNN"
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-3,
    help="Learning rate"
)

parser.add_argument(
    "--test-fraction",
    type=float,
    default=0.15,
    help="Fraction of data used for testing"
)

parser.add_argument(
    "--validation-fraction",
    type=float,
    default=0.15,
    help="Fraction of data used for validation"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)

parser.add_argument(
    "--model-output",
    default="gnn_transition_model.pt",
    help="Output filename for trained model"
)

args = parser.parse_args()


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(args.input, allow_pickle=True)

matrixlist = data["matrixlist"]
targets = np.asarray(
    data["Exponential_Decay_parameters"],
    dtype=np.float32
)

print()
print("Loaded data")
print("----------------------------")
print(f"Number of matrices : {len(matrixlist)}")
print(f"Number of targets  : {len(targets)}")
print(f"Target mean        : {targets.mean():.6g}")
print(f"Target std         : {targets.std():.6g}")
print(f"Target min         : {targets.min():.6g}")
print(f"Target max         : {targets.max():.6g}")


if len(matrixlist) != len(targets):
    raise ValueError(
        "matrixlist and Exponential_Decay_parameters "
        "must have the same length."
    )


# ============================================================
# CONVERT MATRIX TO PYTORCH GEOMETRIC GRAPH
# ============================================================

def matrix_to_graph(matrix, target):
    """
    Convert an n x n transition matrix into a PyG graph.

    Nodes:
        One node for every row/column of the matrix.

    Node features:
        All equal to 1.

    Directed edges:
        i -> j whenever matrix[i,j] != 0.

    Edge features:
        matrix[i,j].

    Target:
        Scalar corresponding to this matrix.
    """

    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D matrix, got shape {matrix.shape}"
        )

    n = matrix.shape[0]

    if matrix.shape[1] != n:
        raise ValueError(
            f"Transition matrix must be square, got {matrix.shape}"
        )

    # --------------------------------------------------------
    # Node features
    # --------------------------------------------------------
    #
    # Every node starts with exactly the same feature:
    #
    #       x_i = 1
    #
    # We deliberately do NOT give nodes their indices.
    #
    x = torch.ones(
        (n, 1),
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Directed edges
    # --------------------------------------------------------
    #
    # matrix[i,j] represents:
    #
    #       i -> j
    #
    rows, cols = np.nonzero(matrix)

    edge_index = torch.tensor(
        np.vstack((rows, cols)),
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Edge attributes
    # --------------------------------------------------------
    #
    # Each edge receives its transition probability as an
    # edge feature.
    #
    edge_attr = torch.tensor(
        matrix[rows, cols].reshape(-1, 1),
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(
            [target],
            dtype=torch.float32
        )
    )

    return graph


# ============================================================
# CREATE GRAPHS
# ============================================================

graphs = []

for matrix, target in zip(matrixlist, targets):

    graph = matrix_to_graph(
        matrix,
        target
    )

    graphs.append(graph)


print()
print("Graph information")
print("----------------------------")

for i in range(min(3, len(graphs))):

    graph = graphs[i]

    print(
        f"Graph {i}: "
        f"{graph.num_nodes} nodes, "
        f"{graph.num_edges} edges"
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

num_graphs = len(graphs)

indices = np.arange(num_graphs)

rng = np.random.default_rng(args.seed)
rng.shuffle(indices)

n_test = int(
    args.test_fraction * num_graphs
)

n_validation = int(
    args.validation_fraction * num_graphs
)

test_indices = indices[:n_test]

validation_indices = indices[
    n_test:n_test + n_validation
]

train_indices = indices[
    n_test + n_validation:
]


train_graphs = [
    graphs[i]
    for i in train_indices
]

validation_graphs = [
    graphs[i]
    for i in validation_indices
]

test_graphs = [
    graphs[i]
    for i in test_indices
]


print()
print("Dataset split")
print("----------------------------")
print(f"Training   : {len(train_graphs)}")
print(f"Validation : {len(validation_graphs)}")
print(f"Testing    : {len(test_graphs)}")


# ============================================================
# TARGET NORMALISATION
# ============================================================

train_targets = np.asarray(
    [
        graphs[i].y.item()
        for i in train_indices
    ],
    dtype=np.float32
)

target_mean = float(
    train_targets.mean()
)

target_std = float(
    train_targets.std()
)

# Avoid division by zero if all training targets happen
# to be identical.
if target_std < 1e-12:
    target_std = 1.0


def normalise_target(y):
    return (
        y - target_mean
    ) / target_std


def denormalise_target(y):
    return (
        y * target_std
    ) + target_mean


for graph in graphs:

    graph.y = (
        graph.y - target_mean
    ) / target_std


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_graphs,
    batch_size=args.batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    validation_graphs,
    batch_size=args.batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_graphs,
    batch_size=args.batch_size,
    shuffle=False
)


# ============================================================
# EDGE NETWORK
# ============================================================

class EdgeNetwork(nn.Module):
    """
    Maps a scalar transition probability T_ij to a
    matrix used by NNConv.

    Input:
        edge_attr = T_ij

    Output:
        flattened [in_channels x out_channels] matrix.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                1,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                in_channels * out_channels
            )
        )

    def forward(self, edge_attr):

        return self.network(edge_attr)


# ============================================================
# GNN MODEL
# ============================================================

class TransitionGNN(nn.Module):

    def __init__(
        self,
        hidden_dim=64
    ):

        super().__init__()

        # ----------------------------------------------------
        # First edge-conditioned convolution
        # ----------------------------------------------------

        edge_network_1 = EdgeNetwork(
            in_channels=1,
            out_channels=hidden_dim,
            hidden_dim=hidden_dim
        )

        self.conv1 = NNConv(
            in_channels=1,
            out_channels=hidden_dim,
            nn=edge_network_1,
            aggr="mean"
        )

        # ----------------------------------------------------
        # Second edge-conditioned convolution
        # ----------------------------------------------------

        edge_network_2 = EdgeNetwork(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            hidden_dim=hidden_dim
        )

        self.conv2 = NNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            nn=edge_network_2,
            aggr="mean"
        )

        # ----------------------------------------------------
        # Third edge-conditioned convolution
        # ----------------------------------------------------

        edge_network_3 = EdgeNetwork(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            hidden_dim=hidden_dim
        )

        self.conv3 = NNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            nn=edge_network_3,
            aggr="mean"
        )

        # ----------------------------------------------------
        # Graph-level predictor
        # ----------------------------------------------------

        self.predictor = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim // 2,
                1
            )
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch
    ):

        # ----------------------------------------------------
        # Message passing
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edge_index,
            edge_attr
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index,
            edge_attr
        )

        x = F.relu(x)

        x = self.conv3(
            x,
            edge_index,
            edge_attr
        )

        x = F.relu(x)

        # ----------------------------------------------------
        # Convert node representation to graph representation
        # ----------------------------------------------------

        x = global_mean_pool(
            x,
            batch
        )

        # ----------------------------------------------------
        # Predict scalar
        # ----------------------------------------------------

        x = self.predictor(x)

        return x.squeeze(-1)


# ============================================================
# CREATE MODEL
# ============================================================

model = TransitionGNN(
    hidden_dim=args.hidden_dim
).to(device)

print()
print("Model")
print("----------------------------")
print(model)


# ============================================================
# OPTIMIZER / LOSS
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.learning_rate
)

criterion = nn.MSELoss()


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_epoch():

    model.train()

    total_loss = 0.0
    total_graphs = 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        prediction = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        loss = criterion(
            prediction,
            batch.y.view(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * batch.num_graphs
        )

        total_graphs += batch.num_graphs

    return total_loss / total_graphs


# ============================================================
# EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate(loader):

    model.eval()

    total_loss = 0.0
    total_graphs = 0

    predictions = []
    actual = []

    for batch in loader:

        batch = batch.to(device)

        prediction = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        loss = criterion(
            prediction,
            batch.y.view(-1)
        )

        total_loss += (
            loss.item()
            * batch.num_graphs
        )

        total_graphs += batch.num_graphs

        predictions.append(
            prediction.cpu().numpy()
        )

        actual.append(
            batch.y.view(-1).cpu().numpy()
        )

    if total_graphs == 0:

        return (
            np.nan,
            np.array([]),
            np.array([])
        )

    predictions = np.concatenate(
        predictions
    )

    actual = np.concatenate(
        actual
    )

    return (
        total_loss / total_graphs,
        predictions,
        actual
    )


# ============================================================
# TRAIN
# ============================================================

best_validation_loss = np.inf
best_state = None

print()
print("Training")
print("----------------------------")

for epoch in range(1, args.epochs + 1):

    train_loss = train_epoch()

    validation_loss, _, _ = evaluate(
        validation_loader
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % 10 == 0
        or epoch == args.epochs
    ):

        print(
            f"Epoch {epoch:4d} | "
            f"Train loss: {train_loss:.6f} | "
            f"Validation loss: {validation_loss:.6f}"
        )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# TEST
# ============================================================

test_loss, test_predictions, test_actual = evaluate(
    test_loader
)


# Convert back to original target units

test_predictions_original = denormalise_target(
    test_predictions
)

test_actual_original = denormalise_target(
    test_actual
)


# ============================================================
# TEST METRICS
# ============================================================

if len(test_predictions_original) > 0:

    mse = np.mean(
        (
            test_predictions_original
            - test_actual_original
        ) ** 2
    )

    rmse = np.sqrt(mse)

    mae = np.mean(
        np.abs(
            test_predictions_original
            - test_actual_original
        )
    )

    correlation = np.corrcoef(
        test_predictions_original,
        test_actual_original
    )[0, 1]

else:

    mse = np.nan
    rmse = np.nan
    mae = np.nan
    correlation = np.nan


print()
print("Test results")
print("----------------------------")
print(f"Normalised MSE : {test_loss:.6g}")
print(f"MSE            : {mse:.6g}")
print(f"RMSE           : {rmse:.6g}")
print(f"MAE            : {mae:.6g}")
print(f"Correlation    : {correlation:.6g}")


# ============================================================
# PREDICTION DIAGNOSTIC
# ============================================================

if len(test_predictions_original) > 0:

    print()
    print("Prediction diagnostic")
    print("----------------------------")

    print(
        f"Actual mean       : "
        f"{np.mean(test_actual_original):.6g}"
    )

    print(
        f"Prediction mean   : "
        f"{np.mean(test_predictions_original):.6g}"
    )

    print(
        f"Actual std        : "
        f"{np.std(test_actual_original):.6g}"
    )

    print(
        f"Prediction std    : "
        f"{np.std(test_predictions_original):.6g}"
    )

    print()
    print("Predictions")
    print("----------------------------")

    for actual, prediction in zip(
        test_actual_original,
        test_predictions_original
    ):

        print(
            f"Actual = {actual:.6g}    "
            f"Predicted = {prediction:.6g}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "hidden_dim": args.hidden_dim,

        "target_mean": target_mean,

        "target_std": target_std,

        "model_type": "NNConv",

        "node_features": "ones",

        "edge_features": "transition_probability",

    },
    args.model_output
)


print()
print(
    f"Model saved to: {args.model_output}"
)
