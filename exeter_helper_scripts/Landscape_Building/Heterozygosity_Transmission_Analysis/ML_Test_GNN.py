
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Predict heterozygosity decay using a graph neural network"
)

parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="Path to input .npz file"
)

parser.add_argument(
    "--log-target",
    action="store_true",
    help="Train on log-transformed decay parameters"
)

args = parser.parse_args()


# ============================================================
# RANDOM SEEDS
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# LOAD DATA
# ============================================================

data = np.load(
    args.input,
    allow_pickle=True
)

matrixlist = data["matrixlist"]

decay_parameters = data[
    "Exponential_Decay_parameters"
]


print("=" * 60)
print("DATA")
print("=" * 60)

print(
    "Number of matrices:",
    len(matrixlist)
)

print(
    "Number of targets:",
    len(decay_parameters)
)


X = np.asarray(
    matrixlist,
    dtype=np.float32
)

y = np.asarray(
    decay_parameters,
    dtype=np.float32
)


print()
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# CHECK DATA
# ============================================================

if X.ndim != 3:

    raise ValueError(
        "Expected matrixlist to have shape "
        "(N, N_nodes, N_nodes)"
    )


if X.shape[1] != X.shape[2]:

    raise ValueError(
        "Transition matrices must be square."
    )


if len(X) != len(y):

    raise ValueError(
        "Number of matrices and targets do not match."
    )


n_samples = X.shape[0]
n_nodes = X.shape[1]


print(
    "Number of nodes:",
    n_nodes
)


# ============================================================
# TARGET TRANSFORMATION
# ============================================================

if args.log_target:

    if np.any(y <= 0):

        raise ValueError(
            "Cannot log-transform target because "
            "some decay parameters are <= 0."
        )

    y_model = np.log(y)

    print()
    print("Using log(target).")

else:

    y_model = y.copy()

    print()
    print("Using raw target.")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_model,
    test_size=0.2,
    random_state=42
)


print()
print(
    "Training landscapes:",
    len(X_train)
)

print(
    "Test landscapes:",
    len(X_test)
)


# ============================================================
# TARGET STANDARDISATION
#
# IMPORTANT:
# Mean and standard deviation are calculated only
# from the training set.
# ============================================================

y_mean = y_train.mean()
y_std = y_train.std()


if y_std == 0:

    raise ValueError(
        "Target has zero standard deviation."
    )


y_train_scaled = (
    (y_train - y_mean)
    / y_std
)

y_test_scaled = (
    (y_test - y_mean)
    / y_std
)


print()
print("Model target mean:", y_mean)
print("Model target std: ", y_std)


# ============================================================
# DATASET
# ============================================================

class TransitionMatrixDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        matrices,
        targets
    ):

        self.matrices = matrices
        self.targets = targets


    def __len__(self):

        return len(self.matrices)


    def __getitem__(self, index):

        T = self.matrices[index]

        target = self.targets[index]


        T = torch.tensor(
            T,
            dtype=torch.float32
        )


        target = torch.tensor(
            target,
            dtype=torch.float32
        )


        return T, target


train_dataset = TransitionMatrixDataset(
    X_train,
    y_train_scaled
)

test_dataset = TransitionMatrixDataset(
    X_test,
    y_test_scaled
)


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# GRAPH NEURAL NETWORK LAYER
# ============================================================

class GraphLayer(nn.Module):
    """
    One message-passing layer.

    T[i,j] is interpreted as the probability of
    transitioning from node i to node j.

    Therefore the incoming message to node j is

        sum_i T[i,j] h_i

    which is calculated as

        T.T @ H

    for each graph.

    The same operation is performed for every node,
    so the network is permutation equivariant.
    """

    def __init__(
        self,
        input_features,
        output_features
    ):

        super().__init__()


        self.message_network = nn.Sequential(

            nn.Linear(
                input_features,
                output_features
            ),

            nn.ReLU(),

            nn.Linear(
                output_features,
                output_features
            )
        )


        self.update_network = nn.Sequential(

            nn.Linear(
                input_features + output_features,
                output_features
            ),

            nn.ReLU(),

            nn.Linear(
                output_features,
                output_features
            )
        )


    def forward(
        self,
        H,
        T
    ):
        """
        H:
            [batch, nodes, features]

        T:
            [batch, nodes, nodes]
        """


        # Transform node features before
        # sending messages.

        messages = self.message_network(H)


        # T.T @ messages
        #
        # For every destination node j:
        #
        # message_j = sum_i T[i,j] message_i

        aggregated = torch.bmm(
            T.transpose(1, 2),
            messages
        )


        # Combine the original node state
        # with its aggregated neighbourhood.

        combined = torch.cat(
            [
                H,
                aggregated
            ],
            dim=2
        )


        H_new = self.update_network(
            combined
        )


        return H_new


# ============================================================
# GRAPH NEURAL NETWORK
# ============================================================

class TransitionMatrixGNN(nn.Module):

    def __init__(
        self,
        n_nodes,
        hidden_features=64
    ):

        super().__init__()


        self.n_nodes = n_nodes


        # ----------------------------------------------------
        # Initial node features
        #
        # We give each node simple features derived from T.
        #
        # row_sum:
        #   total probability leaving node
        #
        # column_sum:
        #   total probability entering node
        #
        # diagonal:
        #   probability of remaining at node
        #
        # These are all permutation-equivariant quantities.
        # ----------------------------------------------------

        self.input_features = 3


        self.layer1 = GraphLayer(
            self.input_features,
            hidden_features
        )


        self.layer2 = GraphLayer(
            hidden_features,
            hidden_features
        )


        self.layer3 = GraphLayer(
            hidden_features,
            hidden_features
        )


        # ----------------------------------------------------
        # Graph-level prediction
        # ----------------------------------------------------

        self.output_network = nn.Sequential(

            nn.Linear(
                hidden_features,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                1
            )
        )


    def forward(self, T):

        # ----------------------------------------------------
        # Construct node features
        # ----------------------------------------------------

        row_sum = T.sum(
            dim=2,
            keepdim=True
        )


        column_sum = T.sum(dim=1).unsqueeze(-1)

        diagonal = torch.diagonal(
            T,
            dim1=1,
            dim2=2
        ).unsqueeze(-1)


        H = torch.cat(
            [
                row_sum,
                column_sum,
                diagonal
            ],
            dim=2
        )


        # ----------------------------------------------------
        # Message passing
        # ----------------------------------------------------

        H = self.layer1(
            H,
            T
        )


        H = self.layer2(
            H,
            T
        )


        H = self.layer3(
            H,
            T
        )


        # ----------------------------------------------------
        # Global pooling
        #
        # Mean pooling makes the representation independent
        # of node ordering.
        # ----------------------------------------------------

        graph_embedding = H.mean(
            dim=1
        )


        # ----------------------------------------------------
        # Predict graph-level quantity
        # ----------------------------------------------------

        output = self.output_network(
            graph_embedding
        )


        return output


# ============================================================
# CREATE MODEL
# ============================================================

model = TransitionMatrixGNN(
    n_nodes=n_nodes,
    hidden_features=64
)


print()
print("=" * 60)
print("MODEL")
print("=" * 60)

print(model)


# ============================================================
# LOSS / OPTIMISER
# ============================================================

loss_function = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    loader,
    y_mean,
    y_std,
    log_target
):

    model.eval()


    predictions_scaled = []
    actual_scaled = []


    with torch.no_grad():

        for X_batch, y_batch in loader:

            prediction = model(
                X_batch
            ).squeeze(-1)


            predictions_scaled.extend(
                prediction.cpu().numpy()
            )


            actual_scaled.extend(
                y_batch.cpu().numpy()
            )


    predictions_scaled = np.asarray(
        predictions_scaled
    )

    actual_scaled = np.asarray(
        actual_scaled
    )


    # Undo standardisation

    predictions_model = (
        predictions_scaled * y_std
        + y_mean
    )


    actual_model = (
        actual_scaled * y_std
        + y_mean
    )


    # Undo log transformation

    if log_target:

        predictions = np.exp(
            predictions_model
        )

        actual = np.exp(
            actual_model
        )

    else:

        predictions = predictions_model
        actual = actual_model


    r2 = r2_score(
        actual,
        predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )


    mae = mean_absolute_error(
        actual,
        predictions
    )


    return (
        r2,
        rmse,
        mae,
        actual,
        predictions
    )


# ============================================================
# MEAN BASELINE
# ============================================================

if args.log_target:

    baseline_prediction = np.exp(
        y_mean
    )

else:

    baseline_prediction = y_mean


mean_prediction = np.full_like(
    y_test,
    baseline_prediction,
    dtype=np.float32
)


# y_test is currently in model space if
# log transformation was used, so convert
# it back to original space.

if args.log_target:

    y_test_original = np.exp(y_test)

else:

    y_test_original = y_test


baseline_r2 = r2_score(
    y_test_original,
    mean_prediction
)


baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test_original,
        mean_prediction
    )
)


baseline_mae = mean_absolute_error(
    y_test_original,
    mean_prediction
)


print()
print("=" * 60)
print("MEAN BASELINE")
print("=" * 60)

print(
    f"R�   = {baseline_r2:.6f}"
)

print(
    f"RMSE = {baseline_rmse:.6f}"
)

print(
    f"MAE  = {baseline_mae:.6f}"
)


# ============================================================
# TRAINING
# ============================================================

n_epochs = 2000

print_every = 100


epochs_history = []

train_loss_history = []

train_r2_history = []
train_rmse_history = []
train_mae_history = []

test_r2_history = []
test_rmse_history = []
test_mae_history = []


print()
print("=" * 60)
print("TRAINING")
print("=" * 60)
print()


for epoch in range(
    1,
    n_epochs + 1
):


    model.train()


    total_loss = 0.0


    for X_batch, y_batch in train_loader:


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        prediction = model(
            X_batch
        ).squeeze(-1)


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = loss_function(
            prediction,
            y_batch
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
            * len(X_batch)
        )


    average_loss = (
        total_loss
        / len(train_dataset)
    )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if (
        epoch == 1
        or epoch % print_every == 0
    ):


        (
            train_r2,
            train_rmse,
            train_mae,
            _,
            _
        ) = evaluate_model(
            model,
            train_loader,
            y_mean,
            y_std,
            args.log_target
        )


        (
            test_r2,
            test_rmse,
            test_mae,
            _,
            _
        ) = evaluate_model(
            model,
            test_loader,
            y_mean,
            y_std,
            args.log_target
        )


        epochs_history.append(
            epoch
        )


        train_loss_history.append(
            average_loss
        )


        train_r2_history.append(
            train_r2
        )

        train_rmse_history.append(
            train_rmse
        )

        train_mae_history.append(
            train_mae
        )


        test_r2_history.append(
            test_r2
        )

        test_rmse_history.append(
            test_rmse
        )

        test_mae_history.append(
            test_mae
        )


        print(
            f"Epoch {epoch:4d} | "
            f"Loss {average_loss:10.5f} | "
            f"Train R� {train_r2:7.4f} | "
            f"Test R� {test_r2:7.4f} | "
            f"Test RMSE {test_rmse:10.3f}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

(
    final_r2,
    final_rmse,
    final_mae,
    actual_test,
    final_predictions
) = evaluate_model(
    model,
    test_loader,
    y_mean,
    y_std,
    args.log_target
)


print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"R�   = {final_r2:.6f}"
)

print(
    f"RMSE = {final_rmse:.6f}"
)

print(
    f"MAE  = {final_mae:.6f}"
)


# ============================================================
# PREDICTION STATISTICS
# ============================================================

print()
print("=" * 60)
print("PREDICTION STATISTICS")
print("=" * 60)


print("Actual:")

print(
    "  Mean:",
    actual_test.mean()
)

print(
    "  Std: ",
    actual_test.std()
)

print(
    "  Min: ",
    actual_test.min()
)

print(
    "  Max: ",
    actual_test.max()
)


print()

print("Predicted:")

print(
    "  Mean:",
    final_predictions.mean()
)

print(
    "  Std: ",
    final_predictions.std()
)

print(
    "  Min: ",
    final_predictions.min()
)

print(
    "  Max: ",
    final_predictions.max()
)


# ============================================================
# EXAMPLE PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("EXAMPLE PREDICTIONS")
print("=" * 60)


print(
    f"{'Actual':>15} "
    f"{'Predicted':>15} "
    f"{'Error':>15}"
)

print("-" * 50)


for actual_value, predicted_value in zip(
    actual_test[:10],
    final_predictions[:10]
):

    error = (
        predicted_value
        - actual_value
    )


    print(
        f"{actual_value:15.3f} "
        f"{predicted_value:15.3f} "
        f"{error:15.3f}"
    )


# ============================================================
# LOW / HIGH TARGET DIAGNOSTIC
# ============================================================

i_low = np.argmin(
    actual_test
)

i_high = np.argmax(
    actual_test
)


print()
print("=" * 60)
print("LOW / HIGH TARGET DIAGNOSTIC")
print("=" * 60)


print()
print("LOW TARGET")

print(
    "Actual:    ",
    actual_test[i_low]
)

print(
    "Predicted: ",
    final_predictions[i_low]
)


print()
print("HIGH TARGET")

print(
    "Actual:    ",
    actual_test[i_high]
)

print(
    "Predicted: ",
    final_predictions[i_high]
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure()


plt.scatter(
    actual_test,
    final_predictions
)


minimum = min(
    actual_test.min(),
    final_predictions.min()
)

maximum = max(
    actual_test.max(),
    final_predictions.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--"
)


plt.xlabel(
    "Actual exponential decay parameter"
)

plt.ylabel(
    "Predicted exponential decay parameter"
)


plt.title(
    f"GNN prediction "
    f"(Test R� = {final_r2:.3f})"
)


plt.tight_layout()

plt.show()


# ============================================================
# R� OVER TRAINING
# ============================================================

plt.figure()


plt.plot(
    epochs_history,
    train_r2_history,
    label="Training R�"
)


plt.plot(
    epochs_history,
    test_r2_history,
    label="Test R�"
)


plt.axhline(
    0,
    linestyle="--"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "R�"
)


plt.title(
    "GNN accuracy over training"
)


plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# RMSE OVER TRAINING
# ============================================================

plt.figure()


plt.plot(
    epochs_history,
    train_rmse_history,
    label="Training RMSE"
)


plt.plot(
    epochs_history,
    test_rmse_history,
    label="Test RMSE"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "RMSE"
)


plt.title(
    "GNN RMSE over training"
)


plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# TRAINING LOSS
# ============================================================

plt.figure()


plt.plot(
    epochs_history,
    train_loss_history
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE loss"
)


plt.title(
    "GNN training loss"
)


plt.tight_layout()

plt.show()


# ============================================================
# PERMUTATION INVARIANCE TEST
# ============================================================

print()
print("=" * 60)
print("PERMUTATION INVARIANCE TEST")
print("=" * 60)


# Pick one test landscape

T_original = X_test[0]

true_value = (
    np.exp(y_test[0])
    if args.log_target
    else y_test[0]
)


n_permutations = 100


permuted_matrices = []


for _ in range(
    n_permutations
):


    order = np.random.permutation(
        n_nodes
    )


    T_permuted = T_original[
        np.ix_(
            order,
            order
        )
    ]


    permuted_matrices.append(
        T_permuted
    )


permuted_matrices = np.asarray(
    permuted_matrices,
    dtype=np.float32
)


permuted_tensors = torch.tensor(
    permuted_matrices,
    dtype=torch.float32
)


model.eval()


with torch.no_grad():

    permutation_predictions_model = (
        model(
            permuted_tensors
        )
        .squeeze(-1)
        .cpu()
        .numpy()
    )


# Undo standardisation

permutation_predictions_model = (
    permutation_predictions_model
    * y_std
    + y_mean
)


# Undo log transformation

if args.log_target:

    permutation_predictions = np.exp(
        permutation_predictions_model
    )

else:

    permutation_predictions = (
        permutation_predictions_model
    )


print(
    "True decay parameter:",
    true_value
)

print(
    "Mean prediction:",
    permutation_predictions.mean()
)

print(
    "Std across permutations:",
    permutation_predictions.std()
)

print(
    "Minimum prediction:",
    permutation_predictions.min()
)

print(
    "Maximum prediction:",
    permutation_predictions.max()
)
