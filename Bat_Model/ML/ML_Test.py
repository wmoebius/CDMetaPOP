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
    description="Predict heterozygosity decay from transition matrices"
)

parser.add_argument(
    "-i",
    "--input",
    required=True,
    help="Path to input .npz file"
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

data = np.load(args.input, allow_pickle=True)

matrixlist = data["matrixlist"]
decay_parameters = data["Exponential_Decay_parameters"]

print("=" * 60)
print("DATA")
print("=" * 60)

print("Number of matrices:", len(matrixlist))
print("Number of targets:", len(decay_parameters))


X = np.asarray(matrixlist, dtype=np.float32)
y = np.asarray(decay_parameters, dtype=np.float32)

print()
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# CHECK DATA
# ============================================================

if X.ndim != 3:
    raise ValueError(
        "Expected matrixlist to have shape (N, N_nodes, N_nodes)"
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

print("Number of nodes:", n_nodes)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print()
print("Training landscapes:", len(X_train))
print("Test landscapes:", len(X_test))


# ============================================================
# TARGET STANDARDISATION
# ============================================================

y_mean = y_train.mean()
y_std = y_train.std()

if y_std == 0:
    raise ValueError(
        "Target has zero standard deviation."
    )

y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

print()
print("Target mean:", y_mean)
print("Target std:", y_std)


# ============================================================
# DATASET
# ============================================================

class TransitionMatrixDataset(torch.utils.data.Dataset):

    def __init__(self, matrices, targets):

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
# MODEL
# ============================================================

class TransitionMatrixModel(nn.Module):

    def __init__(self, n_nodes):

        super().__init__()

        n_inputs = n_nodes * n_nodes

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                n_inputs,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
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

    def forward(self, x):

        return self.network(x)


model = TransitionMatrixModel(n_nodes)


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
    lr=0.001
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    loader,
    y_mean,
    y_std
):

    model.eval()

    predictions_scaled = []
    actual_scaled = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            prediction = model(
                X_batch
            ).squeeze()

            predictions_scaled.extend(
                prediction.numpy()
            )

            actual_scaled.extend(
                y_batch.numpy()
            )

    predictions_scaled = np.asarray(
        predictions_scaled
    )

    actual_scaled = np.asarray(
        actual_scaled
    )

    # Convert back to original units

    predictions = (
        predictions_scaled * y_std
        + y_mean
    )

    actual = (
        actual_scaled * y_std
        + y_mean
    )

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

    return r2, rmse, mae


# ============================================================
# MEAN BASELINE
# ============================================================

mean_prediction = np.full_like(
    y_test,
    y_train.mean()
)

baseline_r2 = r2_score(
    y_test,
    mean_prediction
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        mean_prediction
    )
)

baseline_mae = mean_absolute_error(
    y_test,
    mean_prediction
)


print()
print("=" * 60)
print("MEAN BASELINE")
print("=" * 60)

print(f"R²   = {baseline_r2:.6f}")
print(f"RMSE = {baseline_rmse:.6f}")
print(f"MAE  = {baseline_mae:.6f}")


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


for epoch in range(1, n_epochs + 1):

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        prediction = model(
            X_batch
        ).squeeze()

        loss = loss_function(
            prediction,
            y_batch
        )

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
    # EVALUATE
    # --------------------------------------------------------

    if epoch == 1 or epoch % print_every == 0:

        train_r2, train_rmse, train_mae = evaluate_model(
            model,
            train_loader,
            y_mean,
            y_std
        )

        test_r2, test_rmse, test_mae = evaluate_model(
            model,
            test_loader,
            y_mean,
            y_std
        )


        epochs_history.append(epoch)

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
            f"Train R² {train_r2:7.4f} | "
            f"Test R² {test_r2:7.4f} | "
            f"Test RMSE {test_rmse:10.3f}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

final_r2 = test_r2_history[-1]
final_rmse = test_rmse_history[-1]
final_mae = test_mae_history[-1]


print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"R²   = {final_r2:.6f}")
print(f"RMSE = {final_rmse:.6f}")
print(f"MAE  = {final_mae:.6f}")


# ============================================================
# FINAL PREDICTIONS
# ============================================================

model.eval()

with torch.no_grad():

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    final_predictions_scaled = (
        model(X_test_tensor)
        .squeeze()
        .numpy()
    )


final_predictions = (
    final_predictions_scaled
    * y_std
    + y_mean
)


# ============================================================
# PREDICTION STATISTICS
# ============================================================

print()
print("=" * 60)
print("PREDICTION STATISTICS")
print("=" * 60)

print("Actual:")
print("  Mean:", y_test.mean())
print("  Std: ", y_test.std())
print("  Min: ", y_test.min())
print("  Max: ", y_test.max())

print()
print("Predicted:")
print("  Mean:", final_predictions.mean())
print("  Std: ", final_predictions.std())
print("  Min: ", final_predictions.min())
print("  Max: ", final_predictions.max())


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
    y_test[:10],
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
# LOWEST / HIGHEST TARGET DIAGNOSTIC
# ============================================================

i_low = np.argmin(y_test)
i_high = np.argmax(y_test)


print()
print("=" * 60)
print("LOW / HIGH TARGET DIAGNOSTIC")
print("=" * 60)


print()
print("LOW TARGET")

print("Actual:    ", y_test[i_low])
print("Predicted: ", final_predictions[i_low])


print()
print("HIGH TARGET")

print("Actual:    ", y_test[i_high])
print("Predicted: ", final_predictions[i_high])


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure()

plt.scatter(
    y_test,
    final_predictions
)

minimum = min(
    y_test.min(),
    final_predictions.min()
)

maximum = max(
    y_test.max(),
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
    f"PyTorch prediction "
    f"(Test R² = {final_r2:.3f})"
)

plt.tight_layout()

plt.show()


# ============================================================
# R² OVER TRAINING
# ============================================================

plt.figure()

plt.plot(
    epochs_history,
    train_r2_history,
    label="Training R²"
)

plt.plot(
    epochs_history,
    test_r2_history,
    label="Test R²"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Epoch")

plt.ylabel("R²")

plt.title(
    "Model accuracy over training"
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

plt.xlabel("Epoch")

plt.ylabel("RMSE")

plt.title(
    "RMSE over training"
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

plt.xlabel("Epoch")

plt.ylabel("MSE loss")

plt.title(
    "Training loss"
)

plt.tight_layout()

plt.show()
