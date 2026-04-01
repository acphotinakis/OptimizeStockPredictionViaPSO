import torch
import torch.nn as nn
import logging
from pathlib import Path
from tqdm import tqdm

# Configure module-level logger
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    This mechanism is critical for financial time-series to prevent the model from
    memorizing noise (overfitting) rather than learning market patterns.
    """

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Performs one full pass over the training dataset.
    """
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        # Move tensors to the configured device (CPU/CUDA)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)

        # Standard PyTorch training step
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set without updating weights.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def run_training(model, train_loader, val_loader, cfg, device):
    """
    Orchestrates the full training pipeline, including early stopping and checkpointing.

    Args:
        model (nn.Module): The LSTM architecture to train.
        train_loader (DataLoader): Training data batches.
        val_loader (DataLoader): Validation data batches.
        cfg: Hydra configuration (cfg.model.training and cfg.paths).
        device: The device (cpu or cuda) to run training on.
    """
    logger.info("Initializing LSTM training environment...")

    # Use Mean Squared Error (MSE) as the loss function for price prediction
    criterion = nn.MSELoss()

    # Configure optimizer based on proposal (Adam/RMSprop)
    optimizer_name = cfg.model.training.optimizer
    lr = cfg.model.training.learning_rate
    wd = cfg.model.training.weight_decay

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        logger.warning(
            f"Optimizer {optimizer_name} not explicitly supported. Defaulting to Adam."
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Ensure output directory exists for model artifacts
    save_dir = Path(cfg.paths.model_save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_model.pt"

    # Initialize Early Stopping monitoring
    early_stopping = EarlyStopping(
        patience=cfg.model.training.early_stopping_patience,
        verbose=True,
        path=str(best_model_path),
    )

    epochs = cfg.model.training.epochs
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Check if validation loss improved
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logger.info("Early stopping criteria met. Terminating training loop.")
            break

    # Load the best weights discovered during training before returning
    model.load_state_dict(torch.load(str(best_model_path)))
    logger.info(f"Training complete. Best model state saved to {best_model_path}")

    return model, val_loss
