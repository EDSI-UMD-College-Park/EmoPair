import pairadigm as pdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr
from pairadigm import Pairadigm
import copy
import random
import polars as pl
import math
from contextlib import nullcontext

import sys
from datetime import datetime
import os

# Setup logging to file for tmux sessions

# Create logs directory if it doesn't exist
os.makedirs("../Model Training/training_logs", exist_ok=True)

# Create log file with timestamp
log_file = f"../Model Training/training_logs/training_log_emobank_reward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Redirect stdout and stderr to log file while keeping console output
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    def isatty(self):
        return False

log_file_handle = open(log_file, 'w')
sys.stdout = Tee(sys.stdout, log_file_handle)
sys.stderr = Tee(sys.stderr, log_file_handle)

print(f"Logging to: {log_file}")
print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Set random seeds for reproducibility
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Detect best device available
def get_best_device(cuda_index: int = 0, verbose: bool = True):
    """Detect best device: CUDA (with index) > MPS > CPU"""
    if torch.cuda.is_available():
        if cuda_index < torch.cuda.device_count():
            device = torch.device(f"cuda:{cuda_index}")
            device_name = f"cuda:{cuda_index}"
            if verbose:
                try:
                    gpu_name = torch.cuda.get_device_name(cuda_index)
                    print(f"Using NVIDIA GPU {cuda_index}: {gpu_name}")
                except Exception:
                    print(f"Using NVIDIA GPU {cuda_index}")
        else:
            raise ValueError(
                f"CUDA device {cuda_index} not available. "
                f"Available: 0 to {torch.cuda.device_count()-1}"
            )

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "mps"
        if verbose:
            print("Using Apple Silicon GPU via MPS")
    else:
        device = torch.device("cpu")
        device_name = "cpu"
        if verbose:
            print("No GPU found, using CPU")
    return device, device_name

# Set seeds and get device
seed_num = 42
set_seeds(seed=seed_num)
device, device_name = get_best_device(cuda_index=0, verbose=True)

# Load emopair dataset (contains id, split, V, text, Aro/Dom BT scores)
print("Loading data...")
emopair = pl.read_csv("../Results/emopair.csv").to_pandas()
emopair_pairs = pl.read_csv("../Results/emopair_pairs_w_breakdowns.csv").to_pandas()

# Load EmoBank-based pairwise indicator labels (A_better_indicator: 1 if Item_A > Item_B)
A_probs = pl.read_csv("../Data/pairwise_probs_A.csv").to_pandas()
D_probs = pl.read_csv("../Data/pairwise_probs_D.csv").to_pandas()
for df in [A_probs, D_probs]:
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

print(f"Data loaded. {len(emopair)} EmoPair items. {len(emopair_pairs)} pairs.")

def merge_indicator(pairs_df, probs_df):
    """
    Merge A_better_indicator from pairwise_probs onto pairs_df using order-aware matching.
    Handles both (item1=Item_A, item2=Item_B) and the reversed ordering (flipping the indicator).
    """
    # Direct match: item1 == Item_A, item2 == Item_B
    direct = pairs_df[['item1', 'item2']].merge(
        probs_df[['Item_A', 'Item_B', 'A_better_indicator']],
        left_on=['item1', 'item2'],
        right_on=['Item_A', 'Item_B'],
        how='left'
    )['A_better_indicator']

    # Reversed match: item1 == Item_B, item2 == Item_A → flip indicator
    probs_reversed = probs_df[['Item_A', 'Item_B', 'A_better_indicator']].copy()
    probs_reversed['A_better_indicator'] = 1 - probs_reversed['A_better_indicator']
    reverse = pairs_df[['item1', 'item2']].merge(
        probs_reversed,
        left_on=['item1', 'item2'],
        right_on=['Item_B', 'Item_A'],
        how='left'
    )['A_better_indicator']

    return direct.fillna(reverse)

# Add EmoBank-based labels; pairs absent from pairwise_probs will be NaN
print("Merging pairwise indicator labels...")
emopair_pairs['label_A'] = merge_indicator(emopair_pairs, A_probs)
emopair_pairs['label_D'] = merge_indicator(emopair_pairs, D_probs)
print("Merge complete. Sample of pairs with labels:")

# Drop rows with null texts or pairs not found in pairwise_probs
n_before = len(emopair_pairs)
emopair_pairs = emopair_pairs[
    emopair_pairs['item1_text'].notna() &
    emopair_pairs['item2_text'].notna() &
    emopair_pairs['label_A'].notna() &
    emopair_pairs['label_D'].notna()
].reset_index(drop=True)
print(f"Dropped {n_before - len(emopair_pairs)} rows (null text or missing pairwise label). {len(emopair_pairs)} pairs remaining.")

# Split data into train/dev/test based on 'split' column in emopair_pairs
print("Preparing data...")
train_set_pairs = emopair_pairs[(emopair_pairs['item1_split'] == 'train') & (emopair_pairs['item2_split'] == 'train')].copy()
print(f"Training set prepared. Number of pairs: {len(train_set_pairs)}")

eval_set_pairs = emopair_pairs[(emopair_pairs['item1_split'] == 'dev') & (emopair_pairs['item2_split'] == 'dev')].copy()
print(f"Evaluation set prepared. Number of pairs: {len(eval_set_pairs)}")

test_set_pairs = emopair_pairs[(emopair_pairs['item1_split'] == 'test') & (emopair_pairs['item2_split'] == 'test')].copy()
print(f"Test set prepared. Number of pairs: {len(test_set_pairs)}")

def make_tuples(data):
    """Convert dataframe to list of tuples with V gold scores and A/D EmoBank indicator labels.
    
    Returns:
        List of tuples: (text_1, text_2, V_1, V_2, label_A, label_D)
        where label_A/label_D = 1.0 if item1 has higher arousal/dominance, 0.0 otherwise
    """
    return list(
        zip(
            data['item1_text'].tolist(),
            data['item2_text'].tolist(),
            data['item1_V'].tolist(),
            data['item2_V'].tolist(),
            data['label_A'].tolist(),
            data['label_D'].tolist()
        )
    )

# Functions for evaluation of regression metrics on the splits to compare with PERT, etc.
def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, dim_names: List[str] = ['valence', 'arousal', 'dominance']) -> Dict[str, float]:
    """Compute MAE, R2, and Pearson metrics (overall and per-dimension).
    
    Args:
        predictions: Array of shape (N, D)
        targets: Array of shape (N, D)
        dim_names: Names for each dimension (length must match D)
    """
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    pearson = pearsonr(targets.flatten(), predictions.flatten())[0]

    n_dims = targets.shape[1]
    mae_per_dim = [mean_absolute_error(targets[:, i], predictions[:, i]) for i in range(n_dims)]
    r2_per_dim = [r2_score(targets[:, i], predictions[:, i]) for i in range(n_dims)]
    pearson_per_dim = [pearsonr(targets[:, i], predictions[:, i])[0] for i in range(n_dims)]

    result = {'mae': mae, 'r2': r2, 'pearson': pearson}
    for i, name in enumerate(dim_names[:n_dims]):
        result[f'mae_{name}'] = mae_per_dim[i]
        result[f'r2_{name}'] = r2_per_dim[i]
        result[f'pearson_{name}'] = pearson_per_dim[i]

    return result

def evaluate_AD_bt_correlation_on_split(
    emopair_df: pd.DataFrame,
    split: str,
    reward_model: 'RewardModel',
    batch_size: int = 32,
    use_full_bt: bool = False
) -> Dict[str, float]:
    """
    Score texts for a split and compute Pearson correlation of model A/D outputs
    against Bradley-Terry scores.

    Args:
        emopair_df: EmoPair dataframe
        split: 'train', 'dev', or 'test'
        reward_model: Trained reward model
        batch_size: Batch size for scoring
        use_full_bt: If True, use A_BT_score_full/D_BT_score_full (for test set);
                     if False, use A_BT_score_split/D_BT_score_split (for dev set)
    """
    a_col = 'A_BT_score_full' if use_full_bt else 'A_BT_score_split'
    d_col = 'D_BT_score_full' if use_full_bt else 'D_BT_score_split'

    subset = emopair_df[emopair_df['split'] == split].copy()
    subset = subset.dropna(subset=['text', a_col, d_col])
    if subset.empty:
        raise ValueError(f"No rows found for split '{split}' with BT score labels.")

    texts = subset['text'].tolist()
    targets_A = subset[a_col].values.astype(np.float32)
    targets_D = subset[d_col].values.astype(np.float32)

    preds_dict = reward_model.score_batch(texts, batch_size=batch_size)

    pearson_A = pearsonr(targets_A, preds_dict['arousal'])[0]
    pearson_D = pearsonr(targets_D, preds_dict['dominance'])[0]

    bt_type = 'full' if use_full_bt else 'split'
    print(f"  [{split}] A/D BT correlation ({bt_type}): Arousal r={pearson_A:.4f}, Dominance r={pearson_D:.4f}")

    return {
        'pearson_arousal_bt': float(pearson_A),
        'pearson_dominance_bt': float(pearson_D),
        'bt_score_type': bt_type
    }


def evaluate_valence_regression_on_split(
    emopair_df: pd.DataFrame,
    split: str,
    reward_model: 'RewardModel',
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Score texts for a split and compute regression metrics vs gold V.

    Valence is trained directly against gold 1-5 scores, so metrics should
    be computed on the raw valence head output without post-hoc normalization.
    """
    subset = emopair_df[emopair_df['split'] == split].copy()
    subset = subset.dropna(subset=['text', 'V'])
    if subset.empty:
        raise ValueError(f"No rows found for split '{split}' with V labels.")

    texts = subset['text'].tolist()
    targets_V = subset['V'].values.astype(np.float32)

    preds_dict = reward_model.score_batch(texts, batch_size=batch_size)
    preds_V = preds_dict['valence'].astype(np.float32)

    return {k: float(v) for k, v in calculate_metrics(
        preds_V.reshape(-1, 1),
        targets_V.reshape(-1, 1),
        dim_names=['valence']
    ).items()}

class RewardModel(nn.Module):
    """
    Unified class for training and using a reward model for text scoring.
    
    This class handles:
    - Model initialization and configuration
    - Dataset creation and management
    - Training loop with pairwise comparisons
    - Scoring individual texts or batches
    - Score normalization
    """
    
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-large",
        dropout: float = 0.1,
        max_length: int = 384,
        device: Optional[str] = None,
        Pairadigm: Optional['Pairadigm'] = None,
        seed: int = 42
    ):
        """
        Initialize the reward model trainer.
        
        Args:
            model_name: HuggingFace model identifier
            dropout: Dropout rate for reward head
            max_length: Maximum sequence length for tokenization
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()
        # Set seeds at model initialization
        set_seeds(seed)

        self.model_name = model_name
        self.max_length = max_length
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"Model using device: {self.device}")
        self.device_type = self.device.type if isinstance(self.device, torch.device) else self.device.split(':')[0]
        self.use_amp = self.device_type == 'cuda'
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = self._build_model(dropout)
        self.model.to(self.device)

        # If a Pairadigm instance is provided, link it
        self.pairadigm = Pairadigm
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler(self.device_type, enabled=self.device_type == 'cuda')
        self.training_history = []
        # self.loss_weights = {'V': 0.4, 'A': 0.3, 'D': 0.3}

    def _autocast_context(self):
        if self.device_type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if self.device_type == "cpu":
            return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
        return nullcontext()
    
    def _build_model(self, dropout: float) -> nn.Module:
        """Build the reward model architecture with separate heads for V (pointwise) and A/D (pairwise)."""
        class _EncoderWithHead(nn.Module):
            def __init__(inner_self, model_name, dropout):
                super().__init__()

                inner_self.encoder = AutoModel.from_pretrained(model_name)
                hidden_size = inner_self.encoder.config.hidden_size
                inner_self.dropout = nn.Dropout(dropout)

                inner_self.valence_head = nn.Linear(hidden_size, 1)
                inner_self.arousal_dominance_head = nn.Linear(hidden_size, 2)

                nn.init.xavier_uniform_(inner_self.valence_head.weight, gain=1.0)
                nn.init.zeros_(inner_self.valence_head.bias)
                nn.init.xavier_uniform_(inner_self.arousal_dominance_head.weight, gain=1.0)
                nn.init.zeros_(inner_self.arousal_dominance_head.bias)

            def forward(inner_self, input_ids, attention_mask):
                outputs = inner_self.encoder(input_ids=input_ids, attention_mask=attention_mask)

                pooled_output = outputs.last_hidden_state[:, 0, :]
                pooled_output = inner_self.dropout(pooled_output)

                valence = inner_self.valence_head(pooled_output)
                arousal_dominance = inner_self.arousal_dominance_head(pooled_output)

                return valence, arousal_dominance

        return _EncoderWithHead(self.model_name, dropout)

    @staticmethod
    def _uncertainty_weight(prob: torch.Tensor, min_weight: float = 0.25, power: float = 0.5) -> torch.Tensor:
        """Compute softened informativeness weights from pairwise probabilities.

        Base informativeness is |2 * (p - 0.5)| in [0, 1]. We soften it by:
        - applying a square-root transform (power < 1) to reduce harsh downweighting near 0.5
        - adding a minimum floor so uncertain pairs still contribute
        """
        base = (2 * (prob - 0.5)).abs().clamp(0.0, 1.0)
        softened = base.pow(power)
        return min_weight + (1.0 - min_weight) * softened
    
    def prepare_data(
        self,
        pairs: Optional[List[Tuple[str, str, Union[int, float]]]] = None,
        batch_size: int = 16,
        shuffle: bool = True,
        decision_col: str = "decision",
        score_col: str = "Bradley_Terry_Score",
        margins: bool = True,
        test_size: float = 0.15,
        eval_size: float = 0.15,
        stratify: Optional[List] = None,
        random_state: Optional[int] = 42
    ) -> Dict[str, DataLoader]:
        """
        Prepare training, evaluation, and test data from pairwise comparisons.

        Args:
            pairs: Optional list of (text_winner, text_loser, margin) tuples. If None and a Pairadigm instance is linked, will use Pairadigm.
            batch_size: Batch size for training
            shuffle: Whether to shuffle the training data
            decision_col: Column name for decision in pairwise_df
            score_col: Column name for scores in scored_df
            margins: Whether to compute margins from scores
            test_size: Proportion of data for test set (default 0.15)
            eval_size: Proportion of data for eval set (default 0.15)
            stratify: Optional list of labels for stratified splitting
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'eval', and 'test' DataLoaders
        """
        if pairs is None and self.pairadigm:
            pairs = self._prepare_pairadigm(
                decision_col=decision_col, 
                score_col=score_col, 
                margins=margins
            )
        elif pairs is None:
            raise ValueError("No pairs provided and no Pairadigm instance linked.")
        
        # Split data: first split into train+eval and test
        train_eval_pairs, test_pairs = train_test_split(
            pairs,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state
        )
        
        # Then split train+eval into train and eval
        # Adjust eval_size to be relative to train+eval size
        adjusted_eval_size = eval_size / (1 - test_size)
        train_pairs, eval_pairs = train_test_split(
            train_eval_pairs,
            test_size=adjusted_eval_size,
            stratify=None,  # stratify only on first split
            random_state=random_state
        )
        
        print(f"Data split - Train: {len(train_pairs)}, Eval: {len(eval_pairs)}, Test: {len(test_pairs)}")
        
        # Create datasets and dataloaders
        train_dataset = self._PairwiseDataset(train_pairs, self.tokenizer, self.max_length)
        eval_dataset = self._PairwiseDataset(eval_pairs, self.tokenizer, self.max_length)
        test_dataset = self._PairwiseDataset(test_pairs, self.tokenizer, self.max_length)
        
        return {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
            'eval': DataLoader(eval_dataset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }
    
    class _PairwiseDataset(Dataset):
        """Internal dataset class for pairwise comparisons with V/A/D probabilities."""
        
        def __init__(self, pairs, tokenizer, max_length):
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            text_A, text_B, V_A, V_B, label_A, label_D = self.pairs[idx]
            
            encoding_A = self.tokenizer(
                text_A,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            encoding_B = self.tokenizer(
                text_B,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids_A': encoding_A['input_ids'].squeeze(0),
                'attention_mask_A': encoding_A['attention_mask'].squeeze(0),
                'input_ids_B': encoding_B['input_ids'].squeeze(0),
                'attention_mask_B': encoding_B['attention_mask'].squeeze(0),
                'V_A': torch.tensor(V_A, dtype=torch.float),
                'V_B': torch.tensor(V_B, dtype=torch.float),
                'label_A': torch.tensor(label_A, dtype=torch.float),
                'label_D': torch.tensor(label_D, dtype=torch.float)
            }

    def fit(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        log_interval: int = 50,
        early_stopping_patience: int = 3,
        accumulation_steps: int = 1,
        max_steps_per_epoch: int = None
    ):
        """
        Train the reward model on pairwise comparison data with optional early stopping and gradient accumulation.

        Args:
            train_loader: DataLoader with training pairs (mini-batch size)
            eval_loader: DataLoader for evaluation (required for early stopping)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for scheduler
            log_interval: Log metrics every N steps
            early_stopping_patience: Number of epochs with no improvement on eval loss before stopping early.
                                     Set to None or 0 to disable early stopping.
            accumulation_steps: Number of gradient accumulation steps (effective_batch_size = mini_batch_size * accumulation_steps)
        Returns:
            The model (self.model) restored to the best-performing weights observed on eval data.
        """
        self.model.train()
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

        steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        if max_steps_per_epoch is not None:
            steps_per_epoch = math.ceil(max_steps_per_epoch / accumulation_steps)

        total_optimization_steps = steps_per_epoch * epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimization_steps
        )

        # Early stopping bookkeeping
        best_state = None
        best_eval_loss = float('inf')
        epochs_without_improve = 0
        use_early_stopping = bool(early_stopping_patience and eval_loader is not None and early_stopping_patience > 0)
        
        for epoch in range(epochs):
            epoch_loss = 0
            grad_norms = []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            grad_norm = 0.0
            
            for step, batch in enumerate(progress_bar):
                # Perform forward/backward pass, accumulate gradients
                loss = self._training_step(batch, accumulation_steps)
                epoch_loss += loss

                is_accum_step = (step + 1) % accumulation_steps == 0
                is_last_batch = (step + 1) == len(train_loader)
                is_max_step = max_steps_per_epoch is not None and (step + 1) == max_steps_per_epoch

                if is_accum_step or is_last_batch or is_max_step:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    grad_norms.append(grad_norm.item())

                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scaler.get_scale() >= scale_before:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                
                if (step + 1) % log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'grad_norm': f'{grad_norm:.4f}'})

                if is_max_step:
                    break
            
            actual_steps = step + 1
            avg_epoch_loss = epoch_loss / actual_steps
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss,
                'train_grad_norm': float(np.nanmean([g for g in grad_norms if np.isfinite(g)])) if grad_norms else float('nan'),
                'grad_norms': grad_norms
            }
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f}")
            
            # Evaluation
            if eval_loader:
                eval_metrics = self.evaluate(eval_loader)
                epoch_metrics.update(eval_metrics)
                print(f"Epoch {epoch + 1} - Eval Loss: {eval_metrics['eval_loss']:.4f} | Accuracy: {eval_metrics['eval_accuracy']:.2%}")
                print(f"  Loss - V (MSE): {eval_metrics['eval_loss_V']:.4f} | A: {eval_metrics['eval_loss_A']:.4f} | D: {eval_metrics['eval_loss_D']:.4f}")
                print(f"  Acc  - V (EMOBANK): {eval_metrics['eval_accuracy_V']:.2%} | A: {eval_metrics['eval_accuracy_A']:.2%} | D: {eval_metrics['eval_accuracy_D']:.2%}")
                
                # Check for improvement on eval_loss and save best model
                current_eval_loss = eval_metrics.get('eval_loss', float('inf'))
                if current_eval_loss < best_eval_loss:
                    best_eval_loss = current_eval_loss
                    # store a CPU copy of the state dict
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improve = 0
                    print(f"  New best model found (eval_loss improved to {best_eval_loss:.4f}).")
                else:
                    epochs_without_improve += 1
                    print(f"  No improvement for {epochs_without_improve} epoch(s).")
            
            else:
                # If no eval_loader provided, we can't do early stopping / track best by eval
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            self.training_history.append(epoch_metrics)

            # Early stopping check
            if use_early_stopping and epochs_without_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs (no improvement in eval loss for {early_stopping_patience} epochs).")
                break
        
        # Restore best model weights if we tracked them
        if best_state is not None:
            # move tensors back to device as needed when loading
            device_state = {k: v.to(self.device) for k, v in best_state.items()}
            self.model.load_state_dict(device_state)
            print("Best model weights restored based on eval data.")
        
        return self.model
    
    def _training_step(self, batch, accumulation_steps: int = 1) -> float:
        """Single training step: MSE for V (pointwise) and binary cross-entropy for A/D (pairwise).
        
        Args:
            batch: Dictionary containing input_ids, attention_masks,
                   V_A/V_B (gold scores), label_A/label_D (binary decisions: 1.0 = item1 wins, 0.0 = item2 wins)
            accumulation_steps: Number of steps to accumulate gradients over
            
        Returns:
            Loss value for this step
        """
        batch = {key: value.to(self.device) for key, value in batch.items()}
        
        with self._autocast_context():
            valence_A, ad_A = self.model(
                batch['input_ids_A'],
                batch['attention_mask_A']
            )
            valence_B, ad_B = self.model(
                batch['input_ids_B'],
                batch['attention_mask_B']
            )

            pred_V_A = valence_A.squeeze(-1)
            pred_V_B = valence_B.squeeze(-1)
            
            # Valence: Pointwise MSE loss against gold scores (1-5 scale)
            loss_V = (F.mse_loss(pred_V_A, batch['V_A']) + \
                      F.mse_loss(pred_V_B, batch['V_B'])) / 2.0
            
            # Arousal & Dominance: Pairwise binary cross-entropy loss
            # loss = -[label * log(σ(r_A - r_B)) + (1-label) * log(σ(r_B - r_A))]
            # label = 1.0 if item1 wins, 0.0 if item2 wins
            
            # Arousal (index 0 in arousal_dominance output)
            label_A = batch['label_A']
            logit_A = ad_A[:, 0] - ad_B[:, 0]
            loss_A = F.binary_cross_entropy_with_logits(logit_A, label_A)
            
            # Dominance (index 1 in arousal_dominance output)
            label_D = batch['label_D']
            logit_D = ad_A[:, 1] - ad_B[:, 1]
            loss_D = F.binary_cross_entropy_with_logits(logit_D, label_D)
            
            total_loss = (loss_V + loss_A + loss_D) / accumulation_steps
        
        self.scaler.scale(total_loss).backward()
        return total_loss.item() * accumulation_steps
    
    def _compute_pairwise_accuracy(self, batch) -> Dict[str, float]:
        """Compute pairwise accuracy for V (via gold EmoBank scores), A, and D."""
        batch = {key: value.to(self.device) for key, value in batch.items()}
        
        with torch.no_grad():
            valence_A, ad_A = self.model(batch['input_ids_A'], batch['attention_mask_A'])
            valence_B, ad_B = self.model(batch['input_ids_B'], batch['attention_mask_B'])

        pred_V_A = valence_A.squeeze(-1)
        pred_V_B = valence_B.squeeze(-1)

        V_A = batch['V_A']
        V_B = batch['V_B']
        label_A = batch['label_A']
        label_D = batch['label_D']

        correct_V = ((V_A > V_B) & (pred_V_A > pred_V_B)) | \
                    ((V_A < V_B) & (pred_V_A < pred_V_B))
        correct_A = ((label_A == 1.0) & (ad_A[:, 0] > ad_B[:, 0])) | \
                    ((label_A == 0.0) & (ad_B[:, 0] > ad_A[:, 0]))
        correct_D = ((label_D == 1.0) & (ad_A[:, 1] > ad_B[:, 1])) | \
                    ((label_D == 0.0) & (ad_B[:, 1] > ad_A[:, 1]))

        non_tie_V = (V_A != V_B)
        non_tie_A = torch.ones_like(label_A, dtype=torch.bool)  # Binary decisions, no ties
        non_tie_D = torch.ones_like(label_D, dtype=torch.bool)  # Binary decisions, no ties

        accuracy_V = correct_V[non_tie_V].float().mean().item() if non_tie_V.any() else 0.0
        accuracy_A = correct_A[non_tie_A].float().mean().item() if non_tie_A.any() else 0.0
        accuracy_D = correct_D[non_tie_D].float().mean().item() if non_tie_D.any() else 0.0

        return {
            'accuracy_V': accuracy_V,
            'accuracy_A': accuracy_A,
            'accuracy_D': accuracy_D,
            'accuracy_overall': (accuracy_V + accuracy_A + accuracy_D) / 3
        }
    
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model: MSE for V (pointwise), binary cross-entropy for A/D (pairwise)."""
        self.model.eval()
        total_loss = 0
        losses_by_dim = {'V': 0, 'A': 0, 'D': 0}
        
        total_correct_V = 0
        total_correct_A = 0
        total_correct_D = 0
        total_count_V = 0
        total_count_A = 0
        total_count_D = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                
                valence_A, ad_A = self.model(
                    batch['input_ids_A'],
                    batch['attention_mask_A']
                )
                valence_B, ad_B = self.model(
                    batch['input_ids_B'],
                    batch['attention_mask_B']
                )

                pred_V_A = valence_A.squeeze(-1)
                pred_V_B = valence_B.squeeze(-1)
                
                # Valence: MSE loss against gold scores
                loss_V = (F.mse_loss(pred_V_A, batch['V_A']) + \
                         F.mse_loss(pred_V_B, batch['V_B'])) / 2.0
                
                # Arousal: binary cross-entropy loss
                label_A = batch['label_A']
                logit_A = ad_A[:, 0] - ad_B[:, 0]
                loss_A = F.binary_cross_entropy_with_logits(logit_A, label_A)
                
                # Dominance: binary cross-entropy loss
                label_D = batch['label_D']
                logit_D = ad_A[:, 1] - ad_B[:, 1]
                loss_D = F.binary_cross_entropy_with_logits(logit_D, label_D)
                
                batch_loss = loss_V + loss_A + loss_D
                total_loss += batch_loss.item()
                losses_by_dim['V'] += loss_V.item()
                losses_by_dim['A'] += loss_A.item()
                losses_by_dim['D'] += loss_D.item()
                
                # Valence pairwise accuracy via gold V scores
                V_A_gold = batch['V_A']
                V_B_gold = batch['V_B']
                correct_V = ((V_A_gold > V_B_gold) & (pred_V_A > pred_V_B)) | \
                           ((V_A_gold < V_B_gold) & (pred_V_A < pred_V_B))
                non_tie_V = (V_A_gold != V_B_gold)
                total_correct_V += correct_V[non_tie_V].sum().item()
                total_count_V += non_tie_V.sum().item()
                
                # Arousal/Dominance pairwise accuracy
                correct_A = ((label_A == 1.0) & (ad_A[:, 0] > ad_B[:, 0])) | \
                           ((label_A == 0.0) & (ad_B[:, 0] > ad_A[:, 0]))
                correct_D = ((label_D == 1.0) & (ad_A[:, 1] > ad_B[:, 1])) | \
                           ((label_D == 0.0) & (ad_B[:, 1] > ad_A[:, 1]))
                
                total_correct_A += correct_A.sum().item()
                total_correct_D += correct_D.sum().item()
                total_count_A += label_A.shape[0]
                total_count_D += label_D.shape[0]
        
        self.model.train()
        
        accuracy_V = total_correct_V / max(total_count_V, 1)
        accuracy_A = total_correct_A / max(total_count_A, 1)
        accuracy_D = total_correct_D / max(total_count_D, 1)
        accuracy_overall = (accuracy_V + accuracy_A + accuracy_D) / 3
        
        return {
            'eval_loss': total_loss / len(eval_loader),
            'eval_loss_V': losses_by_dim['V'] / len(eval_loader),
            'eval_loss_A': losses_by_dim['A'] / len(eval_loader),
            'eval_loss_D': losses_by_dim['D'] / len(eval_loader),
            'eval_accuracy': accuracy_overall,
            'eval_accuracy_V': accuracy_V,
            'eval_accuracy_A': accuracy_A,
            'eval_accuracy_D': accuracy_D
        }
    
    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score a single text item across all three dimensions (V, A, D).
        
        Args:
            text: Text to score
            
        Returns:
            Dictionary with 'valence', 'arousal', 'dominance' scores
        """
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            valence, arousal_dominance = self.model(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device)
            )
        
        self.model.train()
        return {
            'valence': valence[0, 0].item(),
            'arousal': arousal_dominance[0, 0].item(),
            'dominance': arousal_dominance[0, 1].item()
        }
    
    def score_batch(self, texts: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Score multiple texts efficiently across all three dimensions.
        
        Args:
            texts: List of texts to score
            batch_size: Number of texts to process at once
            
        Returns:
            Dictionary with 'valence', 'arousal', 'dominance' arrays
        """
        self.model.eval()
        scores_V = []
        scores_A = []
        scores_D = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                valence, arousal_dominance = self.model(
                    encodings['input_ids'].to(self.device),
                    encodings['attention_mask'].to(self.device)
                )
            
            scores_V.extend(valence[:, 0].cpu().numpy())
            scores_A.extend(arousal_dominance[:, 0].cpu().numpy())
            scores_D.extend(arousal_dominance[:, 1].cpu().numpy())
        
        self.model.train()
        return {
            'valence': np.array(scores_V),
            'arousal': np.array(scores_A),
            'dominance': np.array(scores_D)
        }
    
    def fit_normalization_params(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Dict[str, Dict[str, float]]:
        """
        Fit normalization parameters using percentiles for robustness."""
        scores = self.score_batch(texts, batch_size=batch_size)
        
        params = {}
        for dim in ['valence', 'arousal', 'dominance']:
            dim_scores = scores[dim]
            # Use 1st/99th percentiles instead of raw min/max
            p_low = float(np.percentile(dim_scores, 1))
            p_high = float(np.percentile(dim_scores, 99))
            params[dim] = {'min': p_low, 'max': p_high}
        
        return params
    
    def normalize_scores_with_params(
        self,
        scores: Dict[str, np.ndarray],
        params: Dict[str, Dict[str, float]],
        scale_min: float = 1.0,
        scale_max: float = 5.0,
        clip: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Normalize scores using pre-computed parameters.
        
        Args:
            scores: Dictionary with 'valence', 'arousal', 'dominance' arrays
            params: Min/max parameters for each dimension from fit_normalization_params
            scale_min: Minimum value of output scale (EmoBank uses 1)
            scale_max: Maximum value of output scale (EmoBank uses 5)
            clip: Whether to clip values to [scale_min, scale_max]
            
        Returns:
            Dictionary with normalized scores
        """
        normalized = {}
        
        for dim in ['valence', 'arousal', 'dominance']:
            dim_min = params[dim]['min']
            dim_max = params[dim]['max']
            
            if dim_max == dim_min:
                # If no variance, use midpoint
                normalized[dim] = np.full_like(scores[dim], (scale_min + scale_max) / 2)
            else:
                # Linear scaling
                norm = (scores[dim] - dim_min) / (dim_max - dim_min)
                norm = norm * (scale_max - scale_min) + scale_min
                
                if clip:
                    norm = np.clip(norm, scale_min, scale_max)
                
                normalized[dim] = norm
        
        return normalized
    
    def normalize_scores(
        self,
        scores: np.ndarray,
        scale_min: float = 0.0,
        scale_max: float = 1.0
    ) -> np.ndarray:
        """
        Normalize raw reward scores to a desired scale.
        This normalizes within the provided scores (relative scaling).
        For consistent scaling across splits, use normalize_scores_with_params instead.
        
        Args:
            scores: Raw scores to normalize
            scale_min: Minimum value of output scale
            scale_max: Maximum value of output scale
            
        Returns:
            Normalized scores
        """
        score_min = scores.min()
        score_max = scores.max()
        
        if score_max == score_min:
            return np.full_like(scores, (scale_min + scale_max) / 2)
        
        normalized = (scores - score_min) / (score_max - score_min)
        normalized = normalized * (scale_max - scale_min) + scale_min
        
        return normalized
    
    def test_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on the test set: MSE for V, binary cross-entropy for A/D."""
        print("\n" + "="*60)
        print("Running Test Evaluation (V, A, D)")
        print("="*60)
        
        self.model.eval()
        total_loss = 0
        losses_by_dim = {'V': 0, 'A': 0, 'D': 0}
        
        total_correct_V = 0
        total_correct_A = 0
        total_correct_D = 0
        total_count_V = 0
        total_count_A = 0
        total_count_D = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                
                valence_A, ad_A = self.model(
                    batch['input_ids_A'],
                    batch['attention_mask_A']
                )
                valence_B, ad_B = self.model(
                    batch['input_ids_B'],
                    batch['attention_mask_B']
                )

                pred_V_A = valence_A.squeeze(-1)
                pred_V_B = valence_B.squeeze(-1)
                
                # Valence: MSE loss
                loss_V = (F.mse_loss(pred_V_A, batch['V_A']) + \
                         F.mse_loss(pred_V_B, batch['V_B'])) / 2.0
                
                # Arousal: binary cross-entropy loss
                label_A = batch['label_A']
                logit_A = ad_A[:, 0] - ad_B[:, 0]
                loss_A = F.binary_cross_entropy_with_logits(logit_A, label_A)
                
                # Dominance: binary cross-entropy loss
                label_D = batch['label_D']
                logit_D = ad_A[:, 1] - ad_B[:, 1]
                loss_D = F.binary_cross_entropy_with_logits(logit_D, label_D)
                
                batch_loss = loss_V + loss_A + loss_D
                total_loss += batch_loss.item()
                losses_by_dim['V'] += loss_V.item()
                losses_by_dim['A'] += loss_A.item()
                losses_by_dim['D'] += loss_D.item()
                
                # Valence pairwise accuracy via gold V scores
                V_A_gold = batch['V_A']
                V_B_gold = batch['V_B']
                correct_V = ((V_A_gold > V_B_gold) & (pred_V_A > pred_V_B)) | \
                           ((V_A_gold < V_B_gold) & (pred_V_A < pred_V_B))
                non_tie_V = (V_A_gold != V_B_gold)
                total_correct_V += correct_V[non_tie_V].sum().item()
                total_count_V += non_tie_V.sum().item()
                
                # Arousal/Dominance pairwise accuracy
                correct_A = ((label_A == 1.0) & (ad_A[:, 0] > ad_B[:, 0])) | \
                           ((label_A == 0.0) & (ad_B[:, 0] > ad_A[:, 0]))
                correct_D = ((label_D == 1.0) & (ad_A[:, 1] > ad_B[:, 1])) | \
                           ((label_D == 0.0) & (ad_B[:, 1] > ad_A[:, 1]))
                
                total_correct_A += correct_A.sum().item()
                total_correct_D += correct_D.sum().item()
                total_count_A += label_A.shape[0]
                total_count_D += label_D.shape[0]
        
        test_loss = total_loss / len(test_loader)
        test_loss_V = losses_by_dim['V'] / len(test_loader)
        test_loss_A = losses_by_dim['A'] / len(test_loader)
        test_loss_D = losses_by_dim['D'] / len(test_loader)
        
        accuracy_V = total_correct_V / max(total_count_V, 1)
        accuracy_A = total_correct_A / max(total_count_A, 1)
        accuracy_D = total_correct_D / max(total_count_D, 1)
        accuracy_overall = (accuracy_V + accuracy_A + accuracy_D) / 3
        
        print(f"\nTest Results:")
        print(f"  Total Loss: {test_loss:.4f} | Accuracy: {accuracy_overall:.2%}")
        print(f"  Loss - V (MSE): {test_loss_V:.4f} | A: {test_loss_A:.4f} | D: {test_loss_D:.4f}")
        print(f"  Acc  - V (EMOBANK): {accuracy_V:.2%} | A: {accuracy_A:.2%} | D: {accuracy_D:.2%}")
        print("\n" + "="*60 + "\n")
        
        results = {
            'test_loss': test_loss,
            'test_loss_V': test_loss_V,
            'test_loss_A': test_loss_A,
            'test_loss_D': test_loss_D,
            'test_accuracy': accuracy_overall,
            'test_accuracy_V': accuracy_V,
            'test_accuracy_A': accuracy_A,
            'test_accuracy_D': accuracy_D
        }
        
        return results
    
    def save(self, path: str):
        """Save model and training state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history,
            'config': {
                'model_name': self.model_name,
                'max_length': self.max_length
            }
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_history = checkpoint.get('training_history', [])
        print(f"Model loaded from {path}")

    def push_to_hub(self, repo_id: str, private: bool = True):
        """
        Push the trained model to HuggingFace Hub.
        
        Args:
            repo_id: Repository ID in format "username/repo-name"
            private: Whether to make the repository private (default True)
        """
        from huggingface_hub import create_repo, upload_folder
        import os
        
        # Create a temporary directory for the model files
        temp_dir = f"./temp_model_{repo_id.split('/')[-1]}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save encoder, tokenizer, and both heads
            self.model.encoder.save_pretrained(os.path.join(temp_dir, "encoder"))
            self.tokenizer.save_pretrained(temp_dir)
            torch.save(
                {
                    'valence_head': self.model.valence_head.state_dict(),
                    'arousal_dominance_head': self.model.arousal_dominance_head.state_dict()
                },
                os.path.join(temp_dir, "task_heads.pt")
            )
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'training_history': self.training_history
            }
            with open(os.path.join(temp_dir, "training_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create and upload to Hub
            create_repo(repo_id, private=private, exist_ok=True)
            upload_folder(repo_name=repo_id, folder_path=temp_dir, repo_type="model")
            
            print(f"Model successfully pushed to HuggingFace Hub: https://huggingface.co/{repo_id}")
        
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

# Train the reward model using the prepared pairwise data
learning_rate = 2e-6
weight_decay = 0.05
epochs = 20
patience = 7
dropout = 0.2
batch_size = 16 # Mini-batch size that fits in GPU memory
accumulation_steps = 4  # Gradient accumulation steps (effective batch size = 16 * 4 = 64)
max_length = 256
seed_num = 42

effective_batch_size = batch_size * accumulation_steps
print(f"Training parameters - Epochs: {epochs}, LR: {learning_rate}, Weight Decay: {weight_decay}, Patience: {patience}, Dropout: {dropout}")
print(f"Batch Configuration - Mini-batch Size: {batch_size}, Accumulation Steps: {accumulation_steps}, Effective Batch Size: {effective_batch_size}, Seed: {seed_num}")

print("Initializing Reward Model...")
reward_model = RewardModel(
    model_name="roberta-large",
    dropout=dropout,
    max_length=max_length,
    device=device 
)
print("Reward Model initialized.")

print("Preparing data loaders...")
num_workers = 4 if torch.cuda.is_available() else 0

# Convert to tuples format (V uses gold scores, A/D use probabilities)
train_tuples = make_tuples(train_set_pairs)
eval_tuples = make_tuples(eval_set_pairs)
test_tuples = make_tuples(test_set_pairs)

# Create datasets manually
train_dataset = reward_model._PairwiseDataset(train_tuples, reward_model.tokenizer, reward_model.max_length)
eval_dataset = reward_model._PairwiseDataset(eval_tuples, reward_model.tokenizer, reward_model.max_length)
test_dataset = reward_model._PairwiseDataset(test_tuples, reward_model.tokenizer, reward_model.max_length)

# Calculate warmup steps based on effective batches (after accumulation)
warmup_steps = int(0.06 * (len(train_dataset) / (batch_size * accumulation_steps)) * epochs)
log_interval = 50

# Create dataloaders with fixed generator for reproducibility
generator = torch.Generator()
generator.manual_seed(seed_num)

data_loaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=num_workers,
                       generator=generator, worker_init_fn=lambda _: set_seeds(seed_num)),
    'eval': DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
}

print("Data loaders ready.")
print(f"Train pairs: {len(train_tuples)}")
print(f"Eval pairs: {len(eval_tuples)}")
print(f"Test pairs: {len(test_tuples)}")

date = datetime.now().strftime("%Y%m%d")

print("Saving training arguments...")
os.makedirs("../Results/trained models", exist_ok=True)
training_args = {
    'model_name': reward_model.model_name,
    'num_epochs': epochs,
    'batch_size': batch_size,
    'accumulation_steps': accumulation_steps,
    'effective_batch_size': effective_batch_size,
    'learning_rate': learning_rate,
    'device': device_name,
    'weight_decay': weight_decay,
    'warmup_steps': warmup_steps,
    'dropout': dropout,
    'max_length': max_length,
    'seed': seed_num,
    'patience': patience,
    'log_interval': log_interval
}
with open(f"../Results/trained models/emobank_reward_model_training_args_{date}.json", "w") as f:
    json.dump(training_args, f, indent=4)

print("Starting training...")
try:
    emobank_reward = reward_model.fit(
        train_loader=data_loaders['train'],
        eval_loader=data_loaders['eval'],
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        log_interval=log_interval,
        early_stopping_patience=patience,
        accumulation_steps=accumulation_steps
    )
    
    # Save immediately after training
    reward_model.save(f"../Results/trained models/emobank_reward_model_{date}.pth")
    
except Exception as e:
    print(f"Training failed: {e}")
    # Save checkpoint even if failed
    reward_model.save(f"../Results/trained models/emobank_reward_model_checkpoint_{date}.pth")
    raise

print("Training completed.")

# Save results

# After loading the best model, evaluate on test set
test_results = reward_model.test_model(data_loaders['test'])

# Fit normalization parameters on training data


# Compute regression-style metrics for Valence against gold V scores
print("\nComputing valence regression metrics vs gold V...")
regression_metrics_test = evaluate_valence_regression_on_split(
    emopair_df=emopair,
    split='test',
    reward_model=reward_model,
    batch_size=32
)
regression_metrics_dev = evaluate_valence_regression_on_split(
    emopair_df=emopair,
    split='dev',
    reward_model=reward_model,
    batch_size=32
)
print(f"Valence Regression Metrics (dev): {regression_metrics_dev}")
print(f"Valence Regression Metrics (test): {regression_metrics_test}")

# Compute A/D BT correlations
print("\nComputing A/D BT score correlations...")
bt_metrics_dev = evaluate_AD_bt_correlation_on_split(
    emopair_df=emopair,
    split='dev',
    reward_model=reward_model,
    batch_size=32,
    use_full_bt=False  # dev: use A_BT_score_split / D_BT_score_split
)
bt_metrics_test = evaluate_AD_bt_correlation_on_split(
    emopair_df=emopair,
    split='test',
    reward_model=reward_model,
    batch_size=32,
    use_full_bt=True   # test: use A_BT_score_full / D_BT_score_full
)

# Save metrics
metrics_payload = {
    'dev': {**{k: float(v) for k, v in regression_metrics_dev.items()}, **bt_metrics_dev},
    'test': {**{k: float(v) for k, v in regression_metrics_test.items()}, **bt_metrics_test}
}
with open(f"../Results/trained models/emobank_reward_model_regression_metrics_{date}.json", "w") as f:
    json.dump(metrics_payload, f, indent=4)

# Save test results
test_results_df = pd.DataFrame([test_results])
test_results_df.to_csv(f"../Results/trained models/emobank_reward_model_test_results_{date}.csv", index=False)

# Save the trained model
reward_model.save(f"../Results/trained models/emobank_reward_model_{date}.pth")

# Save training history to CSV
history_df = pd.DataFrame(reward_model.training_history)
history_df.to_csv(f"../Results/trained models/emobank_reward_model_training_history_{date}.csv", index=False)
# Clean up memory after testing
import gc

print("Cleaning up memory...")
del emobank_reward
del data_loaders
gc.collect()
if isinstance(device, torch.device) and device.type == "cuda":
    gpu_index = device.index if device.index is not None else 0
    torch.cuda.empty_cache()
    available = (
        torch.cuda.get_device_properties(gpu_index).total_memory
        - torch.cuda.memory_allocated(gpu_index)
    )
    print(f"GPU {gpu_index} memory freed. Available: {available}")
print("Memory cleanup complete.")