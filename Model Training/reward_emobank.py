import pairadigm as pdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from pairadigm import Pairadigm
import copy
import random
import polars as pl

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
                    print(f"Using NVIDIA GPU {cuda_index}: {torch.cuda.get_device_name(cuda_index)}")
                except Exception:
                    print(f"Using NVIDIA GPU {cuda_index}")
        else:
            raise ValueError(f"CUDA device {cuda_index} not available. Available devices: 0 to {torch.cuda.device_count()-1}")

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

# Load with Polars (faster)
print("Loading data...")
emobank = pl.read_csv("../Data/emobank.csv").to_pandas()
V = pl.read_csv("../Data/pairwise_probs_V.csv").to_pandas()
A = pl.read_csv("../Data/pairwise_probs_A.csv").to_pandas()
D = pl.read_csv("../Data/pairwise_probs_D.csv").to_pandas()

# Drop the Unnamed column if it exists
for df in [emobank, V, A, D]:
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

print(f"Data loaded.")

def join_texts(pairwise_probs, emobank):
    """Join the texts from emobank to the pairwise probabilities dataframe."""

    texts = emobank[['id', 'text']]

    merge = pairwise_probs.merge(
        texts,
        left_on="Item_A",
        right_on="id",
        how="left"
    ).rename(columns={"text": "Text_A"}).drop(columns=["id"]).merge(
        texts,
        left_on="Item_B",
        right_on="id",
        how="left"
    ).rename(columns={"text": "Text_B"}).drop(columns=["id"])

    return merge

# --- Create Paired Data Sets wrt EmoBank splits ---
 
print("Preparing data...")
def prepare_pairs(split: str, V, A, D, emobank, seed_num: int = 42):
    """
    Prepare pairs from emopair/emobank dataset with V/A/D probabilities.
    
    Args:
        split: 'dev' or 'test' split from emopair
        V, A, D: DataFrames with pairwise probabilities
        emobank: DataFrame with text data
        seed_num: random seed for reproducibility
        
    Returns:
        DataFrame with pairs and probabilities for all three dimensions
    """
    set_seeds(seed_num)
    num_pairs_per_item = 10
    if split == 'train':
        num_pairs_per_item = 15

    # Get IDs 
    emobank_subset = emobank[emobank['split'] == split].copy()
    
    subset_ids = emobank_subset['id'].to_list()
    subset_pairs = pdm.pair_items(subset_ids, 
                                  num_pairs_per_item=num_pairs_per_item,
                                  random_seed=seed_num)
    
    # Rename for consistency
    subset_pairs.rename(columns={'item1': 'Item_A', 'item2': 'Item_B'}, inplace=True)
    subset_pairs = join_texts(subset_pairs, emobank)
    
    # Create pair keys for matching
    subset_pairs['pair_key'] = [
        tuple(sorted(pair)) 
        for pair in zip(subset_pairs['Item_A'], subset_pairs['Item_B'])
    ]
    subset_pair_keys = set(subset_pairs['pair_key'])
    
    # Merge V, A, D probabilities
    for df, dim_name in zip([V, A, D], ['V', 'A', 'D']):
        # Create pair keys
        df['pair_key'] = [
            tuple(sorted(pair)) 
            for pair in zip(df['Item_A'], df['Item_B'])
        ]
        
        # Filter to only subset pairs
        df_subset = df[df['pair_key'].isin(subset_pair_keys)].copy()
        
        # Clean up and rename columns
        df_subset = df_subset[['pair_key', 'Probability_A_greater_B', 'A_better_indicator']].copy()
        df_subset.rename(columns={
            'Probability_A_greater_B': f'Probability_A_greater_B_{dim_name}',
            'A_better_indicator': f'A_better_indicator_{dim_name}'
        }, inplace=True)
        
        # Merge on pair_key
        subset_pairs = subset_pairs.merge(df_subset, on='pair_key', how='left')
    
    # Drop pair_key column
    subset_pairs.drop(columns=['pair_key'], inplace=True)
    
    return subset_pairs

train_set_pairs = prepare_pairs('train', V, A, D, emobank, seed_num=seed_num)
print(f"Training set prepared. Number of pairs: {len(train_set_pairs)}")

eval_set_pairs = prepare_pairs('dev', V, A, D, emobank, seed_num=seed_num)
print(f"Evaluation set prepared. Number of pairs: {len(eval_set_pairs)}")

test_set_pairs = prepare_pairs('test', V, A, D, emobank, seed_num=seed_num)
print(f"Test set prepared. Number of pairs: {len(test_set_pairs)}")

# --- End Paired Data Sets wrt EmoBank splits ---

def make_tuples(data):
    """Convert dataframe to list of tuples with all three probabilities."""
    return list(
        zip(
            data['Text_A'].tolist(),
            data['Text_B'].tolist(),
            data['Probability_A_greater_B_V'].tolist(),
            data['Probability_A_greater_B_A'].tolist(),
            data['Probability_A_greater_B_D'].tolist()
        )
    )

class RewardModel:
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
        self.training_history = []
    
    def _build_model(self, dropout: float) -> nn.Module:
        """Build the reward model architecture."""
        encoder = AutoModel.from_pretrained(self.model_name)
        hidden_size = encoder.config.hidden_size
        
        # Create complete model
        model = nn.Module()
        model.encoder = encoder
        # Output 3 scores: Valence, Arousal, Dominance
        model.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # Output V, A, D scores
        )
        
        # Add forward method
        def forward(input_ids, attention_mask):
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]
            rewards = model.reward_head(pooled_output)  # Shape: (batch_size, 3)
            return rewards
        
        model.forward = forward
        return model
    
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
    
    def _prepare_pairadigm(
            self,
            decision_col: str = "decision", 
            score_col: str = "Bradley_Terry_Score",
            margins: bool = True):
        """
        Prepare pairwise data from linked Pairadigm instance.
        Args:
            decision_col: Column in pairwise_df indicating winner/loser
            score_col: Column in scored_df with precomputed scores
            margins: Whether to compute margins from scores
        Returns:
            Tuples with pairs and margins
        """
        if self.pairadigm is None:
            raise ValueError("No Pairadigm instance linked to RewardModel.")
        if decision_col not in self.pairadigm.pairwise_df.columns:
            raise ValueError(f"Decision column '{decision_col}' not found in pairwise_df. Run generate_pairwise_annotations() first.")
        if "breakdown1" not in self.pairadigm.pairwise_df.columns or "breakdown2" not in self.pairadigm.pairwise_df.columns:
            raise ValueError("Breakdown columns not found in pairwise_df. Run generate_breakdowns() and generate_pairings(breakdowns=True) first.")

        pairwise_df = self.pairadigm.pairwise_df.copy()

        original_text = self.pairadigm.data[[self.pairadigm.item_id_name, 
                                             self.pairadigm.text_name]].copy()
        
        pairwise_df = pd.merge(
            pairwise_df, original_text,
            left_on='item1',
            right_on=self.pairadigm.item_id_name,
            how='left').rename(columns={self.pairadigm.text_name: 'text1'}).drop(columns=[self.pairadigm.item_id_name]).merge(
                original_text,
                left_on='item2',
                right_on=self.pairadigm.item_id_name,
                how='left'
                ).rename(columns={self.pairadigm.text_name: 'text2'}).drop(columns=[self.pairadigm.item_id_name])
        
        # Calculate margins when desired, if scores are available
        if margins: 
            if not self.pairadigm.scored_df:
                raise ValueError("No scored_df found in linked Pairadigm instance.")
            if score_col not in self.pairadigm.scored_df.columns:
                raise ValueError(f"Score column '{score_col}' not found in scored_df. Run score_items() first.")

            pairs = pd.merge(
                pairwise_df[['item1', 'item2', 
                             'text1', 'text2', decision_col]],
                self.pairadigm.scored_df[[self.pairadigm.item_id_name, score_col]],
                    left_on='item1',
                    right_on=self.pairadigm.item_id_name,
                    how='left'
                ).rename(columns={score_col: 'item_A_score'}).drop(columns=[self.pairadigm.item_id_name]).merge(
                    self.pairadigm.scored_df[[self.pairadigm.item_id_name, score_col]],
                    left_on='item2',
                    right_on=self.pairadigm.item_id_name,
                    how='left'
                ).rename(columns={score_col: 'item_B_score'}).drop(columns=[self.pairadigm.item_id_name])
            
            pairs['margin'] = pairs['item_B_score'] - pairs['item_A_score']

            # Make a tuple of breakdown1, breakdown2, margin for each row in prepped_data
            training_pairs = list(zip(
                pairs['text1'],
                pairs['text2'],
                pairs['margin']
            ))

            return training_pairs
        
        # If margins not desired, just return winner/loser pairs
        training_pairs = []
        for _, row in self.pairadigm.pairwise_df.iterrows():
            if row[decision_col] == 1 or row[decision_col] == 'Text1':
                training_pairs.append((row['text1'], row['text2'], 1.0))
            elif row[decision_col] == 2 or row[decision_col] == 'Text2':
                training_pairs.append((row['text2'], row['text1'], 1.0))
            else:
                continue  # Skip ties or invalid decisions

        return training_pairs
    
    class _PairwiseDataset(Dataset):
        """Internal dataset class for pairwise comparisons with V/A/D probabilities."""
        
        def __init__(self, pairs, tokenizer, max_length):
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            # Expect tuples: (text_A, text_B, prob_V, prob_A, prob_D)
            text_A, text_B, prob_V, prob_A, prob_D = self.pairs[idx]
            
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
                'prob_V': torch.tensor(prob_V, dtype=torch.float),
                'prob_A': torch.tensor(prob_A, dtype=torch.float),
                'prob_D': torch.tensor(prob_D, dtype=torch.float)
            }
    
    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        epochs: int = 5,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        log_interval: int = 50,
        early_stopping_patience: int = 3
    ):
        """
        Train the reward model on pairwise comparison data with optional early stopping.

        Args:
            train_loader: DataLoader with training pairs
            eval_loader: DataLoader for evaluation (required for early stopping)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps for scheduler
            log_interval: Log metrics every N steps
            early_stopping_patience: Number of epochs with no improvement on eval loss before stopping early.
                                     Set to None or 0 to disable early stopping.
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
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Early stopping bookkeeping
        best_state = None
        best_eval_loss = float('inf')
        epochs_without_improve = 0
        use_early_stopping = bool(early_stopping_patience and eval_loader is not None and early_stopping_patience > 0)
        
        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            
            for step, batch in enumerate(progress_bar):
                loss = self._training_step(batch)
                epoch_loss += loss
                
                if (step + 1) % log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            epoch_metrics = {'epoch': epoch + 1, 'train_loss': avg_epoch_loss}
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f}")
            
            # Evaluation
            if eval_loader:
                eval_metrics = self.evaluate(eval_loader)
                epoch_metrics.update(eval_metrics)
                print(f"Epoch {epoch + 1} - Eval Loss: {eval_metrics['eval_loss']:.4f}")
                print(f"  V: {eval_metrics['eval_loss_V']:.4f} | A: {eval_metrics['eval_loss_A']:.4f} | D: {eval_metrics['eval_loss_D']:.4f}")
                
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
    
    def _training_step(self, batch) -> float:
        """Single training step with V/A/D dimensions."""
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {key: value.to(self.device) for key, value in batch.items()}
        
        # Get rewards for A and B (shape: batch_size x 3 for V, A, D)
        rewards_A = self.model(
            batch['input_ids_A'],
            batch['attention_mask_A']
        )
        rewards_B = self.model(
            batch['input_ids_B'],
            batch['attention_mask_B']
        )
        
        # Compute margins: score_A - score_B for each dimension
        margin_V = rewards_A[:, 0] - rewards_B[:, 0]
        margin_A = rewards_A[:, 1] - rewards_B[:, 1]
        margin_D = rewards_A[:, 2] - rewards_B[:, 2]
        
        # Use probability as a margin: convert prob to log-odds for stronger signal
        # prob ranges [0, 1], where 0.5 is neutral, >0.5 means A is better
        # Transform: log(prob / (1 - prob)) gives range (-inf, +inf)
        prob_V = batch['prob_V'].clamp(1e-7, 1 - 1e-7)  # Clamp to avoid log(0)
        prob_A = batch['prob_A'].clamp(1e-7, 1 - 1e-7)
        prob_D = batch['prob_D'].clamp(1e-7, 1 - 1e-7)
        
        target_margin_V = torch.log(prob_V / (1 - prob_V))
        target_margin_A = torch.log(prob_A / (1 - prob_A))
        target_margin_D = torch.log(prob_D / (1 - prob_D))
        
        # MSE loss on margins for each dimension
        loss_V = torch.nn.functional.mse_loss(margin_V, target_margin_V)
        loss_A = torch.nn.functional.mse_loss(margin_A, target_margin_A)
        loss_D = torch.nn.functional.mse_loss(margin_D, target_margin_D)
        
        # Weighted loss (can adjust weights based on importance)
        total_loss = loss_V + loss_A + loss_D
        
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    
        return total_loss.item()
    
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a validation set.
        
        Args:
            eval_loader: DataLoader with evaluation pairs
            
        Returns:
            Dictionary with evaluation metrics for V, A, D
        """
        self.model.eval()
        total_loss = 0
        losses_by_dim = {'V': 0, 'A': 0, 'D': 0}
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                
                rewards_A = self.model(
                    batch['input_ids_A'],
                    batch['attention_mask_A']
                )
                rewards_B = self.model(
                    batch['input_ids_B'],
                    batch['attention_mask_B']
                )
                
                # Compute margins for each dimension
                margin_V = rewards_A[:, 0] - rewards_B[:, 0]
                margin_A = rewards_A[:, 1] - rewards_B[:, 1]
                margin_D = rewards_A[:, 2] - rewards_B[:, 2]
                
                # Target margins from probabilities
                prob_V = batch['prob_V'].clamp(1e-7, 1 - 1e-7)
                prob_A = batch['prob_A'].clamp(1e-7, 1 - 1e-7)
                prob_D = batch['prob_D'].clamp(1e-7, 1 - 1e-7)
                
                target_margin_V = torch.log(prob_V / (1 - prob_V))
                target_margin_A = torch.log(prob_A / (1 - prob_A))
                target_margin_D = torch.log(prob_D / (1 - prob_D))
                
                # Losses for each dimension
                loss_V = torch.nn.functional.mse_loss(margin_V, target_margin_V)
                loss_A = torch.nn.functional.mse_loss(margin_A, target_margin_A)
                loss_D = torch.nn.functional.mse_loss(margin_D, target_margin_D)
                
                batch_loss = loss_V + loss_A + loss_D
                total_loss += batch_loss.item()
                losses_by_dim['V'] += loss_V.item()
                losses_by_dim['A'] += loss_A.item()
                losses_by_dim['D'] += loss_D.item()
        
        self.model.train()
        return {
            'eval_loss': total_loss / len(eval_loader),
            'eval_loss_V': losses_by_dim['V'] / len(eval_loader),
            'eval_loss_A': losses_by_dim['A'] / len(eval_loader),
            'eval_loss_D': losses_by_dim['D'] / len(eval_loader)
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
            rewards = self.model(
                encoding['input_ids'].to(self.device),
                encoding['attention_mask'].to(self.device)
            )
        
        return {
            'valence': rewards[0, 0].item(),
            'arousal': rewards[0, 1].item(),
            'dominance': rewards[0, 2].item()
        }
    
    def score_batch(self, texts: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Score multiple texts efficiently across all three dimensions.
        
        Args:
            texts: List of texts to score
            batch_size: Batch size for processing
            
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
                rewards = self.model(
                    encodings['input_ids'].to(self.device),
                    encodings['attention_mask'].to(self.device)
                )
            
            scores_V.extend(rewards[:, 0].cpu().numpy())
            scores_A.extend(rewards[:, 1].cpu().numpy())
            scores_D.extend(rewards[:, 2].cpu().numpy())
        
        return {
            'valence': np.array(scores_V),
            'arousal': np.array(scores_A),
            'dominance': np.array(scores_D)
        }
    
    def normalize_scores(
        self,
        scores: np.ndarray,
        scale_min: float = 0.0,
        scale_max: float = 1.0
    ) -> np.ndarray:
        """
        Normalize raw reward scores to a desired scale.
        
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
        """
        Evaluate the model on the test set and print detailed results.
        
        Args:
            test_loader: DataLoader with test pairs
            
        Returns:
            Dictionary with test metrics
        """
        print("\n" + "="*60)
        print("Running Test Evaluation (V, A, D)")
        print("="*60)
        
        self.model.eval()
        total_loss = 0
        losses_by_dim = {'V': 0, 'A': 0, 'D': 0}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                
                rewards_A = self.model(
                    batch['input_ids_A'],
                    batch['attention_mask_A']
                )
                rewards_B = self.model(
                    batch['input_ids_B'],
                    batch['attention_mask_B']
                )
                
                # Compute margins for each dimension
                margin_V = rewards_A[:, 0] - rewards_B[:, 0]
                margin_A = rewards_A[:, 1] - rewards_B[:, 1]
                margin_D = rewards_A[:, 2] - rewards_B[:, 2]
                
                # Target margins from probabilities
                prob_V = batch['prob_V'].clamp(1e-7, 1 - 1e-7)
                prob_A = batch['prob_A'].clamp(1e-7, 1 - 1e-7)
                prob_D = batch['prob_D'].clamp(1e-7, 1 - 1e-7)
                
                target_margin_V = torch.log(prob_V / (1 - prob_V))
                target_margin_A = torch.log(prob_A / (1 - prob_A))
                target_margin_D = torch.log(prob_D / (1 - prob_D))
                
                # Losses for each dimension
                loss_V = torch.nn.functional.mse_loss(margin_V, target_margin_V)
                loss_A = torch.nn.functional.mse_loss(margin_A, target_margin_A)
                loss_D = torch.nn.functional.mse_loss(margin_D, target_margin_D)
                
                batch_loss = loss_V + loss_A + loss_D
                total_loss += batch_loss.item()
                losses_by_dim['V'] += loss_V.item()
                losses_by_dim['A'] += loss_A.item()
                losses_by_dim['D'] += loss_D.item()
        
        # Calculate metrics
        test_loss = total_loss / len(test_loader)
        test_loss_V = losses_by_dim['V'] / len(test_loader)
        test_loss_A = losses_by_dim['A'] / len(test_loader)
        test_loss_D = losses_by_dim['D'] / len(test_loader)
        
        # Print results
        print(f"\nTest Results:")
        print(f"  Total Loss: {test_loss:.4f}")
        print(f"  Valence Loss: {test_loss_V:.4f}")
        print(f"  Arousal Loss: {test_loss_A:.4f}")
        print(f"  Dominance Loss: {test_loss_D:.4f}")
        print("\n" + "="*60 + "\n")
        
        results = {
            'test_loss': test_loss,
            'test_loss_V': test_loss_V,
            'test_loss_A': test_loss_A,
            'test_loss_D': test_loss_D
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
            # Save encoder and reward head
            self.model.encoder.save_pretrained(os.path.join(temp_dir, "encoder"))
            self.tokenizer.save_pretrained(temp_dir)
            torch.save(self.model.reward_head.state_dict(), os.path.join(temp_dir, "reward_head.pt"))
            
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
learning_rate = 5e-6
weight_decay = 0.2
epochs = 15
patience = 5
dropout = 0.6
print(f"Training parameters - Epochs: {epochs}, LR: {learning_rate}, Weight Decay: {weight_decay}, Patience: {patience}, Dropout: {dropout}")

print("Initializing Reward Model...")
reward_model = RewardModel(
    model_name="roberta-large",
    dropout=dropout,
    max_length=128,
    device=device 
)
print("Reward Model initialized.")

print("Preparing data loaders...")
num_workers = 4 if torch.cuda.is_available() else 0

# Convert to tuples format
train_tuples = make_tuples(train_set_pairs)
eval_tuples = make_tuples(eval_set_pairs)
test_tuples = make_tuples(test_set_pairs)

# Create datasets manually
train_dataset = reward_model._PairwiseDataset(train_tuples, reward_model.tokenizer, reward_model.max_length)
eval_dataset = reward_model._PairwiseDataset(eval_tuples, reward_model.tokenizer, reward_model.max_length)
test_dataset = reward_model._PairwiseDataset(test_tuples, reward_model.tokenizer, reward_model.max_length)

# Create dataloaders with fixed generator for reproducibility
generator = torch.Generator()
generator.manual_seed(seed_num)

data_loaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, 
                       generator=generator, worker_init_fn=lambda _: set_seeds(seed_num)),
    'eval': DataLoader(eval_dataset, batch_size=32, shuffle=False),
    'test': DataLoader(test_dataset, batch_size=32, shuffle=False)
}

print("Data loaders ready.")
print(f"Train pairs: {len(train_tuples)}")
print(f"Eval pairs: {len(eval_tuples)}")
print(f"Test pairs: {len(test_tuples)}")

print("Starting training...")
date = datetime.now().strftime("%Y%m%d")
try:
    emobank_reward = reward_model.train(
        train_loader=data_loaders['train'],
        eval_loader=data_loaders['eval'],
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=100,
        log_interval=50,
        early_stopping_patience=patience
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
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory freed. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)}")
print("Memory cleanup complete.")