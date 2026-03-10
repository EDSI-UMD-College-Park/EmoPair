import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
import random
import requests


import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaModel, get_scheduler
from torch.optim import AdamW
from accelerate import Accelerator
torch.cuda.empty_cache()

from pathlib import Path

# Script directory:
script_dir = Path(__file__).parent 

# Data Diretory for current Repo
data_path = script_dir / ".." / "Data"
results_path = script_dir / ".." / "Results"

# Resolve it to an absolute path for safety
absolute_path = data_path.resolve()

print(f"Loading Data from: {absolute_path}")

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def download_emobank(target_path: Path = Path("../Data/emobank.csv"), overwrite: bool = False):
    """Download EmoBank dataset if not present."""
    url = "https://github.com/JULIELab/EmoBank/raw/master/corpus/emobank.csv"
    if target_path.exists() and not overwrite:
        print(f"EmoBank already exists at {target_path}")
        return target_path
    else:
        print("Downloading EmoBank dataset...")
        resp = requests.get(url)
        resp.raise_for_status()
        target_path.write_text(resp.text, encoding="utf-8")
        print(f"Downloaded EmoBank dataset to {target_path}")
    return target_path


def prep_emopair(emopair_data_path):
    """Fetch Emopair data dataset if not present."""
    df = pd.read_csv(emopair_data_path)
    # Keep only the specified columns
    df = df[['item_id', 'split', 'V', 'text',
             'A_BT_score_split', 'D_BT_score_split',
             'A_BT_score_full', 'D_BT_score_full']]
    return df

class VADDataset(Dataset):
    def __init__(self, texts, vad_labels, tokenizer, max_length=256):
        self.texts = texts
        self.vad_labels = vad_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        label = torch.tensor(self.vad_labels[idx], dtype=torch.float)
        return item, label


class RoBERTaForVAD(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.base_model = RobertaModel.from_pretrained(base_model_name)
        self.dropout = nn.Dropout(p=0.1)
        self.regressor = nn.Linear(self.base_model.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        return self.regressor(cls_embedding)


def train_epoch(
    model, dataloader, optimizer, criterion, lr_scheduler, device,
    grad_accum_steps=1, accelerator=None, scaler=None
):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, leave=False)

    optimizer.zero_grad()
    for step, batch in enumerate(loop, start=1):
        inputs, targets = batch
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        targets = targets.to(device)

        # Mixed precision when scaler is provided (only when not using Accelerator)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
            # normalize loss if accumulating gradients
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()

        if step % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

        total_loss += (loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1))
        loop.set_postfix(loss=loss.item())

    # return average loss per batch (approximate when grad accumulation used)
    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            targets = targets.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def eval_with_metrics(model, dataloader, criterion, device):
    """Evaluate model on dataloader and return (val_loss, predictions, targets, extra_metrics)

    extra_metrics contains 'mae' and 'pearson' computed across flattened VAD outputs by default.
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            targets = targets.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if len(all_predictions) == 0:
        raise RuntimeError('Validation set is empty; cannot compute metrics.')

    val_loss = total_loss / len(dataloader)
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    extra = calculate_metrics(all_predictions, all_targets)
    return val_loss, all_predictions, all_targets, extra


def calculate_metrics(predictions, targets):

    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    pearson = pearsonr(targets.flatten(), predictions.flatten())[0]

    mae_per_dim = [mean_absolute_error(targets[:, i], predictions[:, i]) for i in range(3)]
    r2_per_dim = [r2_score(targets[:, i], predictions[:, i]) for i in range(3)]
    pearson_per_dim = [pearsonr(targets[:, i], predictions[:, i])[0] for i in range(3)]

    return {
        'mae': mae,
        'r2': r2,
        'mae_valence': mae_per_dim[0],
        'mae_arousal': mae_per_dim[1],
        'mae_dominance': mae_per_dim[2],
        'r2_valence': r2_per_dim[0],
        'r2_arousal': r2_per_dim[1],
        'r2_dominance': r2_per_dim[2],
        'pearson': pearson,
        'pearson_valence': pearson_per_dim[0],
        'pearson_arousal': pearson_per_dim[1],
        'pearson_dominance': pearson_per_dim[2]
    }

def make_model_dir(suffix, base_dir="."):
    timestamp = datetime.now().strftime("%m-%d-%y_%H-%M")
    folder_name = f"{timestamp}_{suffix}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


class EarlyStopping:
    """Simple early stopping utility to stop training when validation loss stops improving.

    Args:
        patience: number of epochs with no improvement after which training will be stopped.
        min_delta: minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        self.best_state = None
        self.best_epoch = None

    def step(self, current_loss: float, model_state: dict, epoch: int):
        """Call at end of an epoch. Returns True if training should stop."""
        improved = (self.best_loss - current_loss) > self.min_delta
        if improved:
            self.best_loss = float(current_loss)
            self.num_bad_epochs = 0
            # store a copy of the state dict
            self.best_state = copy.deepcopy(model_state)
            self.best_epoch = epoch
            return False
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                return True
            return False



def save_finetuned_model(model, tokenizer, model_name, suffix,
                         num_epochs, batch_size, learning_rate, criterion, optimizer,
                         val_loss, metrics, epoch_history=None, 
                         lr_scheduler=None, dir_name: str = None):
    """Save the finetuned model into dir_name. If dir_name is None, create one using suffix.

    This function is backward-compatible but prefers an explicit `dir_name` to allow
    pre-creating folders and checkpointing during training.
    """
    if dir_name is None:
        dir_name = make_model_dir(suffix, ".")

    # Save final model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'base_model_name': model_name,
            'hidden_size': model.base_model.config.hidden_size,
            'num_labels': 3
        }
    }, os.path.join(dir_name, 'pytorch_model.bin'))

    # Save tokenizer and base model config
    tokenizer.save_pretrained(dir_name)
    model.base_model.config.save_pretrained(dir_name)

    training_args = {
        'model_name': model_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'lr_scheduler': lr_scheduler.__class__.__name__ if lr_scheduler is not None else None,
        'prompted': True,
        'optimizer': optimizer.__class__.__name__,
        'criterion': criterion.__class__.__name__,
        'final_val_loss': float(val_loss),
        'metrics': {k: float(v) for k, v in metrics.items()}
    }
    with open(os.path.join(dir_name, 'training_args.json'), 'w') as f:
        json.dump(training_args, f, indent=4)

    with open(os.path.join(dir_name, 'epoch_history.json'), 'w') as f:
        json.dump(epoch_history or [], f, indent=4)

    metrics_for_json = {k: float(v) for k, v in metrics.items()}
    with open(os.path.join(dir_name, 'eval_metrics.json'), 'w') as f:
        json.dump(metrics_for_json, f, indent=4)

    print(f"Fine-tuning complete. Model and weights saved to {dir_name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for V-A-D regression")
    parser.add_argument(
        '--emobank_path', type=str, default='../Data/emobank.csv',
        help='Path to emobank.csv'
    )
    parser.add_argument('--emopair_path', 
        type=str, default='../Results/emopair.csv', 
        help='Path to emopair.csv')
    parser.add_argument(
        '--download', action='store_true',
        help='Download EmoBank if missing'
    )
    parser.add_argument(
        '--emo_label_type', type=str, default='emopair',
        help='Type of model, emobank or emopair'
    )
    parser.add_argument(
        '--model_name', type=str, default='roberta-large'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64
    )
    parser.add_argument(
        '--num_epochs', type=int, default=5
    )
    parser.add_argument(
        '--learning_rate', type=float, default=3e-05
    )
    parser.add_argument(
        '--cuda_index', type=int, default=0
    )
    parser.add_argument(
        '--max_length', type=int, default=256
    )
    parser.add_argument(
        '--seed', type=int, default=42
    )
    parser.add_argument(
        '--save_suffix', type=str, default='pertuned_VAD'
    )
    parser.add_argument(
        '--no_save', action='store_true',
        help='Do not save model after training'
    )
    # Early stopping options
    parser.add_argument(
        '--early_stop', action='store_true',
        help='Enable early stopping based on validation loss'
    )
    parser.add_argument(
        '--patience', type=int, default=2,
        help='Epochs with no improvement before stopping'
    )
    parser.add_argument(
        '--min_delta', type=float, default=0.0,
        help='Minimum change in val loss to qualify as improvement'
    )
    parser.add_argument(
        '--monitor', type=str, default='val_loss',
        choices=['val_loss', 'mae', 'pearson'],
        help='Metric to monitor for early stopping (smaller is better except pearson)'
    )
    parser.add_argument(
        '--use_accelerator', action='store_true',
        help='Use HuggingFace Accelerate for device placement and distributed training'
    )
    parser.add_argument(
        '--multi_gpu', action='store_true',
        help='Allow multi-GPU distributed run'
    )
    parser.add_argument(
        '--grad_accum_steps', type=int, default=1,
        help='Gradient accumulation steps to simulate larger batch sizes'
    )
    parser.add_argument(
        '--auto_reduce_batch', action='store_true',
        help='Auto reduce batch size on CUDA OOM and retry'
    )
    parser.add_argument(
        '--min_batch_size', type=int, default=8,
        help='Min batch size to reduce down to when auto-reducing'
    )
    parser.add_argument(
        '--mixed_precision', type=str, default='no',
        choices=['no', 'fp16', 'bf16'],
        help='Mixed precision mode (fp16 for NVIDIA GPUs to reduce memory)'
    )
    parser.add_argument(
        '--gradient_checkpointing', action='store_true',
        help='Enable gradient checkpointing on base model to save memory'
    )
    parser.add_argument(
        '--output_dir', type=str, default=str(results_path / "trained models"),
        help='Base directory where the timestamped run folder will be created'
    )
    args = parser.parse_args(argv)

    # Auto-enable Accelerator when launched under a distributed launcher (e.g., torchrun)
    # This makes it easier to run with torchrun without having to remember --use_accelerator.
    if not args.use_accelerator:
        local_rank = os.environ.get('LOCAL_RANK') or os.environ.get('RANK')
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if (local_rank is not None) or (world_size > 1):
            print(
                "Detected distributed environment (torchrun/local rank). "
                "Enabling --use_accelerator automatically."
            )
            args.use_accelerator = True

    set_seeds(args.seed)
    device, device_name = get_best_device(args.cuda_index)

   
    if args.emo_label_type == 'emobank':
        print("dataset working for this run is Emobank")
        if args.download:
            download_emobank(emobank_path)
        emobank_path = data_path / args.emobank_path        
        if not emobank_path.exists():
                raise FileNotFoundError(
            f"EmoBank not found at {emobank_path}. "
            f"Use --download to fetch it."
        )
        df = pd.read_csv(emobank_path)
        print(f"Dataset shape: {df.shape}")
        # Drop rows with NA in text or VAD columns
        if not set(['text', 'V', 'A', 'D']).issubset(df.columns):
            raise ValueError('Expected columns in the EmoBank CSV: text,V,A,D')
        df = df.dropna(subset=['text', 'V', 'A', 'D'])
        score_cols = ['V', 'A', 'D']
        test_score_cols = score_cols
    elif args.emo_label_type == 'emopair':
        print("dataset working for this run is Emopair")
        df = prep_emopair(str(results_path / args.emopair_path))
        print(f"Dataset shape: {df.shape}")
        # Drop rows with NA in text or VAD columns
        if not set(['text', 'V', 'A_BT_score_split', 'D_BT_score_split', 'A_BT_score_full', 'D_BT_score_full']).issubset(df.columns):
            raise ValueError(
                'Expected columns in Emopair CSV: '
                'text, V, A_BT_score_split, D_BT_score_split, A_BT_score_full, D_BT_score_full'
            )
        df = df.dropna(subset=['text', 'V'])
        score_cols = ['V', 'A_BT_score_split', 'D_BT_score_split']
        test_score_cols = ['V', 'A_BT_score_full', 'D_BT_score_full']
    else:
        raise ValueError(f"Unknown emo_label_type: {args.emo_label_type}. Must be 'emobank' or 'emopair'.")
        
    print(f"VAD Scores are located in columns: {score_cols}")   
    prompted_text = (
        "- Valence: The emotional valence of the context is <mask>. "
        "- Arousal: The level of arousal of the context is <mask>. "
        "- Dominance: The perceived dominance associated with the context is <mask>. "
        "Please predict the missing value for each dimension using the content provided."
    )
    df['prompted_text'] = df['text'].apply(lambda text: f"Context: '{text}' {prompted_text}")

    # Use existing split column if present, else random split
    if 'split' in df.columns:
        train_mask = df['split'] == 'train'
        val_mask = df['split'].isin(['dev', 'val', 'validation'])
        test_mask = df['split'] == 'test'
    else:
        train_mask = pd.Series([True] * len(df))
        val_mask = pd.Series([False] * len(df))
        test_mask = pd.Series([False] * len(df))
        # fallback: do a random split
        df_train, df_val = train_test_split(df, test_size=0.1, random_state=args.seed)
        df = pd.concat([df_train, df_val])
        train_mask = df.index.isin(df_train.index)
        val_mask = df.index.isin(df_val.index)

    df_train = df.loc[train_mask].dropna(subset=score_cols)
    df_val = df.loc[val_mask].dropna(subset=score_cols)
    df_test = df.loc[test_mask].dropna(subset=test_score_cols)

    texts_train = df_train['prompted_text'].tolist()
    texts_val = df_val['prompted_text'].tolist()
    texts_test = df_test['prompted_text'].tolist()
    vad_train = df_train[score_cols].values.astype(np.float32)
    vad_val = df_val[score_cols].values.astype(np.float32)
    vad_test = df_test[test_score_cols].values.astype(np.float32)

    print(f"Train: {len(texts_train)}, Val: {len(texts_val)}, Test: {len(texts_test)} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = VADDataset(texts_train, vad_train, tokenizer, max_length=args.max_length)
    val_dataset = VADDataset(texts_val, vad_val, tokenizer, max_length=args.max_length)
    test_dataset = VADDataset(texts_test, vad_test, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = RoBERTaForVAD(args.model_name)
    # If not using Accelerator, move model to device explicitly
    if not args.use_accelerator:
        model = model.to(device)

    # Enable gradient checkpointing if requested (saves memory at cost of compute)
    if args.gradient_checkpointing:
        try:
            # Works for HuggingFace models that expose gradient checkpointing API
            model.base_model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on base model.")
        except Exception:
            print("Warning: failed to enable gradient checkpointing on base model.")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # Optionally use HuggingFace Accelerator for device placement / distributed training
    accelerator = None
    if args.use_accelerator:
        if args.multi_gpu and torch.cuda.device_count() < 2:
            print("Warning: --multi_gpu requested but fewer than 2 GPUs available. Proceeding with available devices.")
        accelerator = Accelerator()
        # Prepare model, optimizer, and dataloaders. Accelerator will handle device placement.
        model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)
        device = accelerator.device

    # Create LR scheduler after accelerator.prepare so it wraps the prepared optimizer if needed
    num_training_steps = args.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Mixed precision scaler (only used when not using Accelerator)
    scaler = None
    use_scaler = (args.mixed_precision == 'fp16') and (not args.use_accelerator)
    if use_scaler:
        scaler = torch.amp.GradScaler('cuda')

    print(f"Starting training at {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}...")
    best_val_loss = float('inf')
    best_model_state = None
    epoch_history = []
    # Setup early stopping if requested
    early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta) if args.early_stop else None
    best_epoch = None

    # Create a directory for this training run up-front so we can checkpoint each epoch
    updated_suffix = f"{args.save_suffix}_{args.model_name}_{args.emo_label_type}"
    run_dir = make_model_dir(updated_suffix, args.output_dir)
    print(f"Run directory: {run_dir}")

    # Save tokenizer and base config once at start
    tokenizer.save_pretrained(run_dir)
    model.base_model.config.save_pretrained(run_dir)

    # Save initial training args (so we know the config if training is interrupted)
    initial_args = {
        'model_name': args.model_name,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length,
        'seed': args.seed,
        'monitor': args.monitor,
        'early_stop': args.early_stop,
        'patience': args.patience,
        'min_delta': args.min_delta
    }
    with open(os.path.join(run_dir, 'training_args.json'), 'w') as f:
        json.dump(initial_args, f, indent=4)
    # Wrap training in OOM-handling loop if requested
    current_batch_size = args.batch_size
    epoch_start = 0
    while epoch_start < args.num_epochs:
        try:
            for epoch in range(epoch_start, args.num_epochs):
                train_loss = train_epoch(
                    model, train_loader, optimizer, criterion,
                    lr_scheduler, device,
                    grad_accum_steps=args.grad_accum_steps,
                    accelerator=accelerator, scaler=scaler
                )
                # Use eval_with_metrics to get extra metrics for monitoring
                val_loss, _, _, extra = eval_with_metrics(model, val_loader, criterion, device)
                epoch_history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss, **{k: float(v) for k, v in extra.items()}})

                # Track best model state according to validation loss (keep this canonical)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1

                # Save checkpoint for this epoch (both per-epoch and 'last') so we can resume/inspect
                epoch_ckpt = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': float(val_loss)
                }
                epoch_file = os.path.join(run_dir, f'pytorch_model_epoch_{epoch+1}.bin')
                last_file = os.path.join(run_dir, 'pytorch_model_last.bin')
                torch.save(epoch_ckpt, epoch_file)
                torch.save(epoch_ckpt, last_file)

                # Update epoch history and metrics files on disk immediately
                with open(os.path.join(run_dir, 'epoch_history.json'), 'w') as f:
                    json.dump(epoch_history, f, indent=4)

                # Write eval metrics summary for this epoch (use `extra` metrics)
                metrics_for_json = {k: float(v) for k, v in extra.items()}
                metrics_for_json.update({'val_loss': float(val_loss)})
                with open(os.path.join(run_dir, 'eval_metrics.json'), 'w') as f:
                    json.dump(metrics_for_json, f, indent=4)

                # Determine the monitored value (smaller better). For 'pearson' larger is better so invert by -pearson.
                if args.monitor == 'val_loss':
                    monitored_value = val_loss
                elif args.monitor == 'mae':
                    monitored_value = float(extra['mae'])
                elif args.monitor == 'pearson':
                    monitored_value = -float(extra['pearson'])
                else:
                    monitored_value = val_loss

                print(f"Epoch {epoch + 1}/{args.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - MAE: {extra['mae']:.4f} - Pearson: {extra['pearson']:.4f}")

                # Check early stopping (smaller monitored_value is treated as improvement)
                if early_stopper is not None:
                    should_stop = early_stopper.step(monitored_value, model.state_dict(), epoch + 1)
                    if should_stop:
                        print(f"Early stopping triggered (no improvement for {args.patience} epochs) monitoring '{args.monitor}'.")
                        # Prefer restoring the best state kept by the early stopper
                        if early_stopper.best_state is not None:
                            best_model_state = early_stopper.best_state
                            # Update best_val_loss only if stopper recorded a best
                            try:
                                best_val_loss = float(early_stopper.best_loss)
                            except Exception:
                                pass
                            best_epoch = early_stopper.best_epoch
                        epoch_start = args.num_epochs
                        break

            # Completed training loop successfully
            break

        except RuntimeError as e:
            # Handle CUDA OOM by reducing batch size and 
            # retrying (only when requested and not using Accelerator)
            msg = str(e)
            if ('out of memory' in msg.lower() or 'cuda out of memory' in msg.lower()) and args.auto_reduce_batch and not args.use_accelerator:
                # free cached memory and try to reduce batch size
                print("CUDA out of memory detected. Attempting to reduce batch size and retry...")
                torch.cuda.empty_cache()
                new_batch_size = max(args.min_batch_size, current_batch_size // 2)
                if new_batch_size >= current_batch_size:
                    print("Cannot reduce batch size further. Reraising OOM error.")
                    raise
                print(f"Reducing batch size: {current_batch_size} -> {new_batch_size}")
                current_batch_size = new_batch_size
                # Recreate dataloaders with new batch size
                train_loader = DataLoader(train_dataset, batch_size=current_batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=current_batch_size)
                test_loader = DataLoader(test_dataset, batch_size=current_batch_size)
                # If accelerator was used, re-prepare; otherwise move model to device
                if accelerator is not None:
                    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)
                    device = accelerator.device
                else:
                    model = model.to(device)
                # Recreate LR scheduler to match new dataloader length
                num_training_steps = args.num_epochs * len(train_loader)
                lr_scheduler = get_scheduler(
                    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
                )
                # Continue the while loop which will retry from epoch_start 
                # (we keep epoch_history so far)
                continue
            else:
                # Not an OOM we can handle; re-raise
                raise

    # Restore best model state before evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state found. Using current model state.")

    # If not using Accelerator, ensure model is on device for evaluation
    if not args.use_accelerator:
        model = model.to(device)

    model.eval()

    # --- Validation set evaluation ---
    _, _, _, val_metrics = eval_with_metrics(model, val_loader, criterion, device)
    print(f"Validation Metrics: {val_metrics}")

    # --- Test set evaluation ---
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            targets = targets.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    if len(all_predictions) == 0:
        print("Warning: Test set is empty; skipping test metrics.")
        metrics = val_metrics
    else:
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        metrics = calculate_metrics(all_predictions, all_targets)
        print(f"Test Metrics: {metrics}")
        with open(os.path.join(run_dir, 'test_metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)

    if not args.no_save:
        save_finetuned_model(model=model, tokenizer=tokenizer, model_name=args.model_name,
                             suffix=args.save_suffix, num_epochs=args.num_epochs,
                             batch_size=args.batch_size, learning_rate=args.learning_rate,
                             criterion=criterion, optimizer=optimizer, val_loss=best_val_loss,
                             metrics=metrics, epoch_history=epoch_history, lr_scheduler=lr_scheduler,
                             dir_name=run_dir)
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Best epoch: {best_epoch}, best_val_loss: {best_val_loss:.6f}. Monitor: {args.monitor}")


if __name__ == '__main__':
    # Local import placed here to avoid top-level dependency unless running script
    from sklearn.model_selection import train_test_split
    main()
