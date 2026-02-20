# Regularization Strategies for Your Reward Model
Given you're overfitting at epoch 2 with dropout=0.4 and weight_decay=0.2, here are targeted strategies:

## 1. Label Smoothing for Bradley-Terry Loss
Your current cross-entropy loss treats probabilities as ground truth. Add smoothing to prevent overconfidence:

```{python}
def _training_step(self, batch, accumulation_steps: int = 1) -> float:
    # ...existing code...
    
    # Add label smoothing (e.g., epsilon=0.1)
    label_smoothing = 0.1
    prob_A_smooth = prob_A * (1 - label_smoothing) + 0.5 * label_smoothing
    prob_D_smooth = prob_D * (1 - label_smoothing) + 0.5 * label_smoothing
    
    # Use smoothed probs in BT loss
    per_sample_loss_A = -(prob_A_smooth * torch.log(torch.sigmoid(logit_A) + eps) + 
               (1 - prob_A_smooth) * torch.log(torch.sigmoid(-logit_A) + eps))
    # ...existing code...
    ```

## 2. Layer-wise Learning Rate Decay (LLRD)
Lower layers should update very slowly — they hold general linguistic knowledge:

```{python}
def _setup_optimizer(self, learning_rate, weight_decay):
    encoder_layers = self.model.encoder.encoder.layer
    num_layers = len(encoder_layers)
    decay_rate = 0.85  # Each layer gets 85% of the lr of the layer above
    
    param_groups = []
    
    # Embedding layers — very low lr
    param_groups.append({
        'params': self.model.encoder.embeddings.parameters(),
        'lr': learning_rate * (decay_rate ** num_layers)
    })
    
    # Per-layer decay
    for i, layer in enumerate(encoder_layers):
        param_groups.append({
            'params': layer.parameters(),
            'lr': learning_rate * (decay_rate ** (num_layers - i))
        })
    
    # Task heads — full lr
    param_groups.append({'params': self.model.valence_head.parameters(), 'lr': learning_rate})
    param_groups.append({'params': self.model.arousal_dominance_head.parameters(), 'lr': learning_rate})
    
    return AdamW(param_groups, weight_decay=weight_decay)
```

Then in train():

```{python}
self.optimizer = self._setup_optimizer(learning_rate, weight_decay)
```

## 3. Freeze Encoder Early Epochs + Gradual Unfreezing
You have this commented out — uncomment and tune it. It's one of the most effective strategies:

```{python}
# In train(), replace the commented-out block:
if epoch == 0:
    # Train only heads
    for param in self.model.encoder.parameters():
        param.requires_grad = False
elif epoch == 1:
    # Unfreeze top 4 transformer layers
    for i, layer in enumerate(self.model.encoder.encoder.layer):
        if i >= len(self.model.encoder.encoder.layer) - 4:
            for param in layer.parameters():
                param.requires_grad = True
elif epoch == 2:
    # Unfreeze top 8 layers
    for i, layer in enumerate(self.model.encoder.encoder.layer):
        if i >= len(self.model.encoder.encoder.layer) - 8:
            for param in layer.parameters():
                param.requires_grad = True
else:
    # Full fine-tuning
    for param in self.model.encoder.parameters():
        param.requires_grad = True
```

## 4. Stochastic Weight Averaging (SWA)
Average weights across late epochs instead of just taking the best checkpoint:

```{python}
# After initializing optimizer in train():
from torch.optim.swa_utils import AveragedModel, SWALR

swa_model = AveragedModel(self.model)
swa_start_epoch = epochs // 2  # Start SWA at halfway point
swa_scheduler = SWALR(self.optimizer, swa_lr=learning_rate * 0.1)

# In the epoch loop, after evaluation:
if epoch >= swa_start_epoch:
    swa_model.update_parameters(self.model)
    swa_scheduler.step()

# After training, update batch norm and use swa_model
torch.optim.swa_utils.update_bn(train_loader, swa_model)
self.model = swa_model.module
```

## 5. R-Drop Regularization
Forces consistency between two forward passes with different dropout masks:

```{python}
# In _training_step, run forward twice and add KL divergence:
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    val_A1, ad_A1 = self.model(batch['input_ids_A'], batch['attention_mask_A'])
    val_A2, ad_A2 = self.model(batch['input_ids_A'], batch['attention_mask_A'])  # second pass
    
    # R-Drop KL consistency loss
    rdrop_alpha = 0.5
    kl_loss = nn.functional.kl_div(
        torch.log_softmax(ad_A1, dim=-1),
        torch.softmax(ad_A2, dim=-1),
        reduction='batchmean'
    )
    total_loss = (loss_V + loss_A + loss_D + rdrop_alpha * kl_loss) / accumulation_steps
```

Quick Parameter Recommendations
Parameter	Current	Suggested
learning_rate	1e-5	5e-6 with LLRD
dropout	0.4	0.3 (LLRD compensates)
weight_decay	0.2	0.1 (LLRD compensates)
warmup_steps	~6%	10-15%
patience	11	5 (with SWA you don't need long patience)

Most impactful in order: LLRD → Gradual Unfreezing → Label Smoothing → SWA