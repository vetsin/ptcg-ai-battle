from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from agent.features import (
    NUM_STATE_FEATURES,
    NUM_OPTION_FEATURES,
    encode_state,
    encode_option,
    encode_observation,
    _zero_state,
    _zero_option,
)
from cg.api import to_observation_class


class PTCGPolicyNet(nn.Module):
    def __init__(
        self,
        state_dim: int = NUM_STATE_FEATURES,
        option_dim: int = NUM_OPTION_FEATURES,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.option_encoder = nn.Sequential(
            nn.Linear(option_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_features: torch.Tensor, option_features: torch.Tensor, option_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        state_enc = self.state_encoder(state_features)
        state_enc = state_enc.unsqueeze(1)

        B, N, D = option_features.shape
        option_enc = self.option_encoder(option_features.view(B * N, D)).view(B, N, -1)

        attn_out, _ = self.attention(
            option_enc, option_enc, option_enc,
            key_padding_mask=option_mask,
        )
        combined_state = state_enc.expand(-1, N, -1)
        pair_features = torch.cat([combined_state, attn_out], dim=-1)
        action_scores = self.action_head(pair_features).squeeze(-1)
        pooled_state = state_enc.squeeze(1)
        value = self.value_head(pooled_state).squeeze(-1)

        if option_mask is not None:
            action_scores = action_scores.masked_fill(option_mask, float('-inf'))

        return action_scores, value


class PTCGSimpleNet(nn.Module):
    def __init__(
        self,
        state_dim: int = NUM_STATE_FEATURES,
        option_dim: int = NUM_OPTION_FEATURES,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.option_encoder = nn.Sequential(
            nn.Linear(option_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_features: torch.Tensor, option_features: torch.Tensor, option_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        state_enc = self.state_encoder(state_features)
        option_enc = self.option_encoder(option_features)

        B, N, D = option_enc.shape
        state_expanded = state_enc.unsqueeze(1).expand(-1, N, -1)
        pair_features = torch.cat([state_expanded, option_enc], dim=-1)
        action_scores = self.action_head(pair_features).squeeze(-1)
        value = self.value_head(state_enc).squeeze(-1)

        if option_mask is not None:
            action_scores = action_scores.masked_fill(option_mask, float('-inf'))

        return action_scores, value


class PTCGDataset(Dataset):
    def __init__(self, data: list[dict], max_options: int = 20):
        self.samples = []
        self.max_options = max_options

        for record in data:
            state_features, option_features_list, action_indices, reward = self._process_record(record)
            if state_features is not None and option_features_list:
                self.samples.append((state_features, option_features_list, action_indices, reward))

    def _process_record(self, record: dict):
        state_features = record.get("state_features")
        option_features = record.get("option_features")
        action = record.get("action", [])
        reward = record.get("reward", 0.0)

        if state_features is None or option_features is None:
            return None, None, None, 0.0

        return state_features, option_features, action, reward

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state_feat, option_feats, action_idx, reward = self.samples[idx]

        state_tensor = torch.tensor(state_feat, dtype=torch.float32)

        num_options = len(option_feats)
        padded_options = torch.zeros(self.max_options, NUM_OPTION_FEATURES, dtype=torch.float32)
        mask = torch.ones(self.max_options, dtype=torch.bool)

        for i, opt in enumerate(option_feats[:self.max_options]):
            padded_options[i] = torch.tensor(opt, dtype=torch.float32)
            mask[i] = False

        action_target = 0
        if action_idx:
            a = action_idx[0] if isinstance(action_idx, list) else action_idx
            action_target = min(max(a, 0), num_options - 1, self.max_options - 1)

        return state_tensor, padded_options, mask, action_target, num_options, reward


def collate_fn(batch):
    states = torch.stack([b[0] for b in batch])
    options = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    actions = torch.tensor([b[3] for b in batch], dtype=torch.long)
    num_options = torch.tensor([b[4] for b in batch], dtype=torch.long)
    rewards = torch.tensor([b[5] for b in batch], dtype=torch.float32)
    return states, options, masks, actions, num_options, rewards


def load_trajectory_data(data_dir: str | Path) -> list[dict]:
    data_dir = Path(data_dir)
    records = []

    for jsonl_file in data_dir.rglob("*.jsonl"):
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue

    return records


def prepare_training_data(trajectories_dir: str | Path, max_options: int = 20) -> list[dict]:
    from agent.selfplay import GameTrajectory

    trajectories_dir = Path(trajectories_dir)
    all_records = []

    for jsonl_file in trajectories_dir.rglob("all_trajectories.jsonl"):
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    steps = record.get("steps", [])
                    reward = record.get("reward", 0.0)
                    outcome = record.get("outcome", -1)
                    my_index = record.get("my_index", 0)

                    for step in steps:
                        train_record = {
                            "game_id": record.get("game_id", ""),
                            "reward": reward,
                            "outcome": outcome,
                            "my_index": my_index,
                            "select_type": step.get("select_type", 0),
                            "select_context": step.get("select_context", 0),
                            "num_options": step.get("num_options", 1),
                            "action": step.get("action", []),
                            "state_score": step.get("state_score", 0.0),
                            "turn": step.get("turn", 0),
                            "difficulty": record.get("difficulty", 0.0),
                        }
                        all_records.append(train_record)
                except (json.JSONDecodeError, KeyError):
                    continue

    return all_records


def train_policy(
    train_data: list[dict],
    val_data: list[dict] | None = None,
    model_type: str = "simple",
    hidden_dim: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 20,
    batch_size: int = 64,
    max_options: int = 20,
    device: str = "auto",
    reward_weight: float = 0.5,
    value_weight: float = 0.25,
    save_path: str | None = None,
) -> dict:
    if len(train_data) < 5:
        return {"train_loss": [0.0], "train_acc": [0.0]}

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PTCGDataset(train_data, max_options=max_options)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    if model_type == "attention":
        model = PTCGPolicyNet(hidden_dim=hidden_dim).to(device)
    else:
        model = PTCGSimpleNet(hidden_dim=hidden_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for states, options, masks, actions, num_options, rewards in dataloader:
            B = states.size(0)
            states = states.to(device)
            options = options.to(device)
            masks = masks.to(device)
            actions = actions.to(device)
            num_options = num_options.to(device)
            rewards = rewards.to(device)

            action_scores, values = model(states, options, masks)

            for b in range(B):
                n = num_options[b].item()
                if n < action_scores.size(1):
                    action_scores[b, n:] = float('-inf')

            valid_mask = actions >= 0
            if valid_mask.sum() == 0:
                continue

            clamped_actions = actions.clamp(min=0)

            loss_action = nn.functional.cross_entropy(action_scores, clamped_actions)

            value_targets = rewards.clone()
            value_targets = value_targets.clamp(0.0, 1.0)

            loss_value = nn.functional.mse_loss(values, value_targets)

            loss = loss_action + reward_weight * loss_value

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * B
            predicted = action_scores.argmax(dim=-1)
            correct = (predicted == clamped_actions).sum().item()
            total_correct += correct
            total_samples += B

        scheduler.step()

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(accuracy)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={accuracy:.3f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "hidden_dim": hidden_dim,
            "max_options": max_options,
            "state_dim": NUM_STATE_FEATURES,
            "option_dim": NUM_OPTION_FEATURES,
            "history": history,
        }, save_path)

    return history


def score_actions(
    model: nn.Module,
    state_features: list[float],
    option_features: list[list[float]],
    max_options: int = 20,
    device: str = "cpu",
) -> tuple[list[float], float]:
    model.eval()
    with torch.no_grad():
        state_tensor = torch.tensor([state_features], dtype=torch.float32, device=device)

        num_options = len(option_features)
        padded = torch.zeros(1, max_options, NUM_OPTION_FEATURES, dtype=torch.float32, device=device)
        mask = torch.ones(1, max_options, dtype=torch.bool, device=device)

        for i, opt in enumerate(option_features[:max_options]):
            padded[0, i] = torch.tensor(opt, dtype=torch.float32)
            mask[0, i] = False

        action_scores, value = model(state_tensor, padded, mask)

        scores = action_scores[0, :num_options].softmax(dim=-1).cpu().tolist()
        v = value[0].item()

    return scores, v


def load_model(path: str | Path, device: str = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_type = checkpoint.get("model_type", "simple")
    hidden_dim = checkpoint.get("hidden_dim", 128)

    if model_type == "attention":
        model = PTCGPolicyNet(hidden_dim=hidden_dim)
    else:
        model = PTCGSimpleNet(hidden_dim=hidden_dim)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model