import os
import shutil
from pathlib import Path
import av
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

BASE_DIR = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.parent / "data").exists() else Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "data" / "openc3_pusht_expert"
OUTPUT_DIR = BASE_DIR / "outputs" / "openc3_pusht_diffusion"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T_OBS = 4
T_PRED = 16
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
TRAIN_STEPS = 40000
LR = 1e-4


class PushTPhysicalDataset(Dataset):
    def __init__(self, data_dir: Path, obs_horizon: int = 2, pred_horizon: int = 16):
        self.data_dir = data_dir
        self.obs_h = obs_horizon
        self.pred_h = pred_horizon
        self.df = pd.read_parquet(data_dir / "data.parquet")

        self.valid_indices = []
        self.states_per_ep = {}
        self.actions_per_ep = {}
        self.video_frames = {}

        for ep_id in self.df["episode_index"].unique():
            self.video_frames[ep_id] = {}
            for cam in ["top", "wrist"]:
                v_path = self.data_dir / "videos" / f"observation.images.{cam}_episode_{ep_id:06d}.mp4"
                container = av.open(str(v_path))
                frames = [
                    torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1)
                    for f in container.decode(video=0)
                ]
                self.video_frames[ep_id][cam] = torch.stack(frames)
                container.close()

        for ep_id, group in self.df.groupby("episode_index"):
            n_pq = len(group)
            n_top = len(self.video_frames[ep_id]["top"])
            n_wrist = len(self.video_frames[ep_id]["wrist"])
            ep_len = min(n_pq, n_top, n_wrist)

            self.states_per_ep[ep_id] = np.stack(group["observation.state"].values)[:ep_len].astype(np.float32)
            self.actions_per_ep[ep_id] = np.stack(group["action"].values)[:ep_len].astype(np.float32)

            for t in range(self.obs_h - 1, ep_len - self.pred_h + 1):
                self.valid_indices.append((ep_id, t))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ep_id, t = self.valid_indices[idx]

        states_np = self.states_per_ep[ep_id][t - self.obs_h + 1 : t + 1]
        actions_np = self.actions_per_ep[ep_id][t : t + self.pred_h]

        top_imgs = self.video_frames[ep_id]["top"][t - self.obs_h + 1 : t + 1].float() / 255.0
        wrist_imgs = self.video_frames[ep_id]["wrist"][t - self.obs_h + 1 : t + 1].float() / 255.0

        return {
            "observation.state": torch.from_numpy(states_np).float(),
            "action": torch.from_numpy(actions_np).float(),
            "observation.images.top": top_imgs,
            "observation.images.wrist": wrist_imgs,
        }


def compute_dataset_stats(dataset: PushTPhysicalDataset) -> dict:
    all_states_np = np.concatenate(list(dataset.states_per_ep.values()), axis=0)
    all_actions_np = np.concatenate(list(dataset.actions_per_ep.values()), axis=0)

    all_states = torch.from_numpy(all_states_np).float()
    all_actions = torch.from_numpy(all_actions_np).float()

    return {
        "observation.state": {
            "min": all_states.min(dim=0).values.cpu(),
            "max": all_states.max(dim=0).values.cpu(),
        },
        "action": {
            "min": all_actions.min(dim=0).values.cpu(),
            "max": all_actions.max(dim=0).values.cpu(),
        },
    }


def normalize_tensor(x: torch.Tensor, stats: dict) -> torch.Tensor:
    min_v = stats["min"].to(x.device)
    max_v = stats["max"].to(x.device)
    diff = max_v - min_v
    scale = torch.where(diff < 1e-4, torch.ones_like(diff), diff)
    norm = 2.0 * (x - min_v) / scale - 1.0
    return torch.clamp(norm, -1.0, 1.0)


def train():
    print(f"=== 开始训练 Open_C3 7D Push-T Diffusion Policy (设备: {DEVICE}) ===")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = PushTPhysicalDataset(DATASET_DIR, obs_horizon=T_OBS, pred_horizon=T_PRED)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    stats = compute_dataset_stats(dataset)

    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }

    cfg = DiffusionConfig(
        n_obs_steps=T_OBS,
        horizon=T_PRED,
        n_action_steps=12,
        input_features=input_features,
        output_features=output_features,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        down_dims=[256, 512, 1024],
        num_train_timesteps=100,
        num_inference_steps=16,
    )

    policy = DiffusionPolicy(cfg)
    policy.to(DEVICE)
    policy.train()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=1e-6)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS, eta_min=1e-6)

    step = 0
    optimizer.zero_grad()
    pbar = tqdm(total=TRAIN_STEPS, desc="Training Policy")

    while step < TRAIN_STEPS:
        for batch_idx, batch in enumerate(dataloader):
            states = normalize_tensor(batch["observation.state"].to(DEVICE), stats["observation.state"])
            actions = normalize_tensor(batch["action"].to(DEVICE), stats["action"])
            top_imgs = batch["observation.images.top"].to(DEVICE)
            wrist_imgs = batch["observation.images.wrist"].to(DEVICE)

            action_is_pad = torch.zeros((actions.shape[0], T_PRED), dtype=torch.bool, device=DEVICE)

            model_input = {
                "observation.state": states,
                "action": actions,
                "action_is_pad": action_is_pad,
                "observation.images.top": top_imgs,
                "observation.images.wrist": wrist_imgs,
            }

            loss, _ = policy(model_input)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                step += 1
                pbar.update(1)
                cur_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{loss.item() * GRAD_ACCUM_STEPS:.4f}", "lr": f"{cur_lr:.2e}"})

                if step % 5000 == 0:
                    ckpt_path = OUTPUT_DIR / f"checkpoint_{step}"
                    policy.save_pretrained(str(ckpt_path))
                    torch.save(stats, ckpt_path / "stats.pt")
                    torch.save(stats, ckpt_path / "dataset_stats.pt")

                if step >= TRAIN_STEPS:
                    break

    pbar.close()

    save_path = OUTPUT_DIR / "final_model"
    policy.eval()
    policy.save_pretrained(str(save_path))
    torch.save(stats, save_path / "stats.pt")
    torch.save(stats, save_path / "dataset_stats.pt")
    print(f"\n✅ 视觉表征充分收敛！模型已保存至: {save_path}")


if __name__ == "__main__":
    train()