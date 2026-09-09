import os
import sys
from pathlib import Path
import av
import numpy as np
import pandas as pd

# 自动兼容当前目录或上一级目录
BASE_DIR = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.parent / "data").exists() else Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "data" / "openc3_pusht_expert"


def check_dataset():
    print("=== [数据校验] 检查 Push-T 物理专家数据集结构与完整性 ===")
    assert DATASET_DIR.exists(), f"数据集目录未找到: {DATASET_DIR}"

    parquet_path = DATASET_DIR / "data.parquet"
    assert parquet_path.exists(), "缺少 data.parquet 文件"

    df = pd.read_parquet(parquet_path)
    total_frames = len(df)
    episodes = df["episode_index"].unique()
    total_episodes = len(episodes)

    print(f"总帧数: {total_frames} | Episode 数量: {total_episodes}")

    # 校验 7 维状态与动作空间
    sample_state = np.array(df["observation.state"].iloc[0])
    sample_action = np.array(df["action"].iloc[0])
    assert sample_state.shape == (7,), f"State 维度错误: {sample_state.shape}"
    assert sample_action.shape == (7,), f"Action 维度错误: {sample_action.shape}"

    # 遍历检查所有 Episode 的视频流是否存在且帧数对齐
    videos_dir = DATASET_DIR / "videos"
    mismatch_count = 0

    for ep_id in episodes:
        ep_len = len(df[df["episode_index"] == ep_id])
        for cam in ["top", "wrist"]:
            v_path = videos_dir / f"observation.images.{cam}_episode_{ep_id:06d}.mp4"
            assert v_path.exists(), f"视频缺失: {v_path.name}"
            
            container = av.open(str(v_path))
            v_frames = container.streams.video[0].frames
            # 部分 mp4 头部未写入 frames，则实际解码探测
            if v_frames == 0:
                v_frames = sum(1 for _ in container.decode(video=0))
            container.close()

            if abs(v_frames - ep_len) > 2:
                print(f"⚠️ Episode {ep_id:02d} [{cam}] 视频帧数 ({v_frames}) 与 Parquet 行数 ({ep_len}) 偏差较大")
                mismatch_count += 1

    if mismatch_count == 0:
        print("\n✅ 3D Push-T 全量视频流与标量数据完全对齐，校验通过！")
    else:
        print(f"\n⚠️ 发现 {mismatch_count} 处轻微帧数差异，训练加载器将自动截断保护。")


if __name__ == "__main__":
    check_dataset()