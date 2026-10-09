"""LeRobot PushT 점검: 데이터셋 로드·비디오 디코딩, 그리고 gym-pusht에서 데모 행동 재생.

    uv run python scripts/check_pusht.py

lerobot/pusht는 관측 상태에 에이전트 위치(2차원)만 있고 T 블록의 포즈는 없어서 데모 전체를
재현할 수는 없다. 대신 "행동 = 에이전트 목표 위치(절대 좌표, 10Hz, PD 추종)"라는 행동 공간의 의미를
확인한다. gym-pusht의 에이전트는 KINEMATIC 강체라 블록과 부딪혀도 밀리지 않으므로, 블록을 어디에
두든 데모 행동을 열린 루프로 넣으면 에이전트 궤적은 데이터셋 상태와 일치해야 한다.
"""

import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import gym_pusht  # noqa: F401  (gymnasium에 PushT-v0 등록)
import gymnasium as gym
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "lerobot/pusht"
RESULTS = Path(__file__).resolve().parents[1] / "results"
EPISODE = 0
BLOCK_POSE = (100.0, 400.0, 0.0)  # 데이터셋에 없으므로 임의 값


def main() -> None:
    t = time.perf_counter()
    ds = LeRobotDataset(REPO_ID)
    load_s = time.perf_counter() - t

    ep = ds.meta.episodes[EPISODE]
    lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
    t = time.perf_counter()
    frames = [ds[i] for i in range(lo, hi)]
    decode_ms = (time.perf_counter() - t) / len(frames) * 1e3
    images = np.stack([(f["observation.image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for f in frames])
    states = np.stack([f["observation.state"].numpy() for f in frames])
    actions = np.stack([f["action"].numpy() for f in frames])

    # 데이터셋에 기록된 성공 라벨과 보상 (에피소드 단위)
    cols = ds.hf_dataset.with_format("numpy", columns=["episode_index", "next.success", "next.reward"])
    epi, succ, rew = (np.asarray(cols[k]) for k in ("episode_index", "next.success", "next.reward"))
    ep_success = np.zeros(ds.num_episodes, bool)
    np.logical_or.at(ep_success, epi, succ)
    ep_max_reward = np.zeros(ds.num_episodes)
    np.maximum.at(ep_max_reward, epi, rew)

    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
    obs, _ = env.reset(seed=0, options={"reset_to_state": [*states[0], *BLOCK_POSE]})
    replay_pos, replay_img = [], [obs["pixels"]]
    t = time.perf_counter()
    for a in actions[:-1]:
        obs, *_ = env.step(a.astype(np.float64))
        replay_pos.append(obs["agent_pos"])
        replay_img.append(obs["pixels"])
    step_hz = (len(actions) - 1) / (time.perf_counter() - t)
    err = np.linalg.norm(np.array(replay_pos) - states[1:], axis=1)
    env.close()

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": {
            "repo_id": REPO_ID,
            "codebase_version": ds.meta.info.codebase_version,
            "episodes": ds.num_episodes,
            "frames": ds.num_frames,
            "fps": ds.fps,
            "features": {k: [v["dtype"], list(v["shape"])] for k, v in ds.features.items()},
            "load_s": round(load_s, 2),
            "episodes_with_success_label": int(ep_success.sum()),
            "episode_max_reward": {
                "median": round(float(np.median(ep_max_reward)), 3),
                "min": round(float(ep_max_reward.min()), 3),
                "max": round(float(ep_max_reward.max()), 3),
            },
        },
        "video_backend": ds._video_backend,
        "decode_ms_per_frame": round(decode_ms, 2),
        "episode": {
            "index": EPISODE,
            "length": int(hi - lo),
            "task": ep["tasks"][0],
            "demo_success": bool(frames[-1]["next.success"]),
            "action_range": [actions.min(0).round(1).tolist(), actions.max(0).round(1).tolist()],
        },
        "replay": {
            "agent_pos_err_px": {"mean": round(float(err.mean()), 2), "max": round(float(err.max()), 2)},
            "workspace_px": 512,
            "env_step_hz": round(step_hz),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    import matplotlib.pyplot as plt

    idx = np.linspace(0, len(images) - 1, 8).astype(int)
    fig, axes = plt.subplots(3, 8, figsize=(16, 7), gridspec_kw={"height_ratios": [1, 1, 1.4]})
    for j, i in enumerate(idx):
        axes[0, j].imshow(images[i])
        axes[0, j].set_title(f"t={i}", fontsize=9)
        axes[1, j].imshow(replay_img[i])
    for ax in axes[:2].flat:
        ax.axis("off")
    axes[0, 0].text(-10, 48, "dataset", rotation=90, va="center", ha="right")
    axes[1, 0].text(-10, 48, "replay", rotation=90, va="center", ha="right")
    gs = axes[2, 0].get_gridspec()
    for ax in axes[2]:
        ax.remove()
    ax = fig.add_subplot(gs[2, :])
    ax.plot(states[:, 0], states[:, 1], label="dataset observation.state", lw=2)
    ax.plot(*np.array(replay_pos).T, "--", label="replay agent_pos", lw=1.5)
    ax.plot(actions[:, 0], actions[:, 1], ":", label="action (target)", lw=1, alpha=0.7)
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"episode {EPISODE}: agent trajectory (mean err {err.mean():.1f}px / 512)")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    tag = "mac" if platform.system() == "Darwin" else "linux"
    fig.savefig(RESULTS / f"pusht-replay-{tag}.png", dpi=80)
    (RESULTS / f"pusht-{tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
