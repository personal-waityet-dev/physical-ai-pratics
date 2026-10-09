"""MuJoCo 오프스크린 렌더링 점검: GL 백엔드별 동작, 접촉 시뮬레이션, 카메라 2대 RGB·depth 렌더.

    uv run python scripts/check_mujoco.py            # 기본 백엔드(macOS: cgl, Linux: egl)로 점검
    uv run python scripts/check_mujoco.py --probe    # 백엔드별 동작 여부도 함께 확인

MUJOCO_GL은 mujoco를 import하기 전에 정해져야 하므로, 백엔드 비교는 하위 프로세스로 한다.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_GL = "cgl" if sys.platform == "darwin" else "egl"
os.environ.setdefault("MUJOCO_GL", DEFAULT_GL)

import mujoco
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"

SCENE = """
<mujoco model="tabletop">
  <option timestep="0.002"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse=".9 .9 .9"/>
    <geom type="plane" size="1 1 .1" material="grid"/>
    <body name="table" pos="0 0 .2">
      <geom type="box" size=".3 .3 .2" rgba=".6 .5 .4 1"/>
    </body>
    <body name="red_cube" pos="-.1 0 .6">
      <freejoint/>
      <geom type="box" size=".04 .04 .04" rgba=".9 .1 .1 1"/>
    </body>
    <body name="green_ball" pos=".1 .05 .8">
      <freejoint/>
      <geom type="sphere" size=".04" rgba=".1 .8 .2 1"/>
    </body>
    <camera name="front" pos="0 -1.0 .75" xyaxes="1 0 0 0 .45 .9"/>
    <camera name="top" pos="0 0 1.4" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


def probe_backends() -> dict[str, str]:
    code = (
        "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'); "
        "r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); "
        "print('ok', r.render().shape)"
    )
    out = {}
    for gl in ("cgl", "glfw", "egl", "osmesa"):
        p = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "MUJOCO_GL": gl},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        lines = (p.stdout or p.stderr).strip().splitlines()
        out[gl] = "ok" if p.returncode == 0 else (lines[-1][:120] if lines else f"exit {p.returncode}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="GL 백엔드별 동작 여부 확인")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)

    # 접촉 시뮬레이션: 물체가 떨어져 테이블 위에 멈추는지
    steps = 1500
    t = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(model, data)
    sim_steps_per_s = steps / (time.perf_counter() - t)
    cube_z = float(data.body("red_cube").xpos[2])
    ball_z = float(data.body("green_ball").xpos[2])

    # 렌더: 카메라 2대 RGB + 정면 depth
    renderer = mujoco.Renderer(model, height=480, width=640)
    frames = {}
    for cam in ("front", "top"):
        renderer.update_scene(data, camera=cam)
        frames[cam] = renderer.render()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="front")
    depth = renderer.render()
    renderer.disable_depth_rendering()

    fps = {}
    for h, w in ((224, 224), (480, 640)):
        r = mujoco.Renderer(model, height=h, width=w)
        n = 100
        t = time.perf_counter()
        for _ in range(n):
            r.update_scene(data, camera="front")
            r.render()
        fps[f"{h}x{w}"] = round(n / (time.perf_counter() - t), 1)
        r.close()

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "MUJOCO_GL": os.environ["MUJOCO_GL"],
        "sim_steps_per_s": round(sim_steps_per_s),
        "rest_height_m": {"red_cube": round(cube_z, 3), "green_ball": round(ball_z, 3)},
        "contact_ok": abs(cube_z - 0.44) < 0.01 and abs(ball_z - 0.44) < 0.01,  # 테이블 윗면 0.4 + 반경 0.04
        "rgb_mean": {k: round(float(v.mean()), 1) for k, v in frames.items()},
        "depth_range_m": [round(float(depth.min()), 3), round(float(depth.max()), 3)],  # 최댓값은 배경(원평면)
        "render_fps": fps,
    }
    if args.probe:
        report["backends"] = probe_backends()
    print(json.dumps(report, indent=2, ensure_ascii=False))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(frames["front"])
    axes[1].imshow(frames["top"])
    im = axes[2].imshow(np.where(depth < 3.0, depth, np.nan), cmap="viridis")  # 3m 밖 배경은 비워 둔다
    fig.colorbar(im, ax=axes[2], label="depth (m)")
    for ax, title in zip(axes, ("front RGB", "top RGB", "front depth")):
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"MuJoCo {mujoco.__version__} · MUJOCO_GL={os.environ['MUJOCO_GL']}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    RESULTS.mkdir(exist_ok=True)
    tag = "mac" if sys.platform == "darwin" else "linux"
    fig.savefig(RESULTS / f"mujoco-render-{tag}.png", dpi=80)
    (RESULTS / f"mujoco-{tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
