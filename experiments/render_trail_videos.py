# Original concept & implementation: Henry Cao
# Project: Digital Fly / MANC Learning Architecture — started 2026-09-19
# Project ID: henrycao-2026-f003b4
"""Re-render an episode with a coloured motion trail and an authorship watermark.

The trail is the thorax path projected through whatever camera the frame was rendered
with, so it is exact for the moving follow camera as well as the fixed top-down one: at
every rendered frame we record the fly's position together with that camera's pose, then
project the path so far with the standard pinhole model.

    python experiments/render_trail_videos.py --run experiments/results/manc_v2 \
        --seed 20000000 --color "#1f77b4" --out-dir ~/digital-fly-share/videos/foo

Reads a trained run, writes only into --out-dir. Never modifies existing results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import RESULTS  # noqa: I001  (sets sys.path, MUJOCO_GL)

import imageio.v3 as iio  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from brain._provenance import AUTHOR_NAME, PROJECT_DATE  # noqa: E402
from brain.interface import run_episode  # noqa: E402
from brain.model import make_brain  # noqa: E402
from environment.fly_interface import EnvConfig, FlyInterface  # noqa: E402

FONT = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")


def project(points_xyz, cam_pos, cam_mat, fovy_deg, width, height):
    """World points -> pixel coordinates for one MuJoCo camera (looks along its -z)."""
    rel = (np.asarray(points_xyz) - cam_pos) @ cam_mat  # cam_mat columns are the axes
    depth = -rel[:, 2]
    tan_y = np.tan(np.deg2rad(fovy_deg) / 2)
    tan_x = tan_y * width / height
    px = width / 2 * (1 + rel[:, 0] / (depth * tan_x))
    py = height / 2 * (1 - rel[:, 1] / (depth * tan_y))
    return np.stack([px, py], axis=1), depth


def draw_frame(frame, path_px, colour, label, tail=None):
    """Draw the trail (older = fainter) and the watermark onto one RGB frame."""
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    n = len(path_px)
    if n >= 2:
        keep = tail if tail else n
        start = max(0, n - keep)
        for i in range(start + 1, n):
            frac = (i - start) / max(1, n - start)  # 0 oldest .. 1 newest
            alpha = int(40 + 175 * frac)
            d.line([tuple(path_px[i - 1]), tuple(path_px[i])], fill=colour + (alpha,), width=3)
        d.ellipse([*(path_px[-1] - 4), *(path_px[-1] + 4)], outline=colour + (230,), width=2)
    font = ImageFont.truetype(FONT, max(10, img.size[0] // 30))
    x0, y0, x1, y1 = font.getbbox(label)
    tx, ty = img.size[0] - (x1 - x0) - 8, img.size[1] - (y1 - y0) - 8
    d.text((tx + 1, ty + 1), label, font=font, fill=(0, 0, 0, 110))
    d.text((tx, ty), label, font=font, fill=(255, 255, 255, 150))
    return np.array(Image.alpha_composite(img, overlay).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="training run dir (uses its config + theta_best.npy)")
    ap.add_argument("--brain", help="reference brain instead of --run: reflex | random")
    ap.add_argument("--seed", type=int, default=20_000_000)
    ap.add_argument("--n-obstacles", type=int, default=0)
    ap.add_argument("--colour", default="#1f77b4")
    ap.add_argument("--tail", type=int, default=0, help="0 = keep the whole path")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.run:
        saved = json.loads((Path(args.run) / "config.json").read_text())
        env = EnvConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved["env"].items()})
        spec, params = saved["brain"], np.load(Path(args.run) / "theta_best.npy")
    else:
        env, spec, params = EnvConfig(), {"type": args.brain}, None
    if args.run and args.brain:  # reference brain in a trained run's environment
        spec, params = {"type": args.brain}, None
    env.n_obstacles = args.n_obstacles

    fly = FlyInterface(env, render=True)
    brain = make_brain(spec)
    if params is not None and len(params):
        brain.set_params(params)

    cams = fly.renderer._cameras_names2id  # name -> internal MuJoCo camera id
    samples = {name: [] for name in cams}  # per camera: (thorax xyz, cam pos, cam mat, fovy)
    original_render = fly.sim.render_as_needed

    def render_and_record():
        rendered = original_render()
        if rendered:
            m, d = fly.sim.mj_model, fly.sim.mj_data
            thorax = fly.sim.get_body_positions(fly.fly.name)[fly._thorax_idx].copy()
            for name, cam_id in cams.items():
                samples[name].append((thorax, d.cam_xpos[cam_id].copy(),
                                      d.cam_xmat[cam_id].reshape(3, 3).copy(), float(m.cam_fovy[cam_id])))
        return rendered

    fly.sim.render_as_needed = render_and_record
    result = run_episode(brain, fly, seed=args.seed)

    colour = tuple(int(args.colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    label = f"{AUTHOR_NAME} — {PROJECT_DATE}"
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frames in fly.renderer.frames.items():
        rec = samples[name]
        n = min(len(frames), len(rec))
        h, w = frames[0].shape[:2]
        out_frames = []
        for k in range(n):
            thoraxes = np.array([rec[i][0] for i in range(k + 1)])
            _, cam_pos, cam_mat, fovy = rec[k]
            px, depth = project(thoraxes, cam_pos, cam_mat, fovy, w, h)
            px = px[depth > 1e-6]
            out_frames.append(draw_frame(frames[k], px, colour, label, args.tail or None))
        path = out_dir / f"{name.replace(chr(47), chr(95))}.mp4"
        iio.imwrite(path, np.array(out_frames), fps=fly.renderer.output_fps, codec="libx264", quality=8)
        written.append(str(path))
    fly.close()
    print(json.dumps({"seed": args.seed, "success": bool(result["success"]),
                      "time_to_target": result["time_to_target"], "videos": written}, indent=2))


if __name__ == "__main__":
    main()
