#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stability_assist.evaluation import evaluate_actor
from stability_assist.model import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically evaluate QWAS Easy Mode")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("doc/stability_assist_evaluation.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(4100, 4140)))
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ActorCritic().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Evaluation start: checkpoint={args.checkpoint}, seeds={len(args.seeds)}, device={device}", flush=True)
    metrics = evaluate_actor(model.actor, args.seeds, device, args.duration, include_all_modes=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for mode in ("no_assist", "zero", "trained"):
        result = metrics[mode]
        print(f"{mode}: survival={result['mean_survival_time']:.3f}s, "
              f"crash_rate={result['crash_rate']:.3f}, max_tilt={result['mean_maximum_tilt_degrees']:.1f}deg, "
              f"assist={result['mean_assist_magnitude']:.4f}", flush=True)
    print(f"No-input sink rate: {metrics['no_input_sink']['mean_vertical_sink_rate_mps']:.3f} m/s", flush=True)
    print(f"Player-agency test: {'pass' if metrics['player_agency']['pass'] else 'fail'}", flush=True)
    print(f"Evaluation complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
