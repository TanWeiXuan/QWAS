#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stability_assist.model import ActorCritic
from stability_assist.weights import write_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a QWAS PPO actor to the portable qwasmlp format")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic()
    model.load_state_dict(checkpoint["model_state"])
    path = write_model(model.actor, args.output)
    print(f"Exported {path}", flush=True)


if __name__ == "__main__":
    main()
