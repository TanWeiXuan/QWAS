#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stability_assist.weights import bake_header


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake portable QWAS Easy Mode weights into a C++17 header")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = bake_header(args.input, args.output)
    print(f"Baked {args.input} into {path}", flush=True)


if __name__ == "__main__":
    main()
