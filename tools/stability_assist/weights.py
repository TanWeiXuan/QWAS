from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .model import Actor
from .spec import ACTION_SIZE, ARCHITECTURE, HIDDEN_SIZE, OBSERVATION_SCALE, OBSERVATION_SIZE

MAGIC = b"QWASMLP\0"
FORMAT_VERSION = 2
HEADER = struct.Struct("<8s5I")
FLOAT_COUNT = OBSERVATION_SIZE + OBSERVATION_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * ACTION_SIZE + ACTION_SIZE
FILE_SIZE = HEADER.size + FLOAT_COUNT * 4


def actor_arrays(actor: Actor) -> list[np.ndarray]:
    linear = [module for module in actor.layers if hasattr(module, "weight")]
    arrays: list[np.ndarray] = [OBSERVATION_SCALE]
    for layer in linear:
        arrays.append(layer.weight.detach().cpu().numpy().astype("<f4", copy=False))
        arrays.append(layer.bias.detach().cpu().numpy().astype("<f4", copy=False))
    return arrays


def write_model(actor: Actor, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = HEADER.pack(MAGIC, FORMAT_VERSION, *ARCHITECTURE)
    payload = b"".join(np.ascontiguousarray(array, dtype="<f4").tobytes() for array in actor_arrays(actor))
    if len(header) + len(payload) != FILE_SIZE:
        raise ValueError("internal model-size mismatch")
    output.write_bytes(header + payload)
    return output


def read_model(path: str | Path) -> dict[str, np.ndarray]:
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"unexpected file length: {len(data)} (expected {FILE_SIZE})")
    magic, version, input_size, hidden1, hidden2, output_size = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("invalid magic bytes")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported format version: {version}")
    if len(data) != FILE_SIZE:
        raise ValueError(f"unexpected file length: {len(data)} (expected {FILE_SIZE})")
    if (input_size, hidden1, hidden2, output_size) != ARCHITECTURE:
        raise ValueError("unexpected model dimensions")
    floats = np.frombuffer(data, dtype="<f4", offset=HEADER.size)
    if not np.all(np.isfinite(floats)):
        raise ValueError("non-finite model value")
    offset = 0

    def take(count: int, shape: tuple[int, ...]) -> np.ndarray:
        nonlocal offset
        result = floats[offset:offset + count].reshape(shape).copy()
        offset += count
        return result

    scale = take(OBSERVATION_SIZE, (OBSERVATION_SIZE,))
    if np.any(scale <= 0.0):
        raise ValueError("invalid observation scale")
    return {
        "observation_scale": scale,
        "weights1": take(HIDDEN_SIZE * OBSERVATION_SIZE, (HIDDEN_SIZE, OBSERVATION_SIZE)),
        "bias1": take(HIDDEN_SIZE, (HIDDEN_SIZE,)),
        "weights2": take(HIDDEN_SIZE * HIDDEN_SIZE, (HIDDEN_SIZE, HIDDEN_SIZE)),
        "bias2": take(HIDDEN_SIZE, (HIDDEN_SIZE,)),
        "weights3": take(ACTION_SIZE * HIDDEN_SIZE, (ACTION_SIZE, HIDDEN_SIZE)),
        "bias3": take(ACTION_SIZE, (ACTION_SIZE,)),
    }


def numpy_forward(model: dict[str, np.ndarray], raw_observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(raw_observation, dtype=np.float32)
    if not np.all(np.isfinite(observation)):
        return np.zeros((*observation.shape[:-1], ACTION_SIZE), dtype=np.float32)
    observation = np.clip(observation / model["observation_scale"], -4.0, 4.0)
    hidden1 = np.tanh(observation @ model["weights1"].T + model["bias1"])
    hidden2 = np.tanh(hidden1 @ model["weights2"].T + model["bias2"])
    return np.tanh(hidden2 @ model["weights3"].T + model["bias3"]).astype(np.float32)


def load_actor(path: str | Path) -> Actor:
    """Reconstruct a PyTorch actor from a portable model for evaluation."""
    model = read_model(path)
    actor = Actor()
    linear = [module for module in actor.layers if hasattr(module, "weight")]
    with torch.no_grad():
        for layer, weight_name, bias_name in zip(
                linear, ("weights1", "weights2", "weights3"), ("bias1", "bias2", "bias3")):
            layer.weight.copy_(torch.from_numpy(model[weight_name]))
            layer.bias.copy_(torch.from_numpy(model[bias_name]))
    return actor


def _format_array(name: str, array: np.ndarray) -> str:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    lines = []
    for start in range(0, values.size, 8):
        literals = []
        for value in values[start:start + 8]:
            literal = f"{float(value):.9g}"
            if "." not in literal and "e" not in literal.lower():
                literal += ".0"
            literals.append(literal + "f")
        chunk = ", ".join(literals)
        lines.append(f"    {chunk},")
    return f"inline constexpr std::array<float, {values.size}> {name} = {{\n" + "\n".join(lines) + "\n};\n"


def bake_header(model_path: str | Path, output_path: str | Path) -> Path:
    model_path = Path(model_path)
    model = read_model(model_path)
    model_id = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "#pragma once\n\n#include <array>\n\n",
        "// Generated by tools/bake_stability_assist_weights.py. Do not edit manually.\n",
        "namespace qwas_baked_stability_assist {\n",
        f"inline constexpr unsigned kFormatVersion = {FORMAT_VERSION};\n",
        f"inline constexpr int kInputSize = {OBSERVATION_SIZE};\n",
        f"inline constexpr int kHidden1Size = {HIDDEN_SIZE};\n",
        f"inline constexpr int kHidden2Size = {HIDDEN_SIZE};\n",
        f"inline constexpr int kOutputSize = {ACTION_SIZE};\n",
        f"inline constexpr char kModelId[] = \"sha256-{model_id}\";\n",
        _format_array("kObservationScale", model["observation_scale"]),
        _format_array("kWeights1", model["weights1"]),
        _format_array("kBias1", model["bias1"]),
        _format_array("kWeights2", model["weights2"]),
        _format_array("kBias2", model["bias2"]),
        _format_array("kWeights3", model["weights3"]),
        _format_array("kBias3", model["bias3"]),
        "}  // namespace qwas_baked_stability_assist\n",
    ]
    output.write_text("".join(parts), encoding="utf-8", newline="\n")
    return output
