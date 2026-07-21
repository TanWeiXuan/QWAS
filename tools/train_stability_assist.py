#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from stability_assist.environment import EnvironmentConfig, VectorizedQwasEnv
from stability_assist.evaluation import evaluate_actor
from stability_assist.model import ActorCritic
from stability_assist.weights import bake_header, write_model


@dataclass
class TrainConfig:
    environments: int
    rollout_steps: int
    total_transitions: int
    ppo_epochs: int
    minibatch_size: int
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    value_coefficient: float = 0.50
    entropy_coefficient: float = 0.001
    max_gradient_norm: float = 0.50


PRESETS = {
    "smoke": TrainConfig(environments=8, rollout_steps=64, total_transitions=4_096,
                         ppo_epochs=2, minibatch_size=256),
    "train": TrainConfig(environments=64, rollout_steps=128, total_transitions=1_048_576,
                         ppo_epochs=4, minibatch_size=1_024),
}


def warm_start_actor(model: ActorCritic, device: torch.device, seed: int, steps: int = 300) -> float:
    """Place PPO in a pitch/roll damping basin without any navigation signal."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 91)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=3.0e-3)
    pitch_mode = torch.tensor([1.0, 1.0, -1.0, -1.0], device=device)
    roll_mode = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device)
    final_loss = 0.0
    model.actor.train()
    for _ in range(steps):
        count = 2_048
        observation = torch.zeros((count, 28), device=device)
        tilt = torch.rand(count, generator=generator, device=device) * torch.deg2rad(torch.tensor(80.0, device=device))
        azimuth = (torch.rand(count, generator=generator, device=device) * 2.0 - 1.0) * torch.pi
        observation[:, 0] = torch.sin(tilt) * torch.cos(azimuth)
        observation[:, 1] = torch.cos(tilt)
        observation[:, 2] = torch.sin(tilt) * torch.sin(azimuth)
        observation[:, 3:6] = torch.rand((count, 3), generator=generator, device=device) * 2.4 - 1.2
        observation[:, 6:10] = torch.rand((count, 4), generator=generator, device=device)
        observation[:, 10:14] = (torch.rand((count, 4), generator=generator, device=device) > 0.5).float()
        observation[:, 14] = torch.rand(count, generator=generator, device=device) * 0.55 + 0.75
        observation[:, 15] = torch.clamp((tilt - 0.35) / 0.70, 0.0, 1.0)
        observation[:, 16:28] = torch.rand((count, 12), generator=generator, device=device) * 0.4 + 0.8
        pitch = -1.5 * observation[:, 2] - 1.5 * observation[:, 3]
        roll = 1.5 * observation[:, 0] - 1.5 * observation[:, 5]
        target = torch.tanh(pitch[:, None] * pitch_mode + roll[:, None] * roll_mode)
        prediction = model.actor(observation)
        loss = torch.mean((prediction - target).square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the QWAS residual stability actor with PPO")
    parser.add_argument("--preset", choices=PRESETS, default="train")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another PyTorch device")
    parser.add_argument("--total-transitions", type=int)
    parser.add_argument("--environments", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--progress-interval", type=int, help="transition interval; default is 5%")
    parser.add_argument("--checkpoint-interval", type=int, help="transition interval; default is 10%")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model-output", type=Path, default=Path("models/stability_assist.qwasmlp"))
    parser.add_argument("--header-output", type=Path, default=Path("include/generated/stability_assist_weights.h"))
    parser.add_argument("--no-auto-export", action="store_true")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def save_checkpoint(path: Path, model: ActorCritic, optimizer: torch.optim.Optimizer,
                    transitions: int, updates: int, best_score: tuple[float, ...] | None,
                    best_metrics: dict | None, config: TrainConfig, args: argparse.Namespace,
                    env: VectorizedQwasEnv) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "transitions": transitions,
        "updates": updates,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "train_config": asdict(config),
        "environment_config": asdict(env.config),
        "seed": args.seed,
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_global_rng_state": np.random.get_state(),
        "numpy_rng_state": env.rng.bit_generator.state,
        "python_rng_state": random.getstate(),
    }, path)


def evaluation_score(metrics: dict) -> tuple[float, float, float]:
    agency = 1.0 if metrics["player_agency"]["pass"] else 0.0
    trained = metrics["trained"]
    return (agency, trained["mean_survival_time"], -trained["mean_assist_magnitude"])


def main() -> int:
    args = parse_args()
    config = TrainConfig(**asdict(PRESETS[args.preset]))
    if args.total_transitions:
        config.total_transitions = args.total_transitions
    if args.environments:
        config.environments = args.environments
    if args.rollout_steps:
        config.rollout_steps = args.rollout_steps
    device = select_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))

    environment_config = EnvironmentConfig(episode_seconds=10.0 if args.preset == "train" else 4.0)
    env = VectorizedQwasEnv(config.environments, args.seed, environment_config)
    model = ActorCritic().to(device)
    if not args.resume:
        warm_loss = warm_start_actor(model, device, args.seed)
        print(f"Pitch/roll-only actor warm start complete: MSE={warm_loss:.6f}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    transitions = 0
    updates = 0
    best_score: tuple[float, ...] | None = None
    best_metrics: dict | None = None

    latest_path = args.output_dir / f"stability_assist_{args.preset}_latest.pt"
    best_path = args.output_dir / f"stability_assist_{args.preset}_best.pt"
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        transitions = int(checkpoint["transitions"])
        updates = int(checkpoint["updates"])
        best_score = tuple(checkpoint["best_score"]) if checkpoint.get("best_score") else None
        best_metrics = checkpoint.get("best_metrics")
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if checkpoint.get("torch_cuda_rng_state_all") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])
        if checkpoint.get("numpy_global_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_global_rng_state"])
        if checkpoint.get("numpy_rng_state") is not None:
            env.rng.bit_generator.state = checkpoint["numpy_rng_state"]
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        print(f"Resumed checkpoint: {args.resume} at {transitions:,} transitions", flush=True)

    batch_size = config.environments * config.rollout_steps
    progress_interval = args.progress_interval or max(batch_size, config.total_transitions // 20)
    checkpoint_interval = args.checkpoint_interval or max(batch_size, config.total_transitions // 10)
    evaluation_interval = max(batch_size, config.total_transitions // (8 if args.preset == "train" else 2))
    next_progress = ((transitions // progress_interval) + 1) * progress_interval
    next_checkpoint = ((transitions // checkpoint_interval) + 1) * checkpoint_interval
    next_evaluation = transitions
    eval_seeds = list(range(4100, 4140)) if args.preset == "train" else [3100, 3101, 3102]
    eval_duration = 8.0 if args.preset == "train" else 4.0

    print("Training start", flush=True)
    print(f"Environment initialized: {config.environments} parallel environments, device={device}, "
          f"seed={args.seed}, target transitions={config.total_transitions:,}", flush=True)
    print(f"PPO batch={batch_size:,}, epochs={config.ppo_epochs}, minibatch={config.minibatch_size}, "
          f"learning_rate={config.learning_rate:g}", flush=True)
    start_time = time.perf_counter()
    observation = env.observe()
    recent_episodes: list[dict] = []
    last_ppo_metrics: dict[str, float] = {}

    while transitions < config.total_transitions:
        observations = np.empty((config.rollout_steps, config.environments, observation.shape[1]), dtype=np.float32)
        actions = np.empty((config.rollout_steps, config.environments, 4), dtype=np.float32)
        log_probabilities = np.empty((config.rollout_steps, config.environments), dtype=np.float32)
        values = np.empty_like(log_probabilities)
        rewards = np.empty_like(log_probabilities)
        dones = np.empty_like(log_probabilities, dtype=np.float32)

        model.eval()
        for step in range(config.rollout_steps):
            observation_tensor = torch.from_numpy(observation).to(device)
            with torch.no_grad():
                action_tensor, log_probability_tensor, value_tensor = model.act(observation_tensor)
            next_observation, reward, done, info = env.step(action_tensor.cpu().numpy(), mode="trained")
            observations[step] = observation
            actions[step] = action_tensor.cpu().numpy()
            log_probabilities[step] = log_probability_tensor.cpu().numpy()
            values[step] = value_tensor.cpu().numpy()
            rewards[step] = reward
            dones[step] = done
            recent_episodes.extend(info["episodes"])
            observation = next_observation

        with torch.no_grad():
            next_value = model.critic(torch.from_numpy(observation).to(device)).squeeze(-1).cpu().numpy()
        advantages = np.zeros_like(rewards)
        gae = np.zeros(config.environments, dtype=np.float32)
        for step in reversed(range(config.rollout_steps)):
            non_terminal = 1.0 - dones[step]
            following_value = next_value if step == config.rollout_steps - 1 else values[step + 1]
            delta = rewards[step] + config.gamma * following_value * non_terminal - values[step]
            gae = delta + config.gamma * config.gae_lambda * non_terminal * gae
            advantages[step] = gae
        returns = advantages + values

        flat_observations = torch.from_numpy(observations.reshape(-1, observations.shape[-1])).to(device)
        flat_actions = torch.from_numpy(actions.reshape(-1, 4)).to(device)
        flat_old_log_probabilities = torch.from_numpy(log_probabilities.reshape(-1)).to(device)
        flat_returns = torch.from_numpy(returns.reshape(-1)).to(device)
        flat_old_values = torch.from_numpy(values.reshape(-1)).to(device)
        flat_advantages = torch.from_numpy(advantages.reshape(-1)).to(device)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        model.train()
        policy_losses = []
        value_losses = []
        entropies = []
        approximate_kls = []
        clip_fractions = []
        gradient_norms = []
        indices = np.arange(batch_size)
        for _ in range(config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, config.minibatch_size):
                batch = torch.as_tensor(indices[start:start + config.minibatch_size], device=device)
                new_log_probability, entropy, new_value = model.evaluate_actions(flat_observations[batch], flat_actions[batch])
                log_ratio = new_log_probability - flat_old_log_probabilities[batch]
                ratio = log_ratio.exp()
                unclipped = ratio * flat_advantages[batch]
                clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range) * flat_advantages[batch]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_prediction_clipped = flat_old_values[batch] + torch.clamp(
                    new_value - flat_old_values[batch], -config.clip_range, config.clip_range)
                value_loss = 0.5 * torch.max(
                    (new_value - flat_returns[batch]).square(),
                    (value_prediction_clipped - flat_returns[batch]).square(),
                ).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy_mean
                if not torch.isfinite(loss):
                    print(f"Material training problem: non-finite loss at update {updates + 1}", flush=True)
                    save_checkpoint(latest_path, model, optimizer, transitions, updates, best_score,
                                    best_metrics, config, args, env)
                    return 3
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
                optimizer.step()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > config.clip_range).float().mean()
                policy_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
                entropies.append(float(entropy_mean.detach()))
                approximate_kls.append(float(approximate_kl.detach()))
                clip_fractions.append(float(clip_fraction.detach()))
                gradient_norms.append(float(gradient_norm.detach()))

        transitions += batch_size
        updates += 1
        returns_variance = float(np.var(returns))
        explained_variance = 1.0 - float(np.var(returns - values)) / returns_variance if returns_variance > 1e-8 else 0.0
        last_ppo_metrics = {
            "policy_loss": float(np.mean(policy_losses)), "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)), "approximate_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clip_fractions)), "gradient_norm": float(np.mean(gradient_norms)),
            "explained_variance": explained_variance,
        }

        if transitions >= next_evaluation or transitions >= config.total_transitions:
            print(f"Deterministic evaluation start at {transitions:,} transitions", flush=True)
            model.eval()
            metrics = evaluate_actor(model.actor, eval_seeds, device, eval_duration, include_all_modes=True)
            score = evaluation_score(metrics)
            print(f"Deterministic evaluation: trained survival={metrics['trained']['mean_survival_time']:.3f}s, "
                  f"no-assist={metrics['no_assist']['mean_survival_time']:.3f}s, "
                  f"zero-policy={metrics['zero']['mean_survival_time']:.3f}s, "
                  f"crash rate={metrics['trained']['crash_rate']:.3f}, agency={'pass' if metrics['player_agency']['pass'] else 'fail'}",
                  flush=True)
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = metrics
                save_checkpoint(best_path, model, optimizer, transitions, updates, best_score,
                                best_metrics, config, args, env)
                print(f"New best deterministic checkpoint: {best_path} "
                      f"(survival={score[1]:.3f}s, agency={'pass' if score[0] else 'fail'})", flush=True)
            next_evaluation += evaluation_interval

        if transitions >= next_checkpoint:
            save_checkpoint(latest_path, model, optimizer, transitions, updates, best_score,
                            best_metrics, config, args, env)
            next_checkpoint += checkpoint_interval

        if transitions >= next_progress or transitions >= config.total_transitions:
            elapsed = time.perf_counter() - start_time
            completed = recent_episodes[-200:]
            mean_survival = float(np.mean([item["survival_time"] for item in completed])) if completed else float("nan")
            crash_rate = float(np.mean([item["crashed"] for item in completed])) if completed else float("nan")
            mean_return = float(np.mean([item["episode_return"] for item in completed])) if completed else float("nan")
            percent = min(100.0, transitions / config.total_transitions * 100.0)
            best_survival = best_metrics["trained"]["mean_survival_time"] if best_metrics else float("nan")
            print(
                f"Training progress: {percent:.1f}% | transitions {transitions:,}/{config.total_transitions:,} | "
                f"updates {updates} | elapsed {elapsed:.1f}s | rollout survival {mean_survival:.3f}s | "
                f"rollout crash {crash_rate:.3f} | mean episode reward {mean_return:.3f} | "
                f"policy loss {last_ppo_metrics['policy_loss']:.4f} | value loss {last_ppo_metrics['value_loss']:.4f} | "
                f"entropy {last_ppo_metrics['entropy']:.4f} | KL {last_ppo_metrics['approximate_kl']:.5f} | "
                f"clip {last_ppo_metrics['clip_fraction']:.3f} | explained variance {last_ppo_metrics['explained_variance']:.3f} | "
                f"lr {config.learning_rate:g} | best eval survival {best_survival:.3f}s",
                flush=True,
            )
            while next_progress <= transitions:
                next_progress += progress_interval

    save_checkpoint(latest_path, model, optimizer, transitions, updates, best_score,
                    best_metrics, config, args, env)
    if not best_path.exists():
        save_checkpoint(best_path, model, optimizer, transitions, updates, best_score,
                        best_metrics, config, args, env)

    elapsed = time.perf_counter() - start_time
    print("Training complete", flush=True)
    print(f"Total transitions: {transitions:,}; PPO updates: {updates}; elapsed: {elapsed:.1f}s", flush=True)
    print(f"Best checkpoint: {best_path}", flush=True)
    if best_metrics:
        print(f"Best evaluation mean survival: {best_metrics['trained']['mean_survival_time']:.3f}s; "
              f"no-assist: {best_metrics['no_assist']['mean_survival_time']:.3f}s; "
              f"crash-rate reduction: {best_metrics['no_assist']['crash_rate'] - best_metrics['trained']['crash_rate']:.3f}; "
              f"mean assist: {best_metrics['trained']['mean_assist_magnitude']:.4f}; "
              f"player agency: {'pass' if best_metrics['player_agency']['pass'] else 'fail'}", flush=True)

    if not args.no_auto_export:
        print(f"Exporting selected actor to {args.model_output}", flush=True)
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        selected = ActorCritic()
        selected.load_state_dict(best_checkpoint["model_state"])
        write_model(selected.actor, args.model_output)
        print(f"Export complete: {args.model_output}", flush=True)
        print(f"Baking selected actor into {args.header_output}", flush=True)
        bake_header(args.model_output, args.header_output)
        print(f"Bake complete: {args.header_output}", flush=True)
        report_path = Path("doc/stability_assist_evaluation.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(best_metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Evaluation report written: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
