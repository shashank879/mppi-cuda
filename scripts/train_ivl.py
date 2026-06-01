"""Train V(s, g) via IVL on an NPZ replay buffer.

The buffer comes from `scripts/collect_ivl_data.py`. Training is just
supervised regression — no env steps, no rollouts during training. This
is what makes IVL fast and the "kernel made the dataset, learning runs
offline" story land.

After training, the checkpoint contains the V network with normalisation
stats baked in as buffers, so inference (CudaMPPIController's
α-blended terminal cost) only needs to call `v_net.v_min(s, g)`.

Typical usage:
    PYTHONPATH=. python scripts/train_ivl.py \\
        --data data/ivl_buffer.npz \\
        --out  data/ivl_value.pt \\
        --steps 50000
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from mppi_cuda.ivl import GCVNetwork, value_loss, polyak_update
from mppi_cuda.ivl_data import IVLDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/ivl_buffer.npz")
    p.add_argument("--out",        default="data/ivl_value.pt")
    p.add_argument("--steps",      type=int,   default=50000)
    p.add_argument("--batch_size", type=int,   default=1024)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--discount",   type=float, default=0.99)
    p.add_argument("--tau",        type=float, default=0.005)
    p.add_argument("--expectile",  type=float, default=0.9)
    p.add_argument("--hidden",     type=int, nargs="+", default=[512, 512, 512],
                   help="V-network hidden widths. Default is 256x2 (small, fast).")
    p.add_argument("--layer_norm", type=bool, default=False,
                   help="Disable layer-norm (default is on; usually a bad idea to disable).")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout.")
    # Goal-relabel mixture
    p.add_argument("--p_cur",    type=float, default=0.20)
    p.add_argument("--p_traj",   type=float, default=0.50)
    p.add_argument("--p_random", type=float, default=0.30)
    p.add_argument("--geom_p",   type=float, default=0.00,
                   help="Geometric parameter for future-target sampling. "
                        "Mean lookahead ~ 1/geom_p ticks.")
    # Logging / checkpointing
    p.add_argument("--log_every",        type=int, default=500)
    p.add_argument("--checkpoint_every", type=int, default=10000)
    p.add_argument("--seed",             type=int, default=0)
    p.add_argument("--device",           default="cuda")
    p.add_argument("--grad_clip",        type=float, default=0.0,
                   help="If > 0, clip gradient L2 norm to this value.")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --------------- data ---------------
    dataset = IVLDataset(
        args.data,
        device=args.device,
        p_cur_goal=args.p_cur,
        p_traj_goal=args.p_traj,
        p_random_goal=args.p_random,
        geom_p=args.geom_p,
    )
    data_example = dataset.sample_batch(5)
    print(dataset)

    norm = dataset.fit_normalisation()
    print(f"State range: mean={norm['state_mean'].mean():.3f}, "
          f"std={norm['state_std'].mean():.3f}")
    print(f"Goal range:  mean={norm['goal_mean']}, std={norm['goal_std']}")

    # --------------- networks ---------------
    state_dim = data_example['s'].shape[-1]
    goal_dim  = data_example['g'].shape[-1]

    v_net = GCVNetwork(
        state_dim=state_dim, goal_dim=goal_dim,
        hidden=tuple(args.hidden),
        use_layer_norm=args.layer_norm,
        dropout=args.dropout,
        state_mean=norm["state_mean"], state_std=norm["state_std"],
        goal_mean=norm["goal_mean"],   goal_std=norm["goal_std"],
    ).to(args.device)
    target_v_net = deepcopy(v_net)
    for p_ in target_v_net.parameters():
        p_.requires_grad_(False)

    n_params = sum(p.numel() for p in v_net.parameters())
    print(f"V network: hidden={args.hidden}, "
          f"layer_norm={args.layer_norm}, params={n_params:,}")

    optimizer = torch.optim.Adam(v_net.parameters(), lr=args.lr)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    # --------------- training loop ---------------
    print(f"\nTraining for {args.steps:,} steps, batch={args.batch_size}")
    t_start = time.time()
    last_log_t = t_start
    last_log_step = 0
    log_history: list[dict] = []

    for step in range(args.steps):
        batch = dataset.sample_batch(args.batch_size, generator=generator)
        loss, info = value_loss(
            batch, v_net, target_v_net,
            discount=args.discount, expectile=args.expectile,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(v_net.parameters(), args.grad_clip)
        optimizer.step()

        polyak_update(v_net, target_v_net, args.tau)

        if step % args.log_every == 0 or step == args.steps - 1:
            now = time.time()
            steps_since = step - last_log_step
            rate = steps_since / max(now - last_log_t, 1e-6)
            elapsed = now - t_start
            eta = (args.steps - step - 1) / max(rate, 1e-6)
            info_with_meta = {
                "step": step, "elapsed_s": round(elapsed, 1),
                "steps_per_s": round(rate, 1), **info,
            }
            log_history.append(info_with_meta)
            print(f"  step {step:>7d}  "
                  f"loss={info['value_loss']:7.4f}  "
                  f"v_mean={info['v_mean']:+8.3f}  "
                  f"adv_mean={info['adv_mean']:+7.4f}  "
                  f"r_mean={info['r_mean']:+7.4f}  "
                  f"{rate:5.0f} step/s  eta {eta/60:5.1f} min")
            last_log_t = now
            last_log_step = step

        if (step + 1) % args.checkpoint_every == 0 or step == args.steps - 1:
            ckpt = {
                "step": step + 1,
                "v_net": v_net.state_dict(),
                "target_v_net": target_v_net.state_dict(),
                "config": {
                    "state_dim": state_dim,
                    "goal_dim": goal_dim,
                    "hidden": tuple(args.hidden),
                    "use_layer_norm": args.layer_norm,
                    'dropout': args.dropout,
                    "discount": args.discount,
                    "tau": args.tau,
                    "expectile": args.expectile,
                    "alpha_u": dataset.alpha_u,
                    "alpha_du": dataset.alpha_du,
                },
                "log_history": log_history,
            }
            torch.save(ckpt, out_path)
            print(f"  [checkpoint] {out_path}")

    print(f"\nDone in {(time.time() - t_start)/60:.1f} min.")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
