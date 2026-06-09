# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import re
import argparse
import logging

from models.model_configs import MODEL_CONFIGS
from torchdiffeq._impl.odeint import SOLVERS

logger = logging.getLogger(__name__)


# ── List parsers ─────────────────────────────────────────────────────────────

def parse_str_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(str(m.group(1)), str(m.group(2)) + 1))
        else:
            ranges.append(str(p))
    return ranges


def parse_float_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(float(m.group(1)), str(m.group(2)) + 1))
        else:
            ranges.append(float(p))
    return ranges


def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), str(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ── Argument parser ───────────────────────────────────────────────────────────

def get_args_parser():
    parser = argparse.ArgumentParser("ULD PET training / inference", add_help=False)

    # ── Paths ─────────────────────────────────────────────────────────────────
    path = parser.add_argument_group("Paths")
    path.add_argument("--data_path",   default="./data/uld_h5",        type=str)
    path.add_argument("--splits_json", default="./data/uld_h5_splits.json", type=str)
    path.add_argument("--train_dir",   default="./results/train",       type=str)
    path.add_argument("--test_dir",    default="./results/test",        type=str)
    path.add_argument("--resume",    default="", help="Path to checkpoint to resume from")
    path.add_argument("--d_resume",  default="", help="Path to discriminator checkpoint")

    # ── Dataset / dose ────────────────────────────────────────────────────────
    data = parser.add_argument_group("Dataset")
    data.add_argument("--dataset",    default=list(MODEL_CONFIGS.keys())[0],
                      type=str, choices=list(MODEL_CONFIGS.keys()))
    data.add_argument("--fold",        default=0,   type=int)
    data.add_argument("--ld",          default=100, type=int,
                      help="Low-dose reduction factor (e.g. 100 → 1/100 dose)")
    data.add_argument("--nd",          default=1,   type=int,
                      help="Normal-dose key")
    data.add_argument("--ref",         default="50", type=str,
                      help="Reference dose key used for normalisation")
    data.add_argument("--dose_levels", default=[50, 20, 10, 4], type=parse_int_list,
                      help="Dose levels used during multi-dose training")
    data.add_argument("--val_dose_levels", default=[50, 20, 10], type=parse_int_list)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = parser.add_argument_group("Model")
    model.add_argument("--model", default="fm",
                       choices=["unet", "gan", "dm", "fm", "ld"],
                       help="Model architecture / training paradigm")
    model.add_argument("--use_ema", action="store_true",
                       help="Use EMA weights for evaluation")

    # ── Flow Matching ─────────────────────────────────────────────────────────
    fm = parser.add_argument_group("Flow Matching")
    fm.add_argument("--fm_pred_type", default="velocity", choices=["velocity", "sample"],
                    help="FM prediction target: velocity field or clean sample")
    fm.add_argument("--fm_path", default="cond_ot",
                    choices=["cond_ot", "vp", "linear_vp", "cosine"],
                    help="FM probability path / interpolant")
    fm.add_argument("--ode_method", default="euler",
                    choices=list(SOLVERS.keys()),
                    help="ODE solver for FM inference")
    fm.add_argument("--nfe", default=1, type=int,
                    help="Number of function evaluations at inference")

    # ── Diffusion Model ───────────────────────────────────────────────────────
    dm = parser.add_argument_group("Diffusion Model")
    dm.add_argument("--dm_pred_type", default="sample",
                    choices=["epsilon", "v_prediction", "sample"],
                    help="DM prediction target")
    dm.add_argument("--dm_path", default="vp",
                    choices=["vp", "linear_vp", "cosine"],
                    help="DM noise schedule")
    dm.add_argument("--dm_sampling_type", default="ddpm",
                    choices=["ddpm", "ddim", "pf_ode_euler"],
                    help="DM sampling algorithm")

    # ── Training ──────────────────────────────────────────────────────────────
    train = parser.add_argument_group("Training")
    train.add_argument("--duration",    default=100000,    type=float,
                       help="Total training steps")
    train.add_argument("--batch_size",  default=4,    type=int)
    train.add_argument("--accum_iter",  default=4,    type=int,
                       help="Gradient accumulation steps")
    train.add_argument("--patch_size",  default=64,   type=int,
                       help="Cubic patch size for training")
    train.add_argument("--patches_per_vol", default=2, type=int)
    train.add_argument("--seed",        default=0,    type=int)
    train.add_argument("--time_schedule", default="uniform",
                       choices=["uniform", "beta", "skewed"])
    train.add_argument("--loss_type",   default="l2", choices=["l1", "l2", "huber"])
    train.add_argument("--eps_t",       default=1e-4, type=float,
                       help="Minimum timestep epsilon")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optim = parser.add_argument_group("Optimizer")
    optim.add_argument("--lr",             type=float, default=1e-5)
    optim.add_argument("--optimizer_betas", nargs="+", type=float, default=[0.9, 0.95])
    optim.add_argument("--decay_lr",       action="store_true",
                       help="Apply linear LR decay over training")
    optim.add_argument("--end_lr",         type=float, default=1e-6)

    # ── Consistency / Curriculum ──────────────────────────────────────────────
    cons = parser.add_argument_group("Consistency & Curriculum")
    cons.add_argument("--cons_loss",       type=float, default=0.5,
                      help="Consistency loss weight (lambda)")
    cons.add_argument("--cons_ramp_start", type=float, default=0.20,
                      help="Fraction of total steps before consistency loss starts ramping up")
    cons.add_argument("--cons_ramp_end",   type=float, default=0.60,
                      help="Fraction of total steps when consistency loss reaches full weight")
    cons.add_argument("--gap_init",        type=float, default=0.20,
                      help="Initial max dose-gap ratio for curriculum sampling")
    cons.add_argument("--gap_ramp_start",  type=float, default=0.05,
                      help="Fraction of total steps before dose gap starts ramping up")
    cons.add_argument("--gap_ramp_end",    type=float, default=0.50,
                      help="Fraction of total steps when dose gap reaches maximum")

    # ── Inference / Slab ──────────────────────────────────────────────────────
    infer = parser.add_argument_group("Inference")
    infer.add_argument("--patch_batch_size", default=2, type=int,
                       help="Number of z-slabs processed per forward pass")
    infer.add_argument("--z_patch",   default=64,  type=int)
    infer.add_argument("--z_overlap", default=0.5, type=float)
    infer.add_argument("--sigma_scale", default=0.5, type=float,
                       help="Gaussian overlap-add weight sigma scale")
    infer.add_argument("--file",      default="Siemens_001.h5", type=str,
                       help="Single-file quick test (best: Siemens_001.h5, worst: UI23_363.h5)")

    # ── Validation ────────────────────────────────────────────────────────────
    val = parser.add_argument_group("Validation")
    val.add_argument("--val_n_subjects", type=int, default=50)
    val.add_argument("--val_frequency",  type=int, default=500)

    # ── Logging / Checkpointing ───────────────────────────────────────────────
    log = parser.add_argument_group("Logging & Checkpointing")
    log.add_argument("--save_frequency",  default=10, type=int,
                     help="Save checkpoint every N steps")
    log.add_argument("--print_frequency", default=1,  type=int,
                     help="Log metrics every N steps")
    log.add_argument("--start_step",      default=0,  type=int,
                     help="Step offset when resuming from checkpoint")
    log.add_argument("--test_run", action="store_true",
                     help="Run a single batch for smoke-testing")

    # ── Hardware ──────────────────────────────────────────────────────────────
    hw = parser.add_argument_group("Hardware")
    hw.add_argument("--device",      default="cuda", type=str)
    hw.add_argument("--gpu",         default=None,   type=int)
    hw.add_argument("--num_workers", default=8,      type=int)
    hw.add_argument("--pin_mem",     action="store_true", default=True,
                    help="Pin CPU memory in DataLoader")
    hw.add_argument("--no_pin_mem",  action="store_false", dest="pin_mem")

    return parser