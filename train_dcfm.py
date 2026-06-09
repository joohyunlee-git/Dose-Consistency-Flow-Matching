# train_dfcm.py
#
# Multi-dose training with FILM dose conditioning + Consistency regularization
# fm_pred_type: "velocity" or "sample"
# cons: asymmetric dose-pair sampling + curriculum gap ramp-up
#
# Args:
#   --cons_loss       float, consistency loss weight (default=0.1)
#   --cons_ramp_start float, fraction of total_step (default=0.1)
#   --cons_ramp_end   float, fraction of total_step (default=0.5)
#   --gap_init        float, initial max_gap_ratio   (default=0.3)
#   --gap_ramp_start  float, fraction of total_step (default=0.0)
#   --gap_ramp_end    float, fraction of total_step (default=0.5)
#   --val_frequency   int,   default=save_frequency
#   --val_n_subjects  int,   default=50

import sys

import os
import json
import time
import datetime
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from flow_matching.path import CondOTProbPath, VPProbPath, LinearVPProbPath, CosineProbPath
from flow_matching.utils import expand_tensor_like
from models.ema import EMA
from models.model_configs import instantiate_film, instantiate_discriminator

from train_arg_parser import get_args_parser
from dataset_multi import MultiDoseTrain, collate_dose_dict, InfiniteSampler

from training.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from training.load_and_save import load_model_step, save_model_step, save_best_step, load_discriminator_step, save_discriminator_step

logger = logging.getLogger(__name__)

KEEP_LAST_CKPTS = 10


# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────

def parse_dose_value(drf_str: str) -> float:
    return 1.0 / float(drf_str)


def skewed_timestep_sample(B: int, device, P_mean=-1.2, P_std=1.2):
    rnd   = torch.randn(B, device=device)
    sigma = (rnd * P_std + P_mean).exp()
    return (1 / (1 + sigma)).clamp(1e-4, 1.0).float()


def sample_t(args, B, device, eps_t=1e-4):
    if args.time_schedule == "uniform":
        t = torch.rand(B, device=device)
    elif args.time_schedule == "beta":
        t = torch.distributions.Beta(0.5, 2.0).sample((B,)).to(device)
    elif args.time_schedule == "skewed":
        return skewed_timestep_sample(B, device)
    else:
        return torch.zeros(B, device=device)
    return t * (1 - eps_t - 1e-4) + eps_t


def ramp_linear(step: int, start: int, end: int) -> float:
    if step <= start: return 0.0
    if step >= end:   return 1.0
    return (step - start) / max(1, end - start)


def sample_dose_pair_curriculum(dose_vals, B, max_gap_ratio):
    n       = len(dose_vals)
    max_gap = max(1, int(max_gap_ratio * (n - 1)))
    i_a     = torch.randint(0, n - 1, ()).item()
    delta   = torch.randint(1, max_gap + 1, ()).item()
    i_b     = min(i_a + delta, n - 1)
    return i_a, i_b, dose_vals[i_a].expand(B), dose_vals[i_b].expand(B)


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def squeeze_vol(v):
    return v.squeeze(0) if (v.ndim == 4 and v.shape[0] == 1) else v


def center_slices(vol: np.ndarray):
    D, H, W = vol.shape
    return vol[D // 2], vol[:, H // 2], vol[:, :, W // 2]


def cleanup_old_checkpoints(output_dir: str, keep_last: int = KEEP_LAST_CKPTS, checkpoint_key: str = "checkpoint"):
    ckpt_files = sorted(
        [p for p in Path(output_dir).glob(f"{checkpoint_key}-*.pth")
         if p.stem.split("-")[1].isdigit()],
        key=lambda p: int(p.stem.split("-")[1]),
    )
    for old in ckpt_files[:-keep_last]:
        try:
            old.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove {old.name}: {e}")


def compute_pred_nd_dm(pred, x_t, t_view, path, dm_pred_type):
    if dm_pred_type == "sample":
        return pred
    elif dm_pred_type == "epsilon":
        return path.epsilon_to_target(pred, x_t, t_view)
    elif dm_pred_type == "v_prediction":
        sched   = path.scheduler(t_view.squeeze())
        alpha_t = expand_tensor_like(sched.alpha_t, x_t)
        sigma_t = expand_tensor_like(sched.sigma_t, x_t)
        return alpha_t * x_t - sigma_t * pred
    else:
        raise ValueError(f"Unknown dm_pred_type: {dm_pred_type!r}")


def compute_pred_nd_fm(pred, x_t, t_view, fm_pred_type):
    if fm_pred_type == "velocity":
        return x_t + (1 - t_view) * pred
    else:  # sample
        return pred


# ─────────────────────────────────────────────────────────────
# Forward
# ─────────────────────────────────────────────────────────────

def forward_cons_sym(args, model, path, criterion, device,
                 x_a, x_b, x_nd, d_a, d_b, B, lambda_cons, eps_t):
    """
    dose pair (x_a, x_b) → FM/cons loss calculation
    Returns: loss, loss_recon, loss_cons, vis_pred_nd
    """
    loss_cons   = torch.zeros((), device=device)
    vis_pred_nd = None

    if args.model in ("unet"):
        t_zero = torch.zeros(B, device=device)
        pred_a = model(x_a, t_zero, extra={"dose": d_a})
        pred_b = model(x_b, t_zero, extra={"dose": d_b})
        loss_recon = criterion(pred_a, x_nd) + criterion(pred_b, x_nd)

        if lambda_cons > 0:
            loss_cons = criterion(pred_a, pred_b)
        vis_pred_nd = pred_a

    elif args.model == "fm":
        t      = sample_t(args, B, device, eps_t)
        t_view = t.view(B, 1, 1, 1, 1)
        samp_a = path.sample(t=t, x_0=x_a, x_1=x_nd)
        samp_b = path.sample(t=t, x_0=x_b, x_1=x_nd)

        # batched forward
        x_ab   = torch.cat([x_a, x_b], dim=0)
        d_ab   = torch.cat([d_a, d_b], dim=0)
        xt_ab  = torch.cat([samp_a.x_t, samp_b.x_t], dim=0)
        t_ab   = torch.cat([t, t], dim=0)
        pred_ab = model(xt_ab, t_ab, extra={"dose": d_ab})
        pred_a, pred_b = pred_ab.chunk(2, dim=0)

        def _fm_target(samp):
            return samp.dx_t if args.fm_pred_type == "velocity" else x_nd

        loss_recon = criterion(pred_a, _fm_target(samp_a)) + criterion(pred_b, _fm_target(samp_b))

        if lambda_cons > 0:
            pnd_a = compute_pred_nd_fm(pred_a, samp_a.x_t, t_view, args.fm_pred_type)
            pnd_b = compute_pred_nd_fm(pred_b, samp_b.x_t, t_view, args.fm_pred_type)
            loss_cons = criterion(pnd_a, pnd_b)

        vis_pred_nd = compute_pred_nd_fm(pred_a.detach(), samp_a.x_t, t_view, args.fm_pred_type)

    elif args.model == "dm":
        t      = sample_t(args, B, device, eps_t)
        t_view = t.view(B, 1, 1, 1, 1)
        noise  = torch.randn_like(x_nd)
        samp_a = path.sample(t=t, x_0=noise, x_1=x_nd)
        samp_b = path.sample(t=t, x_0=noise, x_1=x_nd)  # same noise

        def _dm_target(s):
            if args.dm_pred_type == "epsilon":        return s.x_0
            elif args.dm_pred_type == "v_prediction": return s.dx_t
            else:                                     return s.x_1

        x_ab    = torch.cat([x_a, x_b], dim=0)
        d_ab    = torch.cat([d_a, d_b], dim=0)
        xt_ab   = torch.cat([samp_a.x_t, samp_b.x_t], dim=0)
        t_ab    = torch.cat([t, t], dim=0)
        pred_ab = model(xt_ab, t_ab, extra={"dose": d_ab})
        pred_a, pred_b = pred_ab.chunk(2, dim=0)

        loss_recon = criterion(pred_a, _dm_target(samp_a)) + criterion(pred_b, _dm_target(samp_b))

        if lambda_cons > 0:
            pnd_a = compute_pred_nd_dm(pred_a, samp_a.x_t, t_view, path, args.dm_pred_type)
            pnd_b = compute_pred_nd_dm(pred_b, samp_b.x_t, t_view, path, args.dm_pred_type)
            loss_cons = criterion(pnd_a, pnd_b)

        vis_pred_nd = compute_pred_nd_dm(pred_a.detach(), samp_a.x_t, t_view, path, args.dm_pred_type)
    
    elif args.model == "gan":
        t_zero = torch.zeros(B, device=device)
        pred_a = model(x_a, t_zero, extra={"dose": d_a})
        pred_b = model(x_b, t_zero, extra={"dose": d_b})
        loss_recon = torch.zeros((), device=device)  # gan loss는 train loop에서 별도 처리
        if lambda_cons > 0:
            loss_cons = criterion(pred_a, pred_b)
        vis_pred_nd = pred_a

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    loss = loss_recon + lambda_cons * loss_cons
    return loss, loss_recon, loss_cons, vis_pred_nd


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(args, model, path, criterion, device,
                   val_loader, dose_levels, nd_key, dose_vals,
                   eps_t, lambda_cons, max_gap_ratio):
    model.eval()
    loss_sum = recon_sum = cons_sum = 0.0
    n_batches = 0
    first_vis = None

    for batch_dict in val_loader:
        x_nd = batch_dict[nd_key].to(device, non_blocking=True)
        B    = x_nd.shape[0]

        i_a, i_b, d_a_vals, d_b_vals = sample_dose_pair_curriculum(
            dose_vals, B, max_gap_ratio)
        x_a = batch_dict[str(dose_levels[i_a])].to(device, non_blocking=True)
        x_b = batch_dict[str(dose_levels[i_b])].to(device, non_blocking=True)
        d_a = d_a_vals[:B].to(device)
        d_b = d_b_vals[:B].to(device)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            loss, loss_recon, loss_cons, vis_pred_nd = forward_cons_sym(
                args, model, path, criterion, device,
                x_a, x_b, x_nd, d_a, d_b, B, lambda_cons, eps_t)

        loss_sum  += loss.item()
        recon_sum += loss_recon.item()
        cons_sum  += loss_cons.item()
        n_batches += 1

        if first_vis is None:
            first_vis = {
                "x_src":   x_a.detach().cpu(),
                "x_nd":    x_nd.detach().cpu(),
                "pred_nd": vis_pred_nd.detach().cpu(),
                "d":       float(d_a[0].cpu()),
                "x_key":   str(dose_levels[i_a]),
            }

    model.train(True)
    n = max(n_batches, 1)
    return loss_sum/n, recon_sum/n, cons_sum/n, first_vis


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(args):
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_file_path = Path(args.output_dir) / "log.txt"
    logger.addHandler(logging.FileHandler(log_file_path, mode="a", encoding="utf-8"))
    logger.info("{}".format(args).replace(", ", ",\n"))

    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f)

    device = torch.device(args.device)

    # ── Val Config ─────────────────────────────────────────────
    val_n_subjects = getattr(args, "val_n_subjects", 50)
    val_freq       = getattr(args, "val_frequency",  args.save_frequency)

    # ── dose level Config ──────────────────────────────────────
    def _to_key(d) -> str:
        f = float(d)
        return str(int(f)) if f == int(f) else str(f)

    dose_levels = [_to_key(d) for d in args.dose_levels]
    nd_key      = _to_key(args.nd)
    ref_key     = _to_key(args.ref)

    dose_vals = torch.tensor(
        [parse_dose_value(d) for d in dose_levels],
        dtype=torch.float32, device=device,
    )
    n_doses = len(dose_vals)

    # ── Dataset ─────────────────────────────────────────────
    with open(args.splits_json) as f:
        splits = json.load(f)

    train_file_list = splits[f"fold_{args.fold}"]["train"]
    val_file_list   = splits[f"fold_{args.fold}"]["val"][:val_n_subjects]
    logger.info(f"Fold {args.fold}: {len(train_file_list)} train / {len(val_file_list)} val files")

    train_dataset = MultiDoseTrain(
        file_list   = train_file_list,
        data_path   = args.data_path,
        patch_size  = args.patch_size,
        dose_levels = dose_levels,
        nd          = nd_key,
        ref         = ref_key,
    )
    sampler = InfiniteSampler(train_dataset, seed=args.seed, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler            = sampler,
        batch_size         = args.batch_size,
        num_workers        = args.num_workers,
        pin_memory         = args.pin_mem,
        drop_last          = True,
        collate_fn         = collate_dose_dict,
        prefetch_factor    = 2,
        persistent_workers = True,
    )

    val_dataset = MultiDoseTrain(
        file_list   = val_file_list,
        data_path   = args.data_path,
        patch_size  = args.patch_size,
        dose_levels = dose_levels,
        nd          = nd_key,
        ref         = ref_key,
        seed        = 42,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = min(args.num_workers, 4),
        pin_memory  = args.pin_mem,
        drop_last   = False,
        collate_fn  = collate_dose_dict,
    )

    # ── Model ────────────────────────────────────────────────
    logger.info("Initializing Model")
    model = instantiate_film(
        architechture = args.dataset,
        use_ema       = args.use_ema,
    )
    model.to(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    total_p, train_p = count_parameters(model)
    logger.info(f"Total parameters:     {total_p:,}")
    logger.info(f"Trainable parameters: {train_p:,}")

    # ── Path ─────────────────────────────────────────────────
    path = None
    if args.model == "fm":
        path = {"cond_ot": CondOTProbPath, "vp": VPProbPath,
                "linear_vp": LinearVPProbPath, "cosine": CosineProbPath}[args.fm_path]()
    elif args.model == "dm":
        path = {"vp": VPProbPath, "linear_vp": LinearVPProbPath,
                "cosine": CosineProbPath}[args.dm_path]()

    # ── Loss / Optimizer ─────────────────────────────────────
    criterion = {"l1": torch.nn.L1Loss(), "huber": torch.nn.HuberLoss(delta=0.1)}.get(
        args.loss_type, torch.nn.MSELoss()
    )

    total_step  = args.duration
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.optimizer_betas)
    lr_schedule = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer, total_iters=total_step,
            start_factor=1.0, end_factor=1e-8 / args.lr,
        ) if args.decay_lr else
        torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=total_step, factor=1.0)
    )
    loss_scaler = NativeScaler()
    load_model_step(
        args=args, model_without_ddp=model,
        optimizer=optimizer, loss_scaler=loss_scaler, lr_schedule=lr_schedule,
    )
    model.train(True)

    # ── GAN Discriminator ────────────────────────────────────
    if args.model == "gan":
        best_d_loss_path = Path(args.output_dir) / "best_d_loss.json"
        if args.resume and best_d_loss_path.exists():
            with open(best_d_loss_path) as f:
                best_d_loss = json.load(f)["best_d_loss"]
        else:
            best_d_loss = float("inf")

        discriminator = instantiate_discriminator(use_ema=args.use_ema)
        discriminator.to(device)
        d_optimizer   = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, betas=args.optimizer_betas)
        d_lr_schedule = (
            torch.optim.lr_scheduler.LinearLR(
                d_optimizer, total_iters=total_step,
                start_factor=1.0, end_factor=1e-8 / args.lr,
            ) if args.decay_lr else
            torch.optim.lr_scheduler.ConstantLR(d_optimizer, total_iters=total_step, factor=1.0)
        )
        d_loss_scaler = NativeScaler()
        load_discriminator_step(
            args=args, model_without_ddp=discriminator,
            optimizer=d_optimizer, loss_scaler=d_loss_scaler, lr_schedule=d_lr_schedule,
        )
        discriminator.train(True)
        criterion_gan = torch.nn.BCEWithLogitsLoss()
        criterion_l1  = torch.nn.L1Loss()

    # ── Curriculum schedule ───────────────────────────────────
    gap_ramp_start  = int(total_step * getattr(args, "gap_ramp_start",  0.0))
    gap_ramp_end    = int(total_step * getattr(args, "gap_ramp_end",    0.5))
    cons_ramp_start = int(total_step * getattr(args, "cons_ramp_start", 0.1))
    cons_ramp_end   = int(total_step * getattr(args, "cons_ramp_end",   0.5))
    gap_init        = getattr(args, "gap_init",   0.3)
    cons_loss_w     = getattr(args, "cons_loss",  0.1)

    logger.info(f"Cons weight={cons_loss_w}  ramp=[{cons_ramp_start},{cons_ramp_end}]")
    logger.info(f"Gap init={gap_init}  ramp=[{gap_ramp_start},{gap_ramp_end}]")

    # ── Training loop Initialization ─────────────────────────────────
    accum_iter      = args.accum_iter
    cur_step        = args.start_step
    cur_tick        = (cur_step // args.print_frequency) + 1
    tick_start_step = cur_step

    best_loss_path = Path(args.output_dir) / "best_loss.json"
    if args.resume and best_loss_path.exists():
        with open(best_loss_path) as f:
            ckpt_info = json.load(f)
        best_train_loss = ckpt_info.get("best_train_loss", float("inf"))
        best_val_loss   = ckpt_info.get("best_val_loss",   float("inf"))
        logger.info(f"Resumed best_train={best_train_loss:.6f}  best_val={best_val_loss:.6f}")
    else:
        best_train_loss = float("inf")
        best_val_loss   = float("inf")

    last_val_loss = float("nan")

    sample_path = Path(args.output_dir) / "sample"
    os.makedirs(sample_path, exist_ok=True)

    start_time      = time.time()
    tick_start_time = time.time()
    data_iter       = iter(data_loader)

    logger.info(f"Start {cur_step} → {total_step}")
    logger.info(f"Dose levels: {dose_levels}")
    logger.info(f"Val: every {val_freq} steps, {len(val_file_list)} subjects")

    optimizer.zero_grad(set_to_none=True)
    eps_t = args.eps_t

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ─────────────────────────────────────────────────────────
    while True:
        loss_sum       = 0.0
        loss_recon_sum = 0.0
        loss_cons_sum  = 0.0

        if args.model == "gan":
            d_loss_sum = 0.0

        vis_batch = None
        x_key     = None

        # ── curriculum ───────────────────────────────────────
        max_gap_ratio = gap_init + (1.0 - gap_init) * ramp_linear(
            cur_step, gap_ramp_start, gap_ramp_end)
        lambda_cons = cons_loss_w * ramp_linear(
            cur_step, cons_ramp_start, cons_ramp_end)

        for i in range(accum_iter):
            apply_update = (i + 1) % accum_iter == 0

            try:
                batch_dict = next(data_iter)
            except StopIteration:
                data_iter  = iter(data_loader)
                batch_dict = next(data_iter)

            # ── dose pair sample ────────────────────────────────
            i_a, i_b, d_a_vals, d_b_vals = sample_dose_pair_curriculum(
                dose_vals, args.batch_size, max_gap_ratio)
            x_key = str(dose_levels[i_a])

            x_a  = batch_dict[str(dose_levels[i_a])].to(device, non_blocking=True)
            x_b  = batch_dict[str(dose_levels[i_b])].to(device, non_blocking=True)
            x_nd = batch_dict[nd_key].to(device, non_blocking=True)
            B    = x_a.shape[0]
            d_a  = d_a_vals[:B].to(device)
            d_b  = d_b_vals[:B].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):

                if args.model == "gan":
                    t_zero = torch.zeros(B, device=device)

                    for p in discriminator.parameters():
                        p.requires_grad_(False)
                    pred_a = model(x_a, t_zero, extra={"dose": d_a})
                    pred_b = model(x_b, t_zero, extra={"dose": d_b})

                    dis_fake_a = discriminator(pred_a, x_a)
                    dis_fake_b = discriminator(pred_b, x_b)
                    valid      = torch.ones_like(dis_fake_a)
                    loss_gan   = criterion_gan(dis_fake_a, valid) + criterion_gan(dis_fake_b, valid)
                    loss_l1    = criterion_l1(pred_a, x_nd) + criterion_l1(pred_b, x_nd)
                    loss_cons_val = criterion(pred_a, pred_b) if lambda_cons > 0 else torch.zeros((), device=device)
                    loss_recon = loss_gan + 100 * loss_l1
                    loss       = loss_recon + lambda_cons * loss_cons_val

                    for p in discriminator.parameters():
                        p.requires_grad_(True)
                    dis_real   = discriminator(x_nd, x_a)
                    dis_fake2a = discriminator(pred_a.detach(), x_a)
                    dis_fake2b = discriminator(pred_b.detach(), x_b)
                    fake       = torch.zeros_like(dis_real)
                    d_loss     = ((criterion_gan(dis_real, valid) +
                                criterion_gan(dis_fake2a, fake) +
                                criterion_gan(dis_fake2b, fake)) / 3.0) / accum_iter

                    vis_pred_nd = pred_a

                else:
                    loss, loss_recon, loss_cons_val, vis_pred_nd = forward_cons_sym(
                        args, model, path, criterion, device,
                        x_a, x_b, x_nd, d_a, d_b, B, lambda_cons, eps_t)

                loss = loss / accum_iter

            loss_scaler(
                loss, optimizer,
                parameters  = model.parameters(),
                update_grad = apply_update,
                clip_grad   = 1.0,
            )

            loss_sum       += loss.item()
            loss_recon_sum += (loss_recon / accum_iter).item()
            loss_cons_sum  += (loss_cons_val / accum_iter).item()

            if apply_update:
                if isinstance(model, EMA):
                    model.update_ema()
                lr_schedule.step()
                optimizer.zero_grad(set_to_none=True)

                if args.model == "gan":
                    d_loss_scaler(d_loss, d_optimizer, parameters=discriminator.parameters(),
                                  update_grad=True, clip_grad=1.0)
                    d_loss_sum += d_loss.item()
                    d_lr_schedule.step()
                    d_optimizer.zero_grad(set_to_none=True)

            if vis_batch is None:
                vis_batch = {
                    "x_src":   x_a.detach(),
                    "x_nd":    x_nd.detach(),
                    "pred_nd": vis_pred_nd.detach(),
                    "d":       float(d_a[0].cpu()),
                }

        cur_step += 1
        done = cur_step >= total_step
        lr   = optimizer.param_groups[0]["lr"]

        # ── Logging & Visualization ───────────────────────────
        if cur_step >= tick_start_step + args.print_frequency or done:
            elapsed = datetime.timedelta(seconds=int(time.time() - start_time))
            h, rem  = divmod(elapsed.total_seconds(), 3600)
            m, s    = divmod(rem, 60)
            elapsed_str = (
                (f"{int(h)}h" if h else "")
                + (f"{int(m)}m" if m else "")
                + (f"{int(s)}s" if s or not (h or m) else "")
            )

            logger.info(
                f"tick={cur_tick:5d}  step={cur_step:9d}  "
                f"loss={loss_sum:.6f}  fm={loss_recon_sum:.6f}  cons={loss_cons_sum:.6f}  "
                f"val={last_val_loss:.6f}  "
                f"gap={max_gap_ratio:.2f}  λ={lambda_cons:.4f}  "
                f"dose={x_key}  lr={lr:.4e}  time={elapsed_str}"
            )

            # ── Visualization ─────────────────────────────────────
            vb   = vis_batch
            vols = {
                f"x_src (d={vb['d']:.3f})": squeeze_vol(vb["x_src"][0]).cpu().float(),
                "x_nd (target)":            squeeze_vol(vb["x_nd"][0]).cpu().float(),
                "pred_nd":                  squeeze_vol(vb["pred_nd"][0]).cpu().float(),
            }
            nd_np = vb["x_nd"][0].cpu().float().numpy()
            vmin  = np.percentile(nd_np, 0.01)
            vmax  = np.percentile(nd_np, 99.99)

            n_rows = len(vols)
            fig    = plt.figure(figsize=(13, 3.5 * n_rows))
            fig.suptitle(
                f"[Train] Step {cur_step} | dose={x_key} (d={vb['d']:.3f}) | "
                f"gap={max_gap_ratio:.2f} | λ={lambda_cons:.3f}",
                fontsize=12, fontweight="bold", y=0.99,
            )
            gs = plt.GridSpec(
                n_rows, 3, figure=fig,
                hspace=0.06, wspace=0.04,
                top=0.95, bottom=0.02, left=0.10, right=0.98,
            )
            col_labels = ["Axial", "Coronal", "Sagittal"]
            for row_i, (lbl, vol) in enumerate(vols.items()):
                slices = center_slices(vol.numpy())
                for col_i, (sl, clbl) in enumerate(zip(slices, col_labels)):
                    ax = fig.add_subplot(gs[row_i, col_i])
                    im = ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
                    if row_i == 0: ax.set_title(clbl, fontsize=10, pad=3)
                    if col_i == 0: ax.set_ylabel(lbl, fontsize=9, labelpad=5)
                    ax.set_xticks([]); ax.set_yticks([])
                    if col_i == 2:
                        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                        cb.ax.tick_params(labelsize=7)
            fig.savefig(os.path.join(sample_path, f"step_{cur_step}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

            # ── Best train checkpoint ─────────────────────
            if loss_sum < best_train_loss:
                best_train_loss = loss_sum
                with open(best_loss_path, "w") as f:
                    json.dump({"best_train_loss": best_train_loss,
                               "best_val_loss":   best_val_loss,
                               "step":            cur_step}, f)
                logger.info(f"Best train at step {cur_step}, loss={best_train_loss:.6f}")
                save_best_step(
                    args=args, model=model, model_without_ddp=model,
                    optimizer=optimizer, lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler, step=cur_step, suffix="train",
                )
                if args.model == "gan":
                    best_d_loss = d_loss_sum
                    with open(best_d_loss_path, "w") as f:
                        json.dump({"best_d_loss": best_d_loss, "step": cur_step}, f)
                    torch.save({"step": cur_step, "model_state": discriminator.state_dict()},
                               Path(args.output_dir) / "d-checkpoint-best.pth")

            tick_start_time = time.time()
            cur_tick       += 1
            tick_start_step = cur_step

        # ── Validation ────────────────────────────────────────
        if val_freq > 0 and cur_step % val_freq == 0:
            val_loss, val_fm, val_cons, val_vis = run_validation(
                args, model, path, criterion, device,
                val_loader, dose_levels, nd_key, dose_vals,
                eps_t, lambda_cons, max_gap_ratio,
            )
            last_val_loss = val_loss
            logger.info(
                f"[Val] step={cur_step:9d}  loss={val_loss:.6f}  "
                f"fm={val_fm:.6f}  cons={val_cons:.6f}  "
                f"(best={best_val_loss:.6f})"
            )

            # ── Val Visualization ────────────────────────────────
            if val_vis is not None:
                vv = val_vis
                val_vols = {
                    f"x_src (d={vv['d']:.3f})": squeeze_vol(vv["x_src"][0]).float(),
                    "x_nd (target)":            squeeze_vol(vv["x_nd"][0]).float(),
                    "pred_nd":                  squeeze_vol(vv["pred_nd"][0]).float(),
                }
                vnd_np = vv["x_nd"][0].float().numpy()
                vvmin  = np.percentile(vnd_np, 0.01)
                vvmax  = np.percentile(vnd_np, 99.99)

                n_rows  = len(val_vols)
                val_fig = plt.figure(figsize=(13, 3.5 * n_rows))
                val_fig.suptitle(
                    f"[Val] Step {cur_step} | dose={vv['x_key']} (d={vv['d']:.3f}) | "
                    f"val_loss={val_loss:.6f} | λ={lambda_cons:.3f}",
                    fontsize=12, fontweight="bold", y=0.99,
                )
                val_gs = plt.GridSpec(
                    n_rows, 3, figure=val_fig,
                    hspace=0.06, wspace=0.04,
                    top=0.95, bottom=0.02, left=0.10, right=0.98,
                )
                for row_i, (lbl, vol) in enumerate(val_vols.items()):
                    slices = center_slices(vol.numpy())
                    for col_i, (sl, clbl) in enumerate(zip(slices, ["Axial", "Coronal", "Sagittal"])):
                        ax = val_fig.add_subplot(val_gs[row_i, col_i])
                        im = ax.imshow(sl, cmap="gray", vmin=vvmin, vmax=vvmax, aspect="equal")
                        if row_i == 0: ax.set_title(clbl, fontsize=10, pad=3)
                        if col_i == 0: ax.set_ylabel(lbl, fontsize=9, labelpad=5)
                        ax.set_xticks([]); ax.set_yticks([])
                        if col_i == 2:
                            cb = val_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                            cb.ax.tick_params(labelsize=7)
                val_fig.savefig(os.path.join(sample_path, f"val_step_{cur_step}.png"),
                                dpi=150, bbox_inches="tight")
                plt.close(val_fig)

            # ── Best val checkpoint ───────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                with open(best_loss_path, "w") as f:
                    json.dump({"best_train_loss": best_train_loss,
                               "best_val_loss":   best_val_loss,
                               "step":            cur_step}, f)
                logger.info(f"  → New best val at step {cur_step}, val_loss={best_val_loss:.6f}")
                save_best_step(
                    args=args, model=model, model_without_ddp=model,
                    optimizer=optimizer, lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler, step=cur_step, suffix="val",
                )

        # ── Checkpoint ───────────────────────────────────────
        if args.output_dir and (
            (args.save_frequency > 0 and cur_step % args.save_frequency == 0)
            or args.test_run or done
        ):
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{cur_step}.pth")
            if not os.path.exists(ckpt_path):
                save_model_step(
                    args=args, model=model, model_without_ddp=model,
                    optimizer=optimizer, lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler, step=cur_step,
                )
                cleanup_old_checkpoints(args.output_dir, keep_last=KEEP_LAST_CKPTS)

                if args.model == "gan":
                    save_discriminator_step(
                        args=args, model_without_ddp=discriminator,
                        optimizer=d_optimizer, lr_schedule=d_lr_schedule,
                        loss_scaler=d_loss_scaler, step=cur_step)
                    cleanup_old_checkpoints(args.output_dir, checkpoint_key="d-checkpoint")

        if done:
            break

    logger.info(f"Training time {datetime.timedelta(seconds=int(time.time() - start_time))}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    base_dir = Path(args.train_dir)
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    if args.resume:
        old_dir  = Path(args.resume).parent
        old_name = old_dir.name

        try:
            parts        = old_name.rsplit("_", 2)
            run_id_str   = parts[-1]
            old_duration = int(parts[-2])
            prefix_base  = parts[-3] if len(parts) > 2 else old_name
        except Exception:
            old_duration = args.duration
            prefix_base  = old_name
            run_id_str   = "00000"

        if args.duration > old_duration:
            import shutil
            new_name = f"{prefix_base}_{int(args.duration)}_{run_id_str}"
            new_dir  = base_dir / new_name
            new_dir.mkdir(parents=True, exist_ok=True)
            for fname in ["best_loss.json", "log.txt"]:
                src = old_dir / fname
                if src.exists():
                    shutil.copy2(src, new_dir / fname)
            for ckpt in sorted(old_dir.glob("checkpoint-*.pth")):
                dst = new_dir / ckpt.name
                if not dst.exists():
                    shutil.copy2(ckpt, dst)
            save_name = new_name
            logger.info(f"Extended duration {old_duration} → {args.duration}: {old_name} → {new_name}")
        else:
            save_name = old_name
            if args.duration < old_duration:
                logger.warning(
                    f"args.duration ({args.duration}) < folder duration ({old_duration}), "
                    f"resuming in same folder"
                )

    else:
        dir_name = f"{args.dataset}"
        dir_name += f"_{args.model}"
        if args.model == "dm":
            dir_name += f"_{args.dm_pred_type}_{args.dm_path}"
        if args.model == "fm":
            dir_name += f"_{args.fm_pred_type}_{args.fm_path}"
        if args.loss_type != "l2":
            dir_name += f"_{args.loss_type}"
        dir_name += f"_sym_cons_{args.cons_loss}_multi_{int(args.duration)}"

        existing_runs = sorted([
            d for d in base_dir.iterdir()
            if d.is_dir() and f"{dir_name}_0" in d.name
        ])
        run_id    = len(existing_runs)
        save_name = f"{dir_name}_{run_id:05d}"

    args.output_dir = str(base_dir / save_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)