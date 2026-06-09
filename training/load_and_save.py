# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import torch
from training.distributed_mode import is_main_process


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(
    args, img, model, model_without_ddp, optimizer, lr_schedule, loss_scaler
):
    output_dir = Path(args.output_dir)
    img_name = str(img)
    if loss_scaler is not None:
        checkpoint_paths = [
            output_dir / ("checkpoint-%s.pth" % img_name),
            output_dir / "checkpoint.pth",
        ]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedule": lr_schedule.state_dict(),
                "nimg": img,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"nimg": img}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % img_name,
            client_state=client_state,
        )


def save_model_step(
    args, step, model, model_without_ddp, optimizer, lr_schedule, loss_scaler
):
    output_dir = Path(args.output_dir)
    step_name = str(step)
    if loss_scaler is not None:
        checkpoint_paths = [
            output_dir / ("checkpoint-%s.pth" % step_name),
            output_dir / "checkpoint.pth",
        ]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedule": lr_schedule.state_dict(),
                "step": step,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"step": step}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % step_name,
            client_state=client_state,
        )


def save_best_step(
    args, step, model, model_without_ddp, optimizer, lr_schedule, loss_scaler, suffix="best",
):
    output_dir = Path(args.output_dir)
    fname      = f"checkpoint-best-{suffix}.pth" if suffix else "checkpoint-best.pth"
    latest     = f"checkpoint-latest-{suffix}.pth" if suffix else "checkpoint-latest.pth"
    if loss_scaler is not None:
        checkpoint_paths = [
            output_dir / fname,
            output_dir / latest,
        ]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_schedule": lr_schedule.state_dict(),
                "step": step,
                "scaler": loss_scaler.state_dict(),
                "args": args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {"step": step}
        model.save_checkpoint(
            save_dir=args.output_dir,
            tag="checkpoint-%s" % str(step),
            client_state=client_state,
        )


def load_model(args, model_without_ddp, optimizer, loss_scaler, lr_schedule):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        print("Checkpoint keys:", checkpoint.keys())
        if args.use_ema:
            model_without_ddp.load_state_dict(checkpoint["model"])
        else:
            model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if (
            "optimizer" in checkpoint
            and "nimg" in checkpoint
            # and not (hasattr(args, "eval") and args.eval)
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_schedule.load_state_dict(checkpoint["lr_schedule"])
            args.start_nimg = checkpoint["nimg"]
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")


def load_model_step(args, model_without_ddp, optimizer, loss_scaler, lr_schedule):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        print("Checkpoint keys:", checkpoint.keys())
        if args.use_ema:
            model_without_ddp.load_state_dict(checkpoint["model"])
        else:
            model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if (
            "optimizer" in checkpoint
            and "step" in checkpoint
            # and not (hasattr(args, "eval") and args.eval)
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_schedule.load_state_dict(checkpoint["lr_schedule"])
            args.start_step = checkpoint["step"]
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")


def save_discriminator_step(
    args, step, model_without_ddp, optimizer, lr_schedule, loss_scaler
):
    output_dir = Path(args.output_dir)
    step_name = str(step)
    checkpoint_paths = [
        output_dir / ("d-checkpoint-%s.pth" % step_name),
        output_dir / "d-checkpoint.pth",
    ]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_schedule": lr_schedule.state_dict(),
            "step": step,
            "scaler": loss_scaler.state_dict(),
            "args": args,
        }

        save_on_master(to_save, checkpoint_path)


def load_discriminator_step(args, model_without_ddp, optimizer, loss_scaler, lr_schedule):
    if args.d_resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.d_resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.d_resume, map_location="cpu", weights_only=False)
        print("Checkpoint keys:", checkpoint.keys())
        if args.use_ema:
            model_without_ddp.load_state_dict(checkpoint["model"])
        else:
            model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.d_resume)
        if (
            "optimizer" in checkpoint
            and "step" in checkpoint
            # and not (hasattr(args, "eval") and args.eval)
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_schedule.load_state_dict(checkpoint["lr_schedule"])
            args.start_step = checkpoint["step"]
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")
