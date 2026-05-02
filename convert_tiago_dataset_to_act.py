#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Tiago moving_test HDF5 episodes into ACT-compatible format."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="/home/yun/tiago_public_ws/src/moving_test/training/normalized_data",
        help="Root directory containing act_data_*/episode_*.hdf5.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="act_data_*/episode_*.hdf5",
        help="Glob pattern under --input-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/yun/act_ws/act/data/tiago_act",
        help="Output directory for ACT-style episode_*.hdf5 files.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=0,
        help="Target state/action dimension. <=0 means keep source dimension (auto).",
    )
    parser.add_argument(
        "--use-normalized-fields",
        action="store_true",
        help="Use joint_states_norm/actions_norm instead of joint_states/actions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output episode files.",
    )
    return parser.parse_args()


def _pad_to_dim(arr_2d, target_dim):
    time_steps, src_dim = arr_2d.shape
    out = np.zeros((time_steps, target_dim), dtype=np.float32)
    copy_dim = min(src_dim, target_dim)
    out[:, :copy_dim] = arr_2d[:, :copy_dim]
    return out


def convert_one_file(src_path, dst_path, target_dim, use_normalized_fields):
    with h5py.File(src_path, "r") as fin:
        joint_key = "joint_states_norm" if use_normalized_fields and "joint_states_norm" in fin else "joint_states"
        action_key = "actions_norm" if use_normalized_fields and "actions_norm" in fin else "actions"

        if joint_key not in fin or action_key not in fin:
            raise KeyError(
                f"{src_path} missing required datasets. joint_key={joint_key}, action_key={action_key}"
            )
        if "images" not in fin:
            raise KeyError(f"{src_path} missing images dataset.")

        qpos_src = np.asarray(fin[joint_key], dtype=np.float32)
        action_src = np.asarray(fin[action_key], dtype=np.float32)
        main_images = np.asarray(fin["images"], dtype=np.uint8)
        gripper_images = np.asarray(fin["gripper_images"], dtype=np.uint8) if "gripper_images" in fin else None

        steps = min(qpos_src.shape[0], action_src.shape[0], main_images.shape[0])
        if gripper_images is not None:
            steps = min(steps, gripper_images.shape[0])
        if steps <= 0:
            raise ValueError(f"{src_path} has no valid timesteps.")

        qpos_src = qpos_src[:steps]
        action_src = action_src[:steps]
        main_images = main_images[:steps]
        if gripper_images is not None:
            gripper_images = gripper_images[:steps]

        if target_dim is None:
            target_dim = max(qpos_src.shape[1], action_src.shape[1])
        qpos = _pad_to_dim(qpos_src, target_dim)
        action = _pad_to_dim(action_src, target_dim)
        qvel = np.zeros_like(qpos, dtype=np.float32)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst_path, "w") as fout:
            fout.attrs["sim"] = False
            fout.attrs["source_path"] = str(src_path)
            fout.attrs["source_joint_key"] = joint_key
            fout.attrs["source_action_key"] = action_key

            obs = fout.create_group("observations")
            obs.create_dataset("qpos", data=qpos, dtype=np.float32)
            obs.create_dataset("qvel", data=qvel, dtype=np.float32)
            images = obs.create_group("images")
            images.create_dataset("main", data=main_images, dtype=np.uint8, compression="gzip", compression_opts=4)
            if gripper_images is not None:
                images.create_dataset("gripper", data=gripper_images, dtype=np.uint8, compression="gzip", compression_opts=4)

            fout.create_dataset("action", data=action, dtype=np.float32)

    return steps, qpos_src.shape[1], action_src.shape[1], gripper_images is not None


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    target_dim = None if int(args.state_dim) <= 0 else int(args.state_dim)
    src_files = sorted(input_root.glob(args.pattern))

    if not src_files:
        raise RuntimeError(f"No files matched: root={input_root}, pattern={args.pattern}")

    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    total_steps = 0
    with_gripper_cam = 0

    print(f"[convert] input_root={input_root}")
    print(f"[convert] matched_files={len(src_files)}")
    print(f"[convert] output_dir={output_dir}")

    for idx, src_path in enumerate(src_files):
        dst_path = output_dir / f"episode_{idx}.hdf5"
        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue

        steps, qdim, adim, has_gripper = convert_one_file(
            src_path=src_path,
            dst_path=dst_path,
            target_dim=target_dim,
            use_normalized_fields=args.use_normalized_fields,
        )
        converted += 1
        total_steps += steps
        if has_gripper:
            with_gripper_cam += 1

        if (idx + 1) % 20 == 0 or (idx + 1) == len(src_files):
            print(f"[convert] processed {idx + 1}/{len(src_files)} files")
            print(f"[convert] latest src_dim: qpos={qdim}, action={adim}, steps={steps}")

    print("\n[convert] done")
    print(f"[convert] converted_files={converted}")
    print(f"[convert] skipped_existing={skipped}")
    print(f"[convert] total_steps={total_steps}")
    print(f"[convert] episodes_with_gripper_camera={with_gripper_cam}")
    print(f"[convert] recommended num_episodes={len(src_files)}")
    print(f"[convert] recommended episode_len=50 (verify with your data)")


if __name__ == "__main__":
    main()
