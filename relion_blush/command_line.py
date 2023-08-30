import sys
import time

if sys.version_info < (3, 0):
    # This script requires Python 3. A Syntax error here means you are running it in Python 2.
    print('This script supports Python 3 or above.')
    exit(1)

import os
import argparse
import sys
from loguru import logger

try:
    import torch
except ImportError:
    print("PYTHON ERROR: The required python module 'torch' was not found.")
    exit(1)

try:
    import numpy as np
except ImportError:
    print("PYTHON ERROR: The required python module 'numpy' was not found.")
    exit(1)

sys.path.append(os.path.realpath(os.path.dirname(__file__)))
from relion_blush.util import *


EPS = 1e-6


def refine3d(data, model, device, strides, batch_size, debug=False, skip_spectral_tailing=False):
    # Load and prepare volumes --------
    pad = data["padding"]
    t = time.time()
    recons_df_unfil, recons_df = get_reconstructions(data)
    logger.info(f"Resample time {round(time.time() - t, 2)} s")

    # Run model -----------------------
    model_block_size = int(model.block_size)
    model_voxel_size = int(model.voxel_size)
    recons_df_unfil_nv, _ = normalize_voxel_size_fourier(recons_df_unfil, data["pixel_size"], model_voxel_size)
    recons_df_unfil_nv *= (recons_df_unfil_nv.shape[0] / recons_df_unfil.shape[0])**3
    denoise_input = pad_ifft(recons_df_unfil_nv, pad)

    mask_edge_width = 10 * model_voxel_size
    radius = data["particle_diameter"] * model_voxel_size
    radius = min(radius, (denoise_input.shape[0] - mask_edge_width) / 2. + 1)
    radial_mask = get_radial_mask(denoise_input.shape[0], radius, edge_width=mask_edge_width).to(device)

    t = time.time()
    denoised_nv, _ = apply_model(
        model,
        torch.from_numpy(denoise_input).to(device),
        input_mask=radial_mask,
        device=device,
        strides=strides,
        block_size=model_block_size,
        batch_size=batch_size
    )
    logger.info(f"Running model time {round(time.time() - t, 2)} s")

    denoised_nv = (denoised_nv * radial_mask).cpu().numpy()
    denoised_df_nv = pad_fft(denoised_nv, pad)
    denoised_df = rescale_fourier(denoised_df_nv, data["shape"][0])
    denoised_df *= (denoised_df.shape[0] / denoised_df_nv.shape[0])**3

    # Apply mixing ---------------------------------
    max_denoised_index = res_from_fsc(data["fsc_spectra"], res=None, threshold=1. / 7.) - 1

    if max_denoised_index > denoised_df_nv.shape[-1] - 3:
        logger.info(f"Denoiser maximum resolution reached, mixing in data for higher frequencies.")
        crossover_grid = get_crossover_grid(denoised_df_nv.shape[-1] - 3, data, filter_edge_width=3)
        out_df = denoised_df * crossover_grid + recons_df * (1 - crossover_grid)
    else:
        if skip_spectral_tailing:
            logger.info(f"Skipping spectral trailing")
            out_df = denoised_df
        else:
            logger.info(f"Applying spectral trailing")
            crossover_grid = get_crossover_grid(max_denoised_index, data, filter_edge_width=3)
            out_df = denoised_df * crossover_grid

    out = pad_ifft(out_df, pad)

    if debug:
        save_mrc(
            pad_ifft(recons_df_unfil_nv, pad),
            data["output_path"][:-4] + "_debug_recons_unfil_nv.mrc",
            model_voxel_size
        )
        save_mrc(
            pad_ifft(recons_df_unfil, pad),
            data["output_path"][:-4] + "_debug_recons_unfil.mrc",
            model_voxel_size
        )
        save_mrc(
            pad_ifft(recons_df, pad),
            data["output_path"][:-4] + "_debug_recons.mrc",
            model_voxel_size
        )
        save_mrc(
            denoised_nv,
            data["output_path"][:-4] + "_debug_denoised_nv.mrc",
            model_voxel_size
        )
        save_mrc(
            pad_ifft(denoised_df, pad),
            data["output_path"][:-4] + "_debug_denoised.mrc",
            data["pixel_size"]
        )

    mask_edge_width = 10 * data["pixel_size"]
    radius = data["particle_diameter"] * data["pixel_size"]
    radius = min(radius, (out.shape[0] - mask_edge_width) / 2.)
    radial_mask = get_radial_mask(out.shape[0], radius, edge_width=mask_edge_width).to(device)
    out *= radial_mask.cpu().numpy()

    logger.info(f"Max denoised spectral index: {max_denoised_index}")

    # Save volume ---------------------
    output_path = data["output_path"]

    if data['mode'] == "refine":
        save_mrc(out, output_path, data["pixel_size"], np.array([0, 0, 0]))
        logger.info(f'Ouput to file {output_path}')

    elif data['mode'] == "refine_final":
        recons = pad_ifft(recons_df, pad)
        save_mrc(recons, output_path, data["pixel_size"], np.array([0, 0, 0]))
        logger.info(f'Final reconstruction output to file {output_path}')

        recons_denoised_path = output_path[:-4] + "_denoised.mrc"
        save_mrc(out, recons_denoised_path, data["pixel_size"], np.array([0, 0, 0]))
        logger.info(f'Final reconstruction denoised output to file {recons_denoised_path}')


def class3d(data, model, device, strides, batch_size, debug=False):
    # Load and prepare volumes --------
    pad = data["padding"]
    t = time.time()
    recons_df_unfil, recons_df = get_reconstructions(data)
    logger.info(f"Resample time {round(time.time() - t, 2)} s")

    # Run model -----------------------
    model_block_size = int(model.block_size)
    model_voxel_size = int(model.voxel_size)
    recons_df_nv, _ = normalize_voxel_size_fourier(recons_df, data["pixel_size"], model_voxel_size)
    recons_df_nv *= (recons_df_nv.shape[0] / recons_df.shape[0])**3
    denoise_input = pad_ifft(recons_df_nv, pad)

    t = time.time()
    mask_edge_width = 10 * model_voxel_size
    radius = data["particle_diameter"] * model_voxel_size
    radius = min(radius, (denoise_input.shape[0] - mask_edge_width) / 2. + 1)
    radial_mask = get_radial_mask(denoise_input.shape[0], radius, edge_width=mask_edge_width).to(device)
    denoised_nv, _ = apply_model(
        model,
        torch.from_numpy(denoise_input).to(device),
        input_mask=radial_mask,
        device=device,
        strides=strides,
        block_size=model_block_size,
        batch_size=batch_size
    )
    logger.info(f"Running model time {round(time.time() - t, 2)} s")

    denoised_nv = (denoised_nv * radial_mask).cpu().numpy()

    # Apply maximum resolution ---------------------
    denoised_df_nv = pad_fft(denoised_nv, pad)

    denoised_df = rescale_fourier(denoised_df_nv, data["shape"][0])
    denoised_df *= (denoised_df.shape[0] / denoised_df_nv.shape[0])**3

    crossover_grid = get_crossover_grid(denoised_df_nv.shape[-1] - 2, data, filter_edge_width=3)
    out_df = denoised_df * crossover_grid + recons_df * (1 - crossover_grid)
    out = pad_ifft(out_df, pad)

    if debug:
        save_mrc(
            denoise_input,
            data["output_path"][:-4] + "_debug_denoise_input.mrc",
            model_voxel_size
        )
        save_mrc(
            denoised_nv,
            data["output_path"][:-4] + "_debug_denoise_nv.mrc",
            model_voxel_size
        )
        save_mrc(
            pad_ifft(denoised_df, pad),
            data["output_path"][:-4] + "_debug_denoise.mrc",
            data["pixel_size"]
        )

    # Save volume ---------------------
    output_path = data["output_path"]
    save_mrc(out, output_path, data["pixel_size"], np.array([0, 0, 0]))
    logger.info(f'Ouput to file {output_path}')


def main():
    # warnings.filterwarnings("error")

    # Arguments -----------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('star_file', nargs='?', default=None)
    parser.add_argument('-m', '--model_name', type=str, default="v1.0")
    parser.add_argument('-s', '--strides', type=int, default=20)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-g', '--gpu', type=str, default=None)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--skip-spectral-tailing', action="store_true")
    args = parser.parse_args()

    # Load data -------------------------
    if args.star_file is None:
        data = None
    else:
        try:
            data = load_input_data(args.star_file)
        except FileNotFoundError:
            print(f"Did not find input star file ({args.star_file}).")
            exit(1)

    # Setup logger -------------------------
    if data is not None:
        logger.remove()
        logger.add(
            data["log_path"],
            level=10,
            enqueue=True,
            rotation="500 MB",
            diagnose=True,
            backtrace=True
        )

    logger.info(f"ARGUMENTS: {args}")

    # Load model bundle ----------------------
    t = time.time()

    model, model_path = install_and_load_model(
        name=args.model_name,
        device="cpu",
        verbose=data is None
    )

    logger.info(f"Loading model time {round(time.time() - t, 2)} s")

    if model is None:
        logger.info("Model name not found!")
        exit(1)

    if data is None:
        logger.info("No job target was specified... exiting!")
        exit(0)

    if data['mode'] == "classification":
        time.sleep(data["volume_id"] * 0.2)
    elif data['mode'] == "refine" and data['mode'] == "refine_final":
        sleep_time = data["volume_id"]
        if data["this_half_index"] is not None:
            sleep_time += (data["this_half_index"] - 1) * 20
        time.sleep(sleep_time * 0.1)

    torch.no_grad()

    if data["padding"] != 1:
        raise RuntimeError("Only skip padding=Yes is supported.")

    # Device assignment --------------
    device_lock = DeviceLock(data["job_dir"], device_str=args.gpu)
    device = device_lock.get_device()
    logger.info(f"Selected device: {device}")
    device = torch.device(device)
    model = model.to(device)

    if data['mode'] == "classification":
        class3d(data, model, device, args.strides, args.batch_size)
    elif data['mode'] == "refine" or data['mode'] == "refine_final":
        refine3d(data, model, device, args.strides, args.batch_size, args.skip_spectral_tailing)
    else:
        raise NotImplementedError(f"Mode not supported: {data['mode']}")

    print("success")


if __name__ == "__main__":
    main()
