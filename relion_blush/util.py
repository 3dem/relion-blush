#!/usr/bin/env python3
import os
import re
import sys
import time
import types

import torch
import numpy as np
import mrcfile as mrc
from collections import OrderedDict
from filelock import Timeout, FileLock


class ScanningBlockIterator:
    def __init__(self, shape, block_size, strides):
        self.z_range = self.get_range(shape[0], block_size, strides)
        self.y_range = self.get_range(shape[1], block_size, strides)
        self.x_range = self.get_range(shape[2], block_size, strides)
        self.z_idx = 0
        self.y_idx = 0
        self.x_idx = 0
        self.idx = 0
        self.count = len(self.z_range) * len(self.y_range) * len(self.x_range)

    @staticmethod
    def get_range(span, block_size, strides):
        r = list(np.arange(0, span - block_size, strides))
        if r[-1] < span - block_size:
            r += [span - block_size]
        return r

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == self.count:
            raise StopIteration

        if self.x_idx == len(self.x_range):
            self.x_idx = 0
            self.y_idx += 1
        if self.y_idx == len(self.y_range):
            self.y_idx = 0
            self.z_idx += 1

        coords = (
            self.z_range[self.z_idx],
            self.y_range[self.y_idx],
            self.x_range[self.x_idx]
        )

        self.x_idx += 1
        self.idx += 1
        return coords, self.idx == self.count


def get_fourier_shells(shape):
    (z, y, x) = shape
    Z, Y, X = np.meshgrid(np.linspace(-z // 2, z // 2 - 1, z),
                          np.linspace(-y // 2, y // 2 - 1, y),
                          np.linspace(0, x - 1, x), indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R = np.fft.ifftshift(R, axes=(0, 1))
    return R


def get_local_std_torch(grid, size):
    grid = torch.unsqueeze(grid, 1).clone()
    grid2 = torch.square(grid)

    ls = torch.linspace(-1.5, 1.5, 2 * size + 1)
    kernel = torch.exp(-torch.square(ls)).to(grid.device)
    kernel /= torch.sum(kernel)

    kernel = kernel[None, None, :, None, None]
    for i in range(3):
        grid = grid.permute(0, 1, 4, 2, 3)
        grid = torch.nn.functional.conv3d(grid, kernel, padding='same')
        grid2 = grid2.permute(0, 1, 4, 2, 3)
        grid2 = torch.nn.functional.conv3d(grid2, kernel, padding='same')
    std = torch.sqrt(torch.clip(grid2 - grid ** 2, min=0))

    return torch.squeeze(std, 1)


def get_std_layer(grid):
    grid = get_local_std_torch(grid.unsqueeze(0), size=10)[0]
    grid = grid / torch.mean(grid)
    return grid


def make_weight_box(size, margin=3):
    margin = margin - 1 if margin > 0 else 1
    s = size - margin*2

    z, y, x = np.meshgrid(np.linspace(-s // 2, s // 2, s),
                          np.linspace(-s // 2, s // 2, s),
                          np.linspace(-s // 2, s // 2, s),
                          indexing="ij")
    r = np.max([np.abs(x), np.abs(y), np.abs(z)], axis=0)
    r = np.cos(r/np.max(r) * np.pi/2)

    w = np.zeros((size, size, size))
    m = margin
    w[m:size-m, m:size-m, m:size-m] = r

    w = np.clip(w, 1e-6, None)

    # import matplotlib.pyplot as plt
    # plt.imshow(w[48])
    # plt.show()

    return w


def get_radial_mask(box_size, radius, edge_width=0, device="cpu"):
    bz2 = box_size / 2.
    ls = torch.linspace(-bz2, bz2, box_size).to(device)
    r = torch.stack(torch.meshgrid(ls, ls, ls, indexing="ij"), -1)
    r = torch.sqrt(torch.sum(torch.square(r), -1))
    scale = torch.ones_like(r)

    if edge_width > 0:
        edge_low  = radius - edge_width // 2
        edge_high = radius + edge_width // 2
        scale[r > edge_high] = 0

        scale[(r >= edge_low) & (r <= edge_high)] = 0.5 + 0.5 * torch.cos(
            np.pi * (r[(r >= edge_low) & (r <= edge_high)] - edge_low) / edge_width)
    else:
        scale[r > radius] = 0

    return scale


@torch.no_grad()
def apply_model(model, volume, device, input_mask=None, strides=48, block_size=128, batch_size=2):
    B = volume.shape

    in_channels = 2

    blocks = torch.zeros(
        [batch_size, in_channels] + [block_size] * 3,
        dtype=torch.float32
    ).to(device)
    coords = np.zeros((batch_size, 3), dtype=int)

    original_shape = np.array(B)
    pad_offset = None

    # Make sure window box fit in volume
    if np.any(original_shape <= block_size):
        shape_ = np.clip(original_shape, block_size + strides, None)

        si = original_shape // 2
        so = shape_ // 2

        pad_offset = np.array([
            so[0] - si[0],
            so[1] - si[1],
            so[2] - si[2]
        ])

        v = torch.zeros(list(shape_)).to(device)
        v[pad_offset[0]: so[0] + si[0], pad_offset[1]: so[1] + si[1], pad_offset[2]: so[2] + si[2]] = volume
        volume = v

        if input_mask is not None:
            v = torch.zeros(list(shape_)).to(device)
            v[pad_offset[0]: so[0] + si[0], pad_offset[1]: so[1] + si[1], pad_offset[2]: so[2] + si[2]] = input_mask
            input_mask = v

        B = list(shape_)

    weight_block = torch.Tensor(make_weight_box(block_size, 10)).to(device)
    infer_grid = torch.zeros_like(volume).to(device)
    mask_grid = torch.zeros_like(volume).to(device)

    count_grid = torch.zeros(B, dtype=torch.float32).to(device)

    if input_mask is None:
        input_mask = get_radial_mask(volume.shape[-1], volume.shape[-1] / 2 + 1, volume.device)

    in_std = get_std_layer(volume)

    mean = torch.mean(volume)
    std = torch.std(volume)
    volume = (volume - mean) / (std + 1e-8)

    volume *= input_mask
    in_std *= input_mask

    input = torch.cat([volume.unsqueeze(0), in_std.unsqueeze(0)], 0)

    bi = 0
    for (z, y, x), last_block in ScanningBlockIterator(B, block_size, strides):
        mask_ = input_mask[z:z + block_size, y:y + block_size, x:x + block_size]
        if torch.mean(mask_) < 0.3:
            continue

        blocks[bi, ...] = input[:, z:z + block_size, y:y + block_size, x:x + block_size]
        coords[bi, ...] = np.array([z, y, x])
        bi += 1

        if bi == batch_size or last_block:
            map, mask = model(blocks[:bi, 0], blocks[:bi, 1])
            mask = torch.sigmoid(mask)

            for i in range(bi):
                infer_grid[
                coords[i, 0]:coords[i, 0] + block_size,
                coords[i, 1]:coords[i, 1] + block_size,
                coords[i, 2]:coords[i, 2] + block_size
                ] += map[i] * weight_block
                mask_grid[
                coords[i, 0]:coords[i, 0] + block_size,
                coords[i, 1]:coords[i, 1] + block_size,
                coords[i, 2]:coords[i, 2] + block_size
                ] += mask[i] * weight_block
                count_grid[
                    coords[i, 0]:coords[i, 0] + block_size,
                    coords[i, 1]:coords[i, 1] + block_size,
                    coords[i, 2]:coords[i, 2] + block_size
                ] += weight_block
            bi = 0

    infer_grid[count_grid > 0] /= count_grid[count_grid > 0]
    mask_grid[count_grid > 0] /= count_grid[count_grid > 0]
    infer_grid[count_grid < 1e-1] = 0
    mask_grid[count_grid < 1e-1] = 0

    infer_grid *= input_mask
    mask_grid *= input_mask

    infer_grid = infer_grid * (std + 1e-8) + mean

    if pad_offset is not None:
        si = original_shape // 2
        so = np.array(infer_grid.shape) // 2
        infer_grid = infer_grid[
             pad_offset[0]: so[0] + si[0],
             pad_offset[1]: so[1] + si[1],
             pad_offset[2]: so[2] + si[2]
        ]
        mask_grid = mask_grid[
             pad_offset[0]: so[0] + si[0],
             pad_offset[1]: so[1] + si[1],
             pad_offset[2]: so[2] + si[2]
        ]

    return infer_grid, mask_grid


def index_to_res(index, voxel_size, box_size):
    """
    Calculates resolution from the Fourier shell index and voxel size and box size.

    i = index, b = box_size / 2, r = resolution, v = voxel_size
    r = 2 * b * v / i
    """
    if index <= 0:
        return 1e3
    return box_size * voxel_size / index


def res_from_fsc(fsc, res=None, threshold=0.5):
    """
    Calculates the resolution (res) at the FSC (fsc) threshold.
    If 'res' is not provided, the index of the threshold is returned.
    """
    if res is not None:
        assert len(fsc) == len(res)
    i = np.argmax(fsc < threshold)
    if i > 0:
        if res is None:
            return i-1
        return res[i-1]
    else:
        if res is None:
            return len(fsc) - 1
        return res[-1]


def spectra_to_grid_torch(spectra, indices):
    if len(spectra.shape) == 1:  # Has no batch dimension
        grid = torch.gather(spectra, 0, indices.flatten().long())
    elif len(spectra.shape) == 2:  # Has batch dimension
        indices = indices.unsqueeze(0).expand([spectra.shape[0]] + list(indices.shape))
        grid = torch.gather(spectra.flatten(1), 1, indices.flatten(1).long())
    else:
        raise RuntimeError("Spectra must be at most two-dimensional (one batch dimension).")
    return grid.view(indices.shape)


def grid_spectral_sum_torch(grid, indices):
    if len(grid.shape) == len(indices.shape) and np.all(grid.shape == indices.shape):  # Has no batch dimension
        spectrum = torch.zeros(int(torch.max(indices)) + 1).to(grid.device)
        spectrum.scatter_add_(0, indices.long().flatten(), grid.flatten())
    elif len(grid.shape) == len(indices.shape) + 1 and np.all(grid.shape[1:] == indices.shape):  # Has batch dimension
        spectrum = torch.zeros([grid.shape[0], int(torch.max(indices)) + 1]).to(grid.device)
        indices = indices.long().unsqueeze(0).expand([grid.shape[0]] + list(indices.shape))
        spectrum.scatter_add_(1, indices.flatten(1), grid.flatten(1))
    else:
        raise RuntimeError("Shape of grid must match spectral_indices, except along the batch dimension.")
    return spectrum


def grid_spectral_average_torch(grid, indices):
    indices = indices.long()
    spectrum = grid_spectral_sum_torch(grid, indices)
    norm = grid_spectral_sum_torch(torch.ones_like(indices).float(), indices)
    return spectrum / norm[None, :]


def rescaled_boxsize_from_voxelsize(box_size, input_voxel_size, target_voxel_size):
    out_sz = int(round(box_size * input_voxel_size / target_voxel_size))
    if out_sz % 2 != 0:
        vs1 = input_voxel_size * box_size / (out_sz + 1)
        vs2 = input_voxel_size * box_size / (out_sz - 1)
        if np.abs(vs1 - target_voxel_size) < np.abs(vs2 - target_voxel_size):
            out_sz += 1
        else:
            out_sz -= 1

    out_voxel_sz = input_voxel_size * box_size / out_sz
    return out_sz, out_voxel_sz


def normalize_voxel_size_fourier(grid, in_voxel_sz, target_voxel_size=1., smooth_lowpass_width=0):
    (iz, iy, ix) = np.shape(grid)

    assert iz % 2 == 0
    assert (ix - 1) * 2 == iy == iz
    out_sz, out_voxel_sz = rescaled_boxsize_from_voxelsize(iz, in_voxel_sz, target_voxel_size)
    density = rescale_fourier(grid, out_sz, smooth_lowpass_width)

    return density, out_voxel_sz


def get_lowpass_filter(
    size,
    ires_filter: int = None,
    bfac: float = 0,
    filter_edge_width: int = 3,
    use_cosine_kernel: bool = True,
):
    ls = torch.linspace(-(size // 2), size // 2 - 1, size)
    lsx = torch.linspace(0, size // 2 + 1, size // 2 + 1)
    r = torch.stack(torch.meshgrid(ls, ls, lsx, indexing="ij"), 0)
    R = torch.sqrt(torch.sum(torch.square(r), 0))
    spectral_radius = torch.fft.ifftshift(R, dim=(0, 1))

    res = spectral_radius / (size / 2. + 1.)
    scale_spectrum = torch.zeros_like(res)

    if ires_filter is not None:
        filter_edge_halfwidth = filter_edge_width // 2

        edge_low  = np.clip((ires_filter - filter_edge_halfwidth) / size, 0, size / 2. + 1)
        edge_high = np.clip((ires_filter + filter_edge_halfwidth) / size, 0, size / 2. + 1)
        edge_width = edge_high - edge_low
        scale_spectrum[res < edge_low] = 1

        if use_cosine_kernel and edge_width > 0:
            scale_spectrum[(res >= edge_low) & (res <= edge_high)] = 0.5 + 0.5 * torch.cos(
                np.pi
                * (res[(res >= edge_low) & (res <= edge_high)] - edge_low)
                / edge_width
            )
    else:
        scale_spectrum += 1.

    if bfac != 0:
        scale_spectrum *= torch.exp(-bfac * res)

    return scale_spectrum


def rescale_fourier(box, out_sz, smooth_lowpass_width=0):
    if out_sz % 2 != 0:
        raise Exception("Bad output size")
    if box.shape[0] != box.shape[1] or \
            box.shape[1] != (box.shape[2]-1)*2:
        raise Exception("Input must be cubic")

    ibox = np.fft.ifftshift(box, axes=(0, 1))
    obox = np.zeros((out_sz, out_sz, out_sz//2+1), dtype=box.dtype)

    si = np.array(ibox.shape) // 2
    so = np.array(obox.shape) // 2

    if so[0] < si[0]:
        obox = ibox[
             si[0] - so[0]: si[0] + so[0],
             si[1] - so[1]: si[1] + so[1],
             :obox.shape[2]
             ]
    elif so[0] > si[0]:
        obox[
            so[0] - si[0]: so[0] + si[0],
            so[1] - si[1]: so[1] + si[1],
            :ibox.shape[2]
        ] = ibox
    else:
        obox = ibox

    obox = np.fft.ifftshift(obox, axes=(0, 1))

    if smooth_lowpass_width > 0:
        filter_grid = get_lowpass_filter(
            size=obox.shape[0],
            ires_filter=obox.shape[-1] - smooth_lowpass_width/2,
            filter_edge_width=smooth_lowpass_width
        ).numpy()
        obox *= filter_grid

    return obox


def save_mrc(grid, filename, voxel_size=1, origin=0.):
    if isinstance(origin, float) or isinstance(origin, int) or origin is None:
        origin = [origin] * 3
    (z, y, x) = grid.shape
    o = mrc.new(filename, overwrite=True)
    o.header['cella'].x = x * voxel_size
    o.header['cella'].y = y * voxel_size
    o.header['cella'].z = z * voxel_size
    o.header['origin'].x = origin[0]
    o.header['origin'].y = origin[1]
    o.header['origin'].z = origin[2]
    out_box = np.reshape(grid, (z, y, x))
    o.set_data(out_box.astype(np.float32))
    o.update_header_stats()
    o.flush()
    o.close()


def load_mrc(mrc_fn):
    mrc_file = mrc.open(mrc_fn, 'r')
    c = mrc_file.header['mapc']
    r = mrc_file.header['mapr']
    s = mrc_file.header['maps']

    global_origin = mrc_file.header['origin']
    global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
    global_origin[0] += mrc_file.header['nxstart']
    global_origin[1] += mrc_file.header['nystart']
    global_origin[2] += mrc_file.header['nzstart']

    global_origin *= mrc_file.voxel_size.x

    if c == 1 and r == 2 and s == 3:
        grid = mrc_file.data
    elif c == 3 and r == 2 and s == 1:
        grid = np.moveaxis(mrc_file.data, [0, 1, 2], [2, 1, 0])
    elif c == 2 and r == 1 and s == 3:
        grid = np.moveaxis(mrc_file.data, [1, 2, 0], [2, 1, 0])
    else:
        raise RuntimeError("MRC file axis arrangement not supported!")

    return grid, float(mrc_file.voxel_size.x), global_origin


def load_star(filename):
    datasets = OrderedDict()
    current_data = None
    current_colnames = None

    BASE = 0  # Not in a block
    COLNAME = 1  # Parsing column name
    DATA = 2  # Parsing data
    mode = BASE

    for line in open(filename):
        line = line.strip()

        # remove comments
        comment_pos = line.find('#')
        if comment_pos > 0:
            line = line[:comment_pos]

        if line == "":
            if mode == DATA:
                mode = BASE
            continue

        if line.startswith("data_"):
            mode = BASE
            data_name = line[5:]
            current_data = OrderedDict()
            datasets[data_name] = current_data

        elif line.startswith("loop_"):
            current_colnames = []
            mode = COLNAME

        elif line.startswith("_"):
            if mode == DATA:
                mode = BASE
            token = line[1:].split()
            if mode == COLNAME:
                current_colnames.append(token[0])
                current_data[token[0]] = []
            else:
                current_data[token[0]] = token[1]

        elif mode != BASE:
            mode = DATA
            token = line.split()
            if len(token) != len(current_colnames):
                raise RuntimeError(
                    f"Error in STAR file {filename}, number of elements in {token} "
                    f"does not match number of column names {current_colnames}"
                )
            for idx, e in enumerate(token):
                current_data[current_colnames[idx]].append(e)

    return datasets


def load_mrc_(path):
    with mrc.open(path) as m:
        return m.data.astype(np.float32).copy()


def load_input_data(star_file_path):
    star_file = load_star(star_file_path)
    d = {}

    d["volume_id"] = 1
    class_id = re.search("class([0-9]{3})", star_file_path)
    if class_id:
        d["volume_id"] = max(int(class_id.group(1)), 0)
    else:
        body_id = re.search("body([0-9]{3})", star_file_path)
        if body_id:
            d["volume_id"] = max(int(body_id.group(1)), 0)

    d["job_dir"] = os.path.dirname(os.path.realpath(star_file['external_reconstruct_general']['rlnExtReconsResult']))
    d["log_path"] = star_file['external_reconstruct_general']['rlnExtReconsResult'][:-4] + ".log"

    d["fudge"] = float(star_file['external_reconstruct_general']['rlnTau2FudgeFactor'])

    d["real"] = load_mrc_(star_file['external_reconstruct_general']['rlnExtReconsDataReal'])
    d["imag"] = load_mrc_(star_file['external_reconstruct_general']['rlnExtReconsDataImag'])
    d["kernel"] = load_mrc_(star_file['external_reconstruct_general']['rlnExtReconsWeight'])

    d["shape"] = d["kernel"].shape
    d["original_size"] = int(star_file['external_reconstruct_general']['rlnOriginalImageSize'])
    d["padding"] = float(star_file['external_reconstruct_general']['rlnPaddingFactor'])
    d["current_size"] = int(star_file['external_reconstruct_general']['rlnCurrentImageSize'])
    d["max_r"] = d["current_size"] * d["padding"] / 2
    d["particle_diameter"] = float(star_file['external_reconstruct_general']['rlnParticleDiameter'])
    d["output_path"] = star_file['external_reconstruct_general']['rlnExtReconsResult']
    d["pixel_size"] = float(star_file['external_reconstruct_general']['rlnPixelSize'])

    d["spectral1d_index"] = np.array(star_file['external_reconstruct_tau2']['rlnSpectralIndex'], dtype=float)
    d["spectral3d_index"] = get_fourier_shells(d["kernel"].shape)
    d["spectral3d_mask"] = d["spectral3d_index"] < d["max_r"]

    tau2_spectra = np.array(star_file['external_reconstruct_tau2']['rlnReferenceTau2'], dtype=float)
    if np.any(tau2_spectra != 0):
        d["tau2_spectra"] = tau2_spectra
        idx = d["spectral1d_index"] * d["padding"]
        oversampling_correction = d["padding"] ** 3
        val = 1. / (d["tau2_spectra"] * d["fudge"] * oversampling_correction + 1e-20)
        d["reg_grid"] = np.interp(d["spectral3d_index"], idx, val)
    else:
        d["reg_grid"] = None

    fsc_spectra = np.array(star_file['external_reconstruct_tau2']['rlnGoldStandardFsc'], dtype=float)
    if np.any(fsc_spectra != 0):
        d["fsc_spectra"] = fsc_spectra
        idx = d["spectral1d_index"] * d["padding"]
        val = np.clip(d["fsc_spectra"] * d["fudge"], 0, 1)
        d["fsc_grid"] = np.interp(d["spectral3d_index"], idx, val)
    else:
        d["fsc_grid"] = None

    if "half1" in star_file_path:
        d["doing_half_maps"] = True
        d["this_half_index"] = 1
        d["other_half_index"] = 2
    elif "half2" in star_file_path:
        d["doing_half_maps"] = True
        d["this_half_index"] = 2
        d["other_half_index"] = 1
    else:
        d["doing_half_maps"] = False
        d["this_half_index"] = None
        d["other_half_index"] = None

    if d["doing_half_maps"]:
        d["mode"] = "refine"
    elif "fsc_spectra" in d:
        d["mode"] = "refine_final"
    else:
        d["mode"] = "classification"

    return d


def gridding_correct(grid, padding, original_size=None, trilinear_interpolation=True):
    size = grid.shape[-1]
    if original_size is None:
        original_size = size

    ls = torch.linspace(-(size // 2), size // 2 - 1, size)
    c = torch.stack(torch.meshgrid(ls, ls, ls, indexing="ij"), 0)
    r = torch.sqrt(torch.sum(torch.square(c), 0)) / (original_size * padding)
    correction = torch.sinc(r)

    if trilinear_interpolation:
        correction = torch.square(correction)

    if not torch.is_tensor(grid):
        correction = correction.detach().numpy()

    return grid / correction


def ifft(grid_ft):
    return np.fft.ifftshift(np.fft.irfftn(grid_ft)).astype(np.float32)


def fft(grid):
    return np.fft.rfftn(np.fft.fftshift(grid)).astype(np.complex64)


def pad_ifft(grid_ft, data=None):
    grid = ifft(grid_ft)

    if data is not None and data['padding'] > 1:
        in_sz = grid.shape[0]
        out_sz = round(in_sz / data['padding'])
        out_sz += out_sz % 2

        f = in_sz // 2 - out_sz // 2
        t = in_sz // 2 + out_sz // 2
        grid = grid[f:t, f:t, f:t]

    if data is not None:
        grid = gridding_correct(grid, data['padding'], original_size=data['original_size'])

    return grid


def get_reconstructions(data):
    kernel_ft_torch = torch.from_numpy(data['kernel'])
    spectral3d_index_torch = torch.from_numpy(data['spectral3d_index'])
    spectral_avg_torch = grid_spectral_average_torch(kernel_ft_torch, spectral3d_index_torch)
    weight_margin_spectra = spectral_avg_torch * 0.001
    weight_margin = spectra_to_grid_torch(weight_margin_spectra, spectral3d_index_torch).numpy()[0]

    scale = (data["kernel"].shape[0] / data['padding']) ** 2

    mask = data["spectral3d_mask"].astype(float)

    recons_unweight = (data["real"] + 1j * data["imag"]) * mask * scale
    recons_df_unfil = recons_unweight / (data['kernel'] + weight_margin + 1e-30)
    recons_df = recons_unweight / (data['kernel'] + data["reg_grid"] + 1e-30)

    # Free memory
    del data["real"]
    del data["imag"]
    del data["kernel"]
    del data["reg_grid"]

    recons_unfil = pad_ifft(recons_df_unfil.astype(np.complex64), data).astype(float)
    recons = pad_ifft(recons_df.astype(np.complex64), data).astype(float)

    return recons_unfil, recons


def get_crossover_grid(crossover_index, data, filter_edge_width=3):
    size = data["original_size"]

    r = get_fourier_shells((size, size, size // 2 + 1))
    scale_spectrum = np.zeros_like(r)

    filter_edge_halfwidth = filter_edge_width // 2

    edge_low  = np.clip((crossover_index - filter_edge_halfwidth), 0, size / 2. + 1)
    edge_high = np.clip((crossover_index + filter_edge_halfwidth), 0, size / 2. + 1)
    edge_width = edge_high - edge_low
    scale_spectrum[r < edge_low] = 1

    if edge_width > 0:
        scale_spectrum[(r >= edge_low) & (r <= edge_high)] = 0.5 + 0.5 * np.cos(
            np.pi
            * (r[(r >= edge_low) & (r <= edge_high)] - edge_low)
            / edge_width
        )
    return scale_spectrum


def make_gaussian_kernel(sigma):
    ks = int(sigma * 5)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


def fast_gaussian_filter(grid, kernel_size=None, kernel=None):
    if kernel is not None:
        k = kernel
    elif kernel_size is not None:
        k = make_gaussian_kernel(kernel_size).to(grid.device)
    else:
        raise RuntimeError("Either provide sigma or kernel.")
    grid = torch.nn.functional.conv3d(grid, k[None, None, :, None, None], padding='same')
    grid = torch.nn.functional.conv3d(grid, k[None, None, None, :, None], padding='same')
    grid = torch.nn.functional.conv3d(grid, k[None, None, None, None, :], padding='same')

    return grid


def local_correlation(grid1, grid2, kernel_size):
    std = torch.std(grid1) + 1e-28
    grid1 = grid1.unsqueeze(0).unsqueeze(0) / std
    grid2 = grid2.unsqueeze(0).unsqueeze(0) / std

    kernel = make_gaussian_kernel(kernel_size).to(grid1.device)
    def f(a): return fast_gaussian_filter(a, kernel=kernel)

    grid1_mean = grid1 - f(grid1)
    grid2_mean = grid2 - f(grid2)
    norm = torch.sqrt(f(grid1_mean.square()) * f(grid2_mean.square())) + 1e-12
    corr = f(grid1_mean * grid2_mean) / norm

    return corr.squeeze(0).squeeze(0)


def get_device_assignment(job_dir, device_id_list=None, max_retry=300):
    if torch.cuda.is_available():
        timeout = 0

        if device_id_list is None or len(device_id_list) == 0:
            device_id_list = np.arange(torch.cuda.device_count()).tolist()
        else:
            device_id_list = list(set(device_id_list))  # Get unique values

        for retry in range(max_retry):
            for id in device_id_list:
                if id < 0:
                    return "cpu", None
                device_lock_file_path = os.path.join(job_dir, f"device_lock_id{id}")
                device_lock_file = FileLock(device_lock_file_path)
                try:
                    device_lock_file.acquire(timeout=timeout)
                    return f"cuda:{id}", device_lock_file
                except TimeoutError:
                    pass
            timeout = 1

    return "cpu", None


class DeviceLock:
    def __init__(self, job_dir, device_str=None, max_retry=100):
        devices = None
        if device_str is not None:
            device_count = torch.cuda.device_count()
            numbers = re.findall(r"[0-9\-]+", device_str)
            devices = []
            for n in numbers:
                n = int(n)
                if n < 0:
                    devices = [-1]
                    break
                if n < device_count:
                    devices.append(n)

        self.device, self.device_lock_file = get_device_assignment(
            job_dir, device_id_list=devices, max_retry=max_retry)

    def get_device(self):
        return self.device

    def __del__(self):
        if self.device_lock_file is not None:
            self.device_lock_file.release()


def install_and_load_model(
        name: str,
        device: str = "cpu",
        verbose: bool = False,
):
    model_list = {
        "v1.0": [
            "ftp://ftp.mrc-lmb.cam.ac.uk/pub/dari/blush_v1.0.ckpt.gz",
            "6e42a7d80231418bb77170619eeedf67b59be84078972a25a39fc3b82cd9c34e"
        ]
    }

    if name not in model_list.keys():
        return None

    dest_dir = os.path.join(torch.hub.get_dir(), "checkpoints", "relion_blush")
    model_path = os.path.join(dest_dir, f"{name}.ckpt")
    model_path_gz = model_path + ".gz"
    completed_check_path = os.path.join(dest_dir, f"{name}_installed.txt")

    # Download file and install it if not already done
    if not os.path.isfile(completed_check_path):
        if verbose:
            print(f"Installing Blush model ({name})...")
        os.makedirs(dest_dir, exist_ok=True)

        import gzip, shutil
        torch.hub.download_url_to_file(model_list[name][0], model_path_gz, hash_prefix=model_list[name][1])
        with gzip.open(model_path_gz, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(model_path_gz)

        with open(completed_check_path, "w") as f:
            f.write("Successfully downloaded model")

        if verbose:
            print(f"Blush model ({name}) successfully installed in {dest_dir}")

    # Load checkpoint file
    checkpoint = torch.load(model_path, map_location="cpu")

    # Dynamically include model as a module
    # Make sure to check download integrity for this, otherwise major security risk
    model_module = types.ModuleType("blush_model")
    exec(checkpoint['model_definition'], model_module.__dict__)
    sys.modules["blush_model"] = model_module

    # Load the model
    model = model_module.BlushModel().eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.voxel_size = float(checkpoint["voxel_size"])
    model.block_size = int(checkpoint["block_size"])
    model.no_mask = checkpoint["no_mask"]

    if verbose:
        print(f"Blush model ({name}) loaded successfully from checkpoint {model_path}")

    return model, model_path
