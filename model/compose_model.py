#!/usr/bin/python3

import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('definition', type=str, help="Path to model definition")
    parser.add_argument('checkpoint', type=str, help="Path to training checkpoint file")
    parser.add_argument('output', type=str, help="output file")
    parser.add_argument('voxel_size', type=float, help="training voxel size for model")
    parser.add_argument('block_size', type=float, help="training voxel size for model")
    parser.add_argument('--no_mask', action="store_true", help="model has not mask output")
    args = parser.parse_args()

    device = torch.device("cpu")

    state_dict = torch.load(args.checkpoint, map_location="cpu")

    with open(args.definition, "r") as file:
        model_definition = file.read()

    composed_model = {
        'model_state_dict': state_dict['model'],
        'model_definition': model_definition,
        'block_size': args.block_size,
        'voxel_size': args.voxel_size,
        'no_mask': args.no_mask
    }

    torch.save(composed_model, args.output)

    print("Done!")
