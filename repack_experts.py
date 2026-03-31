#!/usr/bin/env python3
"""Repack expert weights from MLX 4-bit safetensors into contiguous per-layer binary files.

Adapted for Qwen3-235B-A22B-4bit (mlx-community).

Creates one binary file per layer: packed_experts/layer_XX.bin
Each file = 128 experts x 10,616,832 bytes = ~1.27 GB
Expert E starts at byte offset E * 10,616,832

Within each expert block, 9 components packed in fixed order:
  gate_proj.weight, gate_proj.scales, gate_proj.biases,
  up_proj.weight,   up_proj.scales,   up_proj.biases,
  down_proj.weight,  down_proj.scales,  down_proj.biases

Source tensors are fused: shape [128, rows, cols] with experts stacked on dim 0.

Usage:
    python repack_experts.py --model PATH              # repack all 94 layers
    python repack_experts.py --model PATH --layers 0-4  # repack layers 0-4
    python repack_experts.py --model PATH --dry-run     # verify without writing
    python repack_experts.py --model PATH --verify-only 0  # verify layer 0
"""

import argparse
import json
import os
import struct
import time
import sys

# Component order and expected sizes for Qwen3-235B-A22B (MOE_INTERMEDIATE=1536)
COMPONENTS = [
    {"name": "gate_proj.weight",  "offset": 0,        "size": 3145728, "dtype": "U32",  "shape": [1536, 512],  "elem": 4},
    {"name": "gate_proj.scales",  "offset": 3145728,  "size": 196608,  "dtype": "BF16", "shape": [1536, 64],   "elem": 2},
    {"name": "gate_proj.biases",  "offset": 3342336,  "size": 196608,  "dtype": "BF16", "shape": [1536, 64],   "elem": 2},
    {"name": "up_proj.weight",    "offset": 3538944,  "size": 3145728, "dtype": "U32",  "shape": [1536, 512],  "elem": 4},
    {"name": "up_proj.scales",    "offset": 6684672,  "size": 196608,  "dtype": "BF16", "shape": [1536, 64],   "elem": 2},
    {"name": "up_proj.biases",    "offset": 6881280,  "size": 196608,  "dtype": "BF16", "shape": [1536, 64],   "elem": 2},
    {"name": "down_proj.weight",  "offset": 7077888,  "size": 3145728, "dtype": "U32",  "shape": [4096, 192],  "elem": 4},
    {"name": "down_proj.scales",  "offset": 10223616, "size": 196608,  "dtype": "BF16", "shape": [4096, 24],   "elem": 2},
    {"name": "down_proj.biases",  "offset": 10420224, "size": 196608,  "dtype": "BF16", "shape": [4096, 24],   "elem": 2},
]

EXPERT_SIZE = 10616832  # bytes per expert
NUM_EXPERTS = 128
NUM_LAYERS = 94
LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE  # 1,358,954,496 bytes (~1.27 GB)


def parse_layers(spec):
    """Parse layer specification like '0-4' or '0,5,10' or 'all'."""
    if spec is None or spec == 'all':
        return list(range(NUM_LAYERS))
    layers = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_len)
        header = json.loads(header_json)
        data_start = 8 + header_len
    # Remove __metadata__ key if present
    header.pop('__metadata__', None)
    return header, data_start


def build_shard_map(model_path):
    """Build mapping from layer -> {component -> (shard_path, abs_offset, expert_stride)}.

    The MLX model stores expert weights as fused 3D tensors: [128, rows, cols].
    Expert E's data starts at tensor_data_offset + E * rows * cols * elem_size.
    """
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    with open(index_path) as f:
        idx = json.load(f)

    weight_map = idx['weight_map']

    # Group expert tensors by layer and component
    # Tensor name pattern: model.layers.{L}.mlp.switch_mlp.{gate_proj|up_proj|down_proj}.{weight|scales|biases}
    shard_map = {}  # layer_idx -> {comp_name -> (shard_file, abs_offset, expert_stride)}

    # Parse headers for all shards that contain expert weights
    shard_headers = {}  # filename -> (header, data_start)
    expert_shards = set()
    for name, filename in weight_map.items():
        if '.mlp.switch_mlp.' in name:
            expert_shards.add(filename)

    print(f"Parsing {len(expert_shards)} shard headers...")
    for filename in sorted(expert_shards):
        filepath = os.path.join(model_path, filename)
        shard_headers[filename] = parse_safetensors_header(filepath)

    # Build the map
    for name, filename in weight_map.items():
        if '.mlp.switch_mlp.' not in name:
            continue

        # Parse: model.layers.{L}.mlp.switch_mlp.{proj}.{part}
        parts = name.split('.')
        layer_idx = int(parts[2])
        proj = parts[5]       # gate_proj, up_proj, down_proj
        part = parts[6]       # weight, scales, biases
        comp_name = f"{proj}.{part}"

        header, data_start = shard_headers[filename]
        meta = header[name]
        tensor_start = data_start + meta['data_offsets'][0]
        tensor_size = meta['data_offsets'][1] - meta['data_offsets'][0]

        # Shape is [128, rows, cols] — expert stride = rows * cols * elem_size
        shape = meta['shape']
        assert shape[0] == NUM_EXPERTS, f"Expected {NUM_EXPERTS} experts, got {shape[0]} for {name}"
        expert_stride = tensor_size // NUM_EXPERTS

        if layer_idx not in shard_map:
            shard_map[layer_idx] = {}
        shard_map[layer_idx][comp_name] = {
            'file': os.path.join(model_path, filename),
            'abs_offset': tensor_start,
            'expert_stride': expert_stride,
            'shape': shape,
        }

    # Verify all layers have all components
    comp_names = {c['name'] for c in COMPONENTS}
    for layer_idx in range(NUM_LAYERS):
        if layer_idx not in shard_map:
            print(f"WARNING: layer {layer_idx} has no expert tensors in index")
            continue
        missing = comp_names - set(shard_map[layer_idx].keys())
        if missing:
            print(f"WARNING: layer {layer_idx} missing components: {missing}")

    return shard_map


def repack_layer(layer_idx, shard_map, output_dir, dry_run=False):
    """Repack all 128 experts for one layer into a contiguous binary file."""
    if layer_idx not in shard_map:
        print(f"  Layer {layer_idx}: NOT FOUND in shard map, skipping")
        return 0, 0.0

    layer_info = shard_map[layer_idx]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if dry_run:
        for expert_idx in range(NUM_EXPERTS):
            for comp in COMPONENTS:
                info = layer_info[comp['name']]
                src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
                dst_offset = expert_idx * EXPERT_SIZE + comp['offset']
        print(f"  Layer {layer_idx:2d}: DRY RUN OK — would write {LAYER_SIZE:,} bytes to {out_path}")
        return LAYER_SIZE, 0.0

    t0 = time.monotonic()

    # Pre-allocate output file
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, LAYER_SIZE)

    bytes_written = 0

    # Open source files (may be multiple shards for boundary layers)
    source_fds = {}  # filepath -> fd

    # Build read plan grouped by source file for locality
    read_plan = []  # (src_path, src_offset, dst_offset, size)
    for expert_idx in range(NUM_EXPERTS):
        for comp in COMPONENTS:
            info = layer_info[comp['name']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * EXPERT_SIZE + comp['offset']
            read_plan.append((info['file'], src_offset, dst_offset, comp['size']))

    # Sort by (src_file, src_offset) for sequential locality
    read_plan.sort(key=lambda x: (x[0], x[1]))

    # Execute
    for src_path, src_offset, dst_offset, size in read_plan:
        if src_path not in source_fds:
            source_fds[src_path] = os.open(src_path, os.O_RDONLY)

        data = os.pread(source_fds[src_path], size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)} at offset {src_offset} in {src_path}")
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    for fd in source_fds.values():
        os.close(fd)

    elapsed = time.monotonic() - t0
    return bytes_written, elapsed


def verify_layer(layer_idx, shard_map, output_dir):
    """Verify packed layer file against source safetensors."""
    layer_info = shard_map[layer_idx]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)
    source_fds = {}
    mismatches = 0

    # Spot check experts 0, 1, 63, 127
    for expert_idx in [0, 1, 63, 127]:
        for comp in COMPONENTS:
            info = layer_info[comp['name']]
            src_path = info['file']
            if src_path not in source_fds:
                source_fds[src_path] = os.open(src_path, os.O_RDONLY)

            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * EXPERT_SIZE + comp['offset']

            original = os.pread(source_fds[src_path], comp['size'], src_offset)
            packed = os.pread(fd_packed, comp['size'], dst_offset)

            if original != packed:
                print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}")
                mismatches += 1

    os.close(fd_packed)
    for fd in source_fds.values():
        os.close(fd)

    if mismatches == 0:
        print(f"  Layer {layer_idx}: verification PASSED (experts 0, 1, 63, 127)")
    else:
        print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")

    return mismatches == 0


def write_layout(output_dir):
    """Write layout.json describing the packed format."""
    layout = {
        "model": "Qwen3-235B-A22B-4bit",
        "expert_size": EXPERT_SIZE,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "moe_intermediate_size": 1536,
        "hidden_size": 4096,
        "group_size": 64,
        "bits": 4,
        "components": COMPONENTS,
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, 'w') as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Repack Qwen3-235B-A22B expert weights into per-layer binary files")
    parser.add_argument('--model', required=True,
                        help='Path to mlx-community/Qwen3-235B-A22B-4bit model directory')
    parser.add_argument('--output', default=None,
                        help='Output directory for packed_experts/ (default: MODEL/packed_experts)')
    parser.add_argument('--layers', default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Verify offsets without writing')
    parser.add_argument('--verify-only', type=int, default=None, metavar='LAYER',
                        help='Verify a specific layer against originals')
    args = parser.parse_args()

    model_path = args.model
    if not os.path.isdir(model_path):
        print(f"ERROR: model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print("Building shard map from safetensors index...")
    shard_map = build_shard_map(model_path)
    print(f"Mapped {len(shard_map)} layers")

    output_dir = args.output or os.path.join(model_path, "packed_experts")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.verify_only is not None:
        layers = [args.verify_only]
    else:
        layers = parse_layers(args.layers)

    print(f"Layers to process: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

    if not args.dry_run and args.verify_only is None:
        total_bytes = len(layers) * LAYER_SIZE
        print(f"Total data to write: {total_bytes / (1024**3):.1f} GB")

        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        needed_gb = total_bytes / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB, needed: {needed_gb:.1f} GB")
        if free_bytes < total_bytes:
            print(f"WARNING: Not enough free space! Need {needed_gb:.1f} GB but only {free_gb:.1f} GB free.")
            hint_layers = int(free_gb / (LAYER_SIZE / (1024**3))) - 1
            print(f"Hint: use --layers to process a subset, e.g. --layers 0-{max(0, hint_layers)}")
            sys.exit(1)

    if args.verify_only is not None:
        verify_layer(args.verify_only, shard_map, output_dir)
        return

    write_layout(output_dir)

    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        bytes_written, elapsed = repack_layer(
            layer_idx, shard_map, output_dir, dry_run=args.dry_run
        )
        total_written += bytes_written

        if not args.dry_run and bytes_written > 0:
            throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float('inf')
            overall_elapsed = time.monotonic() - t_start
            overall_throughput = total_written / overall_elapsed / (1024**3) if overall_elapsed > 0 else 0
            eta = (len(layers) - i - 1) * (overall_elapsed / (i + 1))
            print(f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                  f"({throughput:.1f} GB/s) | "
                  f"Total: {total_written/1024**3:.1f}/{len(layers)*LAYER_SIZE/1024**3:.1f} GB "
                  f"({overall_throughput:.1f} GB/s avg) | "
                  f"ETA: {eta:.0f}s")

            if not verify_layer(layer_idx, shard_map, output_dir):
                print(f"ABORTING: verification failed for layer {layer_idx}")
                sys.exit(1)

    total_elapsed = time.monotonic() - t_start
    if not args.dry_run and total_written > 0:
        print(f"\n{'='*60}")
        print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
        print(f"Time: {total_elapsed:.1f}s")
        print(f"Throughput: {total_written/total_elapsed/1024**3:.1f} GB/s")
        print(f"Output: {output_dir}")
    elif args.dry_run:
        print(f"\nDRY RUN complete: {len(layers)} layers validated")


if __name__ == '__main__':
    main()
