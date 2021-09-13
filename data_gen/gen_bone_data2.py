import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

ntu_skeleton_bone_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)

bone_pairs = {
    'ntu/xview': ntu_skeleton_bone_pairs,
    'ntu/xsub': ntu_skeleton_bone_pairs,

    # NTU 120 uses the same skeleton structure as NTU 60
    'ntu120/xsub': ntu_skeleton_bone_pairs,
    'ntu120/xset': ntu_skeleton_bone_pairs,
}

benchmarks = {
    'ntu': ('ntu/xview', 'ntu/xsub'),
    'ntu120': ('ntu120/xset', 'ntu120/xsub'),
}

parts = { 'train', 'val' }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bone data generation for NTU60/NTU120')
    parser.add_argument('--dataset', choices=['ntu', 'ntu120'], required=True)
    args = parser.parse_args()

    for benchmark in benchmarks[args.dataset]:
        for part in parts:
            print(benchmark, part)
            try:
                data = np.load('../data/{}/{}_data_joint.npy'.format(benchmark, part), mmap_mode='r')
                N, C, T, V, M = data.shape
                fp_sp = open_memmap(
                    '../data/{}/{}_data_bone_outward.npy'.format(benchmark, part),
                    dtype='float32',
                    mode='w+',
                    shape=(N, 3, T, V, M))

                fp_sp[:, :C, :, :, :] = data
                for v1, v2 in tqdm(bone_pairs[benchmark]):
                    v1 -= 1
                    v2 -= 1
                    fp_sp[:, :, :, v1, :] = data[:, :, :, v2, :] - data[:, :, :, v1, :]
            except Exception as e:
                print(f'Run into error: {e}')
                print(f'Skipping ({benchmark} {part})')