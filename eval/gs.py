import os
import cv2
import numpy as np
import argparse
import gs  # geometry-score
import random

def load_images_from_folder(folder, max_images=10000, resize_shape=(28, 28)):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != resize_shape:
                img = cv2.resize(img, resize_shape)
            img = img.astype(np.float32) / 255.0
            images.append(img.flatten())
            if len(images) >= max_images:
                break
    return np.array(images)

def compute_geometry_score(real_dir, args):
    max_images = 10000 # 5940 # 5000 # 48070 5940
    print(f"Loading real images {real_dir}")
    real_data = load_images_from_folder(real_dir, max_images)
    print(f"Loaded {real_data.shape[0]} real images.")

    print("Loading DNA images...")
    dna_data = load_images_from_folder(args.gen_dir, max_images)
    print(f"Loaded {dna_data.shape[0]} generated images.")

    max_images = real_data.shape[0]

    print("Computing MRLT for real data...")
    rlt_real = gs.rlts(real_data, gamma=1.0/128, n=max_images)

    print("Computing MRLT for DNA data...")
    rlt_dna = gs.rlts(dna_data, gamma=1.0/128, n=max_images)

    dna_gs = gs.geom_score(rlt_real, rlt_dna) # np.sum((mrlt_real - mrlt_gen) ** 2)

    print(f"Geometry DNA Score: {dna_gs*1000:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Geometry Score between two image folders.")
    parser.add_argument("--real_dir", type=str, default="../Generated/result/test2_gt")
    parser.add_argument("--gen_dir", type=str, default="../Generated/result/test2")
    args = parser.parse_args()

    compute_geometry_score(args.real_dir, args)
