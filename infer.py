"""
Batch inference / evaluation for EyeCue.

Loads a checkpoint produced by `new_train.py` and runs the full
video + gaze pipeline over a test list, reporting accuracy and writing
per-sample predictions to a csv.

Usage:
    python infer.py \
        --test_list  /path/to/test.csv \
        --checkpoint checkpoints/best_model.pth \
        --out_csv    predictions.csv
"""

import os
import csv
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from data.dataset import VideoGazeDataset
from models.video_encoder import VideoEncoder
from models.gaze_encoder import GazeEncoder
from models.semantic import Semantic
from models.head import ClassificationHead
from new_train import select, collate_fn


def set_seed(seed: int):
    """
    VideoGazeDataset samples a random window of frames, so the seed is what makes
    an inference run reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_models(checkpoint_path, device):
    """
    Instantiate the four modules and load their weights from `checkpoint_path`.

    ClassificationHead.fc1 is created lazily on the first forward pass, so it has
    to be materialized with the right input dimension before `load_state_dict`.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    video_encoder = VideoEncoder().to(device)
    gaze_encoder  = GazeEncoder().to(device)
    semantic      = Semantic().to(device)
    head          = ClassificationHead().to(device)

    video_encoder.load_state_dict(ckpt['video_encoder'])
    gaze_encoder.load_state_dict(ckpt['gaze_encoder'])
    semantic.load_state_dict(ckpt['semantic'])

    head_sd = ckpt['head']
    if 'fc1.weight' in head_sd:
        in_features = head_sd['fc1.weight'].shape[1]
        head.fc1 = nn.Linear(in_features, head.hidden_dim).to(device)
    head.load_state_dict(head_sd)

    for m in (video_encoder, gaze_encoder, semantic, head):
        m.eval()

    meta = {'epoch': ckpt.get('epoch'), 'val_acc': ckpt.get('val_acc')}
    return video_encoder, gaze_encoder, semantic, head, meta


@torch.no_grad()
def run_inference(video_encoder, gaze_encoder, semantic, head,
                  loader, device, image_processor):
    """
    Returns (paths, labels, preds, probs) where `probs` is P(class = 1).
    """
    all_paths, all_labels, all_preds, all_probs = [], [], [], []
    correct = total = 0

    pbar = tqdm(loader, desc="[Inference]", leave=False)
    for frames_list, gazes, labels, paths in pbar:
        processed = image_processor(images=frames_list, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device)

        gazes  = gazes.to(device)
        labels = labels.to(device)

        video_feats, video_tokens, _ = video_encoder(pixel_values)
        gaze_feats, gaze_tokens      = gaze_encoder(gazes)
        select_video_tokens          = select(video_tokens, gazes)
        semantic_feats               = semantic(select_video_tokens, gaze_tokens)

        fused  = torch.cat([video_feats, gaze_feats, semantic_feats], dim=1)
        logits = head(fused)

        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        all_paths.extend(paths)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        pbar.set_postfix(acc=f"{100.0 * correct / total:.2f}%")

    return all_paths, all_labels, all_preds, all_probs


def report(labels, preds):
    """
    Print accuracy and the binary confusion matrix.
    """
    total = len(labels)
    correct = sum(int(y == p) for y, p in zip(labels, preds))

    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\nSamples  : {total}")
    print(f"Accuracy : {100.0 * correct / total:.2f}%  ({correct}/{total})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted)")
    print(f"            pred 0   pred 1")
    print(f"  true 0    {tn:6d}   {fp:6d}")
    print(f"  true 1    {fn:6d}   {tp:6d}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for the EyeCue video + gaze model"
    )
    parser.add_argument('--test_list',   type=str, required=True,
                        help="List file, one `video_path gaze_path label` per line")
    parser.add_argument('--checkpoint',  type=str, default='checkpoints/best_model.pth',
                        help="Checkpoint written by new_train.py")
    parser.add_argument('--out_csv',     type=str, default='predictions.csv',
                        help="Where to write the per-sample predictions")
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--clip_len',    type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--device',      type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/timesformer-base-finetuned-k600"
    )

    test_ds = VideoGazeDataset(args.test_list, clip_len=args.clip_len, transform=None)
    test_loader = DataLoader(test_ds,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=collate_fn)

    print(f"Loading checkpoint: {args.checkpoint}")
    video_encoder, gaze_encoder, semantic, head, meta = build_models(
        args.checkpoint, device
    )
    if meta['epoch'] is not None:
        print(f"  trained to epoch {meta['epoch']}, "
              f"recorded val_acc {meta['val_acc']:.2f}%")
    print(f"Running on {device} over {len(test_ds)} samples\n")

    paths, labels, preds, probs = run_inference(
        video_encoder, gaze_encoder, semantic, head,
        test_loader, device, image_processor
    )

    report(labels, preds)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "pred", "prob_distracted"])
        for p, y, y_hat, prob in zip(paths, labels, preds, probs):
            writer.writerow([p, int(y), int(y_hat), f"{prob:.6f}"])

    print(f"\nPredictions written to: {args.out_csv}")


if __name__ == "__main__":
    main()
