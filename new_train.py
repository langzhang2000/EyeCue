import os
import argparse
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import VideoGazeDataset
from models.video_encoder import VideoEncoder
from models.gaze_encoder import GazeEncoder
from models.head import ClassificationHead
from models.semantic import Semantic

from transformers import AutoImageProcessor

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# Gaze-guided patch selection
#
# The default strategy picks 1 token per frame (8 frames): the gaze point of
# each frame is mapped onto the 14x14 patch grid of the TimeSformer token space,
# and the corresponding token is gathered out of `video_tokens`.
#
# Alternative strategies used in the ablation study (5 / 9 / 25 tokens per frame,
# and 16-frame variants) are kept at the bottom of this file for reference.
# =============================================================================
def select(video_tokens, gazes):
    """
    video_tokens: [B, 1568, D]   (8 frames x 196 patches)
    gazes:        [B, 8, 2]      per-frame gaze point (x, y) in pixels
    Returns:      [B, 8, D]      one selected token per frame
    """
    col = gazes[..., 0]
    row = gazes[..., 1]
    col_clamped = col.clamp(min=1.0, max=959.0)
    row_clamped = row.clamp(min=1.0, max=719.0)
    col_scaled = torch.ceil(col_clamped / 960.0 * 14.0) - 1.0  # shape [B, 8]
    row_scaled = torch.ceil(row_clamped / 720.0 * 14.0) - 1.0  # shape [B, 8]
    col_idx = col_scaled.to(torch.long)  # shape [B, 8], values in {0, ..., 13}
    row_idx = row_scaled.to(torch.long)  # shape [B, 8], values in {0, ..., 13}
    z = row_idx * 14 + col_idx           # shape [B, 8]
    gazes_new = z

    B, _, D = video_tokens.shape
    # 1. Build the frame ids [0..7] and expand to [B, 8]
    frame_ids = torch.arange(8, device=video_tokens.device)       # shape [8]
    frame_ids = frame_ids.unsqueeze(0).expand(B, -1)              # shape [B, 8]
    # 2. Index into dim 1 (length 1568): idxs[b,i] = frame_ids[b,i] * 196 + gazes_new[b,i]
    idxs = frame_ids * 196 + gazes_new                            # shape [B, 8]
    # 3. Expand the index tensor to [B, 8, D] for torch.gather
    idxs_expanded = idxs.unsqueeze(-1).expand(-1, -1, D)          # shape [B, 8, D]
    # 4. Gather the matching tokens along dim 1 -> [B, 8, D]
    selected_tokens = torch.gather(video_tokens, dim=1, index=idxs_expanded)
    return selected_tokens


def collate_fn(batch):
    """
    Each element of `batch` is (frames_list, gaze_tensor, label, video_path).

    Returns:
      - batch_frames: List[List[PIL.Image]]
      - batch_gazes:  torch.Tensor([B, clip_len, 2])
      - batch_labels: torch.Tensor([B])
      - batch_paths:  List[str]
    """
    batch_frames = [item[0] for item in batch]
    batch_gazes = torch.stack([item[1] for item in batch], dim=0)
    batch_labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    batch_paths = [item[3] for item in batch]
    return batch_frames, batch_gazes, batch_labels, batch_paths


def train_one_epoch(video_encoder, gaze_encoder, semantic, head, loader,
                    optimizer, criterion, device,
                    epoch, total_epochs, image_processor):
    video_encoder.train()
    gaze_encoder.train()
    semantic.train()
    head.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader,
                desc=f"[Epoch {epoch}/{total_epochs}] Train",
                leave=False)
    for frames_list, gazes, labels, paths in pbar:
        # frames_list: List[List[PIL.Image]]
        # gazes:       [B, clip_len, 2]
        # labels:      [B]
        # paths:       List[str]  (unused during training)

        processed = image_processor(
            images=frames_list,
            return_tensors="pt"
        )
        pixel_values = processed["pixel_values"].to(device)

        gazes = gazes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        video_feats, video_tokens, cls_attn = video_encoder(pixel_values)  # [B, L1, D]
        gaze_feats, gaze_tokens = gaze_encoder(gazes)                      # [B, L2, D]
        select_video_tokens = select(video_tokens, gazes)
        semantic_feats = semantic(select_video_tokens, gaze_tokens)
        fused = torch.cat([video_feats, gaze_feats, semantic_feats], dim=1)
        logits = head(fused)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / total
        acc = correct / total * 100.0
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}%")

    return running_loss / total, correct / total * 100.0


def evaluate(video_encoder, gaze_encoder, semantic, head, loader,
             criterion, device, epoch, total_epochs, image_processor,
             collect_results=False):
    """
    collect_results=False: plain evaluation, returns (loss, acc)
    collect_results=True:  additionally returns (paths, labels, preds)
    """
    video_encoder.eval()
    gaze_encoder.eval()
    semantic.eval()
    head.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_paths = []
    all_labels = []
    all_preds = []

    pbar = tqdm(loader,
                desc=f"[Epoch {epoch}/{total_epochs}]  Val ",
                leave=False)
    with torch.no_grad():
        for frames_list, gazes, labels, paths in pbar:
            processed = image_processor(
                images=frames_list,
                return_tensors="pt"
            )
            pixel_values = processed["pixel_values"].to(device)

            gazes = gazes.to(device)
            labels = labels.to(device)

            video_feats, video_tokens, cls_attn = video_encoder(pixel_values)
            gaze_feats, gaze_tokens = gaze_encoder(gazes)
            select_video_tokens = select(video_tokens, gazes)
            semantic_feats = semantic(select_video_tokens, gaze_tokens)
            fused = torch.cat([video_feats, gaze_feats, semantic_feats], dim=1)
            logits = head(fused)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if collect_results:
                all_paths.extend(paths)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

            avg_loss = running_loss / total
            acc = correct / total * 100.0
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.1f}%")

    avg_loss = running_loss / total
    avg_acc = correct / total * 100.0

    if collect_results:
        return avg_loss, avg_acc, all_paths, all_labels, all_preds
    else:
        return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(
        description="Train Video+Gaze Fusion Model for Binary Classification"
    )
    parser.add_argument('--train_list',   type=str, required=True)
    parser.add_argument('--val_list',     type=str, required=True)
    parser.add_argument('--batch_size',   type=int, default=4)
    parser.add_argument('--epochs',       type=int, default=15)
    parser.add_argument('--lr',           type=float, default=1e-5)
    parser.add_argument('--clip_len',     type=int, default=8)
    parser.add_argument('--num_workers',  type=int, default=1)
    parser.add_argument('--device',       type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir',     type=str, default='checkpoints')
    args = parser.parse_args()

    # AutoImageProcessor handles resize / crop / normalize
    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/timesformer-base-finetuned-k600"
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Datasets & DataLoaders
    train_ds = VideoGazeDataset(args.train_list, clip_len=args.clip_len, transform=None)
    val_ds   = VideoGazeDataset(args.val_list,   clip_len=args.clip_len, transform=None)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)

    device = torch.device(args.device)

    # Models
    video_encoder = VideoEncoder().to(device)
    gaze_encoder  = GazeEncoder().to(device)
    head          = ClassificationHead().to(device)
    semantic      = Semantic().to(device)

    optimizer = torch.optim.Adam(
        list(video_encoder.parameters()) +
        list(gaze_encoder.parameters()) +
        list(semantic.parameters()) +
        list(head.parameters()),
        lr=args.lr
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            video_encoder, gaze_encoder, semantic, head,
            train_loader, optimizer, criterion,
            device, epoch, args.epochs,
            image_processor
        )
        val_loss, val_acc = evaluate(
            video_encoder, gaze_encoder, semantic, head,
            val_loader, criterion,
            device, epoch, args.epochs,
            image_processor,
            collect_results=False
        )

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%   "
              f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # New best validation accuracy: save the model and dump predictions
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # 1) Save the best checkpoint
            ckpt_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'video_encoder': video_encoder.state_dict(),
                'gaze_encoder':  gaze_encoder.state_dict(),
                'semantic':      semantic.state_dict(),
                'head':          head.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'epoch':         epoch,
                'val_acc':       val_acc,
            }, ckpt_path)

            torch.save(
                video_encoder.state_dict(),
                os.path.join(args.save_dir, 'eye_video_encoder.pth')
            )

            # 2) Re-run the current best model over val_loader to collect results
            _, _, paths, labels, preds = evaluate(
                video_encoder, gaze_encoder, semantic, head,
                val_loader, criterion,
                device, epoch, args.epochs,
                image_processor,
                collect_results=True
            )

            # 3) Write the csv: filename, ground-truth label, predicted label
            csv_path = os.path.join(args.save_dir, f"best_val_predictions_{timestamp}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "label", "pred"])
                for p, y, y_hat in zip(paths, labels, preds):
                    # Use os.path.basename(p) to keep only the file name
                    writer.writerow([p, int(y), int(y_hat)])

            print(f"[BEST] Epoch {epoch}: val_acc={val_acc:.2f}%, "
                  f"saved checkpoint to {ckpt_path}")
            print(f"[BEST] Predictions csv saved to: {csv_path}")


if __name__ == "__main__":
    main()


'''
python new_train.py \
  --train_list /path/to/train.csv \
  --val_list   /path/to/val.csv
'''


# =============================================================================
# Alternative gaze-guided selection strategies (ablation study)
#
# The variants below were used to produce the token-count and clip-length
# ablations reported in the paper. They are kept here for reference; to use one,
# swap it in for the `select` defined at the top of this file.
#
# Note: the 16-frame variants are not recommended -- 8 frames already satisfies
# the 2-frame tubelet requirement of the backbone.
# =============================================================================

# ---- 5 tokens per frame (8 frames): centre, up, down, left, right ----
# def select(video_tokens, gazes):
#     """
#     video_tokens: [B, 1568, D]
#     gazes:        [B, 8, 2]   per-frame gaze point (x, y) in pixels
#     Returns:      [B, 8*5, D] 5 tokens per frame, ordered centre/up/down/left/right
#     """
#     B, _, D = video_tokens.shape
#
#     # Step 1: map the pixel coordinates onto the 14x14 grid
#     x = gazes[..., 0].clamp(min=1.0, max=719.0)
#     y = gazes[..., 1].clamp(min=1.0, max=959.0)
#     x_grid = torch.ceil(x / 720.0 * 14.0) - 1.0  # shape: [B, 8]
#     y_grid = torch.ceil(y / 960.0 * 14.0) - 1.0
#     x_idx = x_grid.to(torch.long).clamp(0, 13)   # in [0, 13]
#     y_idx = y_grid.to(torch.long).clamp(0, 13)
#
#     # Step 2: the 5 positions -- centre, up, down, left, right (clamped at borders)
#     def safe_shift(idx, delta, max_val=13):
#         return (idx + delta).clamp(0, max_val)
#
#     coords = [
#         (x_idx, y_idx),                             # centre
#         (safe_shift(x_idx, -1), y_idx),             # up
#         (safe_shift(x_idx, +1), y_idx),             # down
#         (x_idx, safe_shift(y_idx, -1)),             # left
#         (x_idx, safe_shift(y_idx, +1))              # right
#     ]
#
#     # Step 3: turn the 2D indices into token indices (0..195) -> [B, 8, 5]
#     token_idxs = torch.stack([
#         x_ * 14 + y_ for (x_, y_) in coords
#     ], dim=-1)  # shape: [B, 8, 5]
#
#     # Step 4: add the per-frame offset
#     frame_ids = torch.arange(8, device=video_tokens.device).view(1, 8, 1)  # [1, 8, 1]
#     full_token_idxs = frame_ids * 196 + token_idxs                         # [B, 8, 5]
#
#     # Step 5: gather -> [B, 8*5, D]
#     full_token_idxs = full_token_idxs.view(B, -1)                          # [B, 40]
#     gather_idx = full_token_idxs.unsqueeze(-1).expand(-1, -1, D)           # [B, 40, D]
#     selected_tokens = torch.gather(video_tokens, dim=1, index=gather_idx)  # [B, 40, D]
#     return selected_tokens


# ---- 9 tokens per frame (8 frames): the 3x3 neighbourhood around the gaze ----
# def select(video_tokens, gazes):
#     """
#     Select the 3x3 patch neighbourhood centred on the gaze point of each frame
#     (9 tokens), clamping at the borders.
#
#     Args:
#       video_tokens: [B, 1568, D]  8 frames, 14x14 = 196 tokens per frame
#       gazes:        [B, 8, 2]     one gaze point per frame, in pixels
#
#     Returns:
#       selected_tokens: [B, 72, D]
#     """
#     B, _, D = video_tokens.shape
#
#     # Step 1: map the pixel coordinates onto the 14x14 grid (integers in [0, 13])
#     x = gazes[..., 0].clamp(1.0, 719.0)
#     y = gazes[..., 1].clamp(1.0, 959.0)
#     x_idx = (torch.ceil(x / 720.0 * 14.0) - 1).to(torch.long).clamp(0, 13)
#     y_idx = (torch.ceil(y / 960.0 * 14.0) - 1).to(torch.long).clamp(0, 13)
#
#     # Step 2: the 3x3 offsets (centre + 8 neighbours)
#     dx = torch.tensor([-1,  0,  1, -1, 0, 1, -1, 0, 1], device=video_tokens.device)  # 9
#     dy = torch.tensor([-1, -1, -1,  0, 0, 0,  1, 1, 1], device=video_tokens.device)  # 9
#
#     # shape: [1, 1, 9] -> broadcast to [B, 8, 9]
#     dx = dx.view(1, 1, 9)
#     dy = dy.view(1, 1, 9)
#
#     # Step 3: expand the base coordinates to [B, 8, 1] and add the offsets
#     x_neighbors = x_idx.unsqueeze(-1) + dx   # [B, 8, 9]
#     y_neighbors = y_idx.unsqueeze(-1) + dy   # [B, 8, 9]
#
#     # Clamp every coordinate back into range
#     x_neighbors = x_neighbors.clamp(0, 13)
#     y_neighbors = y_neighbors.clamp(0, 13)
#
#     # Step 4: per-frame patch index x * 14 + y -> [B, 8, 9]
#     local_patch_idx = x_neighbors * 14 + y_neighbors
#
#     # Step 5: add the per-frame offset (196 tokens per frame) -> [B, 8, 9]
#     frame_ids = torch.arange(8, device=video_tokens.device).view(1, 8, 1)  # [1, 8, 1]
#     global_token_idx = frame_ids * 196 + local_patch_idx                   # [B, 8, 9]
#
#     # Step 6: gather -> [B, 72, D]
#     gather_idx = global_token_idx.view(B, -1).unsqueeze(-1).expand(-1, -1, D)  # [B, 72, D]
#     selected_tokens = torch.gather(video_tokens, dim=1, index=gather_idx)      # [B, 72, D]
#     return selected_tokens


# ---- 25 tokens per frame (8 frames): the 5x5 neighbourhood around the gaze ----
# def select(video_tokens, gazes):
#     """
#     Select the 5x5 patch neighbourhood centred on the gaze point of each frame
#     (25 tokens), clamping at the borders.
#
#     Args:
#       video_tokens: [B, 1568, D]  8 frames, 14x14 = 196 tokens per frame
#       gazes:        [B, 8, 2]     one gaze point per frame, in pixels
#
#     Returns:
#       selected_tokens: [B, 200, D]
#     """
#     B, _, D = video_tokens.shape
#
#     # Step 1: pixel coordinates -> 14x14 patch grid index (integers in [0, 13])
#     x = gazes[..., 0].clamp(1.0, 719.0)
#     y = gazes[..., 1].clamp(1.0, 959.0)
#     x_idx = (torch.ceil(x / 720.0 * 14.0) - 1).to(torch.long).clamp(0, 13)
#     y_idx = (torch.ceil(y / 960.0 * 14.0) - 1).to(torch.long).clamp(0, 13)
#
#     # Step 2: the 5x5 neighbour offsets (dx, dy in {-2, -1, 0, 1, 2})
#     range_5 = torch.tensor([-2, -1, 0, 1, 2], device=video_tokens.device)
#     dx, dy = torch.meshgrid(range_5, range_5, indexing='ij')  # dx, dy shape: [5, 5]
#     dx = dx.reshape(-1)  # [25]
#     dy = dy.reshape(-1)  # [25]
#
#     # Step 3: add the offsets and clamp at the borders -> [B, 8, 25]
#     x_neighbors = x_idx.unsqueeze(-1) + dx.view(1, 1, -1)
#     y_neighbors = y_idx.unsqueeze(-1) + dy.view(1, 1, -1)
#     x_neighbors = x_neighbors.clamp(0, 13)
#     y_neighbors = y_neighbors.clamp(0, 13)
#
#     # Step 4: per-frame patch index (x * 14 + y) -> [B, 8, 25]
#     local_patch_idx = x_neighbors * 14 + y_neighbors
#
#     # Step 5: turn into a global index (accounting for the frame) -> [B, 8, 25]
#     frame_ids = torch.arange(8, device=video_tokens.device).view(1, 8, 1)  # [1, 8, 1]
#     global_token_idx = frame_ids * 196 + local_patch_idx                   # [B, 8, 25]
#
#     # Step 6: gather -> [B, 200, D]
#     gather_idx = global_token_idx.view(B, -1).unsqueeze(-1).expand(-1, -1, D)  # [B, 200, D]
#     selected_tokens = torch.gather(video_tokens, dim=1, index=gather_idx)      # [B, 200, D]
#     return selected_tokens


# ---- 1 token per frame (16 frames) ----
# def select(video_tokens, gazes):
#     B, N, D = video_tokens.shape
#     assert N == 16 * 196, f"Expect N=3136, got {N}"
#
#     # 1. Clamp the pixel coordinates to the image bounds
#     x = gazes[..., 0].clamp(min=1.0, max=719.0)
#     y = gazes[..., 1].clamp(min=1.0, max=959.0)
#
#     # 2. Map onto the 14x14 grid and convert to integer indices [0..13]
#     x_idx = (torch.ceil(x / 720.0 * 14.0) - 1.0).to(torch.long)  # shape [B, 16]
#     y_idx = (torch.ceil(y / 960.0 * 14.0) - 1.0).to(torch.long)  # shape [B, 16]
#
#     # 3. Per-frame token offset [0..195]
#     offset = x_idx * 14 + y_idx  # shape [B, 16]
#
#     # 4. Build the frame ids [0..15] and expand to [B, 16]
#     frame_ids = torch.arange(16, device=video_tokens.device)      # [16]
#     frame_ids = frame_ids.unsqueeze(0).expand(B, -1)              # [B, 16]
#
#     # 5. Global index = frame_id * 196 + offset
#     idxs = frame_ids * 196 + offset                              # [B, 16]
#
#     # 6. Expand to [B, 16, D] for gather
#     idxs_expanded = idxs.unsqueeze(-1).expand(-1, -1, D)          # [B, 16, D]
#
#     # 7. Gather along dim 1
#     selected_tokens = torch.gather(video_tokens, dim=1, index=idxs_expanded)  # [B, 16, D]
#     return selected_tokens


# ---- 5 tokens per frame (16 frames) ----
# def select(video_tokens, gazes):
#     """
#     Select 5 tokens per frame (centre, up, down, left, right) from 16 frames.
#
#     Args:
#       video_tokens: Tensor of shape [B, 16*196, D]
#       gazes:        Tensor of shape [B, 16, 2], per-frame gaze point (x, y) in pixels
#
#     Returns:
#       selected:     Tensor of shape [B, 16*5, D]
#     """
#     B, N, D = video_tokens.shape
#     assert N == 16 * 196, f"Expect token dim 16*196=3136, got {N}"
#
#     # 1. Pixels -> 14x14 grid index [0..13]
#     x = gazes[..., 0].clamp(min=1.0, max=719.0)
#     y = gazes[..., 1].clamp(min=1.0, max=959.0)
#     x_grid = torch.ceil(x / 720.0 * 14.0) - 1.0   # [B, 16]
#     y_grid = torch.ceil(y / 960.0 * 14.0) - 1.0   # [B, 16]
#     x_idx = x_grid.to(torch.long).clamp(0, 13)    # [B, 16]
#     y_idx = y_grid.to(torch.long).clamp(0, 13)    # [B, 16]
#
#     # 2. Border-safe shift
#     def safe_shift(idx, delta, max_val=13):
#         return (idx + delta).clamp(0, max_val)
#
#     coords = [
#         ( x_idx,                 y_idx                 ),  # centre
#         ( safe_shift(x_idx, -1), y_idx                 ),  # up
#         ( safe_shift(x_idx, +1), y_idx                 ),  # down
#         ( x_idx,                 safe_shift(y_idx, -1) ),  # left
#         ( x_idx,                 safe_shift(y_idx, +1) )   # right
#     ]
#
#     # 3. Local index -> token offset [0..195], stacked to [B, 16, 5]
#     token_offsets = torch.stack([
#         x_ * 14 + y_ for (x_, y_) in coords
#     ], dim=-1)  # [B, 16, 5]
#
#     # 4. Global index frame_id * 196 + offset -> [B, 16, 5]
#     frame_ids = torch.arange(16, device=video_tokens.device).view(1, 16, 1)  # [1, 16, 1]
#     full_idxs = frame_ids * 196 + token_offsets                              # [B, 16, 5]
#
#     # 5. Flatten to [B, 80], expand to [B, 80, D], then gather
#     flat = full_idxs.view(B, -1)                     # [B, 16*5=80]
#     idxs = flat.unsqueeze(-1).expand(-1, -1, D)      # [B, 80, D]
#     selected = torch.gather(video_tokens, dim=1, index=idxs)  # [B, 80, D]
#
#     return selected


# ---- 9 tokens per frame (16 frames) ----
# def select(video_tokens, gazes):
#     B, N, D = video_tokens.shape
#     assert N == 16 * 196, f"Expect token dim 16*196=3136, got {N}"
#
#     # Step 1: map the pixel coordinates onto the 14x14 grid (integers in [0, 13])
#     x = gazes[..., 0].clamp(1.0, 719.0)
#     y = gazes[..., 1].clamp(1.0, 959.0)
#     x_idx = (torch.ceil(x / 720.0 * 14.0) - 1).to(torch.long).clamp(0, 13)  # [B, 16]
#     y_idx = (torch.ceil(y / 960.0 * 14.0) - 1).to(torch.long).clamp(0, 13)  # [B, 16]
#
#     # Step 2: the 3x3 offsets (centre + 8 neighbours)
#     dx = torch.tensor([-1,  0,  1, -1, 0, 1, -1, 0, 1], device=video_tokens.device)  # 9
#     dy = torch.tensor([-1, -1, -1,  0, 0, 0,  1, 1, 1], device=video_tokens.device)  # 9
#     dx = dx.view(1, 1, 9)  # [1, 1, 9]
#     dy = dy.view(1, 1, 9)  # [1, 1, 9]
#
#     # Step 3: expand the base coordinates to [B, 16, 1] and add the offsets
#     x_neigh = x_idx.unsqueeze(-1) + dx   # [B, 16, 9]
#     y_neigh = y_idx.unsqueeze(-1) + dy   # [B, 16, 9]
#     x_neigh = x_neigh.clamp(0, 13)
#     y_neigh = y_neigh.clamp(0, 13)
#
#     # Step 4: per-frame patch index -> [B, 16, 9]
#     local_idx = x_neigh * 14 + y_neigh
#
#     # Step 5: add the per-frame offset (196 tokens per frame) -> [B, 16, 9]
#     frame_ids = torch.arange(16, device=video_tokens.device).view(1, 16, 1)  # [1, 16, 1]
#     global_idx = frame_ids * 196 + local_idx                                 # [B, 16, 9]
#
#     # Step 6: gather -> [B, 16*9, D]
#     flat_idx = global_idx.view(B, -1)                                        # [B, 144]
#     gather_idx = flat_idx.unsqueeze(-1).expand(-1, -1, D)                    # [B, 144, D]
#     selected_tokens = torch.gather(video_tokens, dim=1, index=gather_idx)    # [B, 144, D]
#
#     return selected_tokens


# ---- 25 tokens per frame (16 frames) ----
# def select(video_tokens, gazes):
#     """
#     Select the 5x5 patch neighbourhood centred on the gaze point of each of the
#     16 frames (25 tokens), clamping at the borders.
#
#     Args:
#       video_tokens: Tensor of shape [B, 16*196, D], 16 frames, 14x14 = 196 tokens each
#       gazes:        Tensor of shape [B, 16, 2], one gaze point per frame, in pixels
#
#     Returns:
#       selected_tokens: Tensor of shape [B, 16*25, D]
#     """
#     B, N, D = video_tokens.shape
#     assert N == 16 * 196, f"Expected token dim 16*196=3136, got {N}"
#
#     # Step 1: pixel coordinates -> 14x14 grid index [0..13]
#     x = gazes[..., 0].clamp(1.0, 719.0)
#     y = gazes[..., 1].clamp(1.0, 959.0)
#     x_idx = (torch.ceil(x / 720.0 * 14.0) - 1).to(torch.long).clamp(0, 13)  # [B, 16]
#     y_idx = (torch.ceil(y / 960.0 * 14.0) - 1).to(torch.long).clamp(0, 13)  # [B, 16]
#
#     # Step 2: the 5x5 neighbour offsets (dx, dy in {-2, -1, 0, 1, 2})
#     range_5 = torch.tensor([-2, -1, 0, 1, 2], device=video_tokens.device)
#     dx, dy = torch.meshgrid(range_5, range_5, indexing='ij')  # each [5, 5]
#     dx = dx.reshape(-1)  # [25]
#     dy = dy.reshape(-1)  # [25]
#
#     # Step 3: add the offsets and clamp at the borders -> [B, 16, 25]
#     x_neigh = x_idx.unsqueeze(-1) + dx.view(1, 1, -1)
#     y_neigh = y_idx.unsqueeze(-1) + dy.view(1, 1, -1)
#     x_neigh = x_neigh.clamp(0, 13)
#     y_neigh = y_neigh.clamp(0, 13)
#
#     # Step 4: local patch index x * 14 + y -> [B, 16, 25]
#     local_idx = x_neigh * 14 + y_neigh
#
#     # Step 5: global index (frame_id * 196 + local_idx) -> [B, 16, 25]
#     frame_ids = torch.arange(16, device=video_tokens.device).view(1, 16, 1)  # [1, 16, 1]
#     global_idx = frame_ids * 196 + local_idx  # [B, 16, 25]
#
#     # Step 6: gather and flatten to [B, 400, D]
#     flat_idx = global_idx.view(B, -1)                     # [B, 16*25=400]
#     gather_idx = flat_idx.unsqueeze(-1).expand(-1, -1, D) # [B, 400, D]
#     selected_tokens = torch.gather(video_tokens, dim=1, index=gather_idx)
#
#     return selected_tokens  # shape [B, 400, D]
