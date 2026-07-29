import random
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image


class VideoGazeDataset(Dataset):
    """
    Loads video frames + gaze points + label.

    Each line of `list_file` has the format: `video_path gaze_path label`
      - video_path: an .mp4 file
      - gaze_path:  a .txt file with n rows and 2 columns, the (x, y) gaze point
                    of each frame
      - label:      an integer

    A random window of `clip_len` consecutive frames is sampled, together with the
    matching gaze points.
    """

    def __init__(self, list_file, clip_len=8, transform=None):
        self.items = []
        with open(list_file, 'r') as f:
            for line in f:
                video_path, gaze_path, label = line.strip().split(' ')
                self.items.append((video_path, gaze_path, int(label)))
        self.clip_len = clip_len
        # Frames are handed to a HuggingFace image processor downstream, so no
        # torchvision transform is applied here.
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        video_path, gaze_path, label = self.items[idx]

        # --- Sample a random window of `clip_len` consecutive frames ---
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames >= self.clip_len:
            start_idx = random.randint(0, total_frames - self.clip_len)
        else:
            start_idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        frames = []
        last_img = None
        for _ in range(self.clip_len):
            ret, img = cap.read()
            if not ret:
                # Ran past the end of the clip: repeat the previous frame
                img = last_img
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                last_img = img

            if img is None:
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                img = np.zeros((h, w, 3), dtype=np.uint8)

            pil_img = Image.fromarray(img)
            frames.append(pil_img)
        cap.release()

        # --- Read the gaze points and slice the matching window ---
        gaze_arr = np.loadtxt(gaze_path, dtype=np.float32)  # [n, 2]
        gaze_tensor = torch.from_numpy(gaze_arr)            # [n, 2]
        end_idx = start_idx + self.clip_len
        if gaze_tensor.size(0) >= end_idx:
            gaze_segment = gaze_tensor[start_idx:end_idx]
        else:
            # Not enough gaze points: pad by repeating the last one
            last = gaze_tensor[-1].unsqueeze(0)
            needed = end_idx - gaze_tensor.size(0)
            pad = last.repeat(needed, 1)
            gaze_segment = torch.cat([gaze_tensor[start_idx:], pad], dim=0)

        # Returns:
        #   - frames:       List[PIL.Image] of length clip_len
        #   - gaze_segment: torch.Tensor([clip_len, 2])
        #   - label:        int
        #   - video_path:   str
        return frames, gaze_segment, label, video_path
