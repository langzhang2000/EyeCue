import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TimesformerConfig, TimesformerModel


class VideoEncoder(nn.Module):
    def __init__(self,
                 pretrained_name: str = "facebook/timesformer-base-finetuned-k600",
                 num_frames: int = 16):
        """
        TimeSformer video encoder.

        The official checkpoint is trained with 8 frames. To support an arbitrary
        clip length we build a fresh model with `num_frames` temporal positions and
        linearly interpolate the pretrained `time_embeddings` onto the new grid;
        every other weight is loaded unchanged.
        """
        super().__init__()
        self.num_frames = num_frames

        # ---- 1. Build the config for the target clip length ----
        config = TimesformerConfig.from_pretrained(pretrained_name)
        config.num_frames = num_frames
        config.output_attentions = True
        self.backbone = TimesformerModel(config)

        # ---- 2. Load the official pretrained model (8 frames) ----
        pretrained = TimesformerModel.from_pretrained(pretrained_name)

        # ---- 3. Interpolate the temporal embeddings to `num_frames` ----
        pt_te = pretrained.embeddings.time_embeddings.data      # [1, old_T, D]
        te = pt_te.permute(0, 2, 1)                             # [1, D, old_T]
        te = F.interpolate(te, size=num_frames,
                           mode="linear", align_corners=False)
        new_te = te.permute(0, 2, 1).contiguous()               # [1, new_T, D]

        # ---- 4. Load every weight except the temporal embeddings ----
        state_dict = pretrained.state_dict()
        state_dict.pop("embeddings.time_embeddings")
        self.backbone.load_state_dict(state_dict, strict=False)

        # ---- 5. Copy in the interpolated temporal embeddings ----
        self.backbone.embeddings.time_embeddings.data.copy_(new_te)

    def forward(self, frames: torch.FloatTensor):
        """
        Args:
            frames: FloatTensor [B, T, C, H, W]
        Returns:
            cls_token      : [B, 1, D]     global clip representation
            video_tokens   : [B, L-1, D]   per-frame patch tokens (T * 196)
            cls_to_patches : [B, L-1]      CLS attention over patches (last layer,
                                           averaged over heads)
        """
        outputs = self.backbone(frames, output_attentions=True)

        # ---- hidden states ----
        last_hidden  = outputs.last_hidden_state    # [B, L, D]
        cls_token    = last_hidden[:, :1, :]        # [B, 1, D]
        video_tokens = last_hidden[:, 1:, :]        # [B, L-1, D]

        # ---- attention ----
        last_attn = outputs.attentions[-1]          # [B, H, L, L]
        mean_attn = last_attn.mean(dim=1)           # [B, L, L]
        cls_all   = mean_attn[:, 0, :]              # [B, L]
        cls_to_patches = cls_all[:, 1:]             # [B, L-1]

        return cls_token, video_tokens, cls_to_patches
