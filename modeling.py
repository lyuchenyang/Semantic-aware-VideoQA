from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import CLIPProcessor, CLIPModel

from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


class SaDRPR(nn.Module):
    def __init__(self, config, CLIPConfig):
        super().__init__()
        self.config = config

        self.temporal_position_embeddings = nn.Embedding(config.n_frames, config.transformer_width)

        self.clip = CLIPModel(CLIPConfig)

        attn_dropout = 0.1
        is_add_bias_kv = True
        is_add_zero_attn = True
        self.temporal_self_attention = nn.MultiheadAttention(config.transformer_width, config.attention_heads,
                                                             dropout=attn_dropout,
                                                             add_bias_kv=is_add_bias_kv,
                                                             add_zero_attn=is_add_zero_attn)

        self.query_multi_attention = nn.MultiheadAttention(config.transformer_width, config.attention_heads,
                                                           dropout=attn_dropout,
                                                           add_bias_kv=is_add_bias_kv,
                                                           add_zero_attn=is_add_zero_attn)

        self.retro_transform = nn.Linear(in_features=config.transformer_width,
                                         out_features=config.transformer_width)
        self.forward_transform = nn.Linear(in_features=config.transformer_width,
                                           out_features=config.transformer_width)
        self.fuse_in_reasoning = nn.Linear(in_features=3 * config.transformer_width,
                                           out_features=config.transformer_width)
        self.fuse_all_features = nn.Linear(in_features=3 * config.transformer_width,
                                           out_features=config.transformer_width)
        self.retro_forward_weight = nn.Linear(in_features=3 * config.transformer_width,
                                              out_features=1)
        self.video_to_multimodal = nn.Linear(in_features=config.transformer_width,
                                             out_features=config.transformer_width)
        self.text_to_multimodal = nn.Linear(in_features=config.transformer_width, out_features=config.transformer_width)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.layer_norm = nn.LayerNorm(config.transformer_width)
        self.dropout = nn.Dropout(p=0.5)

        self.sigmoid = nn.Sigmoid()

        self.loss_fct = CrossEntropyLoss()

    def forward(self, inputs):
        """
        :param inputs:
                    image_frames: (B x 10) x N
                    question: B x 77
                    opt1: B x 77
                    opt2: B x 77
                    opt3: B x 77
                    opt4: B x 77
                    ans: B x 1

        :return: loss when training else None
        """
        image_features = self.encode_image(inputs['image_frames'])

        question, options = self.encode_questions_and_options(inputs)

        question_attn_image_features = \
            self.query_multi_attention(question.transpose(0, 1), image_features, image_features)[0].transpose(0, 1)

        video_features = self.dropout(question_attn_image_features)
        image_features = image_features.transpose(0, 1)

        # Initialize the coverage vector
        coverage = torch.zeros_like(question)

        for _ in range(self.config.n_reasoning_steps):
            # Update the coverage vector
            coverage = coverage + self.softmax(
                torch.bmm(video_features + question, image_features.transpose(1, 2).contiguous()) * 5
            )
            normalized_coverage = coverage / coverage.sum(dim=-1, keepdim=True)

            # Concatenate the coverage vector to the current question
            question_with_coverage = torch.cat([question, normalized_coverage], dim=-1)

            current_frame = self.softmax(
                torch.bmm(video_features + question_with_coverage, image_features.transpose(1, 2).contiguous()) * 5)

            pivot_pos = torch.argmax(current_frame, dim=-1)
            retro_mask = []
            for piv_pos in pivot_pos:
                retro_mask.append(torch.tensor([0] * piv_pos + [-1000000] * (image_features.size(1) - piv_pos)))
            retro_mask = torch.stack(retro_mask, dim=0).to(self.config.device).unsqueeze(1)
            forward_mask = (torch.full_like(retro_mask, 1000000, device=self.config.device) + retro_mask) * -1
            current_frame = self.gelu(torch.bmm(current_frame, image_features))
            retrospective_frame = torch.bmm(
                self.softmax(torch.bmm(self.retro_transform(video_features) + question,
                                       image_features.transpose(1, 2).contiguous()) + retro_mask),
                image_features)
            retrospective_frame = self.gelu(retrospective_frame)
            forward_frame = torch.bmm(
                self.softmax(torch.bmm(self.forward_transform(video_features) + question,
                                       image_features.transpose(1, 2).contiguous()) + forward_mask),
                image_features)
            forward_frame = self.gelu(forward_frame)

            rf_weight = self.retro_forward_weight(torch.cat([current_frame, retrospective_frame, forward_frame], dim=-1))
            rf_weight = self.sigmoid(rf_weight)
            reasoning_frame = rf_weight * retrospective_frame + (1 - rf_weight) * forward_frame
            video_features = self.gelu(self.fuse_in_reasoning(
                torch.cat([video_features, current_frame, reasoning_frame], dim=-1)))
            video_features = self.layer_norm(video_features)
            video_features = self.dropout(video_features)
        video_features = self.gelu(
            self.fuse_all_features(torch.cat([question_attn_image_features, 0.1 * video_features, question], dim=-1)))
        n_video_features = torch.nn.functional.normalize(self.video_to_multimodal(video_features), p=2, dim=-1)
        n_option_features = torch.nn.functional.normalize(self.text_to_multimodal(options), p=2, dim=-1)

        logit_scale = self.logit_scale.exp()

        sim_matrix = torch.bmm(logit_scale * n_video_features, n_option_features.transpose(1, 2)).squeeze(1)

        if 'ans' in inputs:

            labels = inputs['ans']

            loss = self.loss_fct(sim_matrix, labels)

            return loss
        else:
            return sim_matrix

    def encode_image(self, images):
        image_features = self.clip.get_image_features(images)

        temporal_pos = torch.tensor(
            [[i for i in range(self.config.n_frames)] for j in range(images.size(0) // self.config.n_frames)],
            dtype=torch.int, device=self.config.device).view(-1)

        frame_temporal_pos_embed = self.temporal_position_embeddings(temporal_pos)

        image_features = (image_features + frame_temporal_pos_embed).view(images.size(0) // self.config.n_frames,
                                                                          self.config.n_frames, -1)
        # image_features = image_features.view(all_frame_features.size(0), -1, self.config.transformer_width)

        image_features = image_features.transpose(0, 1).contiguous()
        self_attn_image_features = self.temporal_self_attention(image_features, image_features, image_features)[0]

        return self_attn_image_features

    def encode_questions_and_options(self, inputs):
        offset = 30
        inputs['question'] = inputs['question'][:, :77]
        inputs['opt1'] = inputs['opt1'][:, :offset]
        inputs['opt2'] = inputs['opt2'][:, :offset]
        inputs['opt3'] = inputs['opt3'][:, :offset]
        inputs['opt4'] = inputs['opt4'][:, :offset]

        attn_mask = 1 - (inputs['question'] == 0).long()
        question = self.clip.get_text_features(inputs['question']).unsqueeze(1)

        attn_mask = 1 - (inputs['opt1'] == 0).long()
        opt1 = self.clip.get_text_features(inputs['opt1']).unsqueeze(1)

        attn_mask = 1 - (inputs['opt2'] == 0).long()
        opt2 = self.clip.get_text_features(inputs['opt2']).unsqueeze(1)

        attn_mask = 1 - (inputs['opt3'] == 0).long()
        opt3 = self.clip.get_text_features(inputs['opt3']).unsqueeze(1)

        attn_mask = 1 - (inputs['opt4'] == 0).long()
        opt4 = self.clip.get_text_features(inputs['opt4']).unsqueeze(1)

        options = torch.cat([opt1, opt2, opt3, opt4], dim=1)

        return question, options


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x