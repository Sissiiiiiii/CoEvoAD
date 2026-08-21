import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


CLASS_NAME_MAP = {
    "macaroni1": "macaroni",
    "macaroni2": "macaroni",
    "pcb1": "printed circuit board",
    "pcb2": "printed circuit board",
    "pcb3": "printed circuit board",
    "pcb4": "printed circuit board",
    "pipe_fryum": "pipe fryum",
    "chewinggum": "chewing gum",
    "metal_nut": "metal nut",
}


class Fuse_Block(nn.Module):
    def __init__(self, dim_i, dim_hid, dim_out):
        super().__init__()
        self.pre_process = nn.Sequential(
            nn.Linear(dim_i, dim_hid),
            nn.ReLU(),
            nn.Linear(dim_hid, dim_hid),
        )
        self.post_process = nn.Linear(dim_hid, dim_out)

    def forward(self, x):
        x = self.pre_process(x)
        x = torch.mean(x, dim=1)
        return self.post_process(x)


class Zero_Parameter(nn.Module):
    def __init__(self, dim_v, dim_t, dim_out, num_heads=4, k=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dim_out = dim_out
        self.scale = self.head_dim ** -0.5
        self.q_proj_pre = nn.Conv1d(dim_t, dim_out, kernel_size=1)
        self.linear_proj = nn.ModuleList(
            [nn.Linear(dim_v, dim_t, bias=False) for _ in range(k)]
        )

    def forward(self, F_t, F_s, layer):
        B1, N1, _ = F_t.shape
        B2, N2, _ = F_s.shape
        if B1 != B2:
            raise ValueError(f"Batch mismatch in RCA: {B1} vs {B2}")

        F_s = self.linear_proj[layer](F_s.to(F_t.device, dtype=F_t.dtype))
        F_s = F_s / (F_s.norm(dim=-1, keepdim=True) + 1e-12)

        q_t = self.q_proj_pre(F_t.permute(0, 2, 1)).permute(0, 2, 1)
        q_t = q_t.reshape(B1, N1, self.num_heads, self.head_dim)
        k_s = F_s.reshape(B2, N2, self.num_heads, self.head_dim)
        v_s = F_s.reshape(B2, N2, self.num_heads, self.head_dim)

        attn_t = torch.einsum("bnkc,bmkc->bknm", q_t, k_s) * self.scale
        attn_t = attn_t.softmax(dim=-1)
        F_t_a = torch.einsum("bknm,bmkc->bnkc", attn_t, v_s).reshape(B1, N1, self.dim_out)
        F_t_a = F_t_a + F_t
        F_t_a = F_t_a / (F_t_a.norm(dim=-1, keepdim=True) + 1e-12)
        return F_t_a, F_s


class Global_Feature(nn.Module):
    def __init__(self, dim_i, dim_hid, dim_out, k):
        super().__init__()
        self.fuse_modules = nn.Linear(dim_i * k, dim_hid)
        self.compress = nn.Linear(dim_hid, 1)
        self.post_process = nn.Linear(dim_hid, dim_out)

    def forward(self, inps):
        x = torch.cat(inps, dim=2)
        x = self.fuse_modules(x)
        attention_weights = nn.Softmax(dim=1)(self.compress(x))
        x = torch.sum(attention_weights * x, dim=1)
        return self.post_process(x)


class TextEncoder(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        self.clip_model = clip_model
        self.num_tokens = args.prompt_context_len
        self.context_length = clip_model.context_length
        self.eot_id = 49407

    @property
    def dtype(self):
        return self.clip_model.visual.conv1.weight.dtype

    def _unwrap_tokens(self, text):
        if isinstance(text, dict):
            if "input_ids" not in text:
                raise ValueError("Tokenizer output must contain input_ids")
            return text["input_ids"]
        return text

    def forward(self, text, visual_feature):
        text = self._unwrap_tokens(text)
        pos_y = [1] * text.shape[0]
        x = self.clip_model.token_embedding(text).type(self.dtype)
        out_dim = self.clip_model.text_projection.shape[1]
        x_new_array = torch.zeros(
            (x.shape[0], visual_feature.shape[0], out_dim),
            dtype=x.dtype,
            device=x.device,
        )

        for i in range(text.shape[0]):
            x_temp = x[i, :, :] + torch.zeros(
                (visual_feature.shape[0], x.shape[1], x.shape[2]),
                dtype=x.dtype,
                device=x.device,
            )
            x_new = torch.cat(
                [
                    x_temp[:, 0:pos_y[i], :],
                    visual_feature,
                    x_temp[:, (pos_y[i] + 1):(self.context_length - visual_feature.shape[1] + 1), :],
                ],
                dim=1,
            )
            x_new = x_new + self.clip_model.positional_embedding.type(self.dtype)
            x_new = x_new.permute(1, 0, 2)
            out = self.clip_model.transformer(x_new)
            x_new = out[0] if isinstance(out, (list, tuple)) else out
            x_new = x_new.permute(1, 0, 2)
            x_new = self.clip_model.ln_final(x_new).type(self.dtype)

            eot_positions = torch.where(text[i] == self.eot_id)[0]
            if eot_positions.numel() == 0:
                eot_pos = torch.where(text[i] != 0)[0][-1]
            else:
                eot_pos = eot_positions[-1]

            x_new = x_new[:, eot_pos + visual_feature.shape[1] - 1, :] @ self.clip_model.text_projection
            x_new_array[i, :, :] = x_new.reshape(-1, x_new.shape[-1])

        return x_new_array


class Context_Prompting(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vision_width = args.vision_width
        self.text_width = args.text_width
        self.embed_dim = args.embed_dim

        self.prompt_context = nn.Parameter(
            torch.randn(args.prompt_num, args.prompt_context_len, self.text_width)
        )
        self.prompt_state_normal = nn.Parameter(
            torch.randn(args.prompt_num, args.prompt_state_len, self.text_width)
        )
        self.prompt_state_abnormal = nn.Parameter(
            torch.randn(args.prompt_num, args.prompt_state_len, self.text_width)
        )

        self.temperature_pixel = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.temperature_image = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.fuse = Global_Feature(
            self.vision_width,
            self.vision_width // 2,
            self.text_width,
            k=len(self.args.features_list),
        )
        self.RCA = Zero_Parameter(
            dim_v=self.vision_width,
            dim_t=self.text_width,
            dim_out=self.text_width,
            k=len(args.features_list),
        )
        self.per_slot_mapping = getattr(args, "per_slot_mapping", False)
        if self.per_slot_mapping:
            self.slot_mappings = nn.ModuleList(
                [nn.Linear(self.text_width, self.text_width) for _ in range(args.prompt_num)]
            )
        else:
            self.class_mapping = nn.Linear(self.text_width, self.text_width)
        self.image_mapping = nn.Linear(self.text_width, self.text_width)

        self.use_scoring_head = getattr(args, "use_scoring_head", False)
        if self.use_scoring_head:
            self.scoring_head = nn.Sequential(
                nn.Linear(self.text_width * 3, self.text_width),
                nn.ReLU(),
                nn.Linear(self.text_width, 1),
            )
        self._initialize_weights()

        nn.init.trunc_normal_(self.prompt_context, mean=0, std=0.02)
        nn.init.trunc_normal_(self.prompt_state_normal, mean=0.5, std=0.02)
        nn.init.trunc_normal_(self.prompt_state_abnormal, mean=-0.5, std=0.02)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def prompt_names(self, names):
        no_class_anchor = bool(getattr(self.args, "prompt_bank_no_class_anchor", False))
        sentence = []
        for name in names:
            if no_class_anchor:
                sentence.append("X")
                continue
            name_new = CLASS_NAME_MAP.get(name, name)
            sentence.append("X " + str(name_new).lower())
        return sentence

    @staticmethod
    def _normalize_prompt_batch(prompts, batch_size):
        if isinstance(prompts, str):
            return [prompts] * batch_size
        prompts = list(prompts)
        if len(prompts) == 1 and batch_size > 1:
            return prompts * batch_size
        if len(prompts) != batch_size:
            raise ValueError(
                f"Prompt batch size mismatch: expected {batch_size}, got {len(prompts)}"
            )
        return prompts

    def forward_ensemble(
        self,
        model_text_encoder,
        names,
        device,
        tokenizer,
        mode="train",
        prompt_optimizer=None,
        override_prompts=None,
    ):
        del mode
        visual_normal = torch.cat([self.prompt_context, self.prompt_state_normal], dim=1)
        visual_abnormal = torch.cat([self.prompt_context, self.prompt_state_abnormal], dim=1)

        if override_prompts is not None:
            if len(override_prompts) != 2:
                raise ValueError("override_prompts must be (normal_prompts, abnormal_prompts)")
            prompt_names_n = self._normalize_prompt_batch(override_prompts[0], len(names))
            prompt_names_a = self._normalize_prompt_batch(override_prompts[1], len(names))
        else:
            prompt_names_n = self.prompt_names(names)
            prompt_names_a = list(prompt_names_n)
            if prompt_optimizer is not None and not bool(
                getattr(self.args, "prompt_bank_no_class_anchor", False)
            ):
                try:
                    prompt_names_n = prompt_optimizer.optimize(prompt_names_n)
                    prompt_names_a = list(prompt_names_n)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "prompt_optimizer.optimize failed, using default prompts: %s", e
                    )

        prompt_tokens_n = tokenizer(prompt_names_n)
        prompt_tokens_a = tokenizer(prompt_names_a)
        if hasattr(prompt_tokens_n, "to"):
            prompt_tokens_n = prompt_tokens_n.to(device)
            prompt_tokens_a = prompt_tokens_a.to(device)
        elif isinstance(prompt_tokens_n, dict):
            prompt_tokens_n = {k: v.to(device) for k, v in prompt_tokens_n.items()}
            prompt_tokens_a = {k: v.to(device) for k, v in prompt_tokens_a.items()}
        else:
            prompt_tokens_n = torch.as_tensor(prompt_tokens_n, device=device)
            prompt_tokens_a = torch.as_tensor(prompt_tokens_a, device=device)

        normal_embeddings = model_text_encoder(prompt_tokens_n, visual_normal)
        abnormal_embeddings = model_text_encoder(prompt_tokens_a, visual_abnormal)
        text_embeddings = torch.cat([normal_embeddings, abnormal_embeddings], dim=1)
        return text_embeddings / (text_embeddings.norm(dim=-1, keepdim=True) + 1e-12)

    def multi_anchor_prompt_names(self, names):
        sentence = []
        for name in names:
            sentence.append(str(CLASS_NAME_MAP.get(name, name)).lower().strip())
        return sentence

    def forward_multi_anchor_ensemble(self, model_text_encoder, names, device, tokenizer):
        prompt_names = self.multi_anchor_prompt_names(names)
        if len(prompt_names) == 0:
            raise ValueError("forward_multi_anchor_ensemble requires at least one prompt")
        if len(prompt_names) > self.args.prompt_num:
            prompt_names = prompt_names[:self.args.prompt_num]
        base_prompt_names = list(prompt_names)
        while len(prompt_names) < self.args.prompt_num:
            prompt_names.append(base_prompt_names[len(prompt_names) % len(base_prompt_names)])

        normal_embeddings_list = []
        abnormal_embeddings_list = []
        for idx, prompt_name in enumerate(prompt_names):
            prompt_tokens = tokenizer([prompt_name])
            if hasattr(prompt_tokens, "to"):
                prompt_tokens = prompt_tokens.to(device)
            elif isinstance(prompt_tokens, dict):
                prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
            else:
                prompt_tokens = torch.as_tensor(prompt_tokens, device=device)

            visual_normal = torch.cat(
                [self.prompt_context[idx:idx + 1], self.prompt_state_normal[idx:idx + 1]],
                dim=1,
            )
            visual_abnormal = torch.cat(
                [self.prompt_context[idx:idx + 1], self.prompt_state_abnormal[idx:idx + 1]],
                dim=1,
            )
            normal_embedding = model_text_encoder(prompt_tokens, visual_normal)
            abnormal_embedding = model_text_encoder(prompt_tokens, visual_abnormal)
            normal_embeddings_list.append(normal_embedding[:, 0:1, :])
            abnormal_embeddings_list.append(abnormal_embedding[:, 0:1, :])

        normal_embeddings = torch.cat(normal_embeddings_list, dim=1)
        abnormal_embeddings = torch.cat(abnormal_embeddings_list, dim=1)
        text_embeddings = torch.cat([normal_embeddings, abnormal_embeddings], dim=1)
        return text_embeddings / (text_embeddings.norm(dim=-1, keepdim=True) + 1e-12)

    def forward(self, text_features, image_features, patch_tokens, stage, mode):
        del mode
        if self.per_slot_mapping:
            pn = self.args.prompt_num
            mapped_normal = []
            mapped_abnormal = []
            for i in range(pn):
                mapped_normal.append(self.slot_mappings[i](text_features[:, i:i+1, :]))
                mapped_abnormal.append(self.slot_mappings[i](text_features[:, i+pn:i+pn+1, :]))
            text_embeddings_mapping = torch.cat(mapped_normal + mapped_abnormal, dim=1)
        else:
            text_embeddings_mapping = self.class_mapping(text_features)
        text_embeddings_mapping = text_embeddings_mapping / (
            text_embeddings_mapping.norm(dim=-1, keepdim=True) + 1e-12
        )

        if stage == 1:
            image_embeddings_mapping = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)
        else:
            fused = self.fuse(patch_tokens)
            image_embeddings_mapping = self.image_mapping(image_features + fused)
            image_embeddings_mapping = image_embeddings_mapping / (
                image_embeddings_mapping.norm(dim=-1, keepdim=True) + 1e-12
            )

        if self.use_scoring_head:
            image_expanded = image_embeddings_mapping.unsqueeze(1).expand_as(text_embeddings_mapping)
            combined = torch.cat([text_embeddings_mapping, image_expanded,
                                  text_embeddings_mapping * image_expanded], dim=-1)
            pro_img = self.scoring_head(combined)
        else:
            pro_img = self.temperature_image.exp() * text_embeddings_mapping @ image_embeddings_mapping.unsqueeze(2)

        anomaly_maps_list = []
        for layer in range(len(patch_tokens)):
            text_embeddings_update, dense_feature = self.RCA(text_features, patch_tokens[layer].clone(), layer)
            anomaly_map = self.temperature_pixel.exp() * dense_feature @ text_embeddings_update.permute(0, 2, 1)
            B, L, _ = anomaly_map.shape
            H = int(np.sqrt(L))
            anomaly_map = F.interpolate(
                anomaly_map.permute(0, 2, 1).view(B, self.args.prompt_num * 2, H, H),
                size=self.args.image_size,
                mode="bilinear",
                align_corners=True,
            )
            anomaly_maps_list.append(anomaly_map)

        return pro_img, anomaly_maps_list
