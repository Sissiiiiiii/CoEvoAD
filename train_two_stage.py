import os
import json
import argparse
import logging
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from datasets import Makedataset
from loss import FocalLoss, DiceLoss, Orthogonal_Loss
from models.VPB import Context_Prompting, TextEncoder
from models.evaluate import evaluate_pre
from models.EMA import BestCheckpoint
from models.model_CLIP import Load_CLIP, tokenize


def _convert_image_to_rgb(image):
    return image.convert("RGB")


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _transform_test(n_px):
    return Compose(
        [
            Resize((n_px, n_px), interpolation=BICUBIC),
            CenterCrop((n_px, n_px)),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _checkpoint_state_dict(checkpoint):
    return checkpoint["MyModel"] if isinstance(checkpoint, dict) and "MyModel" in checkpoint else checkpoint


def _resolve_validation_plan(dataset: str, source_only_validation: bool = False):
    mvtec_fixed = ["bottle", "hazelnut", "wood", "zipper", "leather"]
    visa_fixed = ["chewinggum", "cashew", "pipe_fryum", "capsules", "candle"]
    if source_only_validation:
        if dataset == "mvtec":
            return "mvtec", mvtec_fixed
        if dataset == "visa":
            return "visa", visa_fixed
        return dataset, None
    if dataset == "mvtec":
        return "visa", visa_fixed
    if dataset == "visa":
        return "mvtec", mvtec_fixed
    return "mvtec", mvtec_fixed


def _maybe_enable_per_slot_mapping_from_checkpoint(args, checkpoint_path: str = "", logger=None) -> bool:
    path = checkpoint_path or getattr(args, "checkpoint_path", "")
    if not path or not os.path.isfile(path):
        return False

    ckpt = torch.load(path, map_location="cpu")
    state_dict = _checkpoint_state_dict(ckpt)
    has_slot_mappings = any(k.startswith("slot_mappings.") for k in state_dict.keys())
    if has_slot_mappings and not getattr(args, "per_slot_mapping", False):
        args.per_slot_mapping = True
        msg = "checkpoint has slot_mappings; auto-enabling per_slot_mapping before model construction"
        if logger is not None:
            logger.info(msg)
        else:
            print(f"[INFO] {msg}")
        return True
    return False


def setup_logger(save_path, log_filename="result_two_stage.txt"):
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, log_filename)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("train_two_stage")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="a+")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class TwoStageTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

        with open(args.config_path, "r") as f:
            model_configs = json.load(f)
        args.vision_width = model_configs["vision_cfg"]["width"]
        args.text_width = model_configs["text_cfg"]["width"]
        args.embed_dim = model_configs["embed_dim"]

        self.logger = setup_logger(args.save_path)
        self._prompt_counter = 0  # round-robin prompt slot counter
        for arg in vars(args):
            self.logger.info(f"{arg}: {getattr(args, arg)}")

        self.model_clip, _, _ = Load_CLIP(args.image_size, args.pretrained_path, device=self.device)
        self.model_clip.to(self.device)
        self.model_clip.eval()
        self.tokenizer = tokenize
        self.text_encoder = TextEncoder(self.model_clip, args)
        self.model = Context_Prompting(args=args).to(self.device)

        preprocess = _transform_test(args.image_size)
        self.make_dataset = Makedataset(
            train_data_path=args.train_data_path,
            preprocess_test=preprocess,
            mode="train",           # 保留 train-time augmentation
            split_override="test",  # 加载 test split（有异常标签，source-supervised）
            image_size=args.image_size,
            mask_guided_crop_prob=args.mask_guided_crop_prob,
        )
        self.make_dataset_val = Makedataset(
            train_data_path=args.val_data_path,
            preprocess_test=preprocess,
            mode="test",
            image_size=args.image_size,
            mask_guided_crop_prob=0.0,
        )

        self.loss_dice = DiceLoss()
        self.loss_focal = FocalLoss()
        self.loss_ort = Orthogonal_Loss()
        self.loss_cross = nn.CrossEntropyLoss()

    def _get_parameter_groups(self):
        prompt_params = []
        linear_params = []
        class_mapping_params = []

        for name, param in self.model.named_parameters():
            if "prompt" in name:
                prompt_params.append(param)
            elif "class_mapping" in name or "slot_mappings" in name or "image_mapping" in name or "fuse" in name:
                class_mapping_params.append(param)
                linear_params.append(param)
            elif "temperature" not in name:
                linear_params.append(param)

        return {
            "prompt": prompt_params,
            "linear": linear_params,
            "class_mapping": class_mapping_params,
            "temperature": [self.model.temperature_pixel, self.model.temperature_image],
        }

    def stage1_train(self, resume_checkpoint: str = ""):
        if self.args.eval_every < 1:
            raise ValueError("--eval_every must be >= 1")

        self.logger.info("=" * 60)
        if resume_checkpoint:
            self.logger.info("Resume Stage 2: loading %s", resume_checkpoint)
        else:
            self.logger.info("Stage 1: deterministic prompt-bank training")
        self.logger.info("=" * 60)
        self.logger.info(
            "Full pixel validation runs every %d epoch(s) and on the final epoch",
            self.args.eval_every,
        )

        param_groups = self._get_parameter_groups()
        optimizer_stage1 = torch.optim.Adam(
            [
                {"params": param_groups["prompt"], "lr": self.args.learning_rate_prompt},
                {"params": param_groups["temperature"], "lr": self.args.learning_rate_temperature},
                {"params": param_groups["linear"], "lr": self.args.learning_rate_linear},
            ],
            lr=self.args.learning_rate,
            betas=(0.5, 0.999),
            weight_decay=self.args.weight_decay,
        )
        optimizer_stage2 = torch.optim.Adam(
            [{"params": param_groups["class_mapping"], "lr": self.args.learning_rate_linear}],
            lr=self.args.learning_rate,
            betas=(0.5, 0.999),
        )

        train_dataloader, _ = self.make_dataset.mask_dataset(
            name=self.args.dataset,
            batchsize=self.args.batch_size,
            product_list=None,
            shuf=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )
        val_dataset_name, val_product_list = _resolve_validation_plan(
            self.args.dataset,
            source_only_validation=bool(getattr(self.args, "source_only_validation", False)),
        )
        self.logger.info(
            "Validation plan: policy=%s dataset=%s product_list=%s",
            "source_only" if getattr(self.args, "source_only_validation", False) else "legacy_cross_domain",
            val_dataset_name,
            val_product_list,
        )
        val_dataloader, val_obj_list = self.make_dataset_val.mask_dataset(
            name=val_dataset_name,
            product_list=val_product_list,
            batchsize=1,
            shuf=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )

        scheduler_stage1 = get_cosine_schedule_with_warmup(
            optimizer_stage1,
            num_warmup_steps=int(0.1 * len(train_dataloader) * self.args.epochs),
            num_training_steps=len(train_dataloader) * self.args.epochs,
        )

        ema = BestCheckpoint(self.model)
        ap_max_pixel = 0.0
        ap_max_image = 0.0
        early_stop_patience = self.args.early_stop_patience
        patience = 0

        if resume_checkpoint:
            ckpt = torch.load(resume_checkpoint, map_location=self.device)
            state_dict = _checkpoint_state_dict(ckpt)
            ckpt_has_slot = any(k.startswith("slot_mappings.") for k in state_dict.keys())
            ckpt_has_cm = any(k.startswith("class_mapping.") for k in state_dict.keys())
            if ckpt_has_slot and not self.model.per_slot_mapping:
                raise ValueError(
                    "resume checkpoint contains slot_mappings.* but model was constructed "
                    "without per_slot_mapping; enable it before constructing TwoStageTrainer"
                )
            if self.model.per_slot_mapping and not ckpt_has_slot and ckpt_has_cm:
                filtered = {k: v for k, v in state_dict.items() if not k.startswith("class_mapping.")}
                self.model.load_state_dict(filtered, strict=False)
                cm_w = state_dict["class_mapping.weight"]
                cm_b = state_dict.get("class_mapping.bias")
                for sl in self.model.slot_mappings:
                    sl.weight.data.copy_(cm_w)
                    if cm_b is not None and sl.bias is not None:
                        sl.bias.data.copy_(cm_b)
                self.logger.info("per_slot_mapping: warm-started slot_mappings from shared class_mapping")
            else:
                self.model.load_state_dict(state_dict, strict=True)
            self.logger.info("Checkpoint loaded, running initial eval before Stage 2 training")
            stage = 2
            optimizer = optimizer_stage2
            scheduler = None
            # 先做一次 eval，用初始 checkpoint 的 image AP 作为 best-checkpoint 基准
            _, init_ap_image = evaluate_pre(
                val_dataloader, self.model_clip, self.model,
                self.device, self.args, val_obj_list,
                self.text_encoder, self.tokenizer, stage,
            )
            ap_max_image = init_ap_image if init_ap_image is not None else 0.0
            self.logger.info("Initial image AP baseline: %.4f", ap_max_image)
            ema.save_check()
        else:
            stage = 1
            optimizer = optimizer_stage1
            scheduler = scheduler_stage1

        for epoch in range(self.args.epochs):
            loss_lists = {"class": [], "text": [], "seg": []}
            self.model.train()

            for items in tqdm(train_dataloader):
                optimizer.zero_grad(set_to_none=True)
                image = items["img"].to(self.device)
                cls_name = items["cls_name"]
                anomaly = items["anomaly"].to(self.device)
                # Keep the batch dimension even when the last batch size is 1.
                gt = items["img_mask"].squeeze(1).to(self.device)
                gt[gt > 0.5], gt[gt < 0.5] = 1, 0

                with torch.no_grad():
                    image_features, _, patch_tokens = self.model_clip.encode_image(image, self.args.features_list)

                text_embeddings = self.model.forward_ensemble(
                    self.text_encoder,
                    cls_name,
                    self.device,
                    self.tokenizer,
                    mode="train",
                )
                pro_img, anomaly_maps_list = self.model(
                    text_embeddings,
                    image_features,
                    patch_tokens,
                    stage=stage,
                    mode="train",
                )
                pro_img = pro_img.squeeze(2)

                pn = self.args.prompt_num
                ip = self._prompt_counter % pn
                self._prompt_counter += 1
                pair_logits = torch.cat(
                    [pro_img[:, ip:ip + 1], pro_img[:, ip + pn:ip + pn + 1]],
                    dim=1,
                )
                loss_class = self.loss_cross(pair_logits, anomaly.long().view(-1))

                loss_foc = 0
                loss_dic = 0
                for anomaly_map in anomaly_maps_list:
                    anomaly_map_select = anomaly_map[:, [ip, ip + pn], :, :]
                    anomaly_map_select = torch.softmax(anomaly_map_select, dim=1)
                    loss_foc += self.loss_focal(anomaly_map_select, gt)
                    loss_dic += self.loss_dice(anomaly_map_select, gt)
                loss_seg = loss_foc + loss_dic
                if self.args.postmap_div:
                    if self.model.per_slot_mapping:
                        pn = self.args.prompt_num
                        mapped_n = [self.model.slot_mappings[i](text_embeddings[:, i:i+1, :]) for i in range(pn)]
                        mapped_a = [self.model.slot_mappings[i](text_embeddings[:, i+pn:i+pn+1, :]) for i in range(pn)]
                        div_input = torch.cat(mapped_n + mapped_a, dim=1)
                    else:
                        div_input = self.model.class_mapping(text_embeddings)
                    div_input = div_input / (div_input.norm(dim=-1, keepdim=True) + 1e-12)
                else:
                    div_input = text_embeddings
                loss_text = self.loss_ort(div_input, self.args)

                if stage == 1:
                    loss = loss_class + loss_seg + self.args.div_loss_weight * loss_text
                else:
                    loss = loss_class
                    if self.args.stage2_div_loss_weight > 0:
                        loss = loss + self.args.stage2_div_loss_weight * loss_text

                # 改动 4: Abnormal-side category-agnostic regularization
                # 以低概率用 "X object" 替代 "X {category}"，只对 abnormal 侧
                # 强迫 visual_abnormal tokens 不过度依赖类别名
                if stage == 1 and self.args.agnostic_reg_prob > 0 and random.random() < self.args.agnostic_reg_prob:
                    agnostic_names = ["X object"] * len(cls_name)
                    agnostic_tokens = self.tokenizer(agnostic_names)
                    if hasattr(agnostic_tokens, "to"):
                        agnostic_tokens = agnostic_tokens.to(self.device)
                    else:
                        agnostic_tokens = torch.as_tensor(agnostic_tokens, device=self.device)
                    visual_abnormal = torch.cat([self.model.prompt_context, self.model.prompt_state_abnormal], dim=1)
                    agnostic_abnormal_emb = self.text_encoder(agnostic_tokens, visual_abnormal)
                    agnostic_abnormal_emb = agnostic_abnormal_emb / (agnostic_abnormal_emb.norm(dim=-1, keepdim=True) + 1e-12)
                    # 用 agnostic abnormal + 原 normal 计算辅助 classification loss
                    normal_emb = text_embeddings[:, :pn, :].detach()  # 断梯度，不影响 normal 侧
                    agnostic_text = torch.cat([normal_emb, agnostic_abnormal_emb], dim=1)
                    agnostic_pro_img = self.model(
                        agnostic_text, image_features, patch_tokens, stage=stage, mode="train",
                    )[0].squeeze(2)
                    agnostic_pair = torch.cat(
                        [agnostic_pro_img[:, ip:ip+1], agnostic_pro_img[:, ip+pn:ip+pn+1]], dim=1,
                    )
                    loss_agnostic = self.args.agnostic_reg_weight * self.loss_cross(
                        agnostic_pair, anomaly.long().view(-1)
                    )
                    loss = loss + loss_agnostic

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                loss_lists["class"].append(loss_class.item())
                loss_lists["text"].append(loss_text.item())
                loss_lists["seg"].append(loss_seg.item())
                torch.cuda.empty_cache()

            run_full_eval = ((epoch + 1) % self.args.eval_every == 0) or ((epoch + 1) == self.args.epochs)
            ap_raw_pixel = None
            ap_raw_image = None
            if run_full_eval:
                ap_raw_pixel, ap_raw_image = evaluate_pre(
                    val_dataloader,
                    self.model_clip,
                    self.model,
                    self.device,
                    self.args,
                    val_obj_list,
                    self.text_encoder,
                    self.tokenizer,
                    stage,
                )
                self.logger.info(
                    "Epoch [%d/%d], loss_seg: %.4f, loss_cls: %.4f, loss_text: %.4f, ap_pixel: %.4f, ap_image: %.4f",
                    epoch + 1,
                    self.args.epochs,
                    np.mean(loss_lists["seg"]),
                    np.mean(loss_lists["class"]),
                    np.mean(loss_lists["text"]),
                    ap_raw_pixel,
                    ap_raw_image,
                )
            else:
                self.logger.info(
                    "Epoch [%d/%d], loss_seg: %.4f, loss_cls: %.4f, loss_text: %.4f, full_eval: skipped (eval_every=%d)",
                    epoch + 1,
                    self.args.epochs,
                    np.mean(loss_lists["seg"]),
                    np.mean(loss_lists["class"]),
                    np.mean(loss_lists["text"]),
                    self.args.eval_every,
                )

            if (epoch + 1) % self.args.save_freq == 0:
                ckp_path = os.path.join(self.args.save_path, f"epoch_post_{epoch + 1}_{stage}.pth")
                torch.save({"MyModel": self.model.state_dict()}, ckp_path)

            if run_full_eval:
                if stage == 1:
                    if ap_raw_pixel >= ap_max_pixel:
                        patience = 0
                        ap_max_pixel = ap_raw_pixel
                        ema.save_check()
                    else:
                        patience += 1
                        if patience == early_stop_patience:
                            ema.load_check()
                            stage = 2
                            optimizer = optimizer_stage2
                            scheduler = None
                            patience = 0
                            self.logger.info(
                                "Stage 1 → Stage 2 transition (pixel AP peaked at %.4f, switching to classification head fine-tuning)",
                                ap_max_pixel,
                            )
                else:
                    if ap_raw_image >= ap_max_image:
                        patience = 0
                        ap_max_image = ap_raw_image
                        ema.save_check()
                    else:
                        patience += 1
                        if patience == early_stop_patience:
                            ema.load_check()
                            self.logger.info("Stage 2 early stop (image AP stalled)")
                            break

        # 恢复最佳权重后再保存（防止循环自然结束时存到末轮权重）
        ema.load_check()
        ckpt_name = "stage1_final.pth" if stage == 1 else "two_stage_final.pth"
        final_ckp_path = os.path.join(self.args.save_path, ckpt_name)
        torch.save({"MyModel": self.model.state_dict()}, final_ckp_path)
        self.logger.info("Training finished (best checkpoint restored): %s", final_ckp_path)
        return final_ckp_path

    def stage2_optimize(self, checkpoint_path: str = ""):
        from optimize_universal import run_stage2_universal

        if checkpoint_path and not getattr(self.args, "checkpoint_path", ""):
            self.args.checkpoint_path = checkpoint_path
        # 兼容两种 checkpoint 命名：stage1_final.pth / two_stage_final.pth
        ckp = getattr(self.args, "checkpoint_path", "")
        if ckp and not os.path.isfile(ckp):
            alt_names = ["stage1_final.pth", "two_stage_final.pth"]
            parent = os.path.dirname(ckp)
            for alt in alt_names:
                alt_path = os.path.join(parent, alt)
                if os.path.isfile(alt_path):
                    self.logger.info("Checkpoint '%s' not found, using '%s'", ckp, alt_path)
                    self.args.checkpoint_path = alt_path
                    break
        if not getattr(self.args, "checkpoint_path", ""):
            raise ValueError("stage2 requires --checkpoint_path or a stage1 result")
        self.args.scorer_type = getattr(self.args, "scorer_type", "prompt_bank")
        self.logger.info("Stage2 delegated to optimize_universal.py with scorer=%s", self.args.scorer_type)
        return run_stage2_universal(self.args)


def main(args):
    _maybe_enable_per_slot_mapping_from_checkpoint(args)
    trainer = TwoStageTrainer(args)
    checkpoint_path = args.checkpoint_path

    if args.stage2_only:
        trainer.stage2_optimize(checkpoint_path=checkpoint_path)
        return

    if getattr(args, "resume_stage2", False):
        checkpoint_path = trainer.stage1_train(resume_checkpoint=checkpoint_path)
        if args.stage1_only:
            return
        trainer.stage2_optimize(checkpoint_path=checkpoint_path)
        return

    if not args.stage2_only:
        checkpoint_path = trainer.stage1_train()
        if args.stage1_only:
            return
        trainer.stage2_optimize(checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PromptBank two-stage training", add_help=True)

    parser.add_argument("--dataset", type=str, default="visa")
    parser.add_argument("--model", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])

    parser.add_argument("--train_data_path", type=str, default="./dataset/mvisa/data")
    parser.add_argument("--val_data_path", type=str, default="./dataset/mvisa/data")
    parser.add_argument("--save_path", type=str, default="./my_exps/train_two_stage_visa")
    parser.add_argument("--config_path", type=str, default="./open_clip_local/model_configs/ViT-L-14-336.json")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_weight/ViT-L-14-336px.pt")
    parser.add_argument("--checkpoint_path", type=str, default="")

    parser.add_argument("--prompt_context_len", type=int, default=5)
    parser.add_argument("--prompt_num", type=int, default=3)
    parser.add_argument("--prompt_state_len", type=int, default=5)
    parser.add_argument("--prompt_bank_no_class_anchor", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--learning_rate_prompt", type=float, default=0.001)
    parser.add_argument("--learning_rate_linear", type=float, default=0.0001)
    parser.add_argument("--learning_rate_temperature", type=float, default=0.01,
                        help="Temperature parameters learning rate (default: 0.01)")
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Epochs to wait before early stopping (default: 3)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for optimizer (default: 0.0, recommended 1e-4 for retraining)")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval_every", type=int, default=1, help="Run full validation every N epochs and on the final epoch")
    parser.add_argument("--mask_guided_crop_prob", type=float, default=0.5,
                        help="Probability of applying mask-guided crop on anomalous training samples")
    parser.add_argument("--agnostic_reg_prob", type=float, default=0.2,
                        help="Probability of applying abnormal-side agnostic regularization in Stage 1")
    parser.add_argument("--agnostic_reg_weight", type=float, default=0.1,
                        help="Loss weight for abnormal-side agnostic regularization in Stage 1")
    parser.add_argument("--inter_role_margin", type=float, default=0.3,
                        help="Target cosine-similarity upper bound between normal and abnormal prompt means")
    parser.add_argument("--inter_role_margin_weight", type=float, default=0.5,
                        help="Loss weight for inter-role margin regularization")
    parser.add_argument("--postmap_div", action="store_true",
                        help="Measure Orthogonal_Loss in post-class_mapping space (default: pre-mapping)")
    parser.add_argument("--per_slot_mapping", action="store_true",
                        help="Use per-slot class_mapping instead of shared (breaks compression bottleneck)")
    parser.add_argument("--div_loss_weight", type=float, default=1.0,
                        help="Weight for diversity loss in Stage1")
    parser.add_argument("--stage2_div_loss_weight", type=float, default=0.0,
                        help="Weight for diversity loss in internal Stage2 (0=disabled)")
    parser.add_argument("--alpha", type=float, default=0.999,
                        help="[DEPRECATED] EMA decay - no longer used; kept for backward compatibility")
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=True,
                        help="Enable DataLoader pin_memory for faster host-to-device transfer")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false",
                        help="Disable DataLoader pin_memory to reduce host memory pressure")

    parser.add_argument("--stage1_only", action="store_true")
    parser.add_argument("--stage2_only", action="store_true")
    parser.add_argument("--resume_stage2", action="store_true",
                        help="Load checkpoint and run only internal Stage 2 (fuse/image_mapping/class_mapping fine-tuning)")
    parser.add_argument("--source_only_validation", action="store_true",
                        help="Use fixed validation categories from the source dataset instead of the legacy cross-domain validation set")
    parser.add_argument("--stage2_split", type=str, default="test",
                        help="Data split for Stage2 prompt search (train/val/test)")
    parser.add_argument("--allow_split_fallback", action="store_true",
                        help="Allow fallback to 'test' split when requested split not in meta JSON")

    parser.add_argument("--evo_population", type=int, default=16)
    parser.add_argument("--evo_generations", type=int, default=5)
    parser.add_argument("--evo_topk", type=int, default=4)
    parser.add_argument("--evo_lambda_diversity", type=float, default=0.2)
    parser.add_argument("--evo_dual_branch", action="store_true")
    parser.add_argument("--evo_val_batches", type=int, default=10)
    parser.add_argument("--candidate_batch_size", type=int, default=2)
    parser.add_argument("--zero_shot_scoring", dest="zero_shot_scoring", action="store_true",
                        help="Compatibility mode for legacy PFL-style text encoders/checkpoints: replace latent prompt bias with zeros")
    parser.add_argument("--no_zero_shot_scoring", dest="zero_shot_scoring", action="store_false",
                        help="If a legacy PFL-compatible scorer/model is present, use its learned latent prompt bias")
    parser.set_defaults(zero_shot_scoring=False)
    parser.add_argument("--stage2_objective", type=str, default="image_pixel", choices=["image_pixel", "image_only", "image_teacher"])
    parser.add_argument("--stage2_weight_image", type=float, default=0.4)
    parser.add_argument("--stage2_weight_pixel_ap", type=float, default=0.3)
    parser.add_argument("--stage2_weight_pixel_f1", type=float, default=0.3)
    parser.add_argument("--stage2_metric_resolution", type=int, default=256)
    parser.add_argument("--pixel_layer_weights", type=float, nargs=4, default=[1, 1, 1, 1])
    parser.add_argument("--stage2_rerank_topk", type=int, default=8)
    parser.add_argument("--stage2_stability_bootstrap", type=int, default=3)
    parser.add_argument("--stage2_stability_weight", type=float, default=0.1)
    parser.add_argument("--use_coevo_prompt", action="store_true")
    parser.add_argument("--coevo_pair_k", type=int, default=3)
    parser.add_argument("--coevo_alpha_auroc", type=float, default=0.85)
    parser.add_argument("--coevo_beta_contrast", type=float, default=0.15)
    parser.add_argument("--shared_bank_eval_batches", type=int, default=2)
    parser.add_argument("--shared_bank_max_candidates_per_role", type=int, default=48)
    parser.add_argument("--shared_bank_size", type=int, default=4)
    parser.add_argument("--enable_shared_bank", action="store_true")
    parser.add_argument("--shared_bank_path", type=str, default="")
    parser.add_argument("--shared_topk", type=int, default=2)
    parser.add_argument("--shared_route_temp", type=float, default=0.07)
    parser.add_argument("--shared_alpha_exact", type=float, default=0.0)
    parser.add_argument("--shared_alpha_semantic", type=float, default=0.35)
    parser.add_argument("--shared_alpha_missing", type=float, default=1.0)
    parser.add_argument("--global_shared_prompts", action="store_true")
    parser.add_argument("--shared_prompt", type=str, default="X object")
    parser.add_argument("--shared_normal_prompt", type=str, default="X normal object")
    parser.add_argument("--shared_abnormal_prompt", type=str, default="X anomalous object")
    parser.add_argument("--enable_role_fallback", action="store_true")
    parser.add_argument("--disable_role_fallback", action="store_true")
    parser.add_argument("--enable_template_transfer", action="store_true")
    parser.add_argument("--enable_semantic_fallback", action="store_true")
    parser.add_argument("--semantic_fallback_template", type=str, default="a photo of {}")
    parser.add_argument("--semantic_fallback_min_sim", type=float, default=0.5)
    parser.add_argument("--semantic_fallback_min_margin", type=float, default=0.0)

    parser.add_argument("--scorer_type", type=str, default="prompt_bank", choices=["prompt_bank", "clip_transfer", "custom"])
    parser.add_argument("--scorer_module", type=str, default="")
    parser.add_argument("--scorer_class", type=str, default="")
    parser.add_argument("--scorer_config", type=str, default="")
    parser.add_argument("--scorer_kwargs_json", type=str, default="")

    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
    setup_seed(args.seed)
    main(args)
