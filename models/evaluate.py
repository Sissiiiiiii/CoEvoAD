import torch
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm


def _prepare_prompt_weights(prompt_num, prompt_weights=None, device=None, dtype=torch.float32):
    if prompt_num <= 0:
        raise ValueError("prompt_num must be positive")
    if prompt_weights is None:
        return torch.full((prompt_num,), 1.0 / float(prompt_num), device=device, dtype=dtype)
    if not torch.is_tensor(prompt_weights):
        weights = torch.tensor(prompt_weights, device=device, dtype=dtype)
    else:
        weights = prompt_weights.to(device=device, dtype=dtype)
    weights = weights.flatten()
    if weights.numel() == 0:
        return torch.full((prompt_num,), 1.0 / float(prompt_num), device=device, dtype=dtype)
    if weights.numel() > prompt_num:
        weights = weights[:prompt_num]
    while weights.numel() < prompt_num:
        weights = torch.cat([weights, weights[-1:].clone()], dim=0)
    weights = torch.clamp(weights, min=0)
    weight_sum = weights.sum()
    if weight_sum.abs() < 1e-8:
        return torch.full((prompt_num,), 1.0 / float(prompt_num), device=device, dtype=dtype)
    return weights / weight_sum


def prompt_weights_from_scores(prompt_scores, prompt_num, tau=1.0, device=None, dtype=torch.float32):
    if prompt_scores is None:
        return None
    if torch.is_tensor(prompt_scores):
        scores = prompt_scores.to(device=device, dtype=dtype).flatten()
    else:
        if len(prompt_scores) == 0:
            return None
        scores = torch.tensor(prompt_scores, device=device, dtype=dtype).flatten()
    if scores.numel() == 0:
        return None
    if scores.numel() > prompt_num:
        scores = scores[:prompt_num]
    while scores.numel() < prompt_num:
        scores = torch.cat([scores, scores[-1:].clone()], dim=0)
    return torch.softmax(scores * float(tau), dim=0)


def prompt_slot_image_scores(pro_img, prompt_num, scorer_temperature=1.0):
    per_prompt = []
    for i in range(prompt_num):
        logits = torch.stack([pro_img[:, i], pro_img[:, i + prompt_num]], dim=1)
        text_probs = (logits / scorer_temperature).softmax(dim=-1)
        per_prompt.append(text_probs[:, 1])
    return torch.stack(per_prompt, dim=1)


def aggregate_prompt_image_score(pro_img, prompt_num, prompt_weights=None, prompt_agg="mean",
                                  scorer_temperature=1.0, prompt_score_mode="softmax_mean"):
    if prompt_score_mode in (None, "", "softmax_mean"):
        per_prompt = [p for p in prompt_slot_image_scores(pro_img, prompt_num, scorer_temperature).unbind(dim=1)]

        if prompt_agg == "max":
            return torch.stack(per_prompt, dim=0).max(dim=0).values

        weights = _prepare_prompt_weights(prompt_num, prompt_weights, device=pro_img.device, dtype=pro_img.dtype)
        result = torch.zeros_like(per_prompt[0])
        for i, p in enumerate(per_prompt):
            result = result + weights[i] * p
        return result

    k = int(prompt_num)
    normal = pro_img[:, :k]
    abnormal = pro_img[:, k: 2 * k]
    margin = abnormal - normal
    if prompt_score_mode == "softmax_max":
        return prompt_slot_image_scores(pro_img, prompt_num, scorer_temperature).max(dim=1).values
    if prompt_score_mode == "margin_mean":
        return margin.mean(dim=1)
    if prompt_score_mode == "margin_max":
        return margin.max(dim=1).values
    if prompt_score_mode == "margin_top2_mean":
        topk = min(2, k)
        return margin.topk(topk, dim=1).values.mean(dim=1)
    if prompt_score_mode == "energy_margin":
        scaled_normal = normal / scorer_temperature
        scaled_abnormal = abnormal / scorer_temperature
        return torch.logsumexp(scaled_abnormal, dim=1) - torch.logsumexp(scaled_normal, dim=1)
    raise ValueError(f"Unsupported prompt_score_mode: {prompt_score_mode}")


def prompt_slot_anomaly_maps(anomaly_map, prompt_num, scorer_temperature=1.0):
    per_prompt = []
    for i in range(prompt_num):
        logits = torch.stack([anomaly_map[:, i, :, :], anomaly_map[:, i + prompt_num, :, :]], dim=1)
        probs = (logits / scorer_temperature).softmax(dim=1)
        per_prompt.append(probs[:, 1, :, :])
    return torch.stack(per_prompt, dim=1)


def aggregate_prompt_anomaly_map(anomaly_maps_list, prompt_num, layer_reduce="mean", prompt_weights=None,
                                  prompt_agg="mean", scorer_temperature=1.0, return_per_slot=False):
    layer_maps = []
    base_device = anomaly_maps_list[0].device
    base_dtype = anomaly_maps_list[0].dtype
    for anomaly_map in anomaly_maps_list:
        per_prompt_tensor = prompt_slot_anomaly_maps(anomaly_map, prompt_num, scorer_temperature)
        if return_per_slot:
            layer_maps.append(per_prompt_tensor)
            continue
        per_prompt = [p for p in per_prompt_tensor.unbind(dim=1)]

        if prompt_agg == "max":
            prompt_probs_abn = torch.stack(per_prompt, dim=0).max(dim=0).values
        else:
            weights = _prepare_prompt_weights(prompt_num, prompt_weights, device=base_device, dtype=base_dtype)
            prompt_probs_abn = torch.zeros_like(per_prompt[0])
            for i, p in enumerate(per_prompt):
                prompt_probs_abn = prompt_probs_abn + weights[i] * p

        layer_maps.append(prompt_probs_abn)

    stacked_layer_maps = torch.stack(layer_maps, dim=0)
    if layer_reduce == "mean":
        return stacked_layer_maps.mean(dim=0)
    if layer_reduce == "sum":
        return stacked_layer_maps.sum(dim=0)
    raise ValueError(f"Unsupported layer_reduce: {layer_reduce}")


def evaluate_pre(
    val_dataloader,
    model_clip,
    MyModel,
    device,
    args,
    val_obj_list_post,
    text_encoder,
    tokenizer,
    stage,
    prompt_optimizer=None,
):
    model_clip.eval()
    MyModel.eval()
    results = {
        "cls_names": [],
        "imgs_masks": [],
        "anomaly_maps": [],
        "pr_sp": [],
        "gt_sp": [],
    }
    print("-------------------start evaluating ---------------------")
    for items in tqdm(val_dataloader):
        image = items["img"].to(device)
        cls_name = items["cls_name"]
        results["cls_names"].append(cls_name[0])
        results["gt_sp"].append(items["anomaly"].item())
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results["imgs_masks"].append(gt_mask)

        with torch.no_grad():
            image_features, _, patch_tokens = model_clip.encode_image(image, args.features_list)
            text_embeddings = MyModel.forward_ensemble(
                text_encoder,
                cls_name,
                device,
                tokenizer,
                mode="val",
                prompt_optimizer=prompt_optimizer,
            )
            pro_img, anomaly_maps_list = MyModel(
                text_embeddings,
                image_features,
                patch_tokens,
                stage=stage,
                mode="val",
            )
            pro_img = pro_img.squeeze(2)
            temp_cls = aggregate_prompt_image_score(pro_img, args.prompt_num)
            anomaly_map = aggregate_prompt_anomaly_map(
                anomaly_maps_list,
                args.prompt_num,
                layer_reduce="mean",
            )[0].cpu().numpy()
            results["anomaly_maps"].append(anomaly_map)
            results["pr_sp"].append(float(temp_cls.detach().squeeze().cpu().item()))

    ap_px_ls = []
    ap_sp_ls = []
    for obj in val_obj_list_post:
        gt_px = []
        pr_px = []
        gt_sp = []
        pr_sp = []
        for idx in range(len(results["cls_names"])):
            if results["cls_names"][idx] == obj:
                gt_px.append(results["imgs_masks"][idx].squeeze(1).numpy())
                pr_px.append(results["anomaly_maps"][idx])
                gt_sp.append(results["gt_sp"][idx])
                pr_sp.append(results["pr_sp"][idx])
        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)
        gt_sp = np.array(gt_sp)
        pr_sp = np.array(pr_sp)
        pr_sp = (pr_sp - pr_sp.min()) / (pr_sp.max() - pr_sp.min() + 1e-8)
        ap_px_ls.append(average_precision_score(gt_px.ravel(), pr_px.ravel()))
        ap_sp_ls.append(average_precision_score(gt_sp.ravel(), pr_sp.ravel()))

    ap_mean_pixel = np.mean(ap_px_ls) if ap_px_ls else 0.0
    ap_mean_image = np.mean(ap_sp_ls) if ap_sp_ls else 0.0
    model_clip.eval()
    MyModel.train()
    torch.cuda.empty_cache()
    print("------------------- end evaluating ---------------------")
    return ap_mean_pixel, ap_mean_image
