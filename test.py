"""
Test script for SFDA-DDFP.

Usage:
    python3 test.py \
        --model_path results/Target_Adapt/.../saved_models/best_model.pth \
        --data_root datasets/chaos \
        --target_site MR \
        --gpu_id 0

    # with prediction overlay PNGs:
    python3 test.py \
        --model_path ... \
        --data_root datasets/chaos \
        --target_site MR \
        --save_vis --vis_dir ./vis_results
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

from models import get_model
from dataloaders import MyDataset147
from utils.metrics import MultiDiceScore, MultiASD


# Actual label order from preprocess_chaos.ipynb: select_label()
# liver=1, R.Kidney=2, L.Kidney=3, Spleen=4
# Note: the yaml organ_list has Liver/Spleen swapped (a labeling bug, not affecting metric values)
ORGAN_LIST = ['Liver', 'R.Kidney', 'L.Kidney', 'Spleen']
NUM_CLASSES = 5

# ── Visualisation colour palette ─────────────────────────────────────────────
# HEX colours per class index; index 0 = background (skipped).
LABEL_COLORS_HEX = [
    None,        # 0: background — not drawn
    "#80AE80",   # 1: Liver        — green
    "#F1D691",   # 2: Right Kidney — yellow
    "#B17A65",   # 3: Left Kidney  — brown-red
    "#6FB8D2",   # 4: Spleen       — blue
]
LABEL_ALPHA       = 0.35   # fill transparency
CONTOUR_THICKNESS = 1      # contour line width (px)
# ─────────────────────────────────────────────────────────────────────────────


def _hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def overlay_labels(bgr, label):
    """Blend semi-transparent colour fill + thin contour per class."""
    result = bgr.copy().astype(np.float32)
    for cls_idx, hex_color in enumerate(LABEL_COLORS_HEX):
        if hex_color is None:
            continue
        mask = (label == cls_idx).astype(np.uint8)
        if mask.sum() == 0:
            continue
        bgr_color = _hex_to_bgr(hex_color)
        color_layer = np.zeros_like(result)
        color_layer[mask == 1] = bgr_color
        result = np.where(
            mask[:, :, None] == 1,
            result * (1 - LABEL_ALPHA) + color_layer * LABEL_ALPHA,
            result,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, bgr_color, CONTOUR_THICKNESS, cv2.LINE_AA)
    return result.clip(0, 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint (.pth)')
    parser.add_argument('--data_root', type=str,
                        default='datasets/chaos',
                        help='Root directory of the dataset')
    parser.add_argument('--target_site', type=str, default='MR',
                        help='Target domain site name (e.g. MR, CT)')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--arch', type=str, default='UNet')
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pmt_size', type=int, default=256,
                        help='Prompt size for Pmt_UNet / Pmt_DeepLab')
    parser.add_argument('--pmt_type', type=str, default='Data',
                        help='Prompt type for Pmt_UNet / Pmt_DeepLab')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save per-slice prediction overlay PNGs')
    parser.add_argument('--vis_dir', type=str, default='./vis_results',
                        help='Output directory for prediction overlay PNGs')
    return parser.parse_args()


def build_model(args, device):
    import json, pathlib
    # Try to load cfg from config.json saved alongside the checkpoint
    cfg = {
        'arch': args.arch,
        'input_dim': args.input_dim,
        'num_classes': args.num_classes,
        'pmt_size': args.pmt_size,
        'pmt_type': args.pmt_type,
        'doing': 'test',
    }
    ckpt_dir = pathlib.Path(args.model_path).parent.parent
    config_path = ckpt_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            saved_cfg = json.load(f)
        cfg.update(saved_cfg)
        cfg['doing'] = 'test'   # always override to test mode
    model = get_model(cfg)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f'Loaded checkpoint: {args.model_path}')
    return model


def collect_predictions(model, dataloader, device, save_vis=False):
    """Run inference and group slice predictions (and images) by patient."""
    sample_dict = {}
    with torch.no_grad():
        for images, segs, names in tqdm(dataloader, desc='Inference'):
            images = images.to(device)
            _, predicts, *_ = model(images)
            for i, name in enumerate(names):
                # Strip extension if present (MyDataset147 keeps full filename)
                base = name.rsplit('.', 1)[0] if '.' in name else name
                parts = base.split('_')
                patient_id = parts[0]
                slice_idx = int(parts[1])
                img_cpu = images[i].cpu() if save_vis else None
                entry = (predicts[i].cpu(), segs[i].cpu(), slice_idx, img_cpu)
                sample_dict.setdefault(patient_id, []).append(entry)
    return sample_dict


def build_volumes(sample_dict):
    """Sort slices per patient and stack into 3D volumes, skip all-zero slices."""
    pred_volumes, gt_volumes = [], []
    for patient_id in sorted(sample_dict.keys()):
        slices = sorted(sample_dict[patient_id], key=lambda x: x[2])
        preds, targets = [], []
        for pred, target, _, _img in slices:
            if target.sum() == 0:
                continue
            preds.append(pred)
            targets.append(target)
        if len(preds) == 0:
            continue
        pred_volumes.append(torch.stack(preds, dim=-1))    # (C, H, W, D)
        gt_volumes.append(torch.stack(targets, dim=-1))    # (H, W, D)
    return pred_volumes, gt_volumes


def save_vis_slices(sample_dict, target_site, vis_dir):
    """Save per-slice prediction overlay PNGs for all patients.

    Output: {vis_dir}/{target_site}_{patient_id}/slice_{idx:04d}_pred.png
    Images are horizontally flipped (left-right) before saving.
    """
    for patient_id in sorted(sample_dict.keys()):
        slices = sorted(sample_dict[patient_id], key=lambda x: x[2])
        case_dir = os.path.join(vis_dir, f"{target_site}_{patient_id}")
        os.makedirs(case_dir, exist_ok=True)

        for pred, _gt, slice_idx, img_cpu in slices:
            # Use middle channel of 3-channel input as grayscale base
            img_np = img_cpu.numpy()                              # (C, H, W)
            mid = img_np.shape[0] // 2
            gray = img_np[mid]                                    # (H, W)
            lo, hi = gray.min(), gray.max()
            gray = (gray - lo) / (hi - lo + 1e-8)
            img_u8 = (gray * 255).clip(0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

            pred_label = pred.argmax(dim=0).numpy().astype(np.int32)  # (H, W)
            bgr = overlay_labels(bgr, pred_label)
            bgr = cv2.flip(bgr, 1)

            fname = f"slice_{slice_idx:04d}_pred.png"
            cv2.imwrite(os.path.join(case_dir, fname), bgr)

        print(f"  Vis saved -> {case_dir}  ({len(slices)} slices)")


def compute_metrics(pred_volumes, gt_volumes, num_classes, organ_list):
    """Compute per-class Dice and ASSD for all patients."""
    num_fg = num_classes - 1
    all_dice = np.full((len(pred_volumes), num_fg), np.nan)
    all_assd = np.full((len(pred_volumes), num_fg), np.nan)

    for idx, (pred, gt) in enumerate(zip(pred_volumes, gt_volumes)):
        dice_list = MultiDiceScore(pred, gt, num_classes, include_bg=False)
        for c, d in enumerate(dice_list):
            if not np.isnan(d):
                all_dice[idx, c] = d

        try:
            assd_list = MultiASD(pred, gt, num_classes, include_bg=False)
            for c, a in enumerate(assd_list):
                all_assd[idx, c] = a
        except Exception as e:
            print(f'  [ASSD warning] patient {idx}: {e}')

    results = {}
    print('\n' + '=' * 60)
    print(f'{"Class":<15} {"Dice":>10} {"ASSD":>10}')
    print('-' * 60)
    for c, organ in enumerate(organ_list):
        dice_mean = np.nanmean(all_dice[:, c])
        assd_mean = np.nanmean(all_assd[:, c])
        results[organ] = {'dice': dice_mean, 'assd': assd_mean}
        print(f'{organ:<15} {dice_mean:>10.4f} {assd_mean:>10.4f}')

    mean_dice = np.nanmean(all_dice)
    mean_assd = np.nanmean(all_assd)
    results['mean'] = {'dice': mean_dice, 'assd': mean_assd}
    print('-' * 60)
    print(f'{"Mean (fg)":<15} {mean_dice:>10.4f} {mean_assd:>10.4f}')
    print('=' * 60)
    return results


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Target site: {args.target_site}')

    dataset = MyDataset147(
        rootdir=args.data_root,
        sites=[args.target_site],
        phase='val',
        dataset_name='abdomen',
    )
    print(f'Test dataset size: {len(dataset)}')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = build_model(args, device)
    sample_dict = collect_predictions(model, dataloader, device, save_vis=args.save_vis)
    pred_volumes, gt_volumes = build_volumes(sample_dict)
    print(f'Total patients evaluated: {len(pred_volumes)}')

    organ_list = ORGAN_LIST[:args.num_classes - 1]
    compute_metrics(pred_volumes, gt_volumes, args.num_classes, organ_list)

    if args.save_vis:
        print(f'\nSaving prediction overlays -> {args.vis_dir}')
        save_vis_slices(sample_dict, args.target_site, args.vis_dir)


if __name__ == '__main__':
    main()
