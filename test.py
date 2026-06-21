"""
Test script for SFDA-DDFP.

Usage (CHAOS):
    python3 test.py \
        --model_path results/.../best_model.pth \
        --data_root datasets/chaos \
        --target_site ct \
        --gpu_id 0

Usage (PROSTATE):
    python3 test.py \
        --dataset PROSTATE \
        --domain source \
        --model_path results/.../best_model.pth \
        --data_root /opt/data/private/MedSeg_Data_Process/PROSTATE/processed_new \
        --num_classes 2 \
        --gpu_id 0
"""

import argparse
import datetime
import json
import os
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import get_model
from dataloaders import MyDataset147, ProstateDataset
from utils.metrics import MultiDiceScore, MultiASD


CHAOS_ORGAN_LIST = ['Liver', 'R.Kidney', 'L.Kidney', 'Spleen']
CHAOS_NUM_CLASSES = 5
PROSTATE_ORGAN_LIST = ['Prostate']
PROSTATE_NUM_CLASSES = 2


def _parse_sample_name(name):
    if "_slice_" in name:
        parts = name.split("_slice_")
        case_name = parts[0].replace("vol_", "", 1)
        slice_idx = int(parts[1])
        return case_name, slice_idx
    else:
        base = name.rsplit('.', 1)[0] if '.' in name else name
        parts = base.split('_')
        return parts[0], int(parts[1])


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--dataset', type=str, default='CHAOS',
                        choices=['CHAOS', 'PROSTATE'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='datasets/chaos')
    parser.add_argument('--target_site', type=str, default='ct',
                        help='CHAOS target site')
    parser.add_argument('--domain', type=str, default='source',
                        help='PROSTATE domain (source, target_1, target_2)')
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--arch', type=str, default=None)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pmt_size', type=int, default=None)
    parser.add_argument('--pmt_type', type=str, default='Data')
    args = parser.parse_args()

    if args.dataset == 'PROSTATE':
        if args.num_classes is None:
            args.num_classes = PROSTATE_NUM_CLASSES
        if args.img_size is None:
            args.img_size = 384
        if args.pmt_size is None:
            args.pmt_size = 384
    else:
        if args.num_classes is None:
            args.num_classes = CHAOS_NUM_CLASSES
        if args.img_size is None:
            args.img_size = 256
        if args.pmt_size is None:
            args.pmt_size = 256
    return args


def build_model(args, device):
    cfg = {
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
        saved_cfg.update(cfg)
        cfg = saved_cfg

    if args.arch is not None:
        cfg['arch'] = args.arch
    if 'arch' not in cfg:
        cfg['arch'] = 'UNet'

    cfg['doing'] = 'test'
    print(f"Model config: arch={cfg['arch']}, num_classes={cfg['num_classes']}")

    model = get_model(cfg)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f'Loaded checkpoint: {args.model_path}')
    return model


def collect_predictions(model, dataloader, device):
    sample_dict = {}
    with torch.no_grad():
        for images, segs, names in tqdm(dataloader, desc='Inference'):
            images = images.to(device)
            out = model(images)
            if isinstance(out, tuple):
                predicts = out[1]
            else:
                predicts = out
            for i, name in enumerate(names):
                patient_id, slice_idx = _parse_sample_name(name)
                entry = (predicts[i].cpu(), segs[i].cpu(), slice_idx)
                sample_dict.setdefault(patient_id, []).append(entry)
    return sample_dict


def build_volumes(sample_dict):
    pred_volumes, gt_volumes = [], []
    patient_ids = []
    for patient_id in sorted(sample_dict.keys()):
        slices = sorted(sample_dict[patient_id], key=lambda x: x[2])
        preds, targets = [], []
        for pred, target, _ in slices:
            if target.sum() == 0:
                continue
            preds.append(pred)
            targets.append(target)
        if len(preds) == 0:
            continue
        pred_volumes.append(torch.stack(preds, dim=-1))
        gt_volumes.append(torch.stack(targets, dim=-1))
        patient_ids.append(patient_id)
    return pred_volumes, gt_volumes, patient_ids


def compute_metrics(pred_volumes, gt_volumes, patient_ids, num_classes, organ_list):
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
            print(f'  [ASSD warning] patient {patient_ids[idx]}: {e}')

    print('\n' + '=' * 72)
    print(f'{"Patient":<20}', end='')
    for organ in organ_list:
        print(f' {organ+" Dice":>12}', end='')
    print(f' {"Avg Dice":>12}')
    print('-' * 72)
    for idx, pid in enumerate(patient_ids):
        print(f'{pid:<20}', end='')
        for c in range(num_fg):
            print(f' {all_dice[idx, c]:>12.4f}', end='')
        print(f' {np.nanmean(all_dice[idx]):>12.4f}')

    print('\n' + '=' * 60)
    print(f'{"Class":<15} {"Dice":>10} {"ASSD":>10}')
    print('-' * 60)
    for c, organ in enumerate(organ_list):
        dice_mean = np.nanmean(all_dice[:, c])
        assd_mean = np.nanmean(all_assd[:, c])
        print(f'{organ:<15} {dice_mean:>10.4f} {assd_mean:>10.4f}')

    mean_dice = np.nanmean(all_dice)
    mean_assd = np.nanmean(all_assd)
    print('-' * 60)
    print(f'{"Mean (fg)":<15} {mean_dice:>10.4f} {mean_assd:>10.4f}')
    print('=' * 60)
    return all_dice, all_assd, patient_ids


def save_results_txt(args, domain_name, organ_list, all_dice, all_assd,
                     patient_ids, num_test_samples):
    exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.model_path)))
    model_filename = os.path.basename(args.model_path)
    txt_name = f"test_results_{domain_name}.txt"
    txt_path = os.path.join(exp_dir, txt_name)

    num_fg = args.num_classes - 1
    lines = []
    lines.append(f"Test Results")
    lines.append(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset: {args.dataset}")
    lines.append(f"Domain: {domain_name}")
    lines.append(f"Model: {model_filename}")
    lines.append(f"Model path: {args.model_path}")
    lines.append(f"Test samples: {num_test_samples}")
    lines.append(f"Patients evaluated: {len(patient_ids)}")
    lines.append("")

    header = f"{'Patient':<20}"
    for organ in organ_list:
        header += f" {organ + ' Dice':>12} {organ + ' ASSD':>12}"
    header += f" {'Avg Dice':>12} {'Avg ASSD':>12}"
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    for idx, pid in enumerate(patient_ids):
        row = f"{pid:<20}"
        for c in range(num_fg):
            d = all_dice[idx, c]
            a = all_assd[idx, c]
            row += f" {d:>12.4f} {a:>12.4f}"
        row += f" {np.nanmean(all_dice[idx]):>12.4f} {np.nanmean(all_assd[idx]):>12.4f}"
        lines.append(row)
    lines.append("=" * len(header))
    lines.append("")

    lines.append(f"{'Class':<15} {'Dice':>10} {'ASSD':>10}")
    lines.append("-" * 37)
    for c, organ in enumerate(organ_list):
        dice_mean = np.nanmean(all_dice[:, c])
        assd_mean = np.nanmean(all_assd[:, c])
        lines.append(f"{organ:<15} {dice_mean:>10.4f} {assd_mean:>10.4f}")
    lines.append("-" * 37)
    mean_dice = np.nanmean(all_dice)
    mean_assd = np.nanmean(all_assd)
    lines.append(f"{'Mean (fg)':<15} {mean_dice:>10.4f} {mean_assd:>10.4f}")
    lines.append("=" * 37)

    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nResults saved to: {txt_path}')


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Dataset: {args.dataset}')

    if args.dataset == 'PROSTATE':
        domain_name = args.domain
        print(f'Domain: {domain_name}')
        dataset = ProstateDataset(
            data_root=args.data_root,
            domain_name=domain_name,
            phase='val',
            split_train=False,
            img_size=(args.img_size, args.img_size),
        )
        organ_list = PROSTATE_ORGAN_LIST[:args.num_classes - 1]
    else:
        domain_name = args.target_site
        print(f'Target site: {domain_name}')
        dataset = MyDataset147(
            rootdir=args.data_root,
            sites=[domain_name],
            dataset_name='abdomen',
            phase='val',
        )
        organ_list = CHAOS_ORGAN_LIST[:args.num_classes - 1]

    print(f'Test dataset size: {len(dataset)}')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = build_model(args, device)
    sample_dict = collect_predictions(model, dataloader, device)
    pred_volumes, gt_volumes, patient_ids = build_volumes(sample_dict)
    print(f'Total patients evaluated: {len(pred_volumes)}')

    all_dice, all_assd, patient_ids = compute_metrics(
        pred_volumes, gt_volumes, patient_ids, args.num_classes, organ_list)
    save_results_txt(args, domain_name, organ_list, all_dice, all_assd,
                     patient_ids, len(dataset))


if __name__ == '__main__':
    main()
