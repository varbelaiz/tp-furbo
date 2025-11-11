import os
import sys
import yaml
import wandb
import torch
import argparse
import warnings
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
import socket

from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from utils.utils_train_l import train_one_epoch, validation_step
from model.dataloader_l import SoccerNetCalibrationDataset, WorldCup2014Dataset, TSWorldCupDataset, WorldPoseDataset
from model.cls_hrnet_l import get_cls_net
from model.losses import MSELoss

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.RankWarning)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--dataset", type=str, default='SoccerNet', help="Dataset name")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Save directory")
    parser.add_argument("--cuda", type=str, default="0", help="Comma-separated GPU IDs (e.g., '0,1,2')")
    parser.add_argument("--batch", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=2, help="Workers per GPU")
    parser.add_argument("--num_epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--pretrained", type=str, default='', help="Pretrained weights path")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=8, help="Patience for LR scheduler")
    parser.add_argument("--factor", type=float, default=0.5, help="LR scheduler factor")
    parser.add_argument("--wandb_project", type=str, default='', help="Wandb project name")
    return parser.parse_args()


def main(rank, args, world_size, port):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.distributed.init_process_group(backend="nccl", init_method=f"tcp://localhost:{port}", rank=rank,
                                         world_size=world_size)

    if rank == 0:
        wandb.login()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb.init(mode="online" if args.wandb_project else "offline", project=args.wandb_project,
                   config={"timestamp": timestamp,
                           "batch_per_gpu": args.batch, "learning_rate_0": args.lr0,
                           "epochs": args.num_epochs, "pretrained": args.pretrained})

    dataset_splits = {
        "SoccerNet": ("train", "valid"),
        "WorldCup14": ("train_val", "test"),
        "TSWorldCup": ("train", "test"),
        "WorldPose": ("train", "val")
    }
    train_split, val_split = dataset_splits[args.dataset]
    dataset_cls = {
        "SoccerNet": SoccerNetCalibrationDataset,
        "WorldCup14": WorldCup2014Dataset,
        "TSWorldCup": TSWorldCupDataset,
        "WorldPose": WorldPoseDataset
    }[args.dataset]

    transform_module = __import__("model.transforms_l" if "SoccerNet" in args.dataset else "model.transformsWC_l",
                                  fromlist=["transforms", "no_transforms"])
    training_set = dataset_cls(args.root_dir, train_split, transform=transform_module.transforms)
    validation_set = dataset_cls(args.root_dir, val_split, transform=transform_module.no_transforms)

    train_sampler = DistributedSampler(training_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(validation_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(training_set, batch_size=args.batch, sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(validation_set, batch_size=args.batch, sampler=val_sampler, num_workers=args.num_workers)

    cfg = yaml.safe_load(open(args.cfg, 'r'))
    model = get_cls_net(cfg).to(device)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
    model = DDP(model, device_ids=[rank])

    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor,
                                                           mode='min')

    best_vloss = np.inf
    loss_counter = 0
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        avg_loss = train_one_epoch(epoch + 1, train_loader, optimizer, loss_fn, model, device)
        avg_vloss, acc, prec, rec, f1 = validation_step(val_loader, loss_fn, model, device)
        scheduler.step(avg_vloss)

        if rank == 0:
            print(f'LOSS train {avg_loss} valid {avg_vloss} Accuracy {acc} Precision {prec}')
            wandb.log({"train_loss": avg_loss, "val_loss": avg_vloss, "epoch": epoch + 1,
                       'lr': optimizer.param_groups[0]["lr"], 'Accuracy': acc, 'Precision': prec,
                       'Recall': rec, 'F1-score': f1})

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(model.module.state_dict(), args.save_dir + f'_{timestamp}.pt')
                loss_counter = 0
            else:
                loss_counter += 1

            if loss_counter == 16:
                print(f'Early stopping at epoch {epoch + 1}')
                break


def launch_training():
    args = parse_args()
    gpus = list(map(int, args.cuda.split(',')))
    world_size = len(gpus)
    port = find_free_port()
    mp.spawn(main, args=(args, world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    launch_training()

