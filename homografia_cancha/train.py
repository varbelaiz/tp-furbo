import os
import yaml
import wandb
import torch

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import socket
import argparse
import warnings
import numpy as np
import torch.nn as nn
from datetime import datetime
import kornia.augmentation as K
from google.cloud import storage 
import torch.multiprocessing as mp
import kornia.geometry.transform as K_T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

torch.backends.cudnn.benchmark = True

from model.losses import CombMSEAW
from model.cls_hrnet import get_cls_net
from model.dataloader import SoccerNetCalibrationDataset
from utils.utils_train import train_one_epoch, validation_step

warnings.filterwarnings("ignore", category=np.exceptions.RankWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    parser.add_argument("--num_epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--pretrained", type=str, default='', help="Pretrained weights path")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Patience for LR scheduler")
    parser.add_argument("--factor", type=float, default=0.5, help="LR scheduler factor")
    parser.add_argument("--wandb_project", type=str, default='', help="Wandb project name")
    parser.add_argument("--gcs_bucket", type=str, default='', help="GCS Bucket name for saving models")
    return parser.parse_args()

def collate_fn_skip_corrupt(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None 
    
    return torch.utils.data.dataloader.default_collate(batch)

def main(rank, args, world_size, port):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.distributed.init_process_group(backend="nccl", init_method=f"tcp://localhost:{port}", rank=rank,
                                         world_size=world_size)

    gcs_bucket = None
    if rank == 0:
        wandb.login()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        wandb.init(mode="online" if args.wandb_project else "offline", project=args.wandb_project,
                   config={"timestamp": timestamp,
                           "batch_per_gpu": args.batch, "learning_rate_0": args.lr0,
                           "epochs": args.num_epochs, "pretrained": args.pretrained})

        if args.gcs_bucket:
            try:
                storage_client = storage.Client()
                gcs_bucket = storage_client.bucket(args.gcs_bucket)
                print(f"✅ Conectado al bucket de GCS: {args.gcs_bucket}")
            except Exception as e:
                print(f"❌ Error al conectar con GCS Bucket: {e}")
                print("El modelo se guardará solo localmente.")
                gcs_bucket = None

    if args.dataset != "SoccerNet":
        raise ValueError(f"Este script está configurado solo para 'SoccerNet', no '{args.dataset}'")
    
    dataset_cls = SoccerNetCalibrationDataset
    training_set = dataset_cls(args.root_dir, "train", augment=True)
    validation_set = dataset_cls(args.root_dir, "val", augment=False)

    train_sampler = DistributedSampler(training_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(validation_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(training_set, batch_size=args.batch, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_corrupt)
    val_loader = DataLoader(validation_set, batch_size=args.batch, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_skip_corrupt)

    cfg = yaml.safe_load(open(args.cfg, 'r'))

    height, width = 540, 960
    
    data_transforms_gpu = nn.Sequential(
        K_T.Resize((height, width), antialias=True),
        K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]).to(device),
                    std=torch.tensor([0.229, 0.224, 0.225]).to(device))
    ).to(device)

    data_transforms_gpu = data_transforms_gpu    

    model = get_cls_net(cfg).to(device)

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=device))

    model = torch.compile(model) 
    model = DDP(model, device_ids=[rank])

    loss_fn = CombMSEAW(lambda1=1, lambda2=1).to(device)    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor,
                                                           mode='min')

    best_vloss = np.inf
    loss_counter = 0
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
    
        avg_loss = train_one_epoch(epoch + 1, train_loader, optimizer, loss_fn, model, device, data_transforms_gpu)
        avg_vloss, acc, prec, rec, f1 = validation_step(val_loader, loss_fn, model, device, data_transforms_gpu)
        
        scheduler.step(avg_vloss)

        if rank == 0:
            print(f'LOSS train {avg_loss} valid {avg_vloss} Accuracy {acc} Precision {prec}')

            wandb.log({"train_loss": avg_loss, "val_loss": avg_vloss, "epoch": epoch + 1, 'lr': optimizer.param_groups[0]["lr"], 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1})

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss

                local_save_path = args.save_dir + f'_{timestamp}.pt'
                torch.save(model.module.state_dict(), local_save_path)
                print(f"Modelo guardado localmente en: {local_save_path}")

                if gcs_bucket:
                    try:
                        gcs_path = f"models/keypoints_best_epoch{epoch+1}.pt"
                        blob = gcs_bucket.blob(gcs_path)
                        blob.upload_from_filename(local_save_path)
                        print(f"✅ Modelo subido a GCS: {gcs_path}")
                    except Exception as e:
                        print(f"❌ Error al subir a GCS: {e}")

                loss_counter = 0
            else:
                loss_counter += 1

            if loss_counter == 8:
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