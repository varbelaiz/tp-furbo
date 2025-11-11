import torch
from tqdm import tqdm
from time import sleep

from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool
from model.metrics import calculate_metrics

def train_one_epoch(epoch_index, training_loader, optimizer, loss_fn, model, device, transforms_gpu):
    model.train(True)
    running_loss = 0.
    samples = 0

    with (tqdm(enumerate(training_loader), unit="batch", total=len(training_loader)) as tepoch):
        for i, data in tepoch:

            if data is None:
                print(f"Saltando batch {i} corrupto...")
                continue

            tepoch.set_description(f"Epoch {epoch_index}")
            
            img_np, heat_np, mask_np = data

            images_gpu = torch.from_numpy(img_np).to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
            target = torch.from_numpy(heat_np).to(device, non_blocking=True)
            mask = torch.from_numpy(mask_np).to(device, non_blocking=True)

            input = transforms_gpu(images_gpu)

            optimizer.zero_grad()
            outputs = model(input)
            
            loss = loss_fn(outputs, target) * mask.unsqueeze(-1).unsqueeze(-1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            samples += mask.size()[0] 

            tepoch.set_postfix(loss=running_loss / samples)

    avg_loss = running_loss / samples
    return avg_loss


def validation_step(validation_loader, loss_fn, model, device, transforms_gpu):
    running_vloss = 0.0
    acc, prec, rec, f1 = 0, 0, 0, 0
    samples = 0
    model.eval()

    with torch.no_grad():
        for i, vdata in tqdm(enumerate(validation_loader), total=len(validation_loader)):

            if vdata is None:
                print(f"Saltando batch {i} corrupto...")
                continue
            
            img_np, heat_np, mask_np = vdata

            images_gpu = torch.from_numpy(img_np).to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
            target = torch.from_numpy(heat_np).to(device, non_blocking=True)
            mask = torch.from_numpy(mask_np).to(device, non_blocking=True)

            input = transforms_gpu(images_gpu)

            voutputs = model(input)
            vloss = loss_fn(voutputs, target) * mask.unsqueeze(-1).unsqueeze(-1)
            vloss = vloss.mean()

            kp_gt = get_keypoints_from_heatmap_batch_maxpool(target[:,:-1,:,:], return_scores=True, max_keypoints=1)
            kp_pred = get_keypoints_from_heatmap_batch_maxpool(voutputs[:,:-1,:,:], return_scores=True, max_keypoints=1)
            metrics = calculate_metrics(kp_gt, kp_pred, mask) 

            running_vloss += vloss.item() 
            samples += mask.size()[0]

            acc += metrics[0]
            prec += metrics[1]
            rec += metrics[2]
            f1 += metrics[3]

    avg_vloss = running_vloss / samples
    avg_acc = acc / (i+1)
    avg_prec = prec / (i+1)
    avg_rec = rec / (i+1)
    avg_f1 = f1 / (i+1)

    return avg_vloss, avg_acc, avg_prec, avg_rec, avg_f1