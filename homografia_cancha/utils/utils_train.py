import torch
# import torchvision.transforms as T  <- ELIMINADO
from tqdm import tqdm
from time import sleep

from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool
from model.metrics import calculate_metrics

# --- CAMBIO 1: Añadir 'transforms_gpu' a la firma ---
def train_one_epoch(epoch_index, training_loader, optimizer, loss_fn, model, device, transforms_gpu):
    model.train(True)
    running_loss = 0.
    samples = 0
    # transform = T.Resize((540, 960)) <- ELIMINADO (Lógica de CPU)

    with (tqdm(enumerate(training_loader), unit="batch", total=len(training_loader)) as tepoch):
        for i, data in tepoch:
            tepoch.set_description(f"Epoch {epoch_index}")
            
            # --- CAMBIO 2: Desempaquetar numpy arrays y mover/transformar en GPU ---
            # data[0] = img_np [B, 540, 960, 3] (uint8)
            # data[1] = heat_np [B, 58, 270, 480] (float32)
            # data[2] = mask_np [B, 58] (float32)
            img_np, heat_np, mask_np = data

            # Mover a GPU, permutar, y escalar imagen
            images_gpu = torch.from_numpy(img_np).to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
            # Mover targets a GPU
            target = torch.from_numpy(heat_np).to(device, non_blocking=True)
            mask = torch.from_numpy(mask_np).to(device, non_blocking=True)

            # Aplicar Kornia transforms (Resize + Normalize) EN LA GPU
            input = transforms_gpu(images_gpu)
            # --- FIN CAMBIO 2 ---

            # input = input if input.size()[-1] == 960 else transform(input) <- ELIMINADO (Lógica obsoleta)

            optimizer.zero_grad()
            outputs = model(input)
            
            # Esta lógica de loss se mantiene, pero usando los tensores de GPU
            loss = loss_fn(outputs, target) * mask.unsqueeze(-1).unsqueeze(-1)
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            samples += mask.size()[0] # 'mask' ahora es un tensor de GPU

            tepoch.set_postfix(loss=running_loss / samples)
            # sleep(0.1) # <- Opcional: puedes comentar esto para máxima velocidad

    avg_loss = running_loss / samples
    return avg_loss


# --- CAMBIO 3: Añadir 'transforms_gpu' a la firma ---
def validation_step(validation_loader, loss_fn, model, device, transforms_gpu):
    running_vloss = 0.0
    acc, prec, rec, f1 = 0, 0, 0, 0
    samples = 0
    # transform = T.Resize((540, 960)) <- ELIMINADO
    model.eval()

    with torch.no_grad():
        for i, vdata in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            
            # --- CAMBIO 4: Replicar la lógica de Kornia en validación ---
            img_np, heat_np, mask_np = vdata

            # Mover a GPU, permutar, y escalar imagen
            images_gpu = torch.from_numpy(img_np).to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
            # Mover targets a GPU
            target = torch.from_numpy(heat_np).to(device, non_blocking=True)
            mask = torch.from_numpy(mask_np).to(device, non_blocking=True)

            # Aplicar Kornia transforms (Resize + Normalize) EN LA GPU
            input = transforms_gpu(images_gpu)
            # --- FIN CAMBIO 4 ---
            
            # input = input if input.size()[-1] == 960 else transform(input) <- ELIMINADO

            voutputs = model(input)
            vloss = loss_fn(voutputs, target) * mask.unsqueeze(-1).unsqueeze(-1)
            vloss = vloss.mean()

            # Usar los tensores de GPU para las métricas
            kp_gt = get_keypoints_from_heatmap_batch_maxpool(target[:,:-1,:,:], return_scores=True, max_keypoints=1)
            kp_pred = get_keypoints_from_heatmap_batch_maxpool(voutputs[:,:-1,:,:], return_scores=True, max_keypoints=1)
            metrics = calculate_metrics(kp_gt, kp_pred, mask) # 'mask' ya está en GPU

            running_vloss += vloss.item() # .item() es importante aquí
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