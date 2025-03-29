import numpy as np
import torch
import torchvision
import argparse
import torch.utils.tensorboard as tb
from pathlib import Path
import datetime

import numpy as np
import tqdm

import torch.nn.functional as F
import torch.nn as nn

# watch iouloss to see if the model is learning or not, instead of waiting
# until the end of a run and running through the grader for results
class SoftIoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(SoftIoULoss, self).__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] (raw output from model)
        targets: [B, H, W] (ground truth class labels, 0 ≤ target < C)
        """
        # derive number of classes from the shape of logits
        num_classes = logits.shape[1] 

        # convert targets to one-hot encoding
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        one_hot_targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # compute intersection and union
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot_targets, dims)
        union = torch.sum(probs + one_hot_targets, dims) - intersection
        iou = (intersection + self.eps) / (union + self.eps)

        # return 1 - iou for minimization
        return 1.0 - iou.mean()

try:
    from .models import load_model, save_model
except:
    from models import load_model, save_model

try:
    from .datasets.road_utils import load_data
except:
    from datasets.road_dataset import load_data 

def train(
    exp_dir: str = "logs",
    #model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 1969,
    **kwargs
):
    # this has to do two things: 
    ## depth estimation 
    ## road semantic segmentation, which classifies each as left boundary, right boundary, or not boundary

    # depth estimation
    if torch.cuda.is_available():
        print("CUDA available, using GPU")
        device = torch.device("cuda")
    else:  
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    def loss_formula(segm, dpth):
        return segm + dpth**.75

    log_dir = Path(exp_dir) / f"detector_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model("detector", in_channels=3, num_classes=3).to(device)
    if 0:
        model.load_state_dict(torch.load("detector.th"))
    
    model.to(device)

    train_data = load_data(".\\drive_data\\train", batch_size=batch_size, shuffle=True, transform_pipeline="aug")
    val_data = load_data(".\\drive_data\\val", batch_size=batch_size, shuffle=False)

    global_step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    weights = None
    iou_loss = SoftIoULoss()

    for epoch in tqdm.tqdm(range(num_epoch)):
        #epoch+=50
        model.train()

        # change weights for class imbalance
        # add weights after training in case signal is too weak for segmentation
        if epoch / num_epoch < 0.5:
            weights = torch.tensor([1., 3., 3.]).to(device)
        else:
            weights = torch.tensor([1., 7., 7.]).to(device)

        for batch in train_data: #img, track, depth in train_data:
            img, track, depth = batch["image"].to(device), batch["track"].to(device), batch["depth"].to(device)

            model_track, model_depth = model(img)

            segmentation_loss = torch.nn.functional.cross_entropy(model_track, track, weight=weights)
            segmentation_loss_iou = iou_loss(model_track, track)
            depth_loss = torch.nn.functional.l1_loss(model_depth, depth)
            #loss = loss_formula(segmentation_loss+2*segmentation_loss_iou, depth_loss)

            # given that I have weighting mechanism, want to adjust the total loss function
            # to make sure depth loss still gets attention
            loss = loss_formula(segmentation_loss, depth_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("train/loss", loss.item(), global_step=global_step)
            logger.add_scalar("train/segmentation_loss", segmentation_loss.item(), global_step)
            logger.add_scalar("train/iou_loss", segmentation_loss_iou.item(), global_step)
            logger.add_scalar("train/depth_loss", depth_loss.item(), global_step)

            global_step += 1

        with torch.inference_mode():
            model.eval()
            seg_loss_list = []
            iou_loss_list = []
            depth_loss_list = []
            loss_list = []

            for batch in val_data:
                img, track, depth = batch["image"].to(device), batch["track"].to(device), batch["depth"].to(device)
                output_track, output_depth = model(img)

                segmentation_loss = torch.nn.functional.cross_entropy(output_track, track)
                segmentation_loss_iou = iou_loss(output_track, track)
                depth_loss = torch.nn.functional.l1_loss(output_depth, depth)
                loss = loss_formula(segmentation_loss+segmentation_loss_iou, depth_loss)
                seg_loss_list.append(segmentation_loss.item())
                iou_loss_list.append(segmentation_loss_iou.item())
                depth_loss_list.append(depth_loss.item())
                loss_list.append(loss.item())

            logger.add_scalar("val/loss", np.mean(loss_list), epoch)
            logger.add_scalar("val/segmentation_loss", np.mean(seg_loss_list), epoch)
            logger.add_scalar("val/depth_loss", np.mean(depth_loss_list), epoch)
            logger.add_scalar("val/iou_loss", np.mean(iou_loss_list), epoch)

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"detector.th")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    #parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1969)
    parser.add_argument("--batch_size", type=int, default=32)
    #parser.add_argument("--regularizer", type=float, default=0.0)
    #parser.add_argument("--hidden_dim", type=int, default=128)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))      

            

 


