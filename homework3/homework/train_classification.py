import numpy as np
import torch
import torchvision
import argparse
import torch.utils.tensorboard as tb
from pathlib import Path
import datetime

import numpy as np
import tqdm


try:
    from .models import load_model, save_model
except:
    from models import load_model, save_model

try:
    from .datasets.classification_dataset import load_data
except:
    from datasets.classification_dataset import load_data 

def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean()

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

    log_dir = Path(exp_dir) / f"classifier_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    model = model.to(device)

    train_data = load_data(".\\classification_data\\train", batch_size=batch_size, shuffle=True)
    val_data = load_data(".\\classification_data\\val", batch_size=batch_size, shuffle=False)

    global_step = 0
    metrics = {"segm_train_loss": [], "segm_val_loss": [], "segm_train_acc": [], "segm_val_acc": [],
                "depth_train_loss": [], "depth_val_loss": [], "depth_train_acc": [], "depth_val_acc": [],
                "train_loss": [], "val_loss": []}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in tqdm.tqdm(range(num_epoch)):
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data: #img, track, depth in train_data:
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = torch.nn.functional.cross_entropy(output.to(device), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #metrics["segm_train_loss"].append(segmentation_loss.item())
            #metrics["depth_train_loss"].append(depth_loss.item())
            #metrics["train_loss"].append(loss.item())

            logger.add_scalar("train/loss", loss.item(), global_step=global_step)
            logger.add_scalar("train/accuracy", accuracy(output, label), global_step)
            #logger.add_scalar("train/segmentation_loss", segmentation_loss.item(), global_step)
            #logger.add_scalar("train/depth_loss", depth_loss.item(), global_step)

            global_step += 1

        with torch.inference_mode():
            model.eval()
            loss_list = []
            accuracy_list = []

            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                output = model(img)

                loss = torch.nn.functional.cross_entropy(output, label)
                loss_list.append(loss.item())
                accuracy_list.append(accuracy(output, label).item())

            logger.add_scalar("val/loss", np.mean(loss_list), epoch)
            logger.add_scalar("val/accuracy", np.mean(accuracy_list), epoch)

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"classifier.th")

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