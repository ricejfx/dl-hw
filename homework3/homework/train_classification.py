import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

try:
    from .models import ClassificationLoss, load_model, save_model
except:
    from models import ClassificationLoss, load_model, save_model

try:
    from .utils import load_data
except:
    from utils import load_data

import tqdm
import os

def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs
):
    if torch.cuda.is_available():
        print("CUDA available, using GPU")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    print(f"{os.cpu_count()=}")
    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    grader_run = True

    if not grader_run:
        # encode additional kwargs
        if model_name == "linear":
            pass
        if model_name == "mlp":
            pass
        if model_name == "mlp_deep":
            kwargs["optimizer"] = "sgd"
            kwargs["regularizer"] = 0.0
            kwargs["first_layer_dim"] = 128
            kwargs["hidden_dim"] = [128, 64, 32]
        if model_name == "mlp_deep_residual":
            kwargs["regularizer"] = 0.0
            kwargs["optimizer"] = "sgd"
            kwargs["first_layer_dim"] = 128
            kwargs["hidden_dim"] = [512, 128, 64]

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    #print(model)

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=10)
    val_data = load_data("classification_data/val", shuffle=False, num_workers=10)

    # create loss function and optimizer
    loss_func = ClassificationLoss()

    if not grader_run:
        if kwargs["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kwargs["regularizer"])
        elif kwargs["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=kwargs["regularizer"])

    logger.add_text("Arguments", str({"exp_dir": exp_dir, "model_name": model_name, "num_epoch": num_epoch, "lr": lr, "batch_size": batch_size, "seed": seed}))
    logger.add_text("Model kwargs", str(kwargs))

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in tqdm.tqdm(range(num_epoch), desc="Epoch"):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # TODO: implement training step
            output = model(img)
            optimizer.zero_grad()
            loss_value = loss_func(output, label)
            loss_value.backward()
            optimizer.step()
            
            logger.add_scalar("train_loss", loss_value.item(), global_step)
            metrics["train_acc"].append((output.argmax(dim=1) == label).float().mean())
            #raise NotImplementedError("Training step not implemented")
            
            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                output = model(img)
                metrics["val_acc"].append((output.argmax(dim=1) == label).float().mean())
                #raise NotImplementedError("Validation accuracy not implemented")

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        logger.add_scalar("val_accuracy", epoch_val_acc, global_step)

        # print on first, last, every 10th epoch
        if 1 or (epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 2 == 0):
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=128)
    #parser.add_argument("--regularizer", type=float, default=0.0)
    #parser.add_argument("--hidden_dim", type=int, default=128)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
