import csv
import torch
import visdom
import numpy as np
import argparse
import os

from torchinfo import summary
from engine.trainer import train
from config.set_params import params as sp
from modeling.model import HARmodel
from utils.build_dataset import build_dataloader

def main():
    """Driver file for training HAR model."""

    # Solve error in missing engine in quantization
    torch.backends.quantized.engine = 'qnnpack'

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
    )
    args = parser.parse_args()

    params = sp().params
    model = HARmodel(params["input_dim"], params["num_classes"])

    model = torch.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8)


    repr(model) # print +info about model
    print(model)
    # summary(model2, input_size=(1,120,1))
    # print(model)

    if params["use_cuda"]:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    if params["use_cuda"]:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=params["lr"],
                                momentum=params["momentum"],
                                weight_decay=params["weight_decay"])
    
    params["start_epoch"] = 1

    # If checkpoint path is given, load checkpoint data
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))

            checkpoint = torch.load(args.checkpoint)
            params["start_epoch"] = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.checkpoint, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(params["resume"]))

    train_loader, val_loader = build_dataloader(params["root"], params)
    print(len(train_loader))
    # print("DEBUG >>>>>>>>>>>>>>>>>>>>>>>")
    # print(train_loader)
    # print(val_loader)

    logger = visdom.Visdom()

    train(train_loader=train_loader,
          val_loader=val_loader,
          model=model,
          criterion=criterion,
          optimizer=optimizer,
          params=params,
          logger=logger,
          )

if __name__ == "__main__":
    main()