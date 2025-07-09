import argparse
import torch
import datetime
import json
import yaml
import os
from pathlib import Path

from data_preparation import Data_Preparation

from trainer import train_diffusion, train_gan, train_dl

from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from sklearn.model_selection import train_test_split


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="ECG Denoising")
    parser.add_argument("--exp_name", type=str, choices=[
        "DeScoD",
        "DRNN",
        "ECG_GAN",
        ""
    ], default="DRNN", help="Experiment name")
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--n_type', type=int, default=1, help='noise version')
    
    parser.add_argument('--val_interval', type=int, default=1)
    args = parser.parse_args()
    
    config_path = Path("./config") / f"{args.exp_name}.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    foldername = f"./check_points/{args.exp_name}/noise_type_" + str(args.n_type) + "/"
    print('folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.exp_name}/noise_type_" + str(args.n_type) + f"/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Load Data
    [X_train, y_train, X_test, y_test] = Data_Preparation(args.n_type)
    
    X_train = torch.FloatTensor(X_train)
    X_train = X_train.permute(0,2,1)
    
    y_train = torch.FloatTensor(y_train)
    y_train = y_train.permute(0,2,1)
    
    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0,2,1)
    
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0,2,1)
    
    train_val_set = TensorDataset(y_train, X_train)
    test_set = TensorDataset(y_test, X_test)
    
    train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.3)
    train_set = Subset(train_val_set, train_idx)
    val_set = Subset(train_val_set, val_idx)
    
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True)
    test_loader = DataLoader(test_set, batch_size=config['test']['batch_size'])
    
    
    # Load model
    print('Loading model...')
    # DeScoD-ECG
    if (args.exp_name == "DeScoD"):
        from Score_based_ECG_Denoising.main_model import DDPM
        from Score_based_ECG_Denoising.denoising_model_small import ConditionalModel
        
        base_model = ConditionalModel(config['train']['feats']).to(args.device)
        model = DDPM(base_model, config, args.device)
        train_diffusion(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
        
    # DRNN
    elif (args.exp_name == "DRNN"):
        from DRNN.model import DRDNN
        model = DRDNN(input_size=config['model']['input_size'],
                      hidden_size=config['model']['hidden_size'],
                      num_layers=config['model']['num_layers']).to(args.device)
        
        train_dl(model, config['train'], train_loader, args.device, 
        valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
        
    # ECG_GAN
    elif (args.exp_name == "ECG_GAN"):
        from ECG_GAN.model import Generator, Discriminator

        generator = Generator(input_channels=config['generator']['feats']).to(args.device)
        discriminator = Discriminator(input_channels=config['discriminator']['feats']).to(args.device)
        
        train_gan(generator, discriminator, config['train'], train_loader ,args.device,
                  valid_loader=val_loader, valid_epoch_interval=args.val_interval, foldername=foldername, log_dir=log_dir)
    
    
    #eval final
    # print('eval final')
    # evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    #eval best
    # print('eval best')
    # output_path = foldername + "/model.pth"
    # model.load_state_dict(torch.load(output_path))
    # evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    #don't use before final model is determined
    # print('eval test')
    # evaluate(model, test_loader, 1, args.device, foldername=foldername)
    
    
    
    
    
    
    
    
    
    
