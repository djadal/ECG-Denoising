import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import metrics

def train_diffusion(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername="", log_dir=None):

    optimizer_config = config['optimizer']
    optimizer_type = getattr(optim, optimizer_config.get("type", "Adam"))
    optimizer = optimizer_type(model.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})

    #ema = EMA(0.9)
    #ema.register(model)
    
    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
    if config['lr_scheduler'].get("use", False):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=150, gamma=.1, verbose=True
        )
    
    best_valid_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                optimizer.zero_grad()
                
                loss = model(clean_batch, noisy_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                avg_loss += loss.item()
                
                #ema.update(model)
                
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": f"{avg_loss / batch_no:.4f}",
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )
            if lr_scheduler is not None:
                lr_scheduler.step()
        writer.add_scalar('Loss/Train', avg_loss / batch_no, epoch_no)
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        loss = model(clean_batch, noisy_batch)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": f"{avg_loss_valid / batch_no:.4f}",
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )
            
            writer.add_scalar('Loss/Validation', avg_loss_valid / batch_no, epoch_no)
            
            if best_valid_loss > avg_loss_valid/batch_no:
                best_valid_loss = avg_loss_valid/batch_no
                print("\n best loss is updated to ",avg_loss_valid / batch_no,"at Epoch", epoch_no+1)
                
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
    
    torch.save(model.state_dict(), final_path)


def train_gan(generator, discriminator, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername="", log_dir=None):
    
    optimizer_config = config['optimizer']
    optimizer_type = getattr(optim, optimizer_config.get("type", "RMSprop"))
    
    optimizer_G = optimizer_type(generator.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})
    optimizer_D = optimizer_type(discriminator.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})
    
    def lsgan_loss(pred, target):
        return torch.mean((pred - target) ** 2)
    
    if foldername != "":
        gen_output_path = foldername + "/generator.pth"
        disc_output_path = foldername + "/discriminator.pth"
        gen_final_path = foldername + "/generator_final.pth"
        disc_final_path = foldername + "/discriminator_final.pth"
    
    best_valid_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)
    lambda_l1 = config.get('lambda_l1', 100.0)
    
    for epoch_no in range(config["epochs"]):
        avg_g_loss = 0
        avg_d_loss = 0
        generator.train()
        discriminator.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                batch_size = clean_batch.shape[0]
                
                z = torch.randn(batch_size, 512, 8).to(device)
                denoised_ecg = generator(noisy_batch, z)
                
                optimizer_D.zero_grad()
                
                real_pair = torch.cat([clean_batch, noisy_batch], dim=1)
                real_pred = discriminator(real_pair)
                real_loss = lsgan_loss(real_pred, torch.ones_like(real_pred))
                
                fake_pair = torch.cat([denoised_ecg.detach(), noisy_batch], dim=1)
                fake_pred = discriminator(fake_pair)
                fake_loss = lsgan_loss(fake_pred, torch.zeros_like(fake_pred))
                
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_D.step()
                
                optimizer_G.zero_grad()
                
                fake_pair = torch.cat([denoised_ecg, noisy_batch], dim=1)
                fake_pred = discriminator(fake_pair)
                g_adv_loss = lsgan_loss(fake_pred, torch.ones_like(fake_pred))
                
                g_l1_loss = torch.nn.L1Loss()(denoised_ecg, clean_batch)
                
                g_loss = g_adv_loss + lambda_l1 * g_l1_loss
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer_G.step()
                
                avg_g_loss += g_loss.item()
                avg_d_loss += d_loss.item()
                
                it.set_postfix(
                    ordered_dict={
                        "avg_g_loss": f"{avg_g_loss / batch_no:.3f}",
                        "avg_d_loss": f"{avg_d_loss / batch_no:.3f}",
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )
        writer.add_scalar('Loss/Generator', avg_g_loss / batch_no, epoch_no)
        writer.add_scalar('Loss/Discriminator', avg_d_loss / batch_no, epoch_no)
        
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            generator.eval()
            discriminator.eval()
            avg_valid_loss = 0
            
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        batch_size = clean_batch.shape[0]
                        
                        z = torch.randn(batch_size, 512, 8).to(device)
                        denoised_ecg = generator(noisy_batch, z)
                        
                        valid_loss = torch.nn.L1Loss()(denoised_ecg, clean_batch)
                        avg_valid_loss += valid_loss.item()
                        
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_loss": f"{avg_valid_loss / batch_no:.3f}",
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )
            writer.add_scalar('Loss/Validation', avg_valid_loss / batch_no, epoch_no)
            
            if best_valid_loss > avg_valid_loss / batch_no:
                best_valid_loss = avg_valid_loss / batch_no
                print("\n best loss is updated to", f"{avg_valid_loss / batch_no:.4f}", "at Epoch", epoch_no + 1)
                
                if foldername != "":
                    torch.save(generator.state_dict(), gen_output_path)
                    torch.save(discriminator.state_dict(), disc_output_path)
    
    if foldername != "":
        torch.save(generator.state_dict(), gen_final_path)
        torch.save(discriminator.state_dict(), disc_final_path)    
    

def train_dl(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername="", log_dir=None):

    # optimizer config
    optimizer_config = config['optimizer']
    optimizer_type = getattr(optim, optimizer_config.get("type", "Adam"))
    optimizer = optimizer_type(model.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['type']})
    
    # criterion config
    criterion = config.get('criterion', 'MSELoss')
    if criterion == 'MSELoss':
        criterion = torch.nn.MSELoss()
    
    if foldername != "":
        output_path = foldername + "/model.pth"
        final_path = foldername + "/final.pth"
    
    # lr_scheduler config
    if config['lr_scheduler'].get("use", False):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=150, gamma=.1, verbose=True
        )
    else:
        lr_scheduler = None
    
    best_valid_loss = 1e15
    writer = SummaryWriter(log_dir=log_dir)
    
    # training loop
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                optimizer.zero_grad()
                
                denoised_batch = model(noisy_batch)
                loss = criterion(clean_batch, denoised_batch)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": f"{avg_loss / batch_no:.4f}",
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )
            if lr_scheduler is not None:
                lr_scheduler.step()
        
        writer.add_scalar('Loss/Train', avg_loss / batch_no, epoch_no)
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        denoised_batch = model(noisy_batch)
                        loss = criterion(clean_batch, denoised_batch)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": f"{avg_loss_valid / batch_no:.4f}",
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )
            
            writer.add_scalar('Loss/Validation', avg_loss_valid / batch_no, epoch_no)
            
            if best_valid_loss > avg_loss_valid/batch_no:
                best_valid_loss = avg_loss_valid/batch_no
                print("\n best loss is updated to ",f"{avg_loss_valid / batch_no:.4f}","at Epoch", epoch_no+1)
                
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
    
    torch.save(model.state_dict(), final_path)
    
    
   
class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict    
    
    
    
    
    