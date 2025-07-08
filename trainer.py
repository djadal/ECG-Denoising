import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import pickle
import metrics

def train(model, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername=""):

    optimizer_config = config['optimizer']
    optimizer_type = getattr(optim, optimizer_config.get("type", "Adam"))
    optimizer = optimizer_type(model.parameters(), **{k: v for k, v in optimizer_config.items() if k not in ['name']})

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
            
            if best_valid_loss > avg_loss_valid/batch_no:
                best_valid_loss = avg_loss_valid/batch_no
                print("\n best loss is updated to ",avg_loss_valid / batch_no,"at Epoch", epoch_no+1)
                
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
    
    torch.save(model.state_dict(), final_path)


def train_gan(generator, discriminator, config, train_loader, device, valid_loader=None, valid_epoch_interval=5, foldername=""):
    
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
            
            if best_valid_loss > avg_valid_loss / batch_no:
                best_valid_loss = avg_valid_loss / batch_no
                print("\n best loss is updated to", avg_valid_loss / batch_no, "at Epoch", epoch_no + 1)
                
                if foldername != "":
                    torch.save(generator.state_dict(), gen_output_path)
                    torch.save(discriminator.state_dict(), disc_output_path)
    
    if foldername != "":
        torch.save(generator.state_dict(), gen_final_path)
        torch.save(discriminator.state_dict(), disc_final_path)    
    

def evaluate(model, test_loader, shots, device, foldername=""):
    ssd_total = 0
    mad_total = 0
    prd_total = 0
    cos_sim_total = 0
    snr_noise = 0
    snr_recon = 0
    snr_improvement = 0
    eval_points = 0
    
    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
            
            if shots > 1:
                output = 0
                for i in range(shots):
                    output+=model.denoising(noisy_batch)
                output /= shots
            else:
                output = model.denoising(noisy_batch) #B,1,L
            clean_batch = clean_batch.permute(0, 2, 1)
            noisy_batch = noisy_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1) #B,L,1
            out_numpy = output.cpu().detach().numpy()
            clean_numpy = clean_batch.cpu().detach().numpy()
            noisy_numpy = noisy_batch.cpu().detach().numpy()
            
            
            eval_points += len(output)
            ssd_total += np.sum(metrics.SSD(clean_numpy, out_numpy))
            mad_total += np.sum(metrics.MAD(clean_numpy, out_numpy))
            prd_total += np.sum(metrics.PRD(clean_numpy, out_numpy))
            cos_sim_total += np.sum(metrics.COS_SIM(clean_numpy, out_numpy))
            snr_noise += np.sum(metrics.SNR(clean_numpy, noisy_numpy))
            snr_recon += np.sum(metrics.SNR(clean_numpy, out_numpy))
            snr_improvement += np.sum(metrics.SNR_improvement(noisy_numpy, out_numpy, clean_numpy))
            restored_sig.append(out_numpy)
            
            it.set_postfix(
                ordered_dict={
                    "ssd_total": ssd_total/eval_points,
                    "mad_total": mad_total/eval_points,
                    "prd_total": prd_total/eval_points,
                    "cos_sim_total": cos_sim_total/eval_points,
                    "snr_in": snr_noise/eval_points,
                    "snr_out": snr_recon/eval_points,
                    "snr_improve": snr_improvement/eval_points,
                },
                refresh=True,
            )
    
    restored_sig = np.concatenate(restored_sig)
    
    #np.save(foldername + '/denoised.npy', restored_sig)
    
    print("ssd_total: ",ssd_total/eval_points)
    print("mad_total: ", mad_total/eval_points,)
    print("prd_total: ", prd_total/eval_points,)
    print("cos_sim_total: ", cos_sim_total/eval_points,)
    print("snr_in: ", snr_noise/eval_points,)
    print("snr_out: ", snr_recon/eval_points,)
    print("snr_improve: ", snr_improvement/eval_points,)
    
    
   
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
    
    
    
    
    