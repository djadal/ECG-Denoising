import argparse

import torch
import torch.optim as optim
from model import Generator, Discriminator

def train(args=None):

    generator = Generator(input_channels=1)
    discriminator = Discriminator(input_channels=2)

    optimizer_G = optim.RMSprop(generator.parameters(), lr=0.0002)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=0.0002)

    def lsgan_loss(pred, target):
        return torch.mean((pred - target) ** 2)

    for epoch in range(args.num_epochs):
        for clean_ecg, noisy_ecg in train_loader:
            z = torch.randn(clean_ecg.shape[0], 4, 1024).to(args.device)

            denoised_ecg = generator(noisy_ecg, z)
    
            optimizer_D.zero_grad()

            real_pair = torch.cat([clean_ecg, noisy_ecg], dim=1)
            real_loss = lsgan_loss(discriminator(real_pair), torch.ones_like)

            fake_pair = torch.cat([clean_ecg, denoised_ecg.detach()], dim=1)
            fake_loss = lsgan_loss(discriminator(fake_pair), torch.zeros_like)
            
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            
            optimizer_G.zero_grad()

            fake_pair = torch.cat([clean_ecg, denoised_ecg], dim=1)
            g_adv_loss = lsgan_loss(discriminator(fake_pair), torch.ones_like)

            g_l1_loss = torch.nn.L1Loss()(denoised_ecg, clean_ecg)
            
            g_loss = g_adv_loss + args._lambda * g_l1_loss
            g_loss.backward()
            optimizer_G.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ECG-GAN")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--lambda", type=float, default=100.0, help="Weight for L1 loss")
    parser.add_argument("--num_epochs", type=int, default=70, help="Number of epochs to train")
    args = parser.parse_args()


    train(args)