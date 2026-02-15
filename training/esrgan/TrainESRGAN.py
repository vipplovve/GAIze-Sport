import os
import argparse
import time
import threading

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from ESRGANModel import RRDBNet, VGGDiscriminator
from LossFunctions import ESRGANLoss, PSNRLoss
from DatasetLoader import DatasetLoader

def get_config():
    print("=" * 60)
    print(" ESRGAN Training Configuration")
    print("=" * 60)

    phase = int(input("Training phase (1=PSNR, 2=GAN) [1]: ").strip() or "1")
    hr_dir = input("HR images directory [data/hr_frames]: ").strip() or "data/hr_frames"
    epochs = int(input("Number of epochs [50]: ").strip() or "50")
    batch_size = int(input("Batch size [4]: ").strip() or "4")
    lr = float(input("Learning rate [0.0002]: ").strip() or "0.0002")
    hr_crop = int(input("HR crop size [128]: ").strip() or "128")
    num_blocks = int(input("Number of RRDB blocks [8]: ").strip() or "8")
    pretrained_gen = input("Pretrained generator path (leave empty for none): ").strip() or None
    checkpoint_dir = input("Checkpoint directory [checkpoints]: ").strip() or "checkpoints"
    save_every = int(input("Save checkpoint every N epochs [5]: ").strip() or "5")
    num_workers = int(input("DataLoader workers [0]: ").strip() or "0")

    config = argparse.Namespace(
        phase=phase, hr_dir=hr_dir, epochs=epochs, batch_size=batch_size,
        lr=lr, hr_crop=hr_crop, num_blocks=num_blocks,
        pretrained_gen=pretrained_gen, checkpoint_dir=checkpoint_dir,
        save_every=save_every, num_workers=num_workers
    )

    print("\n" + "-" * 40)
    print(f"  Phase:        {config.phase}")
    print(f"  HR Dir:       {config.hr_dir}")
    print(f"  Epochs:       {config.epochs}")
    print(f"  Batch Size:   {config.batch_size}")
    print(f"  LR:           {config.lr}")
    print(f"  HR Crop:      {config.hr_crop}")
    print(f"  RRDB Blocks:  {config.num_blocks}")
    print(f"  Pretrained:   {config.pretrained_gen or 'None'}")
    print(f"  Checkpoints:  {config.checkpoint_dir}")
    print("-" * 40)

    confirm = input("\nProceed? (y/n) [y]: ").strip().lower() or "y"
    if confirm != "y":
        print("Aborted.")
        exit(0)

    return config

def train_phase1(args, log=print, stop_event=None):
    log("=" * 60)
    log(" ESRGAN Phase 1: PSNR Pre-training")
    log("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    dataset = DatasetLoader(args.hr_dir, hr_crop_size=args.hr_crop, scale_factor=4)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)

    generator = RRDBNet(num_blocks=args.num_blocks).to(device)
    if args.pretrained_gen:
        generator.load_state_dict(torch.load(args.pretrained_gen, map_location=device))
        log(f"Loaded pretrained generator from {args.pretrained_gen}")

    criterion = PSNRLoss().to(device)
    optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if stop_event and stop_event.is_set():
            log("Training stopped by user.")
            break

        generator.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = generator(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time

        log(f"Epoch [{epoch}/{args.epochs}] | L1 Loss: {avg_loss:.6f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")

        if epoch % args.save_every == 0:
            path = os.path.join(args.checkpoint_dir, f"phase1_gen_epoch{epoch}.pth")
            torch.save(generator.state_dict(), path)
            log(f"  -> Saved checkpoint: {path}")

    final_path = os.path.join(args.checkpoint_dir, "phase1_gen.pth")
    torch.save(generator.state_dict(), final_path)
    log(f"Phase 1 complete! Final model: {final_path}")
    return final_path

def train_phase2(args, log=print, stop_event=None):
    log("=" * 60)
    log(" ESRGAN Phase 2: GAN Fine-tuning")
    log("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    dataset = DatasetLoader(args.hr_dir, hr_crop_size=args.hr_crop, scale_factor=4)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)

    generator = RRDBNet(num_blocks=args.num_blocks).to(device)
    discriminator = VGGDiscriminator().to(device)

    if args.pretrained_gen:
        generator.load_state_dict(torch.load(args.pretrained_gen, map_location=device))
        log(f"Loaded Phase 1 generator from {args.pretrained_gen}")
    else:
        log("WARNING: No pretrained generator loaded. Starting GAN training from scratch.")

    criterion = ESRGANLoss().to(device)

    opt_gen = optim.Adam(generator.parameters(), lr=args.lr * 0.5, betas=(0.9, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.9, 0.999))

    scheduler_gen = CosineAnnealingLR(opt_gen, T_max=args.epochs, eta_min=1e-7)
    scheduler_disc = CosineAnnealingLR(opt_disc, T_max=args.epochs, eta_min=1e-7)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if stop_event and stop_event.is_set():
            log("Training stopped by user.")
            break

        generator.train()
        discriminator.train()

        g_loss_total = 0.0
        d_loss_total = 0.0
        start_time = time.time()

        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = generator(lr_imgs)

            for p in discriminator.parameters():
                p.requires_grad = True

            real_pred = discriminator(hr_imgs)
            fake_pred = discriminator(sr_imgs.detach())

            d_loss = criterion.discriminator_loss(real_pred, fake_pred)

            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

            for p in discriminator.parameters():
                p.requires_grad = False

            real_pred = discriminator(hr_imgs).detach()
            fake_pred = discriminator(sr_imgs)

            g_loss, loss_dict = criterion.generator_loss(sr_imgs, hr_imgs, real_pred, fake_pred)

            opt_gen.zero_grad()
            g_loss.backward()
            opt_gen.step()

            g_loss_total += loss_dict['total']
            d_loss_total += d_loss.item()

        scheduler_gen.step()
        scheduler_disc.step()

        avg_g = g_loss_total / len(dataloader)
        avg_d = d_loss_total / len(dataloader)
        elapsed = time.time() - start_time

        log(f"Epoch [{epoch}/{args.epochs}] | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f} | "
            f"Time: {elapsed:.1f}s")

        if epoch % args.save_every == 0:
            gen_path = os.path.join(args.checkpoint_dir, f"phase2_gen_epoch{epoch}.pth")
            disc_path = os.path.join(args.checkpoint_dir, f"phase2_disc_epoch{epoch}.pth")
            torch.save(generator.state_dict(), gen_path)
            torch.save(discriminator.state_dict(), disc_path)
            log(f"  -> Saved: {gen_path}, {disc_path}")

    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, "esrgan_generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, "esrgan_discriminator.pth"))
    log(f"Phase 2 complete! Final models saved in {args.checkpoint_dir}/")

def train_from_gui(config_dict, log_callback=print, stop_event=None):
    """Entry point for GUI-based training."""
    args = argparse.Namespace(**config_dict)
    if not hasattr(args, 'save_every'):
        args.save_every = 5
    if not hasattr(args, 'num_workers'):
        args.num_workers = 0
    if not hasattr(args, 'pretrained_gen'):
        args.pretrained_gen = None

    if args.phase == 1:
        return train_phase1(args, log=log_callback, stop_event=stop_event)
    else:
        return train_phase2(args, log=log_callback, stop_event=stop_event)

if __name__ == "__main__":
    config = get_config()

    if config.phase == 1:
        train_phase1(config)
    else:
        train_phase2(config)
