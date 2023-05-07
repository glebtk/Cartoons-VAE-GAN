import yaml
import argparse
from tqdm import tqdm
from vae_utils import *
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from vae_model import VAE, Discriminator
import torchvision.transforms as transforms
from vae_dataset import CartoonFramesDataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter


def mse_loss(x_hat, x, reduction="mean"):
    return nn.functional.mse_loss(x_hat, x, reduction=reduction)


def kld_loss(mu, logvar):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld / mu.numel()


def calc_kld_weight(global_step, kld_weight_start, kld_weight_end, annealing_steps):
    if global_step < annealing_steps:
        return kld_weight_start + (kld_weight_end - kld_weight_start) * global_step / annealing_steps
    else:
        return kld_weight_end


def train(vae, vae_opt, disc, disc_opt, train_loader, test_loader, num_epochs, current_epoch=0, device="cpu", writer=None, train_name=None):
    vae_scaler = GradScaler()
    disc_scaler = GradScaler()

    for epoch in range(current_epoch, num_epochs):

        loss_names = ["disc_loss", "vae_adv_loss", "vae_mse_loss", "vae_kld_loss"]
        current_loss = {name: 0.0 for name in loss_names}

        for idx, real_images in enumerate(tqdm(train_loader)):
            real_images = real_images.to(device)

            global_step = (epoch * len(train_loader) + idx) * len(real_images)

            # -------- Обучение дискриминатора ----
            with autocast():
                reconstructed_images, _, _ = vae(real_images)

                real_validity = disc(real_images.detach())
                fake_validity = disc(reconstructed_images.detach())

                real_disc_loss = mse_loss(real_validity, torch.ones_like(real_validity))
                fake_disc_loss = mse_loss(fake_validity, torch.zeros_like(fake_validity))

            disc_loss = (real_disc_loss + fake_disc_loss) / 2
            current_loss["disc_loss"] += disc_loss.item()

            disc_opt.zero_grad()
            disc_scaler.scale(disc_loss).backward()
            disc_scaler.step(disc_opt)
            disc_scaler.update()

            # -------- Обучение VAE ---------------
            with autocast():
                reconstructed_images, mu, logvar = vae(real_images.detach())
                gen_validity = disc(reconstructed_images.detach())
                vae_adv_loss = mse_loss(gen_validity, torch.ones_like(gen_validity))
                vae_mse_loss = mse_loss(reconstructed_images, real_images)
                vae_kld_loss = kld_loss(mu, logvar)

            # Вычисление текущего веса KLD:
            kld_current_weight = calc_kld_weight(
                global_step,
                config.kld_weight_start,
                config.kld_weight,
                config.kld_annealing_steps
            )

            vae_loss = (
                    vae_mse_loss * config.mse_weight +
                    vae_kld_loss * kld_current_weight +
                    vae_adv_loss * config.adv_weight
            )

            current_loss["vae_mse_loss"] += vae_mse_loss.item()
            current_loss["vae_kld_loss"] += vae_kld_loss.item()
            current_loss["vae_adv_loss"] += vae_adv_loss.item()

            vae_opt.zero_grad()
            vae_scaler.scale(vae_loss).backward()
            vae_scaler.step(vae_opt)
            vae_scaler.update()

            # Обновляем tensorboard:
            if writer is not None and idx % 64 == 0 and idx != 0:
                # Добавляем тестовые изображения:
                test_images = model_test(vae, test_loader, device)
                writer.add_image("Test images", test_images, global_step=global_step)

                # Добавляем текущие изображения на TensorBoard
                current_images = process_and_make_grid(reconstructed_images[:16].detach())
                writer.add_image("Current images", current_images, global_step=global_step)

                # Добавляем loss:
                for loss_name, loss_value in current_loss.items():
                    writer.add_scalar(loss_name, loss_value, global_step=global_step)

            # Обнуляем лоссы:
            current_loss = {loss_name: 0.0 for loss_name in current_loss}

        if config.save_model:
            print("\033[32m=> Saving checkpoint\033[0m")

            # Create directory for saving
            dir_name = get_current_time()
            dir_path = os.path.join(config.checkpoint_dir, dir_name)
            make_directory(dir_path)

            # Save checkpoint
            model_path = os.path.join(dir_path, config.checkpoint_name)
            save_checkpoint(vae, vae_opt, disc, disc_opt, epoch, model_path)


def main():
    if config.seed:
        set_seed(config.seed)

    current_epoch = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = config.num_epochs

    # Инициализируем модели и оптимизаторы:
    weight_init = config.weights_init

    vae = VAE(embedding_size=config.embedding_size, weight_init=weight_init).to(device)
    vae_opt = optim.Adam(params=vae.parameters(), lr=config.vae_learning_rate)

    disc = Discriminator(in_channels=config.in_channels, weight_init=weight_init).to(device)
    disc_opt = optim.Adam(params=disc.parameters(), lr=config.disc_learning_rate)

    # Загружаем датасет:
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = CartoonFramesDataset(root_dir=config.dataset_path, transform=train_transform)
    test_dataset = CartoonFramesDataset(root_dir=config.test_path, transform=test_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Загружаем последний чекпоинт модели:
    if config.load_model:
        print("\033[32m=> Загрузка последнего чекпоинта\033[0m")

        checkpoint_path = get_last_checkpoint(config.checkpoint_dir, config.checkpoint_name)
        vae, vae_opt, disc, disc_opt, current_epoch = load_checkpoint(vae, vae_opt, disc, disc_opt, checkpoint_path, device)

    # Обучаем модель:
    train_name = config.train_name
    if not train_name:
        train_name = get_current_time()

    with SummaryWriter(f"./tb/{train_name}") as writer:
        train(vae, vae_opt, disc, disc_opt,
              train_loader, test_loader,
              num_epochs,
              current_epoch=current_epoch,
              device=device,
              writer=writer)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Train a VAE-GAN model")

    parser.add_argument("--in_channels", type=int, default=default_config["in_channels"], help="Number of input channels")
    parser.add_argument("--image_size", type=int, default=default_config["image_size"], help="Size of input images")
    parser.add_argument("--embedding_size", type=int, default=default_config["embedding_size"], help="Size of embeddings")
    parser.add_argument("--weights_init", type=str, default=default_config["weights_init"], help="Weights initialization method")
    parser.add_argument("--load_model", type=bool, default=default_config["load_model"], help="Load a pretrained model")
    parser.add_argument("--save_model", type=bool, default=default_config["save_model"], help="Save the model")
    parser.add_argument("--num_epochs", type=int, default=default_config["num_epochs"], help="Number of epochs")
    parser.add_argument("--vae_learning_rate", type=float, default=default_config["vae_learning_rate"], help="Learning rate for VAE")
    parser.add_argument("--disc_learning_rate", type=float, default=default_config["disc_learning_rate"], help="Learning rate for discriminator")
    parser.add_argument("--batch_size", type=int, default=default_config["batch_size"], help="Batch size")
    parser.add_argument("--mse_weight", type=float, default=default_config["mse_weight"], help="MSE loss weight")
    parser.add_argument("--kld_weight", type=float, default=default_config["kld_weight"], help="KLD loss weight")
    parser.add_argument("--adv_weight", type=float, default=default_config["adv_weight"], help="Adversarial loss weight")
    parser.add_argument("--kld_weight_start", type=float, default=default_config["kld_weight_start"], help="Starting KLD weight")
    parser.add_argument("--kld_annealing_steps", type=int, default=default_config["kld_annealing_steps"], help="KLD annealing steps")
    parser.add_argument("--num_workers", type=int, default=default_config["num_workers"], help="Number of workers")
    parser.add_argument("--dataset_path", type=str, default=default_config["dataset_path"], help="Path to the dataset")
    parser.add_argument("--test_path", type=str, default=default_config["test_path"], help="Path to the test dataset")
    parser.add_argument("--checkpoint_dir", type=str, default=default_config["checkpoint_dir"], help="Directory for saving checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default=default_config["checkpoint_name"], help="Checkpoint file name")
    parser.add_argument("--seed", type=int, default=default_config["seed"], help="Seed")
    parser.add_argument("--train_name", type=str, default=None, help="Train name for Tensorboard")

    config = parser.parse_args()

    main()

