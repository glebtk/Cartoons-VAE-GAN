import os
import cv2
import sys
import torch
import shutil
import random
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid


def make_directory(dir_path: str) -> None:
    """Создаёт директорию. Если директория существует - перезаписывает."""

    try:
        os.makedirs(dir_path)
    except FileExistsError:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


def save_checkpoint(vae, vae_opt, disc, disc_opt, epoch, model_path) -> None:
    """Сохраняет чекпоинт модели в процессе обучения (модель, оптимизатор, номер эпохи)."""
    checkpoint = {
        "vae_state_dict": vae.state_dict(),
        "vae_opt_state_dict": vae_opt.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "disc_opt_state_dict": disc_opt.state_dict(),
        "epoch": epoch
    }

    torch.save(checkpoint, model_path)


def load_checkpoint(vae, vae_opt, disc, disc_opt, checkpoint_file, device="cpu"):
    """Загружает чекпоинт модели. Возвращает модель, оптимизатор, номер эпохи"""

    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Ошибка: не удалось найти {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, vae.device)

    vae.load_state_dict(checkpoint["vae_state_dict"])
    disc.load_state_dict(checkpoint["disc_state_dict"])
    vae_opt.load_state_dict(checkpoint["vae_opt_state_dict"])
    disc_opt.load_state_dict(checkpoint["disc_opt_state_dict"])
    epoch = checkpoint["epoch"]

    return vae, vae_opt, disc, disc_opt, epoch


def get_last_checkpoint(checkpoint_dir, checkpoint_name) -> str:
    """Возвращает путь к последнему по времени сохранённому чекпоинту."""

    try:
        checkpoints = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
        if not checkpoints:
            raise IndexError
        checkpoints.sort(key=lambda l: os.path.getmtime(os.path.join(checkpoint_dir, l)))  # Сортировка по времени
        path_to_model = os.path.join(checkpoint_dir, checkpoints[-1], checkpoint_name)
        return path_to_model
    except IndexError:
        print(f"Ошибка: в директории {checkpoint_dir} нет сохраненных чекпоинтов")
        sys.exit(1)
    except FileNotFoundError:
        print(f'Ошибка: не удалось загрузить {checkpoint_name}')
        sys.exit(1)


def process_and_make_grid(images):
    images = postprocessing(images)
    images = np.array(images)  # Convert the list of numpy arrays to a single numpy array
    images = make_grid(torch.tensor(images), nrow=4, padding=5, pad_value=1).numpy()
    return images


def model_test(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        for idx, input_batch in enumerate(data_loader):
            input_batch = input_batch.to(device)
            output_batch, mu, logvar = model(input_batch)

            # Генерируем случайный шум и передаем его в декодер
            z = torch.randn_like(mu).to(device)
            random_batch = model.decoder(z)

    input_grid = process_and_make_grid(input_batch.detach())
    output_grid = process_and_make_grid(output_batch.detach())
    random_grid = process_and_make_grid(random_batch.detach())

    # Объединяем сетки изображений
    combined_grid = np.concatenate([input_grid, output_grid, random_grid], axis=1)

    model.train()

    return combined_grid


def postprocessing(tensors, image_size=(227, 128)):
    def denormalize(tensor):
        return tensor.mul_(0.5).add_(0.5)

    def tensor_to_image(tensor, size):
        tensor = denormalize(tensor.clone())
        img = tensor.numpy().transpose((1, 2, 0))
        img = (img * 255).astype(np.uint8)

        if size is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR).transpose((2, 0, 1))

        return img

    tensors = tensors.cpu()

    if len(tensors.shape) == 3:
        return [tensor_to_image(tensors, image_size)]
    elif len(tensors.shape) == 4:
        images = []
        for t in tensors:
            images.append(tensor_to_image(t, image_size))
        return images
    else:
        raise ValueError('Invalid input tensor shape. Must be 3 or 4 dimensions.')


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set as {seed}")
