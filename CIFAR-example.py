import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR100


if __name__ == "__main__":
    # use new transform
    NEW = True

    if NEW:
        transform = transforms.ToTensor()
        script_transform = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        )
        script_transform = torch.jit.script(script_transform)
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset = CIFAR100('/dataset/CIFAR', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    st = time.time()
    for _ in tqdm(range(200)):
        for i, (x, l) in enumerate(dataloader):
            x = x.cuda()
            if NEW:
                script_transform(x)
    et = time.time()
    print(et-st)
