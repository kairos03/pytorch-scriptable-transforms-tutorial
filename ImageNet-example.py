import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageNet


if __name__ == "__main__":
    # use new transform
    NEW = False

    if NEW:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
        ])
        script_transform = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        script_transform = torch.jit.script(script_transform)
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    dataset = ImageNet('/dataset/ImageNet', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    st = time.time()
    for _ in tqdm(range(200)):
        for i, (x, l) in enumerate(dataloader):
            x = x.cuda()
            if NEW:
                script_transform(x)
            if i >= 10:
                break
    et = time.time()
    print(et-st)
