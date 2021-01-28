# pytorch-scriptable-transforms-tutorial

torchvision 0.8's Highlight Feature!  
Scriptable Transforms enhancing your dataloading speed!

Experiment conducted on below enviroments
- NVIDIA RTX-2080ti x 1
- Intel(R) Core(TM) i9-7900X @ 3.30GHz, vCPU 4 core
- Memory 16G
- NVIDIA-Docker
- Pytorch 1.7.1
- Torchvision 0.8.2
- nvidia Driver 440.100
- CUDA 10.1
- cuDNN 7.6

### CIFAR
loading speed 66% enhnaced
| original | scriptable |
| --- | --- |
| 704.09s | 479.32s|

Batch_size 64, num_workers 4, total 200 epoch.  
Conduct three times and an average of them.

### ImageNet
8.5 hours faster
| original | scriptable |
| --- | --- |
| 221.43h | 213.92h |

Batch_size 64, num_workers 4, total 200 epoch. Conduct three times and an average of them.  
Only run 10 bathes at each epoch because it takes too long time to run full batch.   
Above table is approximation value when total batch is 20019.

https://bongjasee.tistory.com/2
