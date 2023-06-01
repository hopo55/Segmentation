from __future__ import division
import os
import torch
import numpy as np
from darts_utils import compute_latency_ms_pytorch as compute_latency
print("use PyTorch for latency test")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

=======
import random
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f
=======
import random
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f
=======
import random
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f
from network.sfnet_resnet import DeepR101_SF_deeply

def main():
    print("begin")
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Configuration ##############
    n_classes = 30
    p = 2
    q = 8
    inputDimension = (1, 3, 512, 672)

    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    model = DeepR101_SF_deeply(n_classes, criterion=None)

    print('loading parameters...')
    # model.load_state_dict(torch.load(save_pth))
    model = model.cuda()
    #####################################################

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    # latency, total_latency = compute_latency(model, inputDimension)
    latency = compute_latency(model, inputDimension, iterations=70)
    # latency = compute_latency(model, inputDimension, iterations=35564)

    methodName = 'SPNet'
    print("{} FPS:{:.2f}".format(methodName, (1000./latency)))
    print("{} Latency:{:.2f}ms / {:.4f}s".format(methodName, latency, (latency/1000.)))
=======
=======
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f
=======
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f
    latency_list = []
    fps_list = []

    numbers = list(range(1, 12346))  # 1부터 12345까지의 숫자 리스트 생성
    random_numbers = random.sample(numbers, 77)  # 리스트에서 10개의 무작위 숫자 추출

    for seed in random_numbers:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # latency, total_latency = compute_latency(model, inputDimension)
        latency = compute_latency(model, inputDimension, iterations=1)
        latency_list.append(latency)
        fps_list.append((1000./latency))

    avg_latency = np.average(latency_list)
    std_latency = np.std(latency_list)
    avg_fps = np.average(fps_list)
    std_fps = np.std(fps_list)
    print(avg_latency)
    print(std_latency)
    print(avg_fps)
    print(std_fps)
    len(latency_list)
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f
=======
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f
=======
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f

    # calculate FLOPS and params
    '''
    model = model.cpu()
    flops, params = profile(model, inputs=(torch.randn(inputDimension),), verbose=False)
    print("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    logging.info("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    '''

if __name__ == '__main__':
    main() 
