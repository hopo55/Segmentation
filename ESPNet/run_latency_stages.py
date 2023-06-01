from __future__ import division
import os
import torch
import numpy as np
from darts_utils import compute_latency_ms_pytorch as compute_latency
print("use PyTorch for latency test")
import random
from Model import ESPNet

def main():
    print("begin")
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Configuration ##############
    n_classes = 20
    p = 2
    q = 8
    inputDimension = (1, 3, 512, 672)
    
    model = ESPNet(n_classes, p, q)

    print('loading parameters...')
    weightsDir = 'pretrained/'
    save_pth = weightsDir + os.sep + 'decoder' + os.sep + 'espnet_p_' + str(p) + '_q_' + str(q) + '.pth'
    model.load_state_dict(torch.load(save_pth))
    #####################################################

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

    # calculate FLOPS and params
    '''
    model = model.cpu()
    flops, params = profile(model, inputs=(torch.randn(inputDimension),), verbose=False)
    print("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    logging.info("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    '''

if __name__ == '__main__':
    main() 
