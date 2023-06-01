from __future__ import division

import os
import sys
import random
import logging
import torch
import numpy as np

from thop import profile
sys.path.append("./")

# from utils.darts_utils import create_exp_dir, plot_op, plot_path_width, objective_acc_lat
# try:
    # from utils.darts_utils import compute_latency_ms_tensorrt as compute_latency
#     print("use TensorRT for latency test")
# except:
#     from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
#     print("use PyTorch for latency test")
from utils.darts_utils import compute_latency_ms_pytorch as compute_latency
print("use PyTorch for latency test")

from models.model_stages_trt import BiSeNet

def main():
    
    print("begin")
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Configuration ##############
    use_boundary_2 = False
    use_boundary_4 = False
    use_boundary_8 = True
    use_boundary_16 = False
    use_conv_last = False
    n_classes = 19
    
    # STDC1Seg-50 250.4FPS on NVIDIA GTX 1080Ti
    backbone = 'STDCNet813'
    methodName = 'STDC1-Seg'
    inputSize = 512
    inputScale = 50
    inputDimension = (1, 3, 512, 672)

    # # STDC1Seg-75 126.7FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet813'
    # methodName = 'STDC1-Seg'
    # inputSize = 768
    # inputScale = 75
    # inputDimension = (1, 3, 768, 1536)

    # # STDC2Seg-50 188.6FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet1446'
    # methodName = 'STDC2-Seg'
    # inputSize = 512
    # inputScale = 50
    # inputDimension = (1, 3, 512, 1024)

    # # STDC2Seg-75 97.0FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet1446'
    # methodName = 'STDC2-Seg'
    # inputSize = 768
    # inputScale = 75
    # inputDimension = (1, 3, 768, 1536)

    latency_list = []
    fps_list = []
    
    model = BiSeNet(backbone=backbone, n_classes=n_classes, 
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
    use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
    input_size=inputSize, use_conv_last=use_conv_last)

    print('loading parameters...')
    respth = './checkpoints/{}/'.format(methodName)
    save_pth = os.path.join(respth, 'model_maxmIOU{}.pth'.format(inputScale))
    model.load_state_dict(torch.load(save_pth))
    model = model.cuda()
    #####################################################

<<<<<<< HEAD
<<<<<<< HEAD
    # latency, total_latency = compute_latency(model, inputDimension)
    latency = compute_latency(model, inputDimension, iterations=70)
    # latency = compute_latency(model, inputDimension, iterations=35564)
=======
    numbers = list(range(1, 12346))  # 1부터 12345까지의 숫자 리스트 생성
    random_numbers = random.sample(numbers, 77)  # 리스트에서 10개의 무작위 숫자 추출
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f
=======
    numbers = list(range(1, 12346))  # 1부터 12345까지의 숫자 리스트 생성
    random_numbers = random.sample(numbers, 77)  # 리스트에서 10개의 무작위 숫자 추출
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f

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
        print("{} Latency:{:.2f}ms / {:.4f}s".format(methodName, latency, (latency/1000.)))

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
