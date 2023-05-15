from __future__ import division
import os
import torch
import numpy as np
from darts_utils import compute_latency_ms_pytorch as compute_latency
print("use PyTorch for latency test")

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
    inputDimension = (1, 3, 512, 1024)

    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    model = ESPNet(n_classes, p, q)

    print('loading parameters...')
    weightsDir = 'pretrained/'
    save_pth = weightsDir + os.sep + 'decoder' + os.sep + 'espnet_p_' + str(p) + '_q_' + str(q) + '.pth'
    model.load_state_dict(torch.load(save_pth))
    model = model.cuda()
    #####################################################

    # latency, total_latency = compute_latency(model, inputDimension)
    # latency = compute_latency(model, inputDimension, iterations=70)
    latency = compute_latency(model, inputDimension, iterations=35564)

    methodName = 'ESPNet'
    print("{} FPS:{:.2f}".format(methodName, (1000./latency)))
    print("{} Latency:{:.2f}ms / {:.4f}s".format(methodName, latency, (latency/1000.)))

    # calculate FLOPS and params
    '''
    model = model.cpu()
    flops, params = profile(model, inputs=(torch.randn(inputDimension),), verbose=False)
    print("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    logging.info("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    '''

if __name__ == '__main__':
    main() 
