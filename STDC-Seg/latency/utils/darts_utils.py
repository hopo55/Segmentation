import os
import math
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import time
from tqdm import tqdm
from utils.genotypes import PRIMITIVES
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pdb import set_trace as bp
import warnings


# except:
#     warnings.warn("TensorRT (or pycuda) is not installed. compute_latency_ms_tensorrt() cannot be used.")
#########################################################################

def compute_latency_ms_pytorch(model, input_size, iterations=None, device=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model.eval()
    # model = model.cpu()
    # input = torch.randn(*input_size)
    input = torch.randn(*input_size).cuda()

    with torch.no_grad():
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        model(input)
        t_start = time.time()
        for _ in tqdm(range(iterations)):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start    # μs ()
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    # FPS = 1000 / latency (in ms)
    return latency

  