import time
import torch
from tqdm import tqdm

def compute_latency_ms_pytorch(model, input_size, iterations=None, device=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model.eval()
    # model = model.cpu()
    # input = torch.randn(*input_size)
    model = model.cuda()
    input = torch.randn(*input_size).cuda()

    with torch.no_grad():
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
<<<<<<< HEAD
<<<<<<< HEAD
=======
        model(input)
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f
=======
        model(input)
>>>>>>> 61a40ddc70810f9a788ef56cfd1db70ead09137f
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