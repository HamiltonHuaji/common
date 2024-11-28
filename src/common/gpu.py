from common.imports import *

def lock_device(device, *, margin_size = 128 * 1024 * 1024):
    if isinstance(device, str):
        if device == "cpu":
            return
        else:
            device = torch.device(device)
    elif isinstance(device, torch.device):
        if device.type != "cuda":
            return
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif device is None:
        device = torch.cuda.current_device()

    # Step 1: get the available memory in bytes
    free_bytes, _ = torch.cuda.mem_get_info(device)

    # Step 2: occupy the memory
    if free_bytes > margin_size:
        try:
            print(f"Free GBs={free_bytes / 1024 ** 3}")
            x = torch.randn((free_bytes - margin_size) // 4, device=device, dtype=torch.float32)
            time.sleep(.01)
            del x
            gc.collect()
        except Exception as e:
            print(f"Warning: failed to lock device: {e}")
    else:
        print("Warning: not enough memory (less than margin_size)")

# deprecated
def occupy_all(device_idx: int, *, margin_size = 128 * 1024 * 1024):
    import gc
    if isinstance(device_idx, str):
        if device_idx == "cpu":
            return
        device_idx = torch.device(device_idx).index
    elif isinstance(device_idx, torch.device):
        if device_idx.type != "cuda":
            return
        device_idx = device_idx.index
    
    if device_idx is None:
        free_bytes, _ = torch.cuda.mem_get_info()
        device = torch.device("cuda")
    else:
        assert isinstance(device_idx, int), f"device_idx must be int, but got {device_idx}({type(device_idx)})"
        import pynvml
        # Step 1: get the available memory in bytes
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_bytes = info.free
        pynvml.nvmlShutdown()
        device = torch.device(f"cuda:{device_idx}")

    # Step 2: occupy the memory
    
    if free_bytes > margin_size:
        try:
            print(f"Free GBs={free_bytes / 1024 ** 3}")
            x = torch.randn((free_bytes - margin_size) // 4, device=device, dtype=torch.float32)
            time.sleep(.01)
            del x
            gc.collect()
        except Exception as e:
            print(f"Warning: occupy_all failed {e}")
    else:
        print("Warning: not enough memory (less than margin_size)")
    return
