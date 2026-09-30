"""Device setup and checkpoint loading shared by spatial experiments."""
from contextlib import nullcontext
import torch
from models.spatial_model import make_model


def amp_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def runtime(device, precision, threads=4):
    torch.set_num_threads(threads)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unsupported on selected GPU; choose fp16 or fp32")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_checkpoint(path, device):
    # Only load trusted locally produced checkpoints (optimizer/RNG state included).
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(ck["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


