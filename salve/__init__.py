import torch
from dotenv import load_dotenv

# Turn off cuDNN's fused attention: on an H100, a soft prompt's bf16 backward
# pass through it gave NaNs on padded batches.
torch.backends.cuda.enable_cudnn_sdp(False)

load_dotenv()
