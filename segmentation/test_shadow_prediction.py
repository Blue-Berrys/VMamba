import torch
import numpy as np
from mmseg.apis import init_model
from mmseg.utils import register_all_modules

register_all_modules()

# Load model
config_path = 'configs/sbu/vssm_tiny_sbu_fixed_40k.py'
checkpoint_path = None  # Not using any checkpoint

model = init_model(config_path, checkpoint=None, device='cuda:0')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 416, 416).cuda()

# Get prediction
with torch.no_grad():
    output = model.infer(dummy_input, data_samples=None, mode='tensor')
    pred = output.argmax(dim=1)

print(f'Prediction shape: {pred.shape}')
print(f'Unique values in prediction: {torch.unique(pred)}')
print(f'Number of shadow pixels: {(pred == 1).sum().item()}')
print(f'Number of non-shadow pixels: {(pred == 0).sum().item()}')
print(f'Shadow ratio: {(pred == 1).sum().item() / pred.numel():.4f}')

# Check raw logits
from mmseg.models.segmentors import EncoderDecoder
if isinstance(model, EncoderDecoder):
    with torch.no_grad():
        features = model.backbone(dummy_input)
        decode_output = model.decode_head.forward_test(features)
        if hasattr(decode_output, 'keys'):
            logits = decode_output[0]
        else:
            logits = decode_output
        print(f'Logits shape: {logits.shape}')
        print(f'Logits min/max: {logits.min():.4f}/{logits.max():.4f}')
        print(f'Logits mean: {logits.mean():.4f}')
