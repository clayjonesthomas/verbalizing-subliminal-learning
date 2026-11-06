"""Regression: merging must not round positional buffers to BF16."""
import os
import tempfile
import torch
from peft import get_peft_model, LoraConfig

with tempfile.TemporaryDirectory() as directory:
    os.environ['PODRUN_DIR'] = directory
    from merge_student import merge_weights

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
            self.register_buffer('freq', torch.tensor([0.123456789], dtype=torch.float32))

        def forward(self, x):
            return self.proj(x)

    model = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, target_modules=['proj']))
    layer = model.base_model.model.proj
    with torch.no_grad():
        layer.lora_B.default.weight.normal_()
    frequencies = model.base_model.model.freq.clone()
    expected = (layer.weight.float() + layer.get_delta_weight('default').float()).bfloat16()
    merged = merge_weights(model)
    assert torch.equal(merged.proj.weight, expected)
    assert torch.equal(merged.freq, frequencies) and merged.freq.dtype == torch.float32
    assert not any('lora_' in name for name, _ in merged.named_parameters())
print('PASS: FP32 merge, BF16 weights, unchanged FP32 buffers, adapters removed')
