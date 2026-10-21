"""Bypass the incompatible MoE converter only for unchanged attention-only keys."""
import re
from unittest.mock import patch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file

PATTERN=re.compile(r'base_model\.model\.model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.lora_[AB]\.weight')

def load_attention_adapter(model,path,name):
    saved=load_file(str(path/'adapter_model.safetensors'))
    assert saved and all(PATTERN.fullmatch(k) for k in saved),'Refuse non-attention or auxiliary adapter weights'
    def unchanged(model,peft_config,adapter_state_dict,adapter_name='default'):
        assert set(peft_config.target_modules)<={'q_proj','k_proj','v_proj','o_proj'}
        assert all(PATTERN.fullmatch(k) for k in adapter_state_dict)
        return adapter_state_dict
    # Transformers changed the expert WeightConverter API. Attention projection
    # names/shapes are identical across layouts, so these keys need no conversion.
    with patch('peft.utils.transformers_weight_conversion.convert_peft_adapter_state_dict_for_transformers',side_effect=unchanged):
        model.load_adapter(str(path),adapter_name=name,is_trainable=False)
    actual=get_peft_model_state_dict(model,adapter_name=name)
    assert set(actual)==set(saved)
    import torch
    for key,value in saved.items():torch.testing.assert_close(actual[key].cpu(),value,rtol=0,atol=0)
    return saved
