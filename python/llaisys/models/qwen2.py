import json
import ctypes
import numpy as np
import torch  # 仅用于权重加载和类型转换 (BF16 -> FP32)
from typing import Sequence, List
from pathlib import Path
from ..libllaisys import LIB_LLAISYS, DeviceType, LlaisysQwen2Meta

try:
    from safetensors import safe_open
except ImportError:
    raise ImportError("pip install safetensors")

class Qwen2:
    def __init__(self, model_path: str, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # 1. 读取 Config
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        with open(config_path, "r") as f:
            cfg = json.load(f)

        # 2. 配置 Meta (填充 C 结构体)
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = 0 # 0 表示 Float32
        self.meta.nlayer = cfg["num_hidden_layers"]
        self.meta.hs = cfg["hidden_size"]
        self.meta.nh = cfg["num_attention_heads"]
        self.meta.nkvh = cfg["num_key_value_heads"]
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = cfg["intermediate_size"]
        self.meta.maxseq = cfg.get("max_position_embeddings", 4096)
        self.meta.voc = cfg["vocab_size"]
        self.meta.epsilon = cfg["rms_norm_eps"]
        self.meta.theta = cfg.get("rope_theta", 1000000.0)
        
        # 处理 EOS Token
        self.meta.end_token = cfg.get("eos_token_id", 151643)
        if isinstance(self.meta.end_token, list):
            self.meta.end_token = self.meta.end_token[0]

        print(f"[LLaisys] Init Qwen2: {self.meta.nlayer}L, {self.meta.hs}H, {self.meta.nh}A/{self.meta.nkvh}KV")

        # 3. 创建 C++ 模型实例
        # 注意：使用 ctypes.byref 传递结构体指针
        self.handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), 
            device.value, 
            None, 
            0
        )
        
        if not self.handle:
            raise RuntimeError("Failed to create C++ Qwen2 Model")

        # 获取 C++ 端的权重容器指针
        self.c_weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.handle).contents

        # 4. 加载权重 (使用 PyTorch 辅助读取 BF16)
        self._load_weights(model_path)

    def _load_weights(self, path):
        """
        加载 Safetensors 权重。
        注意：这里使用 PyTorch 仅仅是因为 NumPy 不支持 bfloat16。
        我们不使用 PyTorch 做任何计算，只是做类型转换。
        """
        weight_files = sorted(list(path.glob("*.safetensors")))
        for f in weight_files:
            print(f"Loading {f.name}...")
            # 使用 framework="pt" 以支持 bfloat16 读取
            with safe_open(f, framework="pt", device="cpu") as st:
                for name in st.keys():
                    # 找到该权重在 C++ 结构体中的位置
                    ptr = self._route(name)
                    if ptr:
                        # 1. 读取为 PyTorch Tensor
                        tensor = st.get_tensor(name)
                        
                        # 2. 核心转换：BFloat16 -> Float32
                        # C++ 后端使用的是 float (FP32)，必须在这里转换
                        tensor = tensor.to(torch.float32)
                        
                        # 3. 转为 NumPy (内存连续)
                        data = tensor.numpy()
                        data = np.ascontiguousarray(data)
                        
                        # 4. 获取 C++ Tensor 的数据指针
                        dst = LIB_LLAISYS.llaisysTensorData(ptr)
                        
                        # 5. 内存拷贝 (Python -> C++)
                        if dst: 
                            ctypes.memmove(dst, data.ctypes.data, data.nbytes)

    def _route(self, name):
        """将 Safetensors 的参数名映射到 C++ 结构体成员"""
        w = self.c_weights
        
        # Global Weights
        if name == "model.embed_tokens.weight": return w.in_embed
        if name == "model.norm.weight": return w.out_norm_w
        if name == "lm_head.weight": return w.out_embed
        
        # Layer Weights
        if name.startswith("model.layers."):
            parts = name.split(".")
            idx = int(parts[2])
            if idx >= self.meta.nlayer: return None
            
            module = parts[3]
            sub = parts[4]
            is_bias = "bias" in parts[-1]

            if module == "self_attn":
                if sub == "q_proj": return w.attn_q_b[idx] if is_bias else w.attn_q_w[idx]
                if sub == "k_proj": return w.attn_k_b[idx] if is_bias else w.attn_k_w[idx]
                if sub == "v_proj": return w.attn_v_b[idx] if is_bias else w.attn_v_w[idx]
                if sub == "o_proj": return w.attn_o_w[idx] # Qwen o_proj 无 bias
            elif module == "mlp":
                if sub == "gate_proj": return w.mlp_gate_w[idx]
                if sub == "up_proj": return w.mlp_up_w[idx]
                if sub == "down_proj": return w.mlp_down_w[idx]
            elif module == "input_layernorm": return w.attn_norm_w[idx]
            elif module == "post_attention_layernorm": return w.mlp_norm_w[idx]
        return None

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.handle)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 20,
        **kwargs
    ) -> List[int]:
        """
        执行推理生成。
        逻辑：完全调用 C++ 接口，Python 不做任何数学运算。
        """
        curr = list(inputs)
        
        # === 阶段 1: Prefill (预填充) ===
        # 将完整的 Prompt 传入，C++ 端会计算并填充 KV Cache
        seq_len = len(curr)
        input_arr = (ctypes.c_int64 * seq_len)(*curr)
        
        # 调用 C++ 推理
        next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, input_arr, seq_len)
        curr.append(next_tok)
        
        # === 阶段 2: Decode (解码) ===
        # 循环生成，每次只传入 1 个 Token
        for _ in range(max_new_tokens - 1):
            if next_tok == self.meta.end_token:
                break
            
            # 准备输入：仅上一个 Token
            input_arr = (ctypes.c_int64 * 1)(next_tok)
            
            # seq_len=1，触发 C++ 端的 KV Cache 增量计算
            next_tok = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, input_arr, 1)
            curr.append(next_tok)
            
        return curr