import numpy as np
from config import NUM_BINS , VALUE_MIN, VALUE_MAX
# 导入 transformers（必需）
from transformers import (
    SiglipVisionModel,
    SiglipImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    GemmaForCausalLM,
)
from typing import List, Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn


# ============================================
# Value 离散化工具函数
# ============================================
def value_to_bin(value: float) -> int:
    """将连续 value 转换为 bin 索引"""
    value = np.clip(value, VALUE_MIN, VALUE_MAX)
    bin_idx = int((value - VALUE_MIN) / (VALUE_MAX - VALUE_MIN) * (NUM_BINS - 1))
    return np.clip(bin_idx, 0, NUM_BINS - 1)


def bin_to_value(bin_idx: int) -> float:
    """将 bin 索引转换回连续 value"""
    return VALUE_MIN + (bin_idx / (NUM_BINS - 1)) * (VALUE_MAX - VALUE_MIN)


def bins_to_value_soft(logits: torch.Tensor) -> torch.Tensor:
    """使用 softmax 加权求和将 logits 转换为连续 value"""
    probs = F.softmax(logits, dim=-1)
    bin_values = torch.linspace(VALUE_MIN, VALUE_MAX, NUM_BINS, device=logits.device)
    return torch.sum(probs * bin_values, dim=-1)

# ============================================
# 基于 SigLIP + Gemma 3 的 Value Function
# ============================================
# SigLIP 模型选择（按大小排列）
SIGLIP_MODELS = {
    "so400m": "google/siglip-so400m-patch14-384",  # ~400M, 1152 dim, patch=14, img=384 (PI0/PI0.5 使用)
}

# Gemma 模型选择（按大小排列）
GEMMA_MODELS = {
    "gemma3-270m": "google/gemma-3-270m-it",       # 270M params 
}


class SigLIPGemmaValueFunction(nn.Module):
    """
    基于 SigLIP + Gemma 的 Value Function
    
    按照 PaliGemma 架构设计：
    - 视觉编码器：SigLIP (独立预训练)
    - 语言模型：Gemma 3 (独立预训练)
    - 融合层：将视觉、语言特征融合
    - Value Head：输出 200 bin 分类
    
    输入：左右两张相机图像 + 文本 prompt
    """
    
    def __init__(
        self,
        num_bins: int = NUM_BINS,
        siglip_variant: str = "so400m",
        gemma_variant: str = "gemma3-270m",
        freeze_vision: bool = False,
        freeze_llm: bool = False,
        hidden_dim: int = 512,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        
        self.num_bins = num_bins
        self.hidden_dim = hidden_dim
        self._text_cache = {}
        
        print("=" * 60)
        print("初始化 SigLIP + Gemma Value Function")
        print("输入: 左右两张图像 + prompt")
        print("=" * 60)
        
        # ==================== SigLIP 视觉编码器 ====================
        self._init_vision_encoder(siglip_variant, freeze_vision)
        
        # ==================== Gemma 语言模型 ====================
        self._init_language_model(gemma_variant, freeze_llm, use_flash_attention)
        
        # ==================== 投影层 ====================
        # 将不同维度的特征投影到统一维度
        self.vision_proj = nn.Sequential(
            nn.Linear(self.vision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        print(f"视觉投影: {self.vision_dim} -> {hidden_dim}")
        print(f"文本投影: {self.text_dim} -> {hidden_dim}")
        
        # ==================== 融合层 ====================
        # 2张图像 + 文本 = 3 * hidden_dim
        fusion_input_dim = 3 * hidden_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        print(f"融合层: {fusion_input_dim} -> {hidden_dim}")
        
        # ==================== Value Head ====================
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_bins),
        )
        print(f"Value Head: {hidden_dim} -> {num_bins} bins")
        
        # 打印参数统计
        self._print_param_stats()
    
    def _init_vision_encoder(self, variant: str, freeze: bool):
        """初始化 SigLIP 视觉编码器"""
        model_name = SIGLIP_MODELS.get(variant, SIGLIP_MODELS["so400m"])
        
        print(f"加载 SigLIP: {model_name}")
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self.vision_dim = self.vision_encoder.config.hidden_size
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
        print(f"  - 隐藏维度: {self.vision_dim}")
        print(f"  - Patch size: {self.vision_encoder.config.patch_size}")
        
        if freeze:
            print("  - 冻结视觉编码器参数")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def _init_language_model(self, variant: str, freeze: bool, use_flash_attention: bool):
        """初始化 Gemma 语言模型"""
        model_name = GEMMA_MODELS.get(variant, GEMMA_MODELS["gemma3-270m"])
        
        print(f"加载 Gemma: {model_name}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        
        self.text_dim = self.language_model.config.hidden_size
        print(f"  - 隐藏维度: {self.text_dim}")
        print(f"  - 层数: {self.language_model.config.num_hidden_layers}")
        
        if freeze:
            print("  - 冻结语言模型参数")
            for param in self.language_model.parameters():
                param.requires_grad = False
    
    def _print_param_stats(self):
        """打印参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 60)
        print(f"总参数量: {total_params / 1e6:.1f}M")
        print(f"可训练参数量: {trainable_params / 1e6:.1f}M")
        print("=" * 60)
    
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像，返回 [batch, hidden_dim]"""
        # SigLIP 输出: [batch, num_patches, hidden_dim]
        outputs = self.vision_encoder(images)
        # 取 CLS token 或 mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output  # [batch, hidden_dim]
        else:
            features = outputs.last_hidden_state.mean(dim=1)  # [batch, hidden_dim]
        
        return features
    
    def _encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        # 只有在语言模型被冻结时才缓存
        llm_frozen = all(not p.requires_grad for p in self.language_model.parameters())
        if not llm_frozen:
            # 不冻结就别缓存，直接正常算
            tokens = self.tokenizer(
                texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            outputs = self.language_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            hs = outputs.hidden_states[-1]
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            return (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # ===== 冻结：按“单条 prompt”缓存 =====
        feats = [None] * len(texts)
        todo_texts, todo_idx = [], []

        for i, t in enumerate(texts):
            if t in self._text_cache:
                feats[i] = self._text_cache[t].to(device)
            else:
                todo_texts.append(t)
                todo_idx.append(i)

        if todo_texts:
            tokens = self.tokenizer(
                todo_texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = self.language_model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                hs = outputs.hidden_states[-1]
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                new_feat = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [N, H]

            for j, i in enumerate(todo_idx):
                t = todo_texts[j]
                self._text_cache[t] = new_feat[j].detach().cpu()
                feats[i] = new_feat[j]

        return torch.stack(feats, dim=0)  # [B, H]


    
    def forward(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            left_image: [batch, C, H, W] 左相机图像
            right_image: [batch, C, H, W] 右相机图像
            prompts: 文本提示列表
            
        Returns:
            logits: [batch, num_bins] 分类 logits
            value: [batch] 预测的连续 value
        """
        device = left_image.device
        
        # 编码两张图像
        left_features = self._encode_images(left_image)     # [batch, vision_dim]
        right_features = self._encode_images(right_image)   # [batch, vision_dim]
        
        # 投影视觉特征
        left_proj = self.vision_proj(left_features)         # [batch, hidden_dim]
        right_proj = self.vision_proj(right_features)       # [batch, hidden_dim]
        
        # 编码文本
        text_features = self._encode_text(prompts, device)  # [batch, text_dim]
        text_proj = self.text_proj(text_features)           # [batch, hidden_dim]
        
        # 融合特征
        combined = torch.cat([
            left_proj,
            right_proj,
            text_proj,
        ], dim=-1)  # [batch, 3 * hidden_dim]
        
        fused = self.fusion(combined)  # [batch, hidden_dim]
        
        # Value Head
        logits = self.value_head(fused)  # [batch, num_bins]
        
        # 计算连续 value
        value = bins_to_value_soft(logits)  # [batch]
        
        return logits, value


# 为了兼容性，保留别名
PaliGemmaValueFunction = SigLIPGemmaValueFunction
