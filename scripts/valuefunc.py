import numpy as np
from config import NUM_BINS , VALUE_MIN, VALUE_MAX
from transformers import (
    SiglipVisionModel,
    SiglipImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import List, Tuple
import torch.nn.functional as F
import torch
import torch.nn as nn


def value_to_bin(value: float) -> int:
    value = np.clip(value, VALUE_MIN, VALUE_MAX)
    bin_idx = int((value - VALUE_MIN) / (VALUE_MAX - VALUE_MIN) * (NUM_BINS - 1))
    return np.clip(bin_idx, 0, NUM_BINS - 1)


def bins_to_value_soft(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    bin_values = torch.linspace(VALUE_MIN, VALUE_MAX, NUM_BINS, device=logits.device)
    return torch.sum(probs * bin_values, dim=-1)

SIGLIP_MODELS = {
    "so400m": "google/siglip-so400m-patch14-384",
}

GEMMA_MODELS = {
    "gemma3-270m": "google/gemma-3-270m-it",
}


class SigLIPGemmaValueFunction(nn.Module):
    
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
        
        self._init_vision_encoder(siglip_variant, freeze_vision)
        
        self._init_language_model(gemma_variant, freeze_llm, use_flash_attention)
        
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
        
        fusion_input_dim = 4 * hidden_dim
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
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_bins),
        )
        self._print_param_stats()
    
    def _init_vision_encoder(self, variant: str, freeze: bool):
        model_name = SIGLIP_MODELS.get(variant, SIGLIP_MODELS["so400m"])
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self.vision_dim = self.vision_encoder.config.hidden_size
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name)
        
        if freeze:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def _init_language_model(self, variant: str, freeze: bool, use_flash_attention: bool):
        model_name = GEMMA_MODELS.get(variant, GEMMA_MODELS["gemma3-270m"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        
        self.text_dim = self.language_model.config.hidden_size
        if freeze:
            for param in self.language_model.parameters():
                param.requires_grad = False
    
    def _print_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"总参数量: {total_params / 1e6:.1f}M")
        print(f"可训练参数量: {trainable_params / 1e6:.1f}M")
    
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_encoder(images)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features
    
    def _encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        llm_frozen = all(not p.requires_grad for p in self.language_model.parameters())
        if not llm_frozen:
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
                new_feat = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            for j, i in enumerate(todo_idx):
                t = todo_texts[j]
                self._text_cache[t] = new_feat[j].detach().cpu()
                feats[i] = new_feat[j]

        return torch.stack(feats, dim=0)


    
    def forward(
        self,
        head_image: torch.Tensor,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = head_image.device
        
        head_features = self._encode_images(head_image)
        left_features = self._encode_images(left_image)
        right_features = self._encode_images(right_image)

        head_proj = self.vision_proj(head_features)
        left_proj = self.vision_proj(left_features)
        right_proj = self.vision_proj(right_features)

        text_features = self._encode_text(prompts, device)
        text_proj = self.text_proj(text_features)

        combined = torch.cat([
            head_proj,
            left_proj,
            right_proj,
            text_proj,
        ], dim=-1)
        fused = self.fusion(combined)
        logits = self.value_head(fused)
        value = bins_to_value_soft(logits)
        return logits, value


PaliGemmaValueFunction = SigLIPGemmaValueFunction
