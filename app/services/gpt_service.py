"""
PyTorch GPT & HuggingFace Instruction Reasoning Service.
--------------------------------------------------------
Manages model selection and inference:
1. "flash_reasoning": Instruction-tuned model for coherent multi-turn responses (Default)
2. "custom_pytorch_gpt": Custom PyTorch Transformer built from scratch (Phase 1 demo)
"""

import os
import sys
import json
import torch
from typing import Optional, List, Dict, Tuple
import app.models.gpt_model as gpt_model_module
from app.models.gpt_model import GPT, GPTConfig
from app.core.config import settings
from app.core.logging import logger

sys.modules['gpt_model'] = gpt_model_module

# HuggingFace Instruction Model
_instruction_model = None
_instruction_tokenizer = None


def get_instruction_model():
    global _instruction_model, _instruction_tokenizer
    if _instruction_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info("Initializing Pretrained Instruction Model (google/flan-t5-small)...")
            _instruction_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            _instruction_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        except Exception as e:
            logger.warning(f"Could not load instruction model: {e}")
            _instruction_model = False
            _instruction_tokenizer = False
    if _instruction_model is False:
        return None, None
    return _instruction_model, _instruction_tokenizer


class GPTInferenceService:
    """Service managing Dual Model Engines: Pretrained Instruction Model vs Custom PyTorch GPT."""

    def __init__(self):
        self.custom_pytorch_model: Optional[GPT] = None
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.device: str = settings.DEVICE

    def load_assets(self):
        """Loads custom PyTorch GPT checkpoint and pre-warms instruction model."""
        logger.info("Pre-warming FLASH Reasoning Assets...")
        get_instruction_model()
        
        # Load Custom PyTorch GPT Checkpoint if available
        ckpt_path = os.path.join(settings.BASE_DIR, "checkpoint.pt")
        meta_path = os.path.join(settings.BASE_DIR, "meta.json")
        
        if os.path.exists(ckpt_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.stoi = meta.get("stoi", {})
                    self.itos = {int(k): v for k, v in meta.get("itos", {}).items()}

                vocab_size = len(self.stoi) if self.stoi else 256
                config = GPTConfig(
                    vocab_size=vocab_size,
                    block_size=128,
                    n_layer=4,
                    n_head=4,
                    n_embd=128,
                )
                self.custom_pytorch_model = GPT(config)
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.custom_pytorch_model.load_state_dict(checkpoint["model_state_dict"])
                self.custom_pytorch_model.to(self.device)
                self.custom_pytorch_model.eval()
                logger.info("Loaded Custom PyTorch Transformer checkpoint successfully.")
            except Exception as e:
                logger.warning(f"Could not load custom PyTorch GPT checkpoint: {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_k: Optional[int] = 5,
        engine_mode: str = "flash_reasoning"
    ) -> Tuple[str, str]:
        """
        Generates text using chosen engine:
        - "flash_reasoning": Pretrained Instruction Model
        - "custom_pytorch_gpt": Custom PyTorch Transformer (built from scratch)
        """
        clean_prompt = prompt.replace("User:", "").replace("Assistant:", "").strip()

        # Engine 1: FLASH Reasoning Engine (Instruction Model)
        if engine_mode == "flash_reasoning":
            model, tokenizer = get_instruction_model()
            if model and tokenizer:
                try:
                    inputs = tokenizer(clean_prompt, return_tensors="pt", max_length=512, truncation=True)
                    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
                    res = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                    if res:
                        return res, "flash_reasoning_engine"
                except Exception as err:
                    logger.error(f"Instruction model inference error: {err}")

        # Engine 2: Custom PyTorch Transformer (Built from scratch)
        if self.custom_pytorch_model is not None and self.stoi:
            try:
                tokens = [self.stoi.get(ch, 0) for ch in clean_prompt]
                if tokens:
                    idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        out_idx = self.custom_pytorch_model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
                    gen_tokens = out_idx[0].tolist()[len(tokens):]
                    gen_text = "".join([self.itos.get(t, "") for t in gen_tokens]).strip()
                    if gen_text:
                        return f"[CUSTOM PYTORCH GPT ENGINE]\n{gen_text}", "custom_pytorch_gpt"
            except Exception as err:
                logger.error(f"Custom PyTorch GPT inference error: {err}")

        return f"Response for query: '{clean_prompt}'", engine_mode


gpt_service = GPTInferenceService()
