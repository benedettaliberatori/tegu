import torch
import numpy as np
from PIL import Image
from typing import List, Union
import copy
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import torchvision.transforms as T

LLAVA_OV_MODELS = {
    "llava-onevision-qwen2-7b-ov-hf": {
        "tokenizer": {
            "path": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        },
        "model": {
            "path": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "torch_dtype": torch.float16,
        },
    },
}


class LLaVAOneVisionModel:
    def __init__(
        self, model_name="llava-onevision-qwen2-7b-ov-hf", device="cuda", cache_dir=None
    ):
        assert (
            model_name in LLAVA_OV_MODELS
        ), f"Model {model_name} not found in LLAVA_OV_MODELS"
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = LLAVA_OV_MODELS[model_name]
        self.load_model()

    def load_model(self):
        model_path = self.model_info["model"]["path"]
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.model_info["model"]["torch_dtype"],
            low_cpu_mem_usage=True,
            device_map="auto",
        ).to()

        self.processor = AutoProcessor.from_pretrained(
            self.model_info["tokenizer"]["path"],
            use_fast=True
        )
        self.tokenizer = self.processor.tokenizer
        self.model.eval()

    def generate(self,
        data: List[Union[str, np.ndarray, torch.Tensor]],
        max_new_tokens: int = 256,
        prompt: str = None,
    ) -> List[str]:

        videos = [d.squeeze().cpu().numpy() if isinstance(d, torch.Tensor) else d for d in data]
        print([v.shape if isinstance(v, np.ndarray) else v for v in videos])

        conversations = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt},
                ],
            }
            for _ in videos
        ]

        prompts = [
        self.processor.apply_chat_template([conv], add_generation_prompt=True)
        for conv in conversations
    ]        
        self.processor.tokenizer.padding_side = "left"
        
        inputs = self.processor(
            videos=videos,         
            text=prompts,         
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device, torch.float16)


        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
            )

        texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        cleaned = [t.split("assistant")[-1].strip() for t in texts]
        return cleaned