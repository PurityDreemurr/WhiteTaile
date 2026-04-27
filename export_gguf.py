from unsloth import FastLanguageModel
import torch

# 1. 加载你刚训好的 LoRA 和原始模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "my_persona_lora", # 你的 LoRA 文件夹
    max_seq_length = 1024,
    load_in_4bit = True,
)

# 2. 导出为 GGUF 格式
# 我们通常选择 q8_0 (接近无损) 或者 q4_k_m (极致速度)
model.save_pretrained_gguf(
    "HaoWei_GGUF",
    tokenizer,
    quantization_method = "q8_0", # 推荐先出个 8-bit 版本保证智商
)