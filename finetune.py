from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch
import os

# 🔥 关键：关闭 compile + 防碎片
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()


# 1. 格式化函数
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n" \
               f"<|im_start|>user\n{input}<|im_end|>\n" \
               f"<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)

    return texts


# 2. 加载模型（限制attention实现）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/kingscat/Documents/WhiteTail_prototype/Qwen3.5-9B",
    max_seq_length=1024,
    load_in_4bit=True,
    fast_inference=False,
    attn_implementation="sdpa",   # 🔥防止fallback炸显存
)


# 3. Token扩展（保留）
actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
actual_tokenizer.add_special_tokens({'additional_special_tokens': ['<state>', '</state>']})
model.resize_token_embeddings(len(actual_tokenizer))


# 4. LoRA（进一步削峰）
model = FastLanguageModel.get_peft_model(
    model,
    r=32,   # 🔥从16降到8
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
)


# 🔥关键：正确开启训练模式（替代你原来的两行）
model = FastLanguageModel.for_training(model)


# 5. 数据集
dataset = load_dataset("json", data_files="dataset_new.json", split="train")


# 6. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=1024,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # 🔥增加，降低峰值
        max_steps=120,
        learning_rate=2e-4,
        bf16=True,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="no",
        max_grad_norm=0.3,   # 🔥防NaN
    ),
)

trainer.train()


# 7. 保存
model.save_pretrained("my_persona_lora")
tokenizer.save_pretrained("my_persona_lora")
print("微调完成！")