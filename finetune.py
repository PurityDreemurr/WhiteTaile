from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch


# 1. 定义格式化函数
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for instruction, input, output in zip(instructions, inputs, outputs):
        # 严格遵循 ChatML 格式拼接
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n" \
               f"<|im_start|>user\n{input}<|im_end|>\n" \
               f"<|im_start|>assistant\n{output}<|im_end|>"
        texts.append(text)

    # 关键：直接返回列表，而不是 {"text": texts}
    return texts

# 1. 加载本地模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/kingscat/Documents/WhiteTail_prototype/Qwen3.5-9B",
    max_seq_length = 1024,
    load_in_4bit = True,
    fix_tokenizer = True,
    # 强制关闭 Unsloth 的编译内核，解决 GatedDeltaNet 显存爆炸问题
    fast_inference = False,
)

# --- 特殊 Token 注入 ---
actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
actual_tokenizer.add_special_tokens({'additional_special_tokens': ['<state>', '</state>']})
model.resize_token_embeddings(len(actual_tokenizer))

# 2. 针对 5070 Ti 设置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
)

# 开启生存模式的显存
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 3. 加载你的数据集
dataset = load_dataset("json", data_files="dataset.json", split="train")

# 4. 设置训练参数
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    formatting_func = formatting_prompts_func,
    max_seq_length = 1024,  # 回到 1024，让 padding/truncation 处理长序列
    packing = False,  # 禁用 packing
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=60,
        learning_rate=2e-4,
        bf16=True,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="no",
        # 不要在这里设置 gradient_checkpointing，因为已经在 get_peft_model 中设置了
    ),
)

trainer.train()

# 5. 保存结果（保存为本地 LoRA 权重）
model.save_pretrained("my_persona_lora")
tokenizer.save_pretrained("my_persona_lora")
print("微调完成！LoRA 权重已保存在 my_persona_lora 文件夹中。")