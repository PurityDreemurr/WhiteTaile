from unsloth import FastLanguageModel
# 加载你刚训好的 LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "my_persona_lora", # 这里指向你保存的文件夹
    max_seq_length = 1024,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # 开启推理模式

# 测试一下
# 修正后的写法：明确指定 text=...
inputs = tokenizer(
    text = ["<|im_start|>user\n阿尾，明天就考试了，但是我还没学完，咋办？<|im_end|>\n<|im_start|>assistant\n"],
    return_tensors = "pt"
).to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens = 128,
    use_cache = True,
    # 告诉模型看到 <|im_end|> 就闭嘴
    eos_token_id = tokenizer.tokenizer.eos_token_id if hasattr(tokenizer, "tokenizer") else tokenizer.eos_token_id
)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=False)
print(response)