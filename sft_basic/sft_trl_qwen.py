"""
这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的LoRA指令微调
2. 使用TRL库的SFTTrainer类进行训练
3. TRL库不使用损失掩码逻辑，NEFT噪声训练
"""
# 基于 Qwen3-4B-Instruct 模型，使用TRL库实现SFT训练
# trl官方文档 https://hf-mirror.com/docs/trl/v0.26.0/en/index

# ====================== 1. 基础设置 ======================
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像加速

# ====================== 2. 导入核心库 ======================
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model
)
# 导入TRL库的SFT相关组件
from trl import (
    SFTTrainer,
    SFTConfig
)

# ====================== 3. 模型配置 ======================
model_name = "Qwen/Qwen3-4B-Instruct-2507"

# 4bit量化配置（降低显存占用）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True, # 加载Qwen、ChatGLM等非Hugging Face官方标准架构的分词器等文件
    padding_side="right"    # 从右侧开始填充
)

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto", # 自动选用设备GPU
    dtype=torch.bfloat16, # 模型权重以及后续计算使用bfloat16
    trust_remote_code=True
)

# 训练前配置
base_model.config.use_cache = False # 训练期间彻底禁用 KV Cache

# ====================== 4. LoRA配置（核心） ======================
peft_config = LoraConfig(
    r=8,  # LoRA秩，控制参数数量
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"], # 指定要微调的模型层
    task_type="CAUSAL_LM",
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

# 打印可训练参数信息
print("LoRA配置:")
print(f"  秩(r): {peft_config.r}")
print(f"  目标模块: {peft_config.target_modules}")
print(f"  LoRA alpha: {peft_config.lora_alpha}")
print(f"  任务类型: {peft_config.task_type}")

# ====================== 5. 数据准备 ======================
# 简单的训练数据 - 直接使用Qwen要求的格式
# 不需要额外的process_data函数处理
sample_data = [
    {
        "text": "<|im_start|>user\n什么是LoRA？<|im_end|>\n<|im_start|>assistant\nLoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数。<|im_end|>"
    },
    {
        "text": "<|im_start|>user\nLoRA有什么优势？<|im_end|>\n<|im_start|>assistant\nLoRA可以大幅减少训练参数，降低显存需求，同时保持较好的微调效果。<|im_end|>"
    }
]

# 转换为Dataset
tokenized_dataset = Dataset.from_list(sample_data)

# ====================== 6. 训练配置（使用TRL的SFTConfig） ======================
# 使用SFTConfig替代TrainingArguments
sft_config = SFTConfig(
    output_dir="./qwen3_simple_trl",  # 模型和检查点的输出目录
    per_device_train_batch_size=1,    # 每个设备（如GPU）的训练批次大小
    gradient_accumulation_steps=4,     # 梯度累积步数
    learning_rate=2e-4,               # 优化器的初始学习率
    num_train_epochs=2,               # 训练总轮数
    bf16=True,                        # 启用BF16混合精度训练（与torch_dtype=torch.bfloat16兼容）
    optim="adamw_bnb_8bit",           # 使用8位量化的AdamW优化器保存优化器状态
    lr_scheduler_type="cosine",       # 学习率调度器类型，`cosine`表示使用余弦退火策略
    warmup_ratio=0.05,                # 学习率预热的比例
    report_to="none",                 # 禁用日志报告
    max_length=1024,                  # 最大序列长度
    logging_steps=1,                  # 日志记录步数
    # 以下是TRL特有的参数
    neftune_noise_alpha=0,            # NEFTune噪声强度，0表示不使用
    packing=False,                    # 是否启用序列打包
    dataset_text_field="text",       # 数据集中用于训练的文本字段
    # 对于Qwen模型，completion_only_loss应该设置为False
    # 因为我们已经在process_data中格式化了完整的prompt和response
    completion_only_loss=False, # 如果想要使用这个，需要使用官方的提示词模板
    
)


# ====================== 7. 训练模型（使用TRL的SFTTrainer） ======================
# 创建LoRA模型（与原始文件保持一致）
peft_model = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()  # 查看可训练参数

trainer = SFTTrainer(
    model=peft_model,
    args=sft_config,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer
)

# 开始训练
print("开始训练...")
trainer.train()
print("训练完成！")

# ====================== 9. 推理测试 ======================
def generate_answer(instruction):
    # 推理前配置：启用use_cache + 切换到eval模式 
    trainer.model.eval()  # 切换到评估模式
    trainer.model.config.use_cache = True

    
    # 构造prompt
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # 分词
    inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
    
    # 生成回答 - 优化生成参数以获得更好的结果
    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )
    
    # 解码 - 不跳过特殊标记，以便正确提取assistant的回答
    answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取assistant的回答部分
    if "<|im_start|>assistant\n" in answer:
        answer = answer.split("<|im_start|>assistant\n")[-1]
        # 移除assistant回答后的结束标记
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0]
    
    # 恢复训练时的配置
    trainer.model.config.use_cache = False
    
    return answer

# 测试推理
print("\n推理测试：")
test_question = "请不留余地的夸赞和鼓励我"
print(f"我的询问：{test_question}")
print(f"AI的回答：{generate_answer(test_question)}")

# ====================== 10. 保存模型 ======================
# 保存完整的微调模型（包括LoRA适配器和基础模型配置）
trainer.save_model("./qwen3_simple_trl")
tokenizer.save_pretrained("./qwen3_simple_trl")
print("\n模型已保存到 ./qwen3_simple_trl")
