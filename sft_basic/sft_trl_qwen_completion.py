"""
这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的LoRA指令微调
2. 使用TRL库的SFTTrainer类进行训练
3. TRL库自动实现了损失掩码逻辑，NEFT噪声训练
"""
# 基于 Qwen3-4B-Instruct 模型，使用TRL库实现SFT训练 - 使用completion_only_loss功能
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
# 使用官方文档推荐的prompt-completion格式数据
# 参考文档：对于prompt-completion格式，SFTTrainer会自动处理并仅计算completion部分的损失

# 简单的训练数据 - 使用prompt-completion格式
# 具体那些开始和结束词的内容采用的是 Qwen 模型的格式
sample_data = [
    {
        "prompt": "<|im_start|>user\n什么是LoRA？<|im_end|>\n<|im_start|>assistant\n",
        "completion": "LoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数。<|im_end|>"
    },
    {
        "prompt": "<|im_start|>user\nLoRA有什么优势？<|im_end|>\n<|im_start|>assistant\n",
        "completion": "LoRA可以大幅减少训练参数，降低显存需求，同时保持较好的微调效果。<|im_end|>"
    },
    {
        "prompt": "<|im_start|>user\n什么是参数高效微调？<|im_end|>\n<|im_start|>assistant\n",
        "completion": "参数高效微调是指在微调大语言模型时，只更新一小部分参数，从而减少计算和存储需求的技术。<|im_end|>"
    },
    {
        "prompt": "<|im_start|>user\n如何使用LoRA微调模型？<|im_end|>\n<|im_start|>assistant\n",
        "completion": "使用LoRA微调模型需要配置LoRA参数（如秩r、目标模块等），然后将LoRA适配器添加到基础模型上进行训练。<|im_end|>"
    }
]

# 转换为Dataset
dataset = Dataset.from_list(sample_data)

# ====================== 6. 训练配置（使用TRL的SFTConfig） ======================
# 使用SFTConfig替代TrainingArguments
sft_config = SFTConfig(
    output_dir="./qwen3_simple_trl_completion",  # 模型和检查点的输出目录
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
    neftune_noise_alpha=10,            # NEFTune噪声强度，0表示不使用
    packing=False,                    # 是否启用序列打包(就是把多个序列合并，减少padding量)
    # 对于prompt-completion格式的数据集，指定字段名
    dataset_text_field=None,          # 使用prompt和completion字段，不使用text字段
    # 启用completion_only_loss，仅计算completion部分的损失
    # 根据官方文档，对于prompt-completion格式的数据集，这个参数默认是True
    # 显式设置为True以确保只计算assistant回复部分的损失
    completion_only_loss=True,
    
)


# ====================== 7. 训练模型（使用TRL的SFTTrainer） ======================
# 创建LoRA模型（与原始文件保持一致）
peft_model = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()  # 查看可训练参数

trainer = SFTTrainer(
    model=peft_model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer
)

# 开始训练
print("开始训练...")
trainer.train()
print("训练完成！")

# ====================== 8. 推理测试 ======================
def generate_answer(instruction):
    # 推理前配置：启用use_cache + 切换到eval模式
    trainer.model.eval()
    trainer.model.config.use_cache = True
    
    # 构造prompt
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages,                 # 对话历史列表，格式为[{"role": "user", "content": instruction}]
                                 #   - "role": 对话角色，这里是用户输入使用"user"
                                 #   - "content": 对话内容，即用户的问题/指令
        add_generation_prompt=True,  # 关键参数：在对话末尾添加模型生成回答所需的提示标记
                                 # 对于Qwen模型，会在对话最后添加<|im_start|>assistant
                                 # 确保模型知道需要生成assistant的回答
        tokenize=False             # 不进行分词，返回原始文本格式的prompt
        )
    
    # 分词
    inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )
    
    # 解码
    # 不跳过特殊标记，以便正确提取assistant的回答
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取assistant的回答部分
    assistant_start = "<|im_start|>assistant\n"
    assistant_end = "<|im_end|>"
    
    if assistant_start in full_text:
        answer = full_text.split(assistant_start)[-1]
        if assistant_end in answer:
            answer = answer.split(assistant_end)[0]
    else:
        # 如果没有找到assistant标记，使用原始解码结果
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 恢复训练时的配置
    trainer.model.config.use_cache = False
    
    return answer

# 测试推理
print("\n推理测试：")
test_questions = [
    "请用优美的话语夸赞和鼓励我：",
    "请用夸张的话语夸赞我",
    "请用搞笑的话语夸赞我："
]

for question in test_questions:
    print(f"我的询问：{question}")
    print(f"AI的回答：{generate_answer(question)}")
    print()

# ====================== 9. 保存模型 ======================
# 保存完整的微调模型（包括LoRA适配器和基础模型配置）
trainer.save_model("./qwen3_simple_trl_completion")
tokenizer.save_pretrained("./qwen3_simple_trl_completion")
print("\n模型已保存到 ./qwen3_simple_trl_completion")
