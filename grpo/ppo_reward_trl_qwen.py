"""
这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的4-bit量化 + LoRA模型训练
2. 使用trl库的RewardTrainer类进行训练 PPO的奖励模型
由于奖励模型是部分GRPO还是PPO的，故这里GRPO也沿用了PPO的奖励模型
"""
# 基于 Qwen3-4B-Instruct 模型

# ====================== 1. 基础设置 ======================
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像加速

# ====================== 2. 导入核心库 ======================
import torch
from datasets import Dataset
from transformers import (
    # AutoModelForSequenceClassification：用于序列分类任务的预训练模型
    AutoModelForSequenceClassification,
    
    # AutoTokenizer：自动加载与预训练模型匹配的Tokenizer
    AutoTokenizer,
    
    # BitsAndBytesConfig：用于配置模型量化参数的类
    BitsAndBytesConfig,
)

# 导入LoRA相关库
from peft import (
    LoraConfig,  # LoRA配置类，用于定义LoRA的参数（如秩、alpha值、目标模块等）
    get_peft_model,  # 创建PeftModel实例的函数，将基础模型与LoRA配置结合
)

# 导入RewardTrainer和RewardConfig
from trl import (
    RewardTrainer,
    RewardConfig
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
    trust_remote_code=True,
    padding_side="right"
)

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载序列分类模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
    num_labels=1  # 奖励模型输出一个标量值
)

# 设置pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# ====================== 4. LoRA配置 ======================
peft_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    task_type="SEQ_CLS",  # 序列分类任务
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)


# ====================== 5. 准备模型 ======================
# 关于是否需要prepare_model_for_kbit_training的分析：
# 1. 在sft_LoRA_qwen.py中，确实没有使用这个函数，直接调用get_peft_model也能正常工作
# 2. 这个函数的主要作用是对模型进行预处理，使其能在4位量化下正常训练
# 3. 它的核心功能包括：启用梯度检查点、处理线性层梯度、配置混合精度等
# 4. 对于不同的模型架构和训练器，必要性不同：
#    - 使用transformers的Trainer类（如sft_LoRA_qwen.py），可能已经内置了必要的支持
#    - 使用trl的RewardTrainer类，早期版本可能需要这个函数来确保兼容性
# 5. 经过测试，可以不使用这个函数，直接进入LoRA模型创建步骤
#    - 测试结果：模型训练正常完成，测试推理也能正常工作
#    - 训练指标：epoch=3.0, loss=0.4720, accuracy=0.5556, margin=2.7801
#    - 测试结果：好回答分数4.25, 差回答分数-5.0, 分数差9.25
# 结论：对于当前版本的库和模型架构，这个函数不是必需的，可以安全移除
# 注意：如果训练出现问题（如梯度计算错误），可以重新启用这个函数

# model = prepare_model_for_kbit_training(model)  #  准备模型进行kbit训练

# 创建LoRA模型
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# ====================== 5. 数据准备 ======================
# 偏好数据集示例（直接使用Qwen聊天模板格式）
sample_data = [
    {
        "chosen": "<|im_start|>user\n什么是LoRA？<|im_end|>\n<|im_start|>assistant\nLoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数，同时保持较好的微调效果。它能够大幅降低显存需求，适用于大规模语言模型的微调。<|im_end|>",
        "rejected": "<|im_start|>user\n什么是LoRA？<|im_end|>\n<|im_start|>assistant\nLoRA是一个公司的名字，主要生产电子产品。<|im_end|>"
    },
    {
        "chosen": "<|im_start|>user\nLoRA有什么优势？<|im_end|>\n<|im_start|>assistant\nLoRA的主要优势包括：1) 大幅减少训练参数；2) 降低显存需求；3) 保持较好的微调效果；4) 支持模型多任务微调；5) 可以与其他参数高效微调方法结合使用。<|im_end|>",
        "rejected": "<|im_start|>user\nLoRA有什么优势？<|im_end|>\n<|im_start|>assistant\nLoRA的优势是它的名字很好听，容易记住。<|im_end|>"
    },
    {
        "chosen": "<|im_start|>user\n如何使用LoRA进行模型微调？<|im_end|>\n<|im_start|>assistant\n使用LoRA进行模型微调的步骤包括：1) 准备训练数据；2) 配置LoRA参数（如秩、目标模块等）；3) 加载预训练模型并应用LoRA配置；4) 设置训练超参数；5) 开始训练；6) 保存和加载微调后的模型。<|im_end|>",
        "rejected": "<|im_start|>user\n如何使用LoRA进行模型微调？<|im_end|>\n<|im_start|>assistant\n使用LoRA很简单，只需要点击一个按钮就可以了。<|im_end|>"
    }
]

# 转换为Dataset
tokenized_dataset = Dataset.from_list(sample_data)

# ====================== 6. 训练配置 ======================
# 创建奖励模型的训练配置对象，用于定义训练过程中的各种参数
reward_config = RewardConfig(
    output_dir="./qwen3_reward_lora", # 指定训练完成后模型和检查点文件的保存路径
    per_device_train_batch_size=1,# 每个设备上的训练批次大小，这里设置为1表示每次只处理1个训练样本
    gradient_accumulation_steps=4,# 梯度累积步数，将4个小批次的梯度累积后再更新模型参数，相当于批次大小为4
    learning_rate=2e-4,# 初始学习率，控制参数更新的步长，2e-4表示0.0002
    num_train_epochs=3,# 训练的总轮数，即整个训练数据集将被遍历3次
    bf16=True,# 启用bfloat16混合精度训练，加快训练速度并减少内存占用
    optim="adamw_bnb_8bit",# 优化器类型，这里使用8位量化版本的AdamW优化器，进一步减少内存占用
    lr_scheduler_type="cosine",# 学习率调度器类型，使用余弦退火调度器，让学习率在训练过程中先升后降
    warmup_ratio=0.05,# 学习率预热比例，前5%的训练步骤中学习率会从0逐渐增加到初始学习率
    report_to="none"# 禁用日志报告，不将训练日志发送到任何外部服务（如WandB）
)

# ====================== 7. 训练模型 ======================
trainer = RewardTrainer(
    model=peft_model,
    args=reward_config,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer  # 将tokenizer传递给processing_class
)

# 开始训练
print("开始训练奖励模型...")
trainer.train()
print("训练完成！")

# ====================== 8. 测试模型 ======================
def test_reward_model():
    print("\n测试奖励模型...")
    
    # 测试问题
    question = "什么是LoRA？"
    
    # 好的回答和差的回答
    good_response = "LoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数，同时保持较好的微调效果。"
    bad_response = "LoRA是一个公司的名字，主要生产电子产品。"
    
    # 格式化输入
    good_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{good_response}<|im_end|>"
    
    bad_prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{bad_response}<|im_end|>"
    
    # 分词：将文本转换为模型可理解的数字编码
    # return_tensors="pt"：返回PyTorch张量格式
    # truncation=True：如果文本长度超过max_length，会截断到max_length
    # max_length=1024：设置最大序列长度为1024
    good_inputs = tokenizer(good_prompt, return_tensors="pt", truncation=True, max_length=1024)
    # 输出示例：good_inputs = {"input_ids": tensor([[151644, 8950, ..., 151645]]), "attention_mask": tensor([[1, 1, ..., 1]])}
    
    bad_inputs = tokenizer(bad_prompt, return_tensors="pt", truncation=True, max_length=1024)
    # 输出示例：bad_inputs = {"input_ids": tensor([[151644, 8950, ..., 151645]]), "attention_mask": tensor([[1, 1, ..., 1]])}
    
    # 移动到设备
    good_inputs = {k: v.to(peft_model.device) for k, v in good_inputs.items()}
    bad_inputs = {k: v.to(peft_model.device) for k, v in bad_inputs.items()}
    
    # 获取奖励分数
    with torch.no_grad():
        good_score = peft_model(**good_inputs).logits.item()
        bad_score = peft_model(**bad_inputs).logits.item()
    
    print(f"问题: {question}")
    print(f"好的回答: {good_response}")
    print(f"好的回答分数: {good_score}")
    print(f"差的回答: {bad_response}")
    print(f"差的回答分数: {bad_score}")
    print(f"分数差: {good_score - bad_score}")

# 测试模型
test_reward_model()

# ====================== 9. 保存模型 ======================
peft_model.save_pretrained("./qwen3_reward_lora")
tokenizer.save_pretrained("./qwen3_reward_lora")
print("\n模型已保存到 ./qwen3_reward_lora")