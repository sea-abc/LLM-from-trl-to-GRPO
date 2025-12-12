"""
这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的4-bit量化 + LoRA指令微调
2. 使用transformers库的Trainer类进行训练
"""
# 基于 Qwen3-4B-Instruct 模型

# ====================== 1. 基础设置 ======================
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像加速

# ====================== 2. 导入核心库 ======================
import torch
from datasets import Dataset
from transformers import (
    # AutoModelForCausalLM：自动加载适用于因果语言建模（如文本生成）的预训练模型
    AutoModelForCausalLM,
    
    # AutoTokenizer：自动加载与预训练模型匹配的Tokenizer（分词器）
    # Tokenizer的核心作用是将自然语言文本转换为模型可识别的数字token（词表索引），同时处理文本截断、填充等操作
    AutoTokenizer,
    
    # BitsAndBytesConfig：用于配置模型量化参数的类
    BitsAndBytesConfig,
    
    # TrainingArguments：用于配置模型训练过程中所有超参数和训练设置的类
    # 包含训练轮数、学习率、批处理大小、保存路径、日志频率等关键参数，是控制训练流程的核心配置
    TrainingArguments,
    
    # Trainer：transformers库提供的高层训练器类
    # 封装了训练循环（前向传播、损失计算、反向传播、参数更新）、验证流程、模型保存等功能，
    Trainer,
    
    # DataCollatorForLanguageModeling：用于因果语言建模任务的数据拼接器
    # 作用是将批量数据中的文本样本拼接成连续序列（按模型最大长度截断/填充），并生成对应的标签（因果建模中标签与输入token相同，仅偏移一位），
    # 确保输入数据格式符合模型训练要求
    DataCollatorForLanguageModeling
)

'''
PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）是针对大模型微调的优化技术，
通过仅训练模型的一小部分参数（而非全部参数），在降低内存占用和计算成本的同时，保持较好的微调效果，LoRA是PEFT中最常用的方法之一
'''
from peft import (
    # LoraConfig：用于配置LoRA（Low-Rank Adaptation，低秩适配）微调参数的类
    # 关键参数包括秩（rank）、缩放因子（lora_alpha）、目标模块（target_modules）等，
    LoraConfig,
    # get_peft_model：根据基础模型和LoRA配置，生成带有LoRA适配器的PEFT模型
    # 该函数会在基础模型的指定层（如Transformer的注意力层）插入LoRA参数，并冻结基础模型的大部分参数，仅保留LoRA参数可训练
    get_peft_model
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
    dtype=torch.bfloat16, # 模型权重以及后续计算使用bf16
    trust_remote_code=True
)

# 训练前配置
base_model.config.use_cache = False # 训练期间彻底禁用 KV Cache

"""
# 在加载 base_model 后，看一下模型都有什么层
for name, module in base_model.named_modules():
    print(name)
    print(module)
    print('-------------------------------------')
# 打印完成后直接退出程序
import sys
sys.exit()
"""

# ====================== 4. LoRA配置（核心） ======================
peft_config = LoraConfig(
    r=8,  # LoRA秩，控制参数数量
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"], # 指定要微调的模型层，这里是Transformer层中的一些配置
    task_type="CAUSAL_LM",
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

# 创建LoRA模型
peft_model = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()  # 查看可训练参数
# trainable params: 16,515,072 || all params: 4,038,983,168 || trainable%: 0.4089
#import sys
#sys.exit()


# ====================== 5. 数据准备 ======================
# 简单的训练数据
sample_data = [
    {
        "instruction": "什么是LoRA？",
        "response": "LoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数。"
    },
    {
        "instruction": "LoRA有什么优势？",
        "response": "LoRA可以大幅减少训练参数，降低显存需求，同时保持较好的微调效果。"
    }
]

# 转换为Dataset
dataset = Dataset.from_list(sample_data)

# 数据处理函数
def process_data(example):
    # 格式化为Qwen要求的prompt
    prompt = f"""<|im_start|>user
{example['instruction']}<|im_end|>
<|im_start|>assistant
{example['response']}<|im_end|>"""
    
    # 分词
    encoding = tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        return_tensors="pt" #返回的对象是tensor
    )
    
    encoding["labels"] = encoding["input_ids"].clone()
    
    return {
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(), # 这个是填充掩码，告诉模型哪些是padding进去的
        "labels": encoding["labels"].flatten() # Trainer在内部会正确处理这个偏移
    }

# 处理数据集
tokenized_dataset = dataset.map(process_data, remove_columns=dataset.column_names) # remove_columns表示扔掉原来的数据，因为已经转换好了

# 数据整理器
# 作用1：padding：
# data_collator会在每个训练步骤中，取当前批次（batch）的所有样本，
#并将它们填充到该批次内最长样本的长度。这与在预处理时就将所有样本填充到一个固定最大长度（如你的代码中的 max_length=1024）相比，
# 能显著减少不必要的填充，提高训练效率，尤其是在样本长度差异较大的情况下
# 作用2：既然padding改了的话，那attention mask 相应也要修改
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 采用因果语言建模，而不是掩码语言建模（bert）
    )

# ====================== 6. 训练配置 ======================
training_args = TrainingArguments(
    output_dir="./qwen3_simple_lora",  # 模型和检查点的输出目录
    per_device_train_batch_size=1,    # 每个设备（如GPU）的训练批次大小。
    gradient_accumulation_steps=4,     # 梯度累积步数
    learning_rate=2e-4,               # 优化器的初始学习率
    num_train_epochs=2,               # 训练总轮数
    fp16=True,                        # 启用FP16混合精度训练
    optim="adamw_bnb_8bit",           # 使用8位量化的AdamW优化器保存优化器状态
    lr_scheduler_type="cosine",       # 学习率调度器类型，`cosine`表示使用余弦退火策略
    warmup_ratio=0.05,                # 学习率预热的比例，训练开始时的前5%步数从0线性增加学习率至初始值
    report_to="none"                  # 禁用日志报告（如不将日志发送到W&B或TensorBoard），仅本地输出
)

# ====================== 7. 训练模型 ======================
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 开始训练
print("开始训练...")
trainer.train()
print("训练完成！")

# ====================== 8. 推理测试 ======================
def generate_answer(instruction):
    # 构造prompt
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # 分词
    inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7
        )
    
    # 解码
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取assistant的回答部分
    answer = answer.split("<|im_start|>assistant\n")[-1]
    return answer

# 测试推理
print("\n推理测试：")
test_question = "请不留余地的夸赞和鼓励我"
print(f"我的询问：{test_question}")
print(f"AI的回答：{generate_answer(test_question)}")

# ====================== 9. 保存模型 ======================
peft_model.save_pretrained("./qwen3_simple_lora")
tokenizer.save_pretrained("./qwen3_simple_lora")
print("\n模型已保存到 ./qwen3_simple_lora")