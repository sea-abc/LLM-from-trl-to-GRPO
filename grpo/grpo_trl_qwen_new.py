"""
GRPO训练示例代码 - 基于Qwen3-4B-Instruct模型

这个代码实现了：
1. 基于Qwen3-4B-Instruct模型(实现4-bit量化)的GRPO训练
2. 支持多种奖励函数：预训练奖励模型和自定义奖励函数
3. 使用LoRA进行参数高效微调
"""

# ====================== 模块1: 导入依赖库 ======================
import os
import shutil
import torch
import warnings
warnings.filterwarnings("ignore")

# 设置国内镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 核心机器学习库
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    GenerationConfig
)

# TRL库相关导入
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# ====================== 模块2: 自定义奖励函数 ======================

def length_reward_func(prompts, completions, **kwargs):
    """
    基于生成长度的奖励函数
    奖励更长的回答（基于token数量）
    """
    return [float(len(completion)) for completion in completions]

def content_quality_reward_func(prompts, completions, **kwargs):
    """
    基于内容质量的奖励函数
    奖励包含特定关键词的回答
    """
    rewards = []
    quality_keywords = ['LoRA', '微调', '参数', '高效', '训练', '模型', '优势', '步骤']
    
    for completion in completions:
        # 计算关键词命中数量
        keyword_count = sum(1 for keyword in quality_keywords if keyword in completion)
        # 归一化到0-1范围
        reward = min(keyword_count / len(quality_keywords), 1.0)
        rewards.append(reward)
    
    return rewards

def format_reward_func(prompts, completions, **kwargs):
    """
    基于格式规范的奖励函数
    检查回答是否符合Qwen对话格式
    """
    rewards = []
    
    for completion in completions:
        # 检查是否包含正确的结束标记
        if '<|im_end|>' in completion and '<|im_start|>' in completion:
            # 检查格式完整性
            if completion.count('<|im_start|>') == completion.count('<|im_end|>'):
                reward = 1.0
            else:
                reward = 0.5
        else:
            reward = 0.0
        
        rewards.append(reward)
    
    return rewards

# ====================== 模块3: 主函数 ======================

if __name__ == "__main__":
    
    # ====================== 3.1 GRPO训练参数配置 ======================
    training_args = GRPOConfig(
        # 基础训练参数
        learning_rate=3e-6,              # 学习率
        output_dir="qwen3-4b-instruct-grpo",  # 输出目录
        per_device_train_batch_size=1,   # 批次大小
        gradient_accumulation_steps=4,    # 梯度累积步数
        num_train_epochs=3,              # 训练轮数
        
        # GRPO特定参数
        num_generations=4,                # 每个提示生成4个回答
        max_completion_length=256,        # 最大生成长度
        temperature=0.7,                  # 采样温度
        top_p=0.9,                        # 核采样参数
        
        # KL散度控制
        beta=0.001,                       # KL系数（参考DeepSeek-R1）
        epsilon=0.2,                      # 裁剪参数
        
        # 损失函数配置
        loss_type="dapo",                  # 使用DAPO损失函数（推荐）
        scale_rewards="group",            # 奖励缩放策略
        
        # 训练优化
        bf16=True,                         # 使用bfloat16
        gradient_checkpointing=True,      # 梯度检查点
        
        # 日志和保存
        logging_steps=10,
        save_steps=100,
        report_to="none"
    )
    
    # ====================== 3.2 清理输出目录 ======================
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    print(f"已清理输出目录: {training_args.output_dir}")
    
    # ====================== 3.3 模型配置 ======================
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    
    # 量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # ====================== 3.4 分词器加载 ======================
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # 确保分词器有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("分词器加载完成")
    
    # ====================== 3.5 策略模型加载 ======================
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    print("策略模型加载完成")
    
    # ====================== 3.6 奖励模型加载 ======================
    reward_model_path = "/root/autodl-tmp/grpo/qwen3_reward_lora"
    
    if os.path.exists(reward_model_path):
        print(f"加载已训练好的奖励模型: {reward_model_path}")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            num_labels=1,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        )
    else:
        print("警告：未找到训练好的奖励模型，将使用基础模型")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16
        )
    
    print("奖励模型加载完成")
    
    # ====================== 3.7 LoRA配置 ======================
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        #target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        target_modules=["q_proj"], # 为了加快训练，所以LoRA只指定了query层
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # ====================== 3.8 数据集准备 ======================
    # 直接创建符合GRPO格式的训练数据集
    train_data = [
        {"prompt": "什么是LoRA？"},
        {"prompt": "LoRA有什么优势？"}, 
        {"prompt": "如何使用LoRA进行模型微调？"},
        {"prompt": "什么是PPO算法？"},
        {"prompt": "RLHF的全称是什么？"},
        {"prompt": "参数高效微调有哪些方法？"},
        {"prompt": "LoRA和全参数微调有什么区别？"},
        {"prompt": "如何选择合适的LoRA秩？"}
    ]
    train_dataset = Dataset.from_list(train_data)
    
    print(f"创建了 {len(train_data)} 个训练样本")
    
    # ====================== 3.9 创建GRPO训练器 ======================
    
    # 定义奖励函数组合
    reward_functions = [
        reward_model,                    # 预训练奖励模型
        length_reward_func,              # 长度奖励
        content_quality_reward_func,     # 内容质量奖励
        format_reward_func               # 格式规范奖励
    ]
    
    trainer = GRPOTrainer(
        model=policy_model,              # 策略模型
        reward_funcs=reward_functions,   # 奖励函数组合
        args=training_args,              # 训练参数
        train_dataset=train_dataset,     # 训练数据集
        processing_class=tokenizer,      # 分词器
        peft_config=peft_config         # LoRA配置
    )
    
    print("GRPO训练器创建完成")
    
    # ====================== 3.10 执行训练 ======================
    print("开始GRPO强化学习训练...")
    print("=" * 50)
    
    trainer.train()
    
    print("=" * 50)
    print("GRPO训练完成!")
    
    # ====================== 3.11 保存训练好的模型 ======================
    trainer.save_model(training_args.output_dir)
    print(f"训练好的模型已保存到: {training_args.output_dir}")
    
    # ====================== 3.12 训练后推理测试 ======================
    
    print("\n" + "=" * 60)
    print("开始推理测试 - 验证训练效果")
    print("=" * 60)
    
    # 加载训练好的模型
    trained_model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    print("训练好的模型加载完成")
    
    # 配置生成参数
    generation_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 测试问题
    test_questions = [
        "请用优美的话语夸赞和鼓励我：",
        "请用夸张的话语夸赞我：",
        "请用搞笑的话语夸赞我："
    ]
    
    print(f"\n准备测试 {len(test_questions)} 个问题...")
    
    for i, question in enumerate(test_questions, 1):
        # 构建Qwen对话格式
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(trained_model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = trained_model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True
            )
        
        # 解码生成的文本
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        
        # 提取assistant的回复部分
        if "<|im_start|>assistant" in generated_text:
            response = generated_text.split("<|im_start|>assistant")[-1]
            response = response.replace("<|im_end|>", "").strip()
        else:
            response = generated_text.replace(prompt, "").strip()
        
        # 输出测试结果
        print(f"\n测试 {i}:")
        print(f"问题: {question}")
        print(f"回答: {response}")
        print("-" * 50)
    
    # ====================== 3.13 测试完成总结 ======================
    print("\n" + "=" * 60)
    print("推理测试完成！")
    print("=" * 60)
    print("\nGRPO训练和推理测试全部完成！")

""" 
    # ====================== 3.14 额外功能：奖励函数测试 ======================
    print("\n" + "=" * 60)
    print("奖励函数测试")
    print("=" * 60)
    
    # 测试自定义奖励函数
    test_prompts = [
        "什么是LoRA？",
        "LoRA有什么优势？", 
        "如何使用LoRA进行模型微调？"
    ]
    test_completions = [
        "LoRA是一种参数高效微调方法，能够大幅减少训练参数。",
        "LoRA很好用。",
        "LoRA的主要优势包括减少参数、降低显存需求、保持微调效果等。"
    ]
    
    # 测试长度奖励
    length_scores = length_reward_func(prompts=test_prompts, completions=test_completions)
    
    # 测试内容质量奖励
    quality_scores = content_quality_reward_func(prompts=test_prompts, completions=test_completions)
    
    # 测试格式奖励
    format_scores = format_reward_func(prompts=test_prompts, completions=test_completions)
    
    print("\n奖励函数测试结果:")
    for i, (comp, len_score, qual_score, fmt_score) in enumerate(zip(test_completions, length_scores, quality_scores, format_scores), 1):
        print(f"\n回答 {i}:")
        print(f"内容: {comp}")
        print(f"长度奖励: {len_score:.2f}")
        print(f"质量奖励: {qual_score:.2f}")
        print(f"格式奖励: {fmt_score:.2f}")
        print("-" * 30)
    
    print("\n脚本执行完毕！")
"""