"""
GRPO简化测试脚本
这个代码实现了：
1. 测试完整的GRPO流程，但只进行少量训练步骤
2. 奖励模型采用的是一个简单的奖励函数，句子位于20-50个词时奖励最高
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
import warnings
warnings.filterwarnings("ignore")

def main():
    print("=" * 60)
    print("GRPO简化测试 - 少量训练步骤")
    print("=" * 60)
    
    # 1. 设置GRPO配置
    print("\n1. 配置GRPO参数...")
    grpo_config = GRPOConfig(
        # 基础学习参数
        learning_rate=3e-6,              # 模型学习的速度，数值越小学习越稳定
        output_dir="./grpo_test_output", # 训练结果保存的文件夹
        per_device_train_batch_size=2,  # 每次训练处理2个样本，必须能被num_generations整除
        gradient_accumulation_steps=1,  # 梯度累积步数，设为1表示不累积，加快测试速度
        num_train_epochs=1,             # 整个数据集只训练1轮
        max_steps=2,                    # 最多训练2步，用于快速测试
        
        # GRPO特有参数
        num_generations=2,              # 每个问题生成2个不同回答进行对比
        max_completion_length=64,      # 生成回答的最大长度，设为64个token以加快测试
        
        # 强化学习控制参数
        beta=0.001,                     # KL散度系数，控制模型不要偏离原始模型太远
        epsilon=0.2,                    # 裁剪参数，防止奖励值变化过大
        loss_type="dapo",               # 使用DAPO损失函数（一种改进的强化学习算法）
        
        # 训练优化参数
        bf16=True,                      # 使用bfloat16精度，节省显存
        
        # 日志和保存设置
        logging_steps=1,                # 每1步记录一次训练日志
        save_steps=10,                  # 每10步保存一次模型
        eval_steps=10,                  # 每10步进行一次评估
        warmup_steps=1                  # 学习率预热步数，让学习率从0慢慢增加到设定值
    )
    print("✓ GRPO配置完成")
    
    # 2. 准备数据集（简化版）
    print("\n2. 准备训练数据...")
    prompts = [
        "什么是LoRA？",
        "LoRA有什么优势？"
    ]
    
    train_data = [{"prompt": prompt} for prompt in prompts]
    train_dataset = Dataset.from_list(train_data)
    print(f"✓ 数据集准备完成，样本数: {len(train_dataset)}")
    
    # 3. 加载分词器
    print("\n3. 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507",
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("✓ 分词器加载完成")
    
    # 4. 配置量化
    print("\n4. 配置模型量化...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    print("✓ 量化配置完成")
    
    # 5. 加载模型并配置LoRA
    print("\n5. 加载模型并配置LoRA...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-4B-Instruct-2507",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        print("✓ 模型加载和LoRA配置成功")
        print(f"  可训练参数数量: {model.print_trainable_parameters()}")
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 6. 定义奖励函数
    print("\n6. 定义奖励函数...")
    def simple_reward_func(completions, **kwargs):
        """
        简化奖励函数，仅基于长度:
        根据文本长度分配奖励值：
            优先奖励长度在 20-50 字符的 “适中文本”（得 1.0 满分）；
            对短于 20 字符的文本，按长度比例降分（如 5 字符得 0.25）；
            对长于 50 字符的文本，按梯度降分（如 60 字符得 0.9，150 字符得 0.0，超过 150 字符仍为 0.0）；
            最终返回与输入文本列表一一对应的奖励值列表（奖励值范围 0.0-1.0）。
        """
        rewards = []
        for completion in completions:
            # 简单奖励：长度适中（20-50字符）得高分
            length = len(completion)
            if 20 <= length <= 50:
                reward = 1.0
            elif length < 20:
                reward = length / 20.0
            else:
                reward = max(0, 1.0 - (length - 50) / 100.0)
            rewards.append(reward)
        return rewards
    
    print("✓ 奖励函数定义完成")
    
    # 7. 创建GRPOTrainer
    print("\n7. 创建GRPOTrainer...")
    try:
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_funcs=simple_reward_func
        )
        print("✓ GRPOTrainer创建成功")
    except Exception as e:
        print(f"✗ GRPOTrainer创建失败: {e}")
        return
    
    # 8. 进行少量训练
    print("\n8. 开始训练...")
    try:
        trainer.train()
        print("✓ 训练完成")
    except Exception as e:
        print(f"✗ 训练过程中出错: {e}")
        print("注意：这可能是由于显存不足或模型配置问题")
        return
    
    # 9. 保存模型
    print("\n9. 保存模型...")
    try:
        trainer.save_model("./grpo_test_model")
        print("✓ 模型保存完成")
    except Exception as e:
        print(f"✗ 模型保存失败: {e}")
    
    print("\n" + "=" * 60)
    print("GRPO简化测试完成！")
    print("=" * 60)
    print("\n测试总结:")
    print("✓ 所有模块初始化成功")
    print("✓ 模型和分词器加载正常")
    print("✓ GRPOTrainer创建成功")
    print("✓ 训练流程可以执行")
    print("\n注意：完整训练需要更多显存和训练时间")

if __name__ == "__main__":
    main()