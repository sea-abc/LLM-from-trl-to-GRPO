"""
这个代码实现了：
1.PPO训练后模型单独的推理脚本
2.使用训练好的策略模型进行文本生成和互动式交流（未添加记忆模块）
"""

# ====================== 1. 基础设置 ======================
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像加速

# ====================== 2. 导入核心库 ======================
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)

def load_trained_model(model_path):
    """加载训练好的PPO模型"""
    
    # 使用4-bit量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",  # Qwen模型使用右填充
        trust_remote_code=True
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载训练好的策略模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16
    )
    
    return tokenizer, model

def generate_response(tokenizer, model, question, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """使用训练好的模型生成回复"""
    
    # 设置生成参数（显式设置Qwen模型的默认值以避免警告）
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=20,  # Qwen模型的默认值
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=151643  # Qwen模型的开始标记ID
    )
    
    # 构建Qwen对话格式
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
    
    # 提取assistant的回复部分
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").strip()
    else:
        response = generated_text.replace(prompt, "").strip()
    
    return response

def interactive_inference(model_path):
    """交互式推理模式"""
    print("正在加载训练好的模型...")
    tokenizer, model = load_trained_model(model_path)
    print("模型加载完成！\n")
    
    print("=" * 60)
    print("PPO训练模型推理模式")
    print("输入 'quit' 或 '退出' 结束对话")
    print("=" * 60)
    
    while True:
        try:
            question = input("\n请输入您的问题: ").strip()
            
            if question.lower() in ['quit', '退出', 'exit']:
                print("感谢使用，再见！")
                break
            
            if not question:
                print("问题不能为空，请重新输入。")
                continue
            
            print("\n正在生成回复...")
            response = generate_response(tokenizer, model, question)
            print(f"\n回答: {response}")
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"生成过程中出现错误: {e}")
            continue

def batch_inference(model_path, questions):
    """批量推理模式"""
    print("正在加载训练好的模型...")
    tokenizer, model = load_trained_model(model_path)
    print("模型加载完成！\n")
    
    print("=" * 60)
    print("批量推理测试")
    print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        print("正在生成回复...")
        
        response = generate_response(tokenizer, model, question)
        print(f"回答: {response}")
        print("-" * 50)
    
    print("\n批量推理完成！")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO训练模型推理脚本")
    parser.add_argument("--model_path", type=str, default="qwen3-4b-instruct-ppo", 
                       help="训练好的模型路径")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch"], default="batch",
                       help="推理模式: interactive(交互式) 或 batch(批量)")
    parser.add_argument("--questions", type=str, nargs="+", 
                       default=["什么是LoRA？", "LoRA有什么优势？", "什么是PPO算法？"],
                       help="批量推理的问题列表")
    
    args = parser.parse_args()
    
    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径 '{args.model_path}' 不存在！")
        print("请先运行训练脚本或指定正确的模型路径。")
        exit(1)
    
    if args.mode == "interactive":
        interactive_inference(args.model_path)
    else:
        batch_inference(args.model_path, args.questions)