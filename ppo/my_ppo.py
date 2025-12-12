"""
这个是官方源码的PPO示例（经简单修改）
"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像加速
import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
from trl.experimental.ppo import PPOConfig, PPOTrainer


if __name__ == "__main__":
    # 硬编码命令行参数
    script_args = ScriptArguments(
        dataset_name="trl-internal-testing/descriptiveness-sentiment-trl-style",
        dataset_train_split="descriptiveness"
    )
    
    training_args = PPOConfig(
        learning_rate=3e-6,
        output_dir="pythia-1b-deduped-descriptiveness-sentiment-trl-style-ppo",
        per_device_train_batch_size=64,
        gradient_accumulation_steps=1,
        total_episodes=100,
        missing_eos_penalty=1.0
    )
    
    model_args = ModelConfig(
        model_name_or_path="EleutherAI/pythia-1b-deduped"
    )
    
    # 递归地删除训练输出目录及其所有内容，并在删除过程中忽略可能出现的错误
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    #当 model_args.dtype为 "auto"或 None时，直接使用该值。
    #"auto"通常表示让框架自动选择最优数据类型（如支持BF16时优先使用），None通常代表使用模型默认的精度（一般是FP32）。
    #当 model_args.dtype是其他字符串时（例如 "float16", "bfloat16"），
    #通过 getattr(torch, model_args.dtype)将这个字符串转换为PyTorch中对应的实际数据类型（如 torch.float16, torch.bfloat16）。
    #这是因为model_args.dtype来自配置输入，通常是字符串，而模型加载函数需要的是torch.dtype类型的对象

    model_kwargs = dict(    
        revision=model_args.model_revision,  # 指定模型的版本或分支，默认为'main'，可用于加载特定版本的模型权重
        attn_implementation=model_args.attn_implementation,  # 指定注意力机制的实现方式，如'eager'、'flash_attention_2'等，影响模型推理和训练效率
        dtype=dtype,  
    )
    quantization_config = get_quantization_config(model_args)  # 根据model_args生成量化配置，用于模型权重的量化压缩
    if quantization_config is not None:  # 检查量化配置是否有效
        model_kwargs["device_map"] = get_kbit_device_map()  # 获取适合量化模型的设备映射，决定模型各部分在哪些设备上运行
        model_kwargs["quantization_config"] = quantization_config  # 将量化配置添加到模型参数中，用于后续模型加载




    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})




    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )

    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )




    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """在训练前对数据集进行预分词（pre-tokenize）；仅在训练过程中进行数据整理（collate）。"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # 仅在主进程（main process）上执行该计算，以实现更快的数据处理（data processing）。
    # 参考：https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)  # 将训练好的模型保存到指定目录

    trainer.generate_completions()  # 使用训练好的模型生成文本完成结果，用于评估模型性能