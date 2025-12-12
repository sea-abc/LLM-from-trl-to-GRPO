# LLM 强化学习与微调示例代码库

这个仓库包含了基于 Qwen3-4B-Instruct 模型的强化学习（PPO/GRPO）和监督微调（SFT）的示例代码，用于学习和实践大语言模型的微调技术。

## 目录结构

```
├── README.md         # 帮助文档
├── 官方文档/          # 相关库和模型的官方文档
├── 学习代码用的提示词/  # 构建代码的提示词和参考链接
│   ├── 官网
│   └── 构建代码
├── grpo/             # GRPO 强化学习训练相关代码
│   ├── grpo_trl_qwen_new.py
│   ├── ppo_reward_trl_qwen.py
│   └── test_grpo_short.py
├── ppo/              # PPO 强化学习训练相关代码
│   ├── inference_qwen_ppo.py
│   ├── my_ppo.py
│   ├── ppo_reward_trl_qwen.py
│   ├── ppo_trl_qwen.py
│   └── ppo_trl_qwen_new.py
└── sft_basic/        # 监督微调（SFT）相关代码
    ├── sft_LoRA_qwen.py
    ├── sft_NEFT_qwen.py
    ├── sft_completions_only_qwen.py
    ├── sft_simple_qwen.py
    ├── sft_trl_qwen.py
    └── sft_trl_qwen_completion.py
```

## 文件夹说明

### 1. 官方文档

包含了各种与项目相关的官方文档，提供了使用相关库和模型的详细信息。
主要是用于让大模型改代码时参考用的。

**所有文件：**
- `AutoModelForCausalLMWithValueHead.txt`：带价值头的因果语言模型使用文档
- `DataCollatorForLanguageModeling官方文档.txt`：语言模型数据拼接器使用说明
- `Dataset Formats.txt`：数据集格式要求说明
- `GRPOConfig.txt`：GRPO 训练配置参数说明
- `GRPOTrainer.txt`：GRPO 训练器使用文档
- `GRPO_quick_start.txt`：GRPO 快速入门指南
- `GRPO_reward_model.txt`：GRPO 奖励模型使用说明
- `PPO官方文档.txt`：PPO 训练算法官方文档
- `RewardModel官方文档.txt`：奖励模型使用说明
- `SFTConfig官方文档.txt`：SFT 训练配置说明
- `SFTTrainer官方文档.txt`：SFT 训练器使用文档
- `ScriptArguments,PPOConfig,ModelConfig.txt`：脚本参数和模型配置说明
- `completion_only_loss参数的官方文档.txt`：completion_only_loss参数使用说明

### 2. 学习代码用的提示词

包含了构建代码的提示词和相关参考链接，用于指导如何构建 PPO 和 GRPO 训练代码。

**主要文件：**
- `官网`：TRL 库官方文档链接
- `构建代码`：详细的代码构建提示词和需求说明

### 3. grpo

包含了基于 Qwen3-4B-Instruct 模型的 GRPO（Generalized Reinforcement Preference Optimization）强化学习训练相关代码。

**主要文件：**

#### `grpo_trl_qwen_new.py`

这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的4-bit量化 + LoRA GRPO训练
2. 使用了多种奖励函数：预训练奖励模型、长度奖励、内容质量奖励和格式规范奖励
3. 包含训练后的模型推理测试功能


#### `ppo_reward_trl_qwen.py`

这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的4-bit量化 + LoRA奖励模型训练
2. 使用TRL库的RewardTrainer进行奖励模型训练
3. 供PPO和GRPO训练共用


#### `test_grpo_short.py`

这个代码实现了：
1. 简化的GRPO训练流程（仅1个训练步骤）
2. 使用简单的奖励函数（句子长度在20-50词时奖励最高）
3. 快速验证GRPO流程是否能正常运行


### 4. ppo

包含了基于 Qwen3-4B-Instruct 模型的 PPO（Proximal Policy Optimization）强化学习训练相关代码。

**主要文件：**

#### `my_ppo.py`

- 官方源码的 PPO 示例（经简单修改）
- 使用 Pythia-1B 模型进行演示
- 包含完整的 PPO 训练流程

#### `ppo_trl_qwen.py`

- 基于 Qwen3-4B-Instruct 模型的 PPO 训练示例代码
- 支持 4-bit 量化 + LoRA 微调
- 包含训练后的自动推理测试功能

#### `ppo_trl_qwen_new.py`

- ppo_trl_qwen.py 的更新版本
- 支持加载已经训练好的奖励模型
- 功能与 ppo_trl_qwen.py 大致相同

#### `inference_qwen_ppo.py`

- PPO 训练后模型单独的推理脚本
- 支持交互式交流和批量推理
- 使用 4-bit 量化加载训练好的模型

### 5. sft_basic

包含了各种基于 Qwen 模型的监督微调（SFT，Supervised Fine-Tuning）相关代码。

**主要文件：**

#### `sft_LoRA_qwen.py`

这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的4-bit量化 + LoRA指令微调
2. 使用transformers库的Trainer类进行训练


#### `sft_NEFT_qwen.py`

这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的LoRA指令微调
2. 手动的损失掩码逻辑，确保只计算生成回答的损失
3. 使用pytorch的传统深度学习预训练框架
4. 添加NEFT噪声训练，实现数据增强


#### `sft_completions_only_qwen.py`

这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的LoRA指令微调
2. 手动的损失掩码逻辑，确保只计算生成回答的损失
3. 使用pytorch的传统深度学习预训练框架


#### `sft_simple_qwen.py`

这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的简单指令微调
2. 手动控制需要训练参数的层
3. 使用pytorch的传统深度学习预训练框架


#### `sft_trl_qwen.py`

这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的LoRA指令微调
2. 使用TRL库的SFTTrainer类进行训练
3. TRL库不使用损失掩码逻辑，NEFT噪声训练


#### `sft_trl_qwen_completion.py`

这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的LoRA指令微调
2. 使用TRL库的SFTTrainer类进行训练
3. TRL库自动实现了损失掩码逻辑，NEFT噪声训练


## 环境要求

**详细的环境搭建流程请参考下一节**

- Python 3.12
- PyTorch
- Transformers
- TRL
- PEFT
- Accelerate
- Datasets
- BitsAndBytes

## 使用说明

1. **环境的详细搭建流程**：

### 推荐使用autodl云端配置

**此环境是在autodl中租赁下创建的，使用的配置为：**

   ```
   镜像 PyTorch  2.5.1 Python  3.12(ubuntu22.04) CUDA  12.4   # 请勿轻易更换
   GPU RTX 3090(24GB) * 1 升降配置  
   CPU 15 vCPU Intel(R) Xeon(R) Platinum 8358P CPU @2.60GHz
   内存 90GB
   硬盘 
   系统盘:30 GB 
   数据盘:免费:50GB   付费:0GB
   ```

**1.1 下载此仓库**
   ```bash
   cd /root/autodl-tmp

   # 克隆到临时文件夹（避免目录冲突）
   git clone https://github.com/sea-abc/LLM-from-trl-to-GRPO.git temp_repo

   # 将临时文件夹中的内容移动到当前目录
   mv temp_repo/* .

   # 删除临时文件夹（可选）
   rm -rf temp_repo
   ```
**1.2 初始环境复制**
   ```bash
   mkdir -p /root/autodl-tmp/llm_env
   conda create --clone /root/miniconda3 -p /root/autodl-tmp/llm_env --yes

   conda init
   source activate /root/autodl-tmp/llm_env
   ```
**1.3 安装核心依赖**
   ```bash
   pip install accelerate==1.12.0 transformers==4.57.3 bitsandbytes==0.48.2 huggingface-hub==0.36.0 tokenizers==0.22.1 ipykernel==7.1.0 peft==0.18.0 datasets==4.4.1 sentencepiece==0.2.1 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
   pip install trl==0.26.0
   ```

2. **运行训练脚本**：
   ```bash
   cd <文件夹>
   python <脚本名>.py
   ```

## 注意事项

- 所有脚本均默认使用 Qwen3-4B-Instruct-2507 模型
- 大部分脚本支持 4-bit 量化，以减少显存占用
- 建议在具有足够显存（至少 16GB）的 GPU 上运行
- 训练时间根据模型大小和训练参数有所不同

## 参考链接

- [TRL 库官方文档](https://hf-mirror.com/docs/trl/v0.26.0/en/ppo_trainer)
- [Qwen3 模型](https://hf-mirror.com/Qwen/Qwen3-4B-Instruct-2507)
- [PEFT 库](https://hf-mirror.com/docs/peft/index)
- [BitsAndBytes 量化](https://hf-mirror.com/docs/transformers/main_classes/quantization)
- 一些原理课程推荐去b占up主RethinkFun处进行学习：包括SFT微调、PPO、GRPO等(https://space.bilibili.com/18235884?spm_id_from=333.788.upinfo.detail.click)
