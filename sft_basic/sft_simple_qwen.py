"""
这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的简单指令微调
2. 手动控制需要训练参数的层
3. 使用pytorch的传统深度学习预训练框架
"""
# 简单的 Qwen3-4B-Instruct 模型微调代码
# https://hf-mirror.com/Qwen/Qwen3-4B-Instruct-2507/tree/main

# ====================== 1. 国内镜像加速（必须放在库导入前面） ======================
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ====================== 2. 导入库 ======================
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer 

# ====================== 2. 模型和分词器 ======================
# 清理显存和缓存
torch.cuda.empty_cache()
gc.collect()


model_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    padding_side="right"
)

# 设置pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id



# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,  # 使用新的参数名 dtype 替代已废弃的 torch_dtype
    trust_remote_code=True
).to("cuda")

"""
# 打印前10个参数名称，了解Qwen模型的参数结构
print("\nQwen模型参数名称结构示例：")
for i, (name, param) in enumerate(model.named_parameters()):
    print(f"{i}. {name}")
    if i >= 9:
        break

    Qwen模型参数名称结构示例：
    0. model.embed_tokens.weight
    1. model.layers.0.self_attn.q_proj.weight
    2. model.layers.0.self_attn.k_proj.weight
    3. model.layers.0.self_attn.v_proj.weight
    4. model.layers.0.self_attn.o_proj.weight
    5. model.layers.0.self_attn.q_norm.weight
    6. model.layers.0.self_attn.k_norm.weight
    7. model.layers.0.mlp.gate_proj.weight
    8. model.layers.0.mlp.up_proj.weight
    9. model.layers.0.mlp.down_proj.weight
"""

# 只让极少数关键参数参与训练，大幅减少显存占用
# 配置：只训练最后2层的关键参数
trainable_params = 0
all_param = 0
num_layers_to_train = 2

# 获取总层数
total_layers = model.config.num_hidden_layers
start_layer = max(0, total_layers - num_layers_to_train)

# 遍历所有参数
for name, param in model.named_parameters():
    # 默认冻结所有参数
    param.requires_grad = False
    all_param += param.numel()
    
    # 1. 启用输出层训练
    if "lm_head" in name:
        param.requires_grad = True
        trainable_params += param.numel()
        continue
    
    # 2. 启用最后num_layers_to_train层的关键参数训练
    if "model.layers." in name:
        # 提取层号
        # 例子：model.layers.0.self_attn.q_proj.weight => ["model.layers.","0.self_attn.q_proj.weight"]
        #                                             => ["0.self_attn.q_proj.weight"]
        #                                             => ["0","self_attn","q_proj","weight"]
        #                                             => ["0"]
        layer_part = name.split("model.layers.")[1].split(".")[0] 
        if layer_part.isdigit(): #检查是否是纯数字字符串
            layer_num = int(layer_part)
            # 检查是否是最后num_layers_to_train层
            if layer_num >= start_layer:
                # 检查是否是关键参数
                if any(key in name for key in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", 
                                              "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]):
                    param.requires_grad = True
                    trainable_params += param.numel()

# 打印参数统计信息（可选）
print(f"\ntrainable params: {trainable_params:,} ({trainable_params/all_param*100:.2f}%)")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# ====================== 3. 训练数据 ======================
dialog = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "什么是LoRA？"},
    {"role": "assistant", "content": "LoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数。"}
]

# 使用apply_chat_template方法将对话历史转换为模型可处理的输入格式
# 该方法是Hugging Face针对聊天模型设计的专用方法，会自动添加模型所需的特殊标记
# 参数说明：
# - dialog: 对话历史列表，每个元素是包含"role"和"content"的字典
#   - "role": 角色标识，通常为"system"、"user"或"assistant"
#   - "content": 对应角色的对话内容
# - return_tensors: 指定返回的张量类型，"pt"表示PyTorch张量
# 返回值：
# - input_ids: 模型可处理的输入序列，已包含适当的特殊标记（如<|im_start|>、<|im_end|>等）
input_ids = tokenizer.apply_chat_template(dialog, return_tensors="pt")

# 准备输入数据
input = {
    "input_ids": input_ids.to("cuda"),
    "attention_mask": torch.ones_like(input_ids).to("cuda")
}

# 设置labels（与inputs一致）
input["labels"] = input["input_ids"].clone()

# ====================== 4. 训练 ======================
# 前向传播
output = model(**input)

# 获取损失
loss = output.loss
print(f"Loss: {loss.item()}")

# 反向传播和优化
loss.backward()
optimizer.step()
optimizer.zero_grad()

# ====================== 5. 推理测试 ======================

print("\n推理测试：")
test_dialog = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "用优美的词来夸一夸我吧"}
]

# 使用apply_chat_template方法准备推理输入
# 参数说明：
# - test_dialog: 测试对话历史，格式与训练数据相同
# - add_generation_prompt=True: 关键参数，用于推理时在对话末尾添加模型生成回答所需的提示标记
#   对于Qwen模型，这会在对话最后添加<|im_start|>assistant
# - return_tensors="pt": 返回PyTorch张量
# - .to("cuda"): 将输入张量移动到GPU上加速推理
# 返回值：
# - test_input: 完整的推理输入序列，包含生成回答所需的所有标记
test_input = tokenizer.apply_chat_template(test_dialog, add_generation_prompt=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    test_output = model.generate(
        test_input,
        max_new_tokens=500,
        temperature=0.7,
        attention_mask=torch.ones_like(test_input)  # 添加attention mask解决警告
    )

# 直接解码输出并提取助手回答部分
answer = tokenizer.decode(test_output[0][test_input.shape[-1]:], skip_special_tokens=True)
print(f"user: {test_dialog[-1]['content']}") # 打印问题
print(f"assistant: {answer}") #打印回答

# 清理显存
torch.cuda.empty_cache()
gc.collect()

# ====================== 6. 保存模型 ======================
# 保存完整模型（由于只训练了部分参数，实际上只有这些参数会被更新）
model.save_pretrained("./qwen3_simple_freeze")
tokenizer.save_pretrained("./qwen3_simple_freeze")
print("\n模型已保存到 ./qwen3_simple_freeze")