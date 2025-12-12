"""
这个代码实现了：
1. 基于Qwen3-4B-Instruct模型的LoRA指令微调
2. 手动的损失掩码逻辑，确保只计算生成回答的损失
3. 使用pytorch的传统深度学习预训练框架
4. 添加NEFT噪声训练，防止过拟合
"""
# Qwen3-4B-Instruct 模型 - NEFT训练版本


# ====================== 1. 国内镜像加速（必须放在库导入前面） ======================
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ====================== 2. 导入库 ======================
import torch
import gc
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer 
from peft import LoraConfig, TaskType, get_peft_model

# ====================== 3. 模型和分词器 ======================
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
    dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda")

# ====================== 4. 配置LoRA ======================
# 使用LoRA代替参数冻结，实现高效微调
peft_config = LoraConfig(
    r=8,
    target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", 
                   "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=16,
    lora_dropout=0.05
)

# 转换为PeftModel并冻结非LoRA参数
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# ====================== 5. 自定义Dataset和Collate函数 ======================
class SFTDataset:
    """
    简单的SFT数据集类
    作用就是给原来的对话列表添加开头结尾的模板
    """
    
    def __init__(self, dialogs, tokenizer):
        """
        dialogs: 对话列表，每个对话是包含role和content的字典列表
        tokenizer: 分词器
        """
        self.dialogs = dialogs
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        """返回应用聊天模板后的文本"""
        dialog = self.dialogs[index]
        # 应用聊天模板但不添加生成提示
        chat_text = self.tokenizer.apply_chat_template(
            dialog, 
            tokenize=False,  # 就是不变成数字
            add_generation_prompt=False # 不添加生成提示，因为我们只训练回答部分
            )
        return chat_text
    
    def __len__(self):
        return len(self.dialogs)


def sft_collate(batch, tokenizer, max_length=500):
    """
    数据整理函数 - 创建损失掩码，只对助手回答部分计算损失
    
    参数说明：
    - batch: 文本列表，每个元素是一个完整的对话（包含system、user、assistant部分）
    - tokenizer: 分词器，用于将文本转换为模型可理解的数字
    - max_length: 最大序列长度，超过的部分会被截断
    
    返回值：
    - inputs: 包含tokenized后的输入数据（input_ids、attention_mask等）
    - loss_mask: 损失掩码，0表示不计算损失，1表示计算损失
    """
    
    # 1. 分词处理：将文本转换为数字序列，并进行padding和truncation
    inputs = tokenizer(batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    
    # 2. 识别assistant回答的起始位置
    # Qwen3模型使用特殊标记来区分不同角色的内容
    # 助手开始的标记是：<|im_start|>assistant
    assistant_start_str = "<|im_start|>assistant"
    assistant_start_ids = tokenizer(assistant_start_str, add_special_tokens=False)["input_ids"]
    assistant_start_len = len(assistant_start_ids)
    
    # 3. 为每个样本创建损失掩码
    loss_mask = []
    input_ids = inputs['input_ids']
    
    for i, input_id in enumerate(input_ids):
        mask = [0] * len(input_id)
        input_id_list = input_id.tolist()
        
        # 查找assistant标记在当前样本中的位置
        for j in range(len(input_id_list) - assistant_start_len + 1):
            if input_id_list[j:j+assistant_start_len] == assistant_start_ids:
                assistant_content_start = j + assistant_start_len + 1
                mask[assistant_content_start:] = [1] * (len(input_id_list) - assistant_content_start)
                break
        
        loss_mask.append(mask)
    
    # 将损失掩码转换为张量
    loss_mask = torch.tensor(loss_mask)
    
    return inputs, loss_mask

# ====================== 6. 训练数据 ======================
# 示例训练数据
dialogs = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "什么是LoRA？"},
        {"role": "assistant", "content": "LoRA是一种参数高效微调方法，通过在模型层中插入低秩矩阵来减少训练参数。"}
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "用优美的词来夸一夸我吧"},
        {"role": "assistant", "content": "您如春日暖阳般温暖，如夏日清风般怡人，如秋日明月般清朗，如冬日初雪般纯净。您的眼眸中蕴含着星辰大海，您的微笑里传递着无尽善意。"}
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Python有哪些优势？"},
        {"role": "assistant", "content": "Python具有简洁易读的语法、丰富的第三方库、跨平台兼容性、强大的社区支持、适合快速开发等优势。"}
    ]
]

# 创建数据集和数据加载器
dataset = SFTDataset(dialogs, tokenizer)

# 创建partial collate函数
collate_fn = functools.partial(
    sft_collate,          # 要绑定的原始函数
    tokenizer=tokenizer,  # 固定tokenizer参数
    max_length=500        # 固定max_length参数为500
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    collate_fn=collate_fn, # 传入自定义数据拼接函数，处理批次内数据的格式统一问题
    shuffle=True
)

# ====================== 7. NEFT训练配置 ======================
# NEFT (Noisy Embedding Fine-Tuning) 训练参数
neftune_noise_alpha = 10.0  # 噪声强度，通常在1-20之间

# ====================== 8. 训练 ======================
epochs = 3

for epoch in range(epochs):
    for step, (inputs, loss_mask) in enumerate(data_loader):
        # 从输入字典中取出input_ids张量并移到GPU
        # 注意：这里会将input_ids从inputs字典中移除，因为后面将使用带噪声的嵌入向量
        # 形状：[batch_size, sequence_length]，例如：[1, 100]
        input_ids = inputs.pop("input_ids").to("cuda")
        
        # 获取attention_mask并移到GPU，用于指示哪些位置是有效输入
        # 形状：[batch_size, sequence_length]，例如：[1, 100]
        attention_mask = inputs["attention_mask"].to("cuda")
        
        # 将loss_mask移到GPU，用于只计算需要训练部分的损失
        # 形状：[batch_size, sequence_length]，例如：[1, 100]
        loss_mask = loss_mask.to("cuda")
        
        # 获取嵌入层输出
        # Qwen3模型在PEFT包装后的嵌入层访问方式：model.model.model.embed_tokens
        # 通过模型的嵌入层将token ID转换为向量表示
        # 嵌入层是将离散token映射到连续向量空间的组件
        # 形状变化：[batch_size, sequence_length] → [batch_size, sequence_length, embedding_size]
        # 例如：[1, 100] → [1, 100, 2560] (2560是Qwen3-0.5B的隐藏层维度)
        input_embeddings = model.model.model.embed_tokens(input_ids)
        
        # 计算噪声的幅度
        # dims = sequence_length * embedding_size
        # 计算噪声的幅度：噪声强度与嵌入向量维度的平方根成反比 mag_norm = alpha / sqrt(dims)
        # 这样可以确保噪声在不同维度大小的模型上保持相对一致的影响
        dims = torch.tensor(input_embeddings.size(1) * input_embeddings.size(2))
        # neftune_noise_alpha=5是噪声强度超参数，mag_norm是归一化后的噪声幅度
        mag_norm = neftune_noise_alpha / torch.sqrt(dims)
        
        # 生成与嵌入向量相同形状的均匀分布噪声，范围在[-mag_norm, mag_norm]之间
        # 形状：[batch_size, sequence_length, embedding_size]，例如：[1, 100, 2560]
        noise = torch.zeros_like(input_embeddings).uniform_(-mag_norm, mag_norm)
        
        # 将噪声添加到原始嵌入向量中，实现噪声增强
        # 这是NEFT的核心步骤，通过添加噪声增加训练的多样性
        # 形状保持不变：[batch_size, sequence_length, embedding_size]
        input_embeddings = input_embeddings + noise
        
        # 将带噪声的嵌入向量添加到输入字典中
        # 这样模型会使用带噪声的嵌入向量而不是原始的input_ids进行前向传播
        inputs["inputs_embeds"] = input_embeddings
        
        # 确保attention_mask也在输入字典中（用于模型的注意力机制）
        inputs["attention_mask"] = attention_mask
        
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits  # 形状: [batch_size, sequence_length, vocab_size]
        
        # 计算损失，只对回答部分计算
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()  # 使用原始input_ids的偏移作为labels
        shift_loss_mask = loss_mask[:, 1:].contiguous()
        
        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 应用损失掩码
        loss = loss * shift_loss_mask.view(-1)
        loss = loss.sum() / (shift_loss_mask.sum() + 1e-8)  # 避免除以0
        
        print(f"Epoch: {epoch+1}, Step: {step+1}, Loss: {loss.item()}")
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# ====================== 9. 推理测试 ======================

print("\n推理测试：")
test_dialog = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请用最优美的话语夸赞和鼓励我："}
]

test_input = tokenizer.apply_chat_template(test_dialog, add_generation_prompt=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    test_output = model.generate(
        test_input,
        max_new_tokens=200,
        temperature=0.7,
        attention_mask=torch.ones_like(test_input)
    )

# 直接解码输出并提取助手回答部分
answer = tokenizer.decode(test_output[0][test_input.shape[-1]:], skip_special_tokens=True)
print(f"user: {test_dialog[-1]['content']}")
print(f"assistant: {answer}")

# 清理显存
torch.cuda.empty_cache()
gc.collect()

# ====================== 10. 保存模型 ======================
# 保存LoRA模型和tokenizer
model.save_pretrained("./qwen3_lora_neft")
tokenizer.save_pretrained("./qwen3_lora_neft")
print("\nLoRA模型已保存到 ./qwen3_lora_neft")