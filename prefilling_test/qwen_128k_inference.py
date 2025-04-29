import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置
model_name = "Qwen/Qwen2-7B"
target_device = "cuda:2"  # 这里指定 GPU2

# 1. 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": target_device},  # 强制放cuda:2
    torch_dtype=torch.bfloat16,       # 建议用bf16，省内存
    trust_remote_code=True
)
model.eval()

# 2. 生成伪输入（比如128k tokens）
seq_len = 128 * 1024  # 128k tokens

input_ids = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(1, seq_len),
    dtype=torch.long,
    device=target_device  # 注意！输入也在cuda:2
)

attention_mask = torch.ones_like(input_ids)

# 3. Prefill 测时间
torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

torch.cuda.synchronize()
end_time = time.time()

prefill_time = end_time - start_time
print(f"\n✅ Prefill {seq_len} tokens on {target_device} took {prefill_time:.2f} seconds")
