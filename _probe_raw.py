import base64, hashlib, time, json, subprocess
from cryptography.fernet import Fernet

master = "dev-secret-change-me-to-32+bytes!!"
key = base64.urlsafe_b64encode(hashlib.sha256(master.encode()).digest())
f = Fernet(key)
out = subprocess.check_output(
    ["docker","exec","ra-postgres","psql","-U","ra","-d","ra_agent","-tA","-c",
     "SELECT api_key FROM user_llm_config WHERE base_url LIKE '%opencode%' ORDER BY created_at DESC LIMIT 1;"]
).decode().strip()
apikey = f.decrypt(out.encode()).decode()

from openai import OpenAI
c = OpenAI(api_key=apikey, base_url="https://opencode.ai/zen/go/v1")

# 用原生 openai SDK 直接看 chunk 的完整字段结构（不做任何转换）
stream = c.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role":"user","content":"3.11 和 3.9 哪个大？"}],
    stream=True,
)
shown_keys = set()
n_reason = 0
sample_reason = []
for i, chunk in enumerate(stream):
    if chunk.choices and chunk.choices[0].delta:
        d = chunk.choices[0].delta
        # 打印第一条 chunk 的完整原始字段名
        if i == 0:
            print("=== 第一条 chunk 的字段(raw) ===")
        rc = getattr(d, "reasoning_content", None)
        cc = d.content
        if rc:
            n_reason += 1
            if len(sample_reason) < 3:
                sample_reason.append(rc[:60])
        if i == 0:
            # 把 delta 转 dict 看所有 key
            print("delta keys:", list(d.__dict__.keys()) if hasattr(d,'__dict__') else "no __dict__")
            print("model_dump:", d.model_dump() if hasattr(d,'model_dump') else "n/a")
        if i > 60:
            break

print("=== 结果 ===")
print("是否为 reasoning_content 流：", n_reason > 0)
print("reasoning 样例：", sample_reason)
