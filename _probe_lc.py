import asyncio, base64, hashlib, subprocess
from cryptography.fernet import Fernet

master = "dev-secret-change-me-to-32+bytes!!"
key = base64.urlsafe_b64encode(hashlib.sha256(master.encode()).digest())
f = Fernet(key)
out = subprocess.check_output(
    ["docker","exec","ra-postgres","psql","-U","ra","-d","ra_agent","-tA","-c",
     "SELECT api_key FROM user_llm_config WHERE base_url LIKE '%opencode%' ORDER BY created_at DESC LIMIT 1;"]
).decode().strip()
apikey = f.decrypt(out.encode()).decode()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=apikey,
    base_url="https://opencode.ai/zen/go/v1",
    streaming=True,
    stream_usage=True,
    request_timeout=60,
    max_retries=0,
)

async def main():
    n = 0
    reason_chars = 0
    content_chars = 0
    saw_reason = False
    sample = []
    import time
    t0 = time.perf_counter()
    first_reason = first_content = None
    async for chunk in model.astream([("user", "3.11 和 3.9 哪个大？")]):
        n += 1
        rc = None
        cc = chunk.content if isinstance(chunk.content, str) else ""
        # 尝试各种位置找 reasoning
        ak = chunk.additional_kwargs or {}
        if "reasoning_content" in ak:
            rc = ak["reasoning_content"]
            saw_reason = True
        if rc:
            if first_reason is None:
                first_reason = time.perf_counter() - t0
                if len(sample) < 3:
                    sample.append(rc[:60])
            reason_chars += len(rc)
        if cc:
            if first_content is None:
                first_content = time.perf_counter() - t0
            content_chars += len(cc)
        if n == 1:
            print("chunk1 additional_kwargs keys:", list(ak.keys()))
            print("chunk1 has __dict__:", hasattr(chunk, '__dict__'))
            if hasattr(chunk, '__dict__'):
                print("chunk1 __dict__ keys:", list(chunk.__dict__.keys()))
    print("=== langchain astream 结果 ===")
    print("chunks:", n)
    print("saw_reasoning in additional_kwargs:", saw_reason)
    print("reasoning_chars:", reason_chars, "content_chars:", content_chars)
    print("first_reason:", first_reason, "first_content:", first_content)
    print("sample:", sample)

asyncio.run(main())
