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
    async for chunk in model.astream([("user", "3.11 和 3.9 哪个大？")]):
        n += 1
        if n <= 2:
            # 深挖 response_metadata 和 model_extra 等所有可能的藏身处
            print(f"--- chunk {n} ---")
            print("content:", repr(chunk.content)[:50])
            print("response_metadata:", chunk.response_metadata)
            print("additional_kwargs:", chunk.additional_kwargs)
            for attr in dir(chunk):
                if 'reason' in attr.lower() or 'think' in attr.lower() or 'extra' in attr.lower():
                    try:
                        v = getattr(chunk, attr)
                        print(f"  .{attr} = {repr(v)[:80]}")
                    except Exception:
                        pass
        if n > 3:
            break
asyncio.run(main())
