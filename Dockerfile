FROM python:3.10-slim

WORKDIR /app

# 多模态音视频的探测、音频规范化与视频抽帧。
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单单独安装：利用 Docker 层缓存（依赖不变就不用重装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制代码
COPY . .

RUN mkdir -p /data/chroma /data/chunks

EXPOSE 8000

CMD ["python", "-m", "app.main"]
