import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

def setup_logging(level: int = logging.INFO, log_file: str | None = "logs/app.log"):
    root = logging.getLogger()          # 根 logger，所有模块的日志汇到这里
    root.setLevel(level)

    fmt = logging.Formatter(_FORMAT)

    sh = logging.StreamHandler(sys.stdout)  # 输出到终端
    sh.setFormatter(fmt)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")   #  同时写文件
        fh.setFormatter(fmt)
        root.addHandler(fh)
    
def get_loger(name: str) -> logging.Logger:
    return logging.getLogger(name)