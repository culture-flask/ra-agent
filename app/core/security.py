# 密码哈希工具
import bcrypt

def hash_password(plain: str) -> str:
    """把明文密码哈希成可存储的字符串。"""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文是否与已存哈希匹配。匹配返回 True，否则 False。"""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))