"""密钥加密工具：AES 对称加密（Fernet）+ 前端掩码。

- 用户 api_key 落库前加密，读出时解密——数据库泄露也看不到明文；
- 密钥由主密钥（jwt_secret）SHA-256 派生：不新增秘密配置；
- 前端只回显掩码（sk-...1234），永不下发明文。
"""

import base64
import hashlib

from cryptography.fernet import Fernet


class SecretCrypto:
    """AES 对称加解密（Fernet = AES-128-CBC + HMAC 完整性校验）。"""

    def __init__(self, master_secret: str):
        # Fernet 需要 32 字节 urlsafe base64 密钥——从主密钥确定性派生
        key = base64.urlsafe_b64encode(hashlib.sha256(master_secret.encode()).digest())
        self._fernet = Fernet(key)

    def encrypt(self, plain: str) -> str:
        """加密：明文 → 密文（落库用）。"""
        return self._fernet.encrypt(plain.encode()).decode()

    def decrypt(self, cipher: str) -> str:
        """解密：密文 → 明文（构建 LLM 客户端时用）。"""
        return self._fernet.decrypt(cipher.encode()).decode()


def mask_secret(plain: str) -> str:
    """掩码：只露前 4 位后 4 位，中间打码——前端展示用。"""
    if len(plain) <= 8:
        return "***"
    return f"{plain[:4]}...{plain[-4:]}"