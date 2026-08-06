import os


def apply_proxy(settings) -> None:
    """把 settings 里的代理配置注入环境变量。
    setdefault：环境变量已存在（如本地终端已 export）时保持原样，不覆盖。"""
    if settings.http_proxy:
        os.environ.setdefault("http_proxy", settings.http_proxy)
        os.environ.setdefault("HTTP_PROXY", settings.http_proxy)
    if settings.https_proxy:
        os.environ.setdefault("https_proxy", settings.https_proxy)
        os.environ.setdefault("HTTPS_PROXY", settings.https_proxy)
    if settings.no_proxy:
        os.environ.setdefault("no_proxy", settings.no_proxy)
        os.environ.setdefault("NO_PROXY", settings.no_proxy)