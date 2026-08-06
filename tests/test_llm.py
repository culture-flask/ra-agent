# 工厂与路由（不调用网络，只断言配置正确）
from app.abstractions.llm import LLMConfig, LLMService
from app.settings import Settings


def test_system_default_fallback():
    s = Settings.load()
    svc = LLMService(system_default=s.llm_system_default, system_api_key=s.llm_api_key)
    assert svc.get_chat_model("u1").model_name == s.llm_system_default["model_id"]


def test_user_config_overrides():
    s = Settings.load()
    svc = LLMService(system_default=s.llm_system_default, system_api_key=s.llm_api_key)
    svc.set_user_config("u2", LLMConfig(provider="openai",
                                        base_url="https://api.openai.com/v1",
                                        model_id="gpt-4o-mini",
                                        api_key="sk-test"))
    assert svc.get_chat_model("u2").model_name == "gpt-4o-mini"