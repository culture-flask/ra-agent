"""生成中断注册表（进程内）：记录用户请求停止的 session_id。

generate 节点逐 token 轮询本表，命中即中断流式生成并保留已生成的
部分答复。单进程部署下够用；停止请求只设置标记，不直接杀任务——
图自己在下一个 token 边界优雅退出（部分答复正常写入 checkpoint）。
"""

_stopped: set[str] = set()


def request_stop(session_id: str) -> None:
    """用户请求终止该会话当前生成（幂等；对未在生成的会话无害）。"""
    _stopped.add(session_id)


def is_stopped(session_id: str) -> bool:
    return session_id in _stopped


def clear_stop(session_id: str) -> None:
    """清除停止标记：每轮对话开始时与 generate 消费后各清一次。

    开始时清保证「上一轮结束后才点的停止」不会误杀本轮对话。
    """
    _stopped.discard(session_id)
