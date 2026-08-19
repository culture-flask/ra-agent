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


# ---------- 后台任务取消（入库 / 重建） ----------
# 与对话停止（_stopped）独立：入库/重建是同步线程池任务，用可复用令牌。
# key = kb_id（入库）/ new_kb_id（重建）。请求端设标记，任务端在
# 文件/批次边界检查，命中抛 OperationCancelled 中断，收尾后 acknowledge 清除。
class OperationCancelled(Exception):
    """入库 / 重建被用户取消：由任务内部抛，业务层捕获后清理并回滚状态。"""

    def __init__(self, key: str):
        super().__init__(f"operation cancelled: {key}")
        self.key = key


_cancelled: set[str] = set()


def request_cancel(key: str) -> None:
    """请求取消正在进行的入库 / 重建（幂等）。"""
    _cancelled.add(key)


def _check_cancel(key: str) -> None:
    """任务内部检查点：命中即抛 OperationCancelled。"""
    if key in _cancelled:
        raise OperationCancelled(key)


def acknowledge_cancel(key: str) -> None:
    """任务收尾后确认取消：清除标记，允许发起下一次操作。"""
    _cancelled.discard(key)

