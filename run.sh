#!/usr/bin/env bash
# ============================================================
# ra-agent 服务启停脚本
#   用法：
#     ./run.sh          启动服务（已运行则跳过）
#     ./run.sh stop     停止服务
#     ./run.sh restart  重启服务
#     ./run.sh status   查看运行状态
#   日志：/tmp/ra-agent.log
# ============================================================
set -uo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/ra-agent.log"
PORT=8000
MATCH="uvicorn app.main:app"   # 进程匹配串（不含 run.sh 自身）

is_running() { pgrep -f "$MATCH" > /dev/null 2>&1; }

check_deps() {
  # 启动前检查基础设施：Postgres(5432) / Redis(6379)
  local ok=1
  for p in 5432 6379; do
    if ! timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/$p" 2> /dev/null; then
      echo "[ra-agent] 依赖未就绪：端口 $p 不可达"
      ok=0
    fi
  done
  if [ "$ok" -eq 0 ]; then
    echo "[ra-agent] 请先启动基础设施容器："
    echo "  docker compose -f $APP_DIR/docker-compose.yml up -d"
    return 1
  fi
  return 0
}

start() {
  if is_running; then
    echo "[ra-agent] 服务已在运行："
    pgrep -af "$MATCH"
    return 0
  fi
  check_deps || return 1
  cd "$APP_DIR"
  echo "[ra-agent] 启动服务 (端口 ${PORT}, 日志 ${LOG_FILE}) ..."
  # setsid + nohup + disown：脱离会话常驻，调用结束后不被回收
  setsid nohup .venv/bin/python -m uvicorn app.main:app \
    --host 0.0.0.0 --port "$PORT" > "$LOG_FILE" 2>&1 < /dev/null &
  disown
  # 轮询 /health，最多等 40s（应用启动时要初始化 DB/向量库/MCP 等）
  for i in $(seq 1 40); do
    if ! is_running; then
      echo "[ra-agent] 进程异常退出，最近日志如下："
      tail -30 "$LOG_FILE"
      return 1
    fi
    if curl -sf -m 2 "http://localhost:${PORT}/health" > /dev/null 2>&1; then
      echo "[ra-agent] 启动成功 (PID $(pgrep -f "$MATCH" | head -1))，/health 通过"
      return 0
    fi
    sleep 1
  done
  echo "[ra-agent] 启动超时（40s 内 /health 未就绪），最近日志如下："
  tail -30 "$LOG_FILE"
  return 1
}

stop() {
  if is_running; then
    pkill -f "$MATCH"
    sleep 1
    if is_running; then
      echo "[ra-agent] 进程仍在，强制结束 ..."
      pkill -9 -f "$MATCH"
      sleep 1
    fi
    echo "[ra-agent] 服务已停止"
  else
    echo "[ra-agent] 服务未在运行"
  fi
}

status() {
  if is_running; then
    echo "[ra-agent] 运行中："
    pgrep -af "$MATCH"
  else
    echo "[ra-agent] 未运行"
  fi
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; start ;;
  status)  status ;;
  *)
    echo "用法: $0 [start|stop|restart|status]（默认 start）" >&2
    exit 1
    ;;
esac
