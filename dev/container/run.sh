#!/usr/bin/env bash
# knowledge-vault 开发专属容器：创建/启动/依赖同步，幂等可重复执行。
# 用法: bash dev/container/run.sh
# 容器内路径：
#   /repo                    代码仓库（宿主 ~/Documents/code_file/knowledge-vault）
#   /root/llwwds_application -> /app_runtime   设备本地运行时（state/testdata/hf_cache）
#   /vault                   Obsidian vault（只读，仅供测试数据抽取）
#   /opt/venv                容器内 Python 3.12 + uv.lock 同步的依赖
# 跑实验：docker exec -w /repo knowledge-vault-dev /opt/venv/bin/python experiments/<脚本>.py
set -euo pipefail

REPO="/Users/llwwds/Documents/code_file/knowledge-vault"
APP="/Users/llwwds/llwwds_application/knowledge-vault"
VAULT="/Users/llwwds/Documents/obsidian_file"
NAME="knowledge-vault-dev"
IMAGE="knowledge-vault-dev:latest"

docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" "$(dirname "$0")"

if docker container inspect "$NAME" >/dev/null 2>&1; then
  docker start "$NAME" >/dev/null
else
  docker run -d --name "$NAME" \
    -v "$REPO":/repo \
    -v "$APP":/app_runtime \
    -v "$VAULT":/vault:ro \
    -e HF_HOME=/app_runtime/hf_cache \
    -e PYTHONDONTWRITEBYTECODE=1 \
    "$IMAGE" sleep infinity
fi

# ~/llwwds_application 语义对齐：容器内 expanduser('~') 指向挂载的运行时
docker exec "$NAME" ln -sfn /app_runtime /root/llwwds_application

# 依赖按 uv.lock 同步（幂等，锁更新后增量；开发容器含 dev 组——pytest 必须可用）
docker exec -w /repo "$NAME" bash -c '
  if [ ! -x /opt/venv/bin/python ]; then uv venv /opt/venv --python 3.12; fi
  VIRTUAL_ENV=/opt/venv uv sync --active --frozen
'

docker exec "$NAME" /opt/venv/bin/python -c "import zvec, FlagEmbedding, jieba; print('container deps ok:', zvec.version('zvec'))"
echo "容器就绪：docker exec -w /repo $NAME /opt/venv/bin/python <脚本>"
