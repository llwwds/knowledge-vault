# knowledge-vault 开发专属容器（开发与部署规范第 1 条：环境受控）
# linux/arm64，依赖由 uv sync 按仓库 uv.lock 安装（zvec 有 manylinux aarch64 wheel）
FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /repo
# 依赖在容器首次启动时由 run.sh 按 uv.lock 同步到 /opt/venv（不烧进镜像层，锁更新后增量同步）
