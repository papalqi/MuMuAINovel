# 独立 Fanqie Uploader MCP Server（Playwright + FastMCP）
#
# 说明：
# - 不修改 MuMuAINovel 主服务代码
# - 单独容器运行，MuMu 通过 MCP 插件接入

FROM mcr.microsoft.com/playwright/python:v1.50.0-jammy

WORKDIR /app

COPY fanqie_uploader_mcp/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY fanqie_uploader_mcp /app/fanqie_uploader_mcp

ENV PORT=9001
ENV FANQIE_DATA_DIR=/data

VOLUME ["/data"]

EXPOSE 9001

CMD ["python", "-m", "fanqie_uploader_mcp"]

