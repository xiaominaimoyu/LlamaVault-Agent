# LlamaVault-Agent

轻量级**单文件本地 CLI Agent**（`agent.py`），专为 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 的 `llama-server`（OpenAI 兼容接口 + 原生 `--tools` 本地工具接口）打造，并深度集成个人开源本地知识库 [**KVault**](https://github.com/xiaominaimoyu/KVault)，为 8GB 显存 / 16K 上下文场景提供**自动上下文压缩、长期会话记忆归档与混合检索召回**能力。

---

## 核心特性

- **零第三方依赖启动**：核心对话、流式输出与工具调用仅依赖 Python 标准库；在未安装 `chromadb` 的系统 Python 环境下，可自动探测 KVault 虚拟环境并通过 `--rag-worker` 后台子进程管道无缝桥接。
- **原生对接 `llama-server --tools all`**：自动获取服务端工具定义与写权限标识；读操作自动放行，文件写入 / 命令执行等高危操作默认触发交互式二次确认（支持 `y` / `n` / `a` 或 `--yolo`）。
- **KVault RAG 上下文压缩与长期记忆**：
  - 接近上下文窗口上限（或单轮工具输出过长）时，按**完整回合**（绝不拆散 `tool_calls` 与 `tool`）将早期对话归档至 KVault（SQLite + ChromaDB + Markdown 会话记录），并调用本地模型生成精简滚动摘要；
  - 支持 **BM25（jieba）+ 向量 RRF 混合检索**；
  - 兼具**提问前自动召回**（`<recalled_context>` 临时注入）与 **`recall_memory` 主动检索工具**，既能召回历史会话，也可直接检索 KVault 主知识库文档。
- **克制的思考模式与实时性能面板**：
  - 默认隐藏冗长的 `<think>` 推理流，终端显示紧凑的 `🤔 思考中... xx tok`，随时通过 `/show-think` 回看或 `/think-view on` 实时显示；
  - 每轮结束自动输出彩色**上下文占用进度条**、**输出速度（t/s）**、**首字延迟（TTFT）**与 **Prefill 速度**，并动态更新终端窗口标题。

---

## 快速开始

### 1. 启动 `llama-server`

以 `Qwen3.5-9B-abliterated.Q4_K_M.gguf`（8GB 显存 + 16K 上下文）为例：

```bash
llama-server \
  -m ./Huihui-Qwen3.5-9B-abliterated.Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 \
  -c 16384 \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  --jinja \
  --tools all \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --repeat-penalty 1.0
```

> 如需配合 [KVault](https://github.com/xiaominaimoyu/KVault) 开启向量 + BM25 混合检索，请同时启动 Embedding 服务（如 BGE-M3 `llama-server` 监听 `:8001` 或 Ollama `:11434`）；未启动 Embedding 服务时会自动平滑降级为 BM25 关键词检索。

### 2. 运行单文件 Agent

```bash
# 克隆仓库（或直接单独下载 agent.py 即可运行）
git clone https://github.com/xiaominaimoyu/LlamaVault-Agent.git
cd LlamaVault-Agent

# 默认交互模式（自动探测 KVault RAG 知识库）
python agent.py

# 指定 KVault 路径
python agent.py --kvault-dir /path/to/KVault

# 未安装 KVault 时，可关闭 RAG 独立运行
python agent.py --no-rag

# 常用参数组合：指定工作目录 / 跳过写确认 / 关闭思考模式
python agent.py --cwd /path/to/project --yolo --no-think

# 一次性提问并退出
python agent.py "统计当前目录下各类型文件的数量"
```

---

## 交互斜杠命令

| 命令 | 说明 |
| :--- | :--- |
| `/help` | 查看全部交互命令 |
| `/think on\|off` | 动态开关模型思考模式（无需重启 `llama-server`） |
| `/show-think` | 查看最近一轮隐藏的思考过程 |
| `/think-view on\|off` | 切换是否实时流式显示思考内容（默认隐藏） |
| `/compact` | 立即手动压缩上下文并归档早期回合至 KVault RAG |
| `/memory` | 查看当前滚动摘要与近期归档记录 |
| `/rag status` | 查看 KVault RAG 存储区、向量数与检索后端状态 |
| `/rag search <关键词>` | 手动检索历史对话记忆与 KVault 主知识库 |
| `/rag auto on\|off` | 开关每轮提问前自动召回 |
| `/rag sync` | Embedding 服务上线后，为此前仅 BM25 索引的回合补齐向量 |
| `/rag wipe` | 清空已归档的对话记忆 |
| `/cwd <路径>` | 切换工具工作目录 |
| `/tools` | 重新拉取服务端工具列表 |
| `/yolo on\|off` | 开关写文件 / 执行命令免确认模式 |
| `/clear` | 归档并清空当前窗口，开启新会话 |
| `/exit` / `/quit` | 退出程序 |

---

## 关联项目

- **本地 RAG 知识库底座**：[xiaominaimoyu/KVault](https://github.com/xiaominaimoyu/KVault) —— 面向个人的本地化知识库桌面应用，提供文档切片、BM25 + 向量混合检索与多工作区管理。

---

## License

[MIT License](./LICENSE)
