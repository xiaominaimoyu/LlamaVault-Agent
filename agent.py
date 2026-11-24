#!/usr/bin/env python3
"""轻量本地 agent —— 对接 llama-server (OpenAI 兼容 API + 内置工具接口) + KVault RAG 记忆。

依赖: Python 标准库 + 已启动的 llama-server (start-qwen35.bat)。
上下文压缩与长期记忆自动接入当前工作区 KVault RAG 知识库：
- 若当前 Python 环境未安装 chromadb，会自动通过 KVault 的虚拟环境启动后台 RAG worker，
  因此直接用系统 python 运行本单文件即可使用完整向量 + BM25 混合检索。
- 对话超出 16K 上下文窗口时，将早期完整回合归档至 KVault RAG 并生成滚动摘要，
  当前窗口只保留最近回合；模型或用户需要历史细节时，通过 RAG 召回（自动召回 + recall_memory 工具）。

用法:
  python agent.py                     # 交互式对话 agent（默认启用 KVault RAG 记忆）
  python agent.py --yolo              # 写文件/执行命令不再确认
  python agent.py --cwd D:\\work      # 指定工具工作目录
  python agent.py --no-tools          # 纯聊天，不启用工具
  python agent.py --no-rag            # 禁用 RAG 记忆，退回简单裁剪
"""

import argparse
import atexit
import ctypes
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_KVAULT_DIR = ""  # 可选：填写本机 KVault 路径，或通过 --kvault-dir / 环境变量 KVAULT_DIR 指定
DEFAULT_EMBED_URL = "http://127.0.0.1:8001"
MAX_TOOL_OUTPUT = 6000          # 单次工具结果最多回填的字符数
DEFAULT_MAX_STEPS = 12          # 一轮用户请求内最多工具调用轮数
CONTEXT_RESERVE = 2000          # 给模型回复预留的 token 数
AUTO_RECALL_MAX_CHARS = 1400    # 单轮自动注入召回片段的字符上限
SUMMARY_MAX_CHARS = 900         # 滚动摘要字符上限
SYSTEM_TEMPLATE = """你是一个运行在本机的轻量命令行 agent，工作目录是 {cwd}。
规则：
- 需要查看、搜索、修改文件或执行命令时，必须调用工具，不要凭空猜测。
- 修改文件前先用 read_file 确认内容；edit_file 的 old_text 必须唯一且精确。
- 相对路径都相对于工作目录；命令在 Windows 上执行。
- Windows 下避免复杂的引号嵌套，需要引号或多行逻辑时先写脚本文件再执行。
- 工具返回错误时先分析原因再重试，不要重复同样的失败调用。
- 绝对不要编造工具输出：输出为空就如实说为空，命令失败就报告失败。
- 用用户的语言简洁回答，工具执行完后用一两句话总结结果，不要复述工具原始输出。"""

RAG_SYSTEM_ADDON = """
记忆与上下文压缩规则：
- 早期对话已自动归档至本地 KVault RAG 知识库，当前对话窗口仅保留最近回合。
- 若存在 <compressed_memory_summary>，它是早期对话的压缩摘要；若存在 <recalled_context>，它是与当前问题相关的自动召回片段。
- 当用户询问或提及早先的对话内容、历史决策、修改过的文件、命令输出或报错细节，而当前上下文信息不足时，必须先调用 `recall_memory` 工具从 RAG 知识库检索后再回答，不要凭空猜测或直接说不知道。"""

RECALL_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "recall_memory",
        "description": (
            "从本地 KVault RAG 知识库中召回已被压缩归档的历史对话内容或知识库文档。"
            "当需要查看早期对话细节、历史修改记录、之前的报错信息或相关背景知识时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询语句或关键词，建议包含具体主题、文件名、报错或专有名词。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的最相关片段数量，默认 5（1-10）。",
                    "default": 5,
                },
                "scope": {
                    "type": "string",
                    "enum": ["conversation", "knowledge", "all"],
                    "description": "检索范围：conversation=历史对话记忆（默认），knowledge=KVault主知识库文档/笔记，all=两者合并检索。",
                    "default": "conversation",
                },
            },
            "required": ["query"],
        },
    },
}

DIM, YELLOW, CYAN, RED, GREEN, RESET = (
    "\x1b[90m", "\x1b[33m", "\x1b[36m", "\x1b[31m", "\x1b[32m", "\x1b[0m")
_ORIGINAL_TITLE = ""


def _set_title(text):
    """仅在交互终端里更新控制台标题（生成时实时显示速度）。"""
    if not sys.stdout.isatty():
        return
    try:
        ctypes.windll.kernel32.SetConsoleTitleW(text)
    except Exception:
        pass


def _capture_title():
    try:
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.kernel32.GetConsoleTitleW(buf, 512)
        return buf.value
    except Exception:
        return ""


def enable_ansi():
    """Windows 控制台启用 ANSI 颜色与 UTF-8 输出。"""
    global _ORIGINAL_TITLE
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        k32 = ctypes.windll.kernel32
        k32.SetConsoleOutputCP(65001)
        k32.SetConsoleMode(k32.GetStdHandle(-11), 7)
    except Exception:
        pass
    _ORIGINAL_TITLE = _capture_title()


def request(method, url, payload=None, headers=None, timeout=600):
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Content-Type": "application/json; charset=utf-8", **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url, timeout=15):
    with request("GET", url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def estimate_tokens(text):
    """保守估算文本 token 数（兼容中文 CJK 与英文/JSON）。"""
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    non_ascii = len(text) - ascii_chars
    return int(non_ascii * 0.9 + ascii_chars * 0.35) + 4


def estimate_message_tokens(msg):
    """估算单条 OpenAI 格式消息占用的 token 数。"""
    tokens = 6 + estimate_tokens(msg.get("content") or "")
    if msg.get("tool_calls"):
        tokens += estimate_tokens(json.dumps(msg["tool_calls"], ensure_ascii=False))
    return tokens


def group_turns(messages):
    """把消息流按 user 消息切分为完整回合，绝不拆散 tool_calls 与 tool。"""
    turns = []
    cur = []
    for msg in messages:
        if msg.get("role") == "user" and cur:
            turns.append(cur)
            cur = [msg]
        else:
            cur.append(msg)
    if cur:
        turns.append(cur)
    return turns


def resolve_kvault_dir(explicit=None):
    """定位 KVault 知识库根目录。"""
    here = Path(__file__).resolve().parent
    candidates = [
        explicit,
        os.environ.get("KVAULT_DIR"),
        os.getcwd(),
        here / "KVault",
        here.parent / "KVault",
        DEFAULT_KVAULT_DIR,
    ]
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand).resolve()
        if (p / "core" / "vector_store.py").is_file() and (p / "core" / "retriever.py").is_file():
            return str(p)
    return None


def find_kvault_python(kvault_dir):
    """查找包含 KVault 依赖（chromadb 等）的 Python 解释器。"""
    if not kvault_dir:
        return None
    base = Path(kvault_dir)
    for rel in (
        ".build-venv/Scripts/python.exe",
        ".env/Scripts/python.exe",
        ".venv/Scripts/python.exe",
        "venv/Scripts/python.exe",
        ".build-venv/bin/python",
        ".venv/bin/python",
    ):
        cand = base / rel
        if cand.is_file():
            return str(cand)
    return None


def _current_has_chromadb():
    try:
        import chromadb  # noqa: F401
        return True
    except Exception:
        return False


# ==============================================================================
# KVault RAG 记忆存储引擎（运行于具备 chromadb 的环境中，或通过 --rag-worker 托管）
# ==============================================================================


class KVaultMemoryEngine:
    """基于当前工作区 KVault 内核的对话归档与混合检索引擎。"""

    def __init__(
        self,
        kvault_dir,
        rag_dir=None,
        embed_backend="auto",
        embed_url="",
        embed_model="",
    ):
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        try:
            import jieba
            jieba.setLogLevel(logging.WARNING)
        except Exception:
            pass

        self.kvault_dir = Path(kvault_dir).resolve()
        if str(self.kvault_dir) not in sys.path:
            sys.path.insert(0, str(self.kvault_dir))

        from core.bm25_retriever import BM25Retriever
        from core.config import Config, HybridSearchConfig, LlamaServerConfig
        from core.embedding_service import EmbeddingService
        from core.fusion import rrf
        from core.metadata_manager import MetadataManager
        from core.text_splitter import KnowledgeTextSplitter
        from core.vector_store import ModelVersionInfo, VectorStore

        self._BM25Retriever = BM25Retriever
        self._rrf = rrf

        self.memory_dir = Path(rag_dir).resolve() if rag_dir else (self.kvault_dir / "data" / "agent_memory")
        self.sessions_dir = self.memory_dir / "sessions"
        self.chroma_dir = self.memory_dir / "chroma_db"
        self.sqlite_path = self.memory_dir / "kb.sqlite"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._build_config(
            Config, HybridSearchConfig, LlamaServerConfig,
            embed_backend=embed_backend, embed_url=embed_url, embed_model=embed_model,
        )
        self.embedder = None
        self.embed_status = "bm25-only"
        self._init_embedder(EmbeddingService)

        model_info = None
        if self.embedder is not None:
            try:
                dim = self.embedder.dimension
                model_info = ModelVersionInfo(model=self.embedder.model_name, dimension=dim)
            except Exception:
                self.embedder = None
                self.embed_status = "bm25-only (embedding probe failed)"

        self.vector_store = VectorStore(persist_dir=str(self.chroma_dir), model_info=model_info)
        self.metadata = MetadataManager(db_path=str(self.sqlite_path), vector_store=self.vector_store)
        self._ensure_agent_tables()
        self.partition_id = self._ensure_partition("对话记忆")
        self.splitter = KnowledgeTextSplitter(chunk_size=600, chunk_overlap=120)
        self._bm25 = None
        self._bm25_dirty = True
        self._doc_turn_cache = {}
        self._load_doc_turn_cache()

        self._main_kb = None
        self._main_kb_checked = False

    def _probe_http_json(self, url, timeout=2.0):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except Exception:
            return None, ""

    def _read_valid_kvault_embed_config(self):
        cfg_path = self.kvault_dir / "config.json"
        if not cfg_path.is_file():
            return None
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        # 忽略 pytest 测试留下的临时配置
        for key in ("files_dir", "chroma_dir", "sqlite_path"):
            val = str(raw.get(key, ""))
            if "pytest-of-" in val or "Temp" in val:
                return None
        if raw.get("embedding_model") in ("model-A", "model-B"):
            return None
        return raw

    def _build_config(
        self, Config, HybridSearchConfig, LlamaServerConfig,
        embed_backend="auto", embed_url="", embed_model="",
    ):
        cfg = Config(
            files_dir=self.sessions_dir,
            chroma_dir=self.chroma_dir,
            sqlite_path=self.sqlite_path,
            logs_dir=self.memory_dir / "logs",
            chunk_size=600,
            chunk_overlap=120,
            top_k=5,
            similarity_threshold=0.25,
            hybrid_search=HybridSearchConfig(
                enabled=True, strategy="rrf", bm25_weight=0.45, vector_weight=0.55, rrf_k=60
            ),
        )
        backend = (embed_backend or "auto").strip().lower()
        if backend == "bm25":
            cfg.embedding_backend = "none"
            return cfg

        if backend == "llama_cpp_server":
            cfg.embedding_backend = "llama_cpp_server"
            cfg.llama_server = LlamaServerConfig(
                base_url=embed_url or DEFAULT_EMBED_URL,
                model=embed_model or "",
                timeout=30,
            )
            return cfg

        if backend == "ollama":
            cfg.embedding_backend = "ollama"
            if embed_url:
                cfg.ollama_base_url = embed_url
            if embed_model:
                cfg.embedding_model = embed_model
            return cfg

        # auto 模式：1) 显式 embed_url -> llama_cpp_server
        if embed_url:
            cfg.embedding_backend = "llama_cpp_server"
            cfg.llama_server = LlamaServerConfig(base_url=embed_url, model=embed_model or "", timeout=30)
            return cfg

        # 2) KVault 有效配置
        kv_raw = self._read_valid_kvault_embed_config()
        if kv_raw and kv_raw.get("embedding_backend"):
            b = kv_raw["embedding_backend"]
            if b == "llama_cpp_server":
                ls = kv_raw.get("llama_server") or {}
                cfg.embedding_backend = "llama_cpp_server"
                cfg.llama_server = LlamaServerConfig(
                    base_url=ls.get("base_url") or DEFAULT_EMBED_URL,
                    model=ls.get("model") or "",
                    timeout=30,
                )
                return cfg
            if b == "ollama":
                cfg.embedding_backend = "ollama"
                cfg.ollama_base_url = kv_raw.get("ollama_base_url") or cfg.ollama_base_url
                cfg.embedding_model = kv_raw.get("embedding_model") or cfg.embedding_model
                return cfg

        # 3) 探测本机 BGE-M3 llama-server (Start-BGE.ps1 默认 :8001)
        status, body = self._probe_http_json(DEFAULT_EMBED_URL + "/v1/models", timeout=1.5)
        if status == 200:
            detected_model = ""
            try:
                models = json.loads(body).get("data") or []
                if models:
                    detected_model = models[0].get("id") or ""
            except Exception:
                pass
            cfg.embedding_backend = "llama_cpp_server"
            cfg.llama_server = LlamaServerConfig(
                base_url=DEFAULT_EMBED_URL,
                model=embed_model or detected_model,
                timeout=30,
            )
            return cfg

        # 4) 探测 Ollama (:11434)
        status, _ = self._probe_http_json("http://127.0.0.1:11434/api/tags", timeout=1.5)
        if status == 200:
            cfg.embedding_backend = "ollama"
            if embed_model:
                cfg.embedding_model = embed_model
            return cfg

        cfg.embedding_backend = "none"
        return cfg

    def _init_embedder(self, EmbeddingService):
        if self.config.embedding_backend == "none":
            self.embedder = None
            self.embed_status = "bm25-only (未检测到 Embedding 服务)"
            return
        try:
            svc = EmbeddingService(config=self.config)
            if svc.is_available() and svc.is_model_available():
                self.embedder = svc
                self.embed_status = f"{svc.backend_type}:{svc.model_name}"
            else:
                self.embedder = None
                self.embed_status = f"bm25-only ({self.config.embedding_backend} 不可用)"
        except Exception as exc:
            self.embedder = None
            self.embed_status = f"bm25-only ({exc})"

    def _ensure_agent_tables(self):
        with self.metadata._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    document_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    cwd TEXT DEFAULT '',
                    user_preview TEXT DEFAULT '',
                    assistant_preview TEXT DEFAULT '',
                    embedded INTEGER DEFAULT 0
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_turns_session ON agent_turns(session_id, turn_index)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_turns_doc ON agent_turns(document_id)"
            )

    def _ensure_partition(self, name):
        for p in self.metadata.list_partitions():
            if p["name"] == name:
                return p["id"]
        return self.metadata.create_partition(name)

    def _load_doc_turn_cache(self):
        with self.metadata._conn() as conn:
            rows = conn.execute(
                "SELECT turn_id, session_id, turn_index, document_id, created_at, user_preview, embedded FROM agent_turns"
            ).fetchall()
        self._doc_turn_cache = {
            r["document_id"]: {
                "turn_id": r["turn_id"],
                "session_id": r["session_id"],
                "turn_index": r["turn_index"],
                "created_at": r["created_at"],
                "user_preview": r["user_preview"],
                "embedded": bool(r["embedded"]),
            }
            for r in rows
        }

    def _ensure_bm25(self):
        if self._bm25 is None or self._bm25_dirty:
            self._bm25 = self._BM25Retriever(self.metadata)
            self._bm25.build_index()
            self._bm25_dirty = False
        return self._bm25

    def archive_turn(self, session_id, turn_index, turn_id, cwd, turn_text, user_preview="", assistant_preview=""):
        """把完整一轮对话写入会话 Markdown，并切片索引到 KVault SQLite + ChromaDB。"""
        if not turn_text or not turn_text.strip():
            return {"ok": True, "skipped": True}

        with self.metadata._conn() as conn:
            existing = conn.execute(
                "SELECT document_id, embedded FROM agent_turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
        if existing:
            return {"ok": True, "already_archived": True, "document_id": existing["document_id"]}

        now = time.time()
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        session_file = self.sessions_dir / f"{session_id}.md"
        is_new = not session_file.exists()
        with session_file.open("a", encoding="utf-8") as f:
            if is_new:
                f.write(f"# 对话会话 {session_id}\n\n- 开始时间: {ts_str}\n- 工作目录: {cwd}\n\n---\n\n")
            f.write(turn_text.rstrip() + "\n\n---\n\n")

        doc_label = f"{session_id}#T{turn_index:04d}"
        doc_id = self.metadata.create_document(
            file_name=doc_label,
            stored_path=str(session_file),
            file_ext=".md",
            file_size=len(turn_text.encode("utf-8")),
            original_path=turn_id,
            partition_id=self.partition_id,
        )

        chunks = self.splitter.split(
            turn_text,
            {
                "document_id": doc_id,
                "file_name": doc_label,
                "session_id": session_id,
                "turn_id": turn_id,
                "turn_index": int(turn_index),
                "timestamp": ts_str,
                "created_at": now,
                "cwd": cwd,
            },
        )
        if not chunks:
            return {"ok": True, "chunks": 0, "document_id": doc_id}

        prefix = f"[{session_id} · Turn {turn_index} · {ts_str}]\n"
        chunk_texts = [
            c.content if i == 0 else prefix + c.content
            for i, c in enumerate(chunks)
        ]
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        embedded_ok = False
        if self.embedder is not None:
            try:
                embeddings = self.embedder.embed_texts(chunk_texts)
                self.vector_store.add_chunks(
                    ids=ids,
                    texts=chunk_texts,
                    embeddings=embeddings,
                    metadatas=[c.metadata for c in chunks],
                )
                embedded_ok = True
            except Exception:
                embedded_ok = False

        chunk_meta = [(i, chunk_texts[i][:4000], ids[i]) for i in range(len(chunks))]
        self.metadata.add_chunks(doc_id, chunk_meta)
        self.metadata.update_status(
            doc_id, "indexed" if embedded_ok else "pending_embed", chunk_count=len(chunks)
        )

        with self.metadata._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_turns
                (turn_id, session_id, turn_index, document_id, created_at, cwd, user_preview, assistant_preview, embedded)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id, session_id, int(turn_index), doc_id, now, cwd,
                    user_preview[:240], assistant_preview[:240], 1 if embedded_ok else 0,
                ),
            )

        self._doc_turn_cache[doc_id] = {
            "turn_id": turn_id,
            "session_id": session_id,
            "turn_index": int(turn_index),
            "created_at": now,
            "user_preview": user_preview[:240],
            "embedded": embedded_ok,
        }
        self._bm25_dirty = True
        return {
            "ok": True,
            "document_id": doc_id,
            "chunks": len(chunks),
            "embedded": embedded_ok,
        }

    def _open_main_kb(self):
        """只读方式尝试打开 KVault 主知识库，供 scope=knowledge/all 查询。"""
        if self._main_kb_checked:
            return self._main_kb
        self._main_kb_checked = True
        try:
            from core.metadata_manager import MetadataManager
            from core.retriever import Retriever
            from core.vector_store import VectorStore
            from core.workspace import WorkspaceManager

            kv_raw = self._read_valid_kvault_embed_config()
            if kv_raw and (self.kvault_dir / "data" / "workspaces").is_dir():
                from core.config import Config
                cfg = Config.load(str(self.kvault_dir / "config.json"))
                ws_mgr = WorkspaceManager(cfg, self.kvault_dir / "data")
                ws = ws_mgr.get_current()
                chroma_dir, sqlite_path = ws.chroma_dir, ws.sqlite_path
            else:
                ws_def = self.kvault_dir / "data" / "workspaces" / "default"
                if (ws_def / "kb.sqlite").is_file() and (ws_def / "kb.sqlite").stat().st_size > 0:
                    chroma_dir, sqlite_path = ws_def / "chroma_db", ws_def / "kb.sqlite"
                else:
                    chroma_dir = self.kvault_dir / "data" / "chroma_db"
                    sqlite_path = self.kvault_dir / "data" / "kb.sqlite"

            if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
                return None
            if self.embedder is None:
                return None

            vs = VectorStore(str(chroma_dir))
            if vs.count() == 0:
                return None
            md = MetadataManager(str(sqlite_path), vector_store=vs)
            ret = Retriever(embedder=self.embedder, vector_store=vs, metadata=md, config=self.config)
            self._main_kb = (vs, md, ret)
        except Exception:
            self._main_kb = None
        return self._main_kb

    def search(
        self,
        query,
        top_k=5,
        scope="conversation",
        exclude_turn_ids=None,
        current_session_id=None,
        min_vector_score=0.0,
    ):
        if not query or not query.strip():
            return {"results": [], "mode": self.embed_status}

        top_k = max(1, min(int(top_k or 5), 15))
        exclude = set(exclude_turn_ids or [])
        results = []

        if scope in ("conversation", "all"):
            results.extend(
                self._search_conversations(
                    query.strip(),
                    top_k=top_k,
                    exclude=exclude,
                    current_session_id=current_session_id,
                    min_vector_score=float(min_vector_score or 0.0),
                )
            )

        if scope in ("knowledge", "all"):
            kb = self._open_main_kb()
            if kb is not None:
                _, _, ret = kb
                try:
                    kb_hits = ret.search(query.strip(), top_k=top_k)
                    for r in kb_hits:
                        results.append({
                            "source": "knowledge",
                            "document_name": r.document_name or r.document_id,
                            "content": r.content,
                            "score": round(float(r.score), 4),
                            "vector_score": round(float(r.score), 4),
                            "hit_source": r.hit_source,
                            "session_id": "",
                            "turn_index": -1,
                            "timestamp": "",
                            "is_current_session": False,
                        })
                except Exception:
                    pass

        if scope == "all":
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            results = results[:top_k]

        return {"results": results, "mode": self.embed_status}

    def _search_conversations(
        self, query, top_k, exclude, current_session_id, min_vector_score
    ):
        fetch_k = max(top_k * 3, top_k + len(exclude) * 2 + 4)

        bm25_hits = []
        if self._BM25Retriever.is_jieba_available():
            bm25 = self._ensure_bm25()
            bm25_hits = bm25.search(query, top_k=fetch_k)

        vector_hits = []
        if self.embedder is not None and self.vector_store.count() > 0:
            try:
                q_emb = self.embedder.embed_query(query)
                vector_hits = self.vector_store.search(query_embedding=q_emb, top_k=fetch_k)
            except Exception:
                vector_hits = []

        # 归并候选并过滤当前上下文窗口中仍活跃的 turn
        chunk_map = {}
        vector_scores = {}
        bm25_scores = {}
        vector_ranking = []
        bm25_ranking = []

        for vh in vector_hits:
            meta = vh.metadata or {}
            doc_id = meta.get("document_id") or vh.chunk_id.rsplit("_", 1)[0]
            info = self._doc_turn_cache.get(doc_id, {})
            turn_id = meta.get("turn_id") or info.get("turn_id", "")
            if turn_id and turn_id in exclude:
                continue
            vector_ranking.append(vh.chunk_id)
            vector_scores[vh.chunk_id] = float(vh.score)
            chunk_map[vh.chunk_id] = {
                "chunk_id": vh.chunk_id,
                "document_id": doc_id,
                "document_name": meta.get("file_name", doc_id),
                "content": vh.content,
                "turn_id": turn_id,
                "session_id": meta.get("session_id") or info.get("session_id", ""),
                "turn_index": meta.get("turn_index") if meta.get("turn_index") is not None else info.get("turn_index", -1),
                "timestamp": meta.get("timestamp") or "",
                "created_at": meta.get("created_at") or info.get("created_at", 0.0),
            }

        for bh in bm25_hits:
            info = self._doc_turn_cache.get(bh.document_id, {})
            turn_id = info.get("turn_id", "")
            if turn_id and turn_id in exclude:
                continue
            bm25_ranking.append(bh.chunk_id)
            bm25_scores[bh.chunk_id] = float(bh.score)
            if bh.chunk_id not in chunk_map:
                ts = ""
                if info.get("created_at"):
                    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["created_at"]))
                chunk_map[bh.chunk_id] = {
                    "chunk_id": bh.chunk_id,
                    "document_id": bh.document_id,
                    "document_name": bh.document_name,
                    "content": bh.content,
                    "turn_id": turn_id,
                    "session_id": info.get("session_id", ""),
                    "turn_index": info.get("turn_index", -1),
                    "timestamp": ts,
                    "created_at": info.get("created_at", 0.0),
                }

        if not chunk_map:
            return []

        if vector_ranking and bm25_ranking:
            fused = self._rrf([bm25_ranking, vector_ranking], weights=[0.45, 0.55], k=60)
            ordered = [(fr.chunk_id, fr.score, "hybrid") for fr in fused]
        elif vector_ranking:
            ordered = [(cid, vector_scores[cid], "vector") for cid in vector_ranking]
        else:
            ordered = [(cid, bm25_scores[cid], "bm25") for cid in bm25_ranking]

        out = []
        for cid, rank_score, mode in ordered:
            item = chunk_map.get(cid)
            if not item:
                continue
            v_score = vector_scores.get(cid)
            b_score = bm25_scores.get(cid)
            # 阈值门控（自动召回时过滤弱相关噪音；若无向量分则看 BM25 强命中）
            if min_vector_score > 0:
                if v_score is not None:
                    if v_score < min_vector_score and (b_score is None or b_score < 2.5):
                        continue
                elif b_score is None or b_score < 2.0:
                    continue

            sess_id = item["session_id"]
            out.append({
                "source": "conversation",
                "chunk_id": cid,
                "document_name": item["document_name"],
                "content": item["content"],
                "score": round(float(v_score if v_score is not None else rank_score), 4),
                "vector_score": round(float(v_score), 4) if v_score is not None else None,
                "bm25_score": round(float(b_score), 4) if b_score is not None else None,
                "hit_source": mode,
                "turn_id": item["turn_id"],
                "session_id": sess_id,
                "turn_index": item["turn_index"],
                "timestamp": item["timestamp"],
                "is_current_session": bool(current_session_id and sess_id == current_session_id),
            })
            if len(out) >= top_k:
                break
        return out

    def sync_pending(self):
        """当 Embedding 服务上线后，为此前仅 BM25 索引的回合补齐向量。"""
        if self.embedder is None:
            from core.embedding_service import EmbeddingService
            self._init_embedder(EmbeddingService)
        if self.embedder is None:
            return {"ok": False, "reason": self.embed_status, "synced": 0}

        with self.metadata._conn() as conn:
            rows = conn.execute(
                "SELECT turn_id, session_id, turn_index, document_id, created_at, cwd FROM agent_turns WHERE embedded = 0"
            ).fetchall()

        synced = 0
        for row in rows:
            doc_id = row["document_id"]
            chunks = self.metadata.get_document_chunks(doc_id)
            if not chunks:
                continue
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["created_at"]))
            doc_label = f"{row['session_id']}#T{row['turn_index']:04d}"
            texts = [c["content_preview"] for c in chunks]
            ids = [c["chroma_id"] for c in chunks]
            metas = [
                {
                    "document_id": doc_id,
                    "file_name": doc_label,
                    "session_id": row["session_id"],
                    "turn_id": row["turn_id"],
                    "turn_index": int(row["turn_index"]),
                    "timestamp": ts_str,
                    "created_at": float(row["created_at"]),
                    "cwd": row["cwd"] or "",
                    "chunk_index": int(c["chunk_index"]),
                    "total_chunks": len(chunks),
                }
                for c in chunks
            ]
            embeddings = self.embedder.embed_texts(texts)
            self.vector_store.add_chunks(ids=ids, texts=texts, embeddings=embeddings, metadatas=metas)
            self.metadata.update_status(doc_id, "indexed", chunk_count=len(chunks))
            with self.metadata._conn() as conn:
                conn.execute("UPDATE agent_turns SET embedded = 1 WHERE turn_id = ?", (row["turn_id"],))
            if doc_id in self._doc_turn_cache:
                self._doc_turn_cache[doc_id]["embedded"] = True
            synced += 1
        return {"ok": True, "synced": synced, "mode": self.embed_status}

    def list_turns(self, limit=10):
        limit = max(1, min(int(limit or 10), 50))
        with self.metadata._conn() as conn:
            rows = conn.execute(
                """
                SELECT turn_id, session_id, turn_index, created_at, user_preview, assistant_preview, embedded
                FROM agent_turns
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "turn_id": r["turn_id"],
                "session_id": r["session_id"],
                "turn_index": r["turn_index"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"])),
                "user_preview": r["user_preview"],
                "assistant_preview": r["assistant_preview"],
                "embedded": bool(r["embedded"]),
            }
            for r in rows
        ]

    def status(self):
        with self.metadata._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS turns,
                       COUNT(DISTINCT session_id) AS sessions,
                       SUM(CASE WHEN embedded = 0 THEN 1 ELSE 0 END) AS pending_embed
                FROM agent_turns
                """
            ).fetchone()
        stats = self.metadata.get_stats()
        main_kb = self._open_main_kb()
        main_kb_count = main_kb[0].count() if main_kb else 0
        return {
            "kvault_dir": str(self.kvault_dir),
            "memory_dir": str(self.memory_dir),
            "embed_status": self.embed_status,
            "jieba_available": self._BM25Retriever.is_jieba_available(),
            "sessions": int(row["sessions"] or 0),
            "turns": int(row["turns"] or 0),
            "pending_embed": int(row["pending_embed"] or 0),
            "chunks": int(stats.get("total_chunks") or 0),
            "vectors": int(self.vector_store.count()),
            "main_kb_vectors": int(main_kb_count),
        }

    def wipe(self):
        """清空所有归档对话记忆。"""
        for doc in self.metadata.list_documents():
            self.metadata.delete_document(doc.id, delete_file=False)
        with self.metadata._conn() as conn:
            conn.execute("DELETE FROM agent_turns")
        for md in self.sessions_dir.glob("*.md"):
            try:
                md.unlink()
            except OSError:
                pass
        self._doc_turn_cache.clear()
        self._bm25_dirty = True
        return {"ok": True}


def run_rag_worker(args):
    """独立 worker 进程：在 KVault venv 内通过 stdin/stdout JSONL 提供 RAG 服务。"""
    proto_out = sys.stdout
    sys.stdout = sys.stderr
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        proto_out.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    try:
        engine = KVaultMemoryEngine(
            kvault_dir=args.kvault_dir,
            rag_dir=args.rag_dir,
            embed_backend=args.embed_backend,
            embed_url=args.embed_url,
            embed_model=args.embed_model,
        )
        proto_out.write(json.dumps({"ready": True, "status": engine.status()}, ensure_ascii=False) + "\n")
        proto_out.flush()
    except Exception as exc:
        proto_out.write(json.dumps({"ready": False, "error": str(exc)}, ensure_ascii=False) + "\n")
        proto_out.flush()
        return 1

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            proto_out.write(json.dumps({"ok": False, "error": f"bad json: {exc}"}, ensure_ascii=False) + "\n")
            proto_out.flush()
            continue

        op = req.get("op")
        params = req.get("params") or {}
        if op == "shutdown":
            proto_out.write(json.dumps({"ok": True}, ensure_ascii=False) + "\n")
            proto_out.flush()
            break
        try:
            fn = getattr(engine, op)
            res = fn(**params)
            proto_out.write(json.dumps({"ok": True, "result": res}, ensure_ascii=False) + "\n")
        except Exception as exc:
            proto_out.write(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False) + "\n")
        proto_out.flush()
    return 0


class RAGClient:
    """统一代理：优先进程内直连 KVaultMemoryEngine，否则自动桥接 KVault venv worker。"""

    def __init__(self, kvault_dir, rag_dir=None, embed_backend="auto", embed_url="", embed_model=""):
        self.kvault_dir = kvault_dir
        self.rag_dir = rag_dir
        self.embed_backend = embed_backend
        self.embed_url = embed_url
        self.embed_model = embed_model
        self.engine = None
        self.proc = None
        self.available = False
        self.error = ""
        self.init_status = {}
        self._start()

    def _start(self):
        if not self.kvault_dir:
            self.error = "未找到 KVault 目录"
            return

        if _current_has_chromadb():
            try:
                self.engine = KVaultMemoryEngine(
                    kvault_dir=self.kvault_dir,
                    rag_dir=self.rag_dir,
                    embed_backend=self.embed_backend,
                    embed_url=self.embed_url,
                    embed_model=self.embed_model,
                )
                self.init_status = self.engine.status()
                self.available = True
                return
            except Exception as exc:
                self.error = str(exc)
                return

        venv_py = find_kvault_python(self.kvault_dir)
        if not venv_py:
            self.error = f"当前 Python 无 chromadb 且未找到 KVault 虚拟环境 ({self.kvault_dir})"
            return

        cmd = [
            venv_py, "-u", os.path.abspath(__file__),
            "--rag-worker",
            "--kvault-dir", str(self.kvault_dir),
            "--embed-backend", str(self.embed_backend or "auto"),
        ]
        if self.rag_dir:
            cmd.extend(["--rag-dir", str(self.rag_dir)])
        if self.embed_url:
            cmd.extend(["--embed-url", str(self.embed_url)])
        if self.embed_model:
            cmd.extend(["--embed-model", str(self.embed_model)])

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["ANONYMIZED_TELEMETRY"] = "False"

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            atexit.register(self.close)
            line = self.proc.stdout.readline()
            if not line:
                self.error = "RAG worker 启动失败（无握手输出）"
                self.close()
                return
            hello = json.loads(line)
            if not hello.get("ready"):
                self.error = hello.get("error") or "RAG worker 初始化失败"
                self.close()
                return
            self.init_status = hello.get("status") or {}
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            self.close()

    def call(self, op, **params):
        if not self.available:
            return None
        if self.engine is not None:
            return getattr(self.engine, op)(**params)
        if not self.proc or self.proc.poll() is not None:
            return None
        try:
            payload = json.dumps({"op": op, "params": params}, ensure_ascii=False)
            self.proc.stdin.write(payload + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                return None
            resp = json.loads(line)
            if not resp.get("ok"):
                return {"error": resp.get("error", "unknown worker error")}
            return resp.get("result")
        except Exception as exc:
            return {"error": str(exc)}

    def close(self):
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    self.proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                    self.proc.stdin.flush()
                    self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


# ==============================================================================
# Agent 主体
# ==============================================================================


class Agent:
    def __init__(self, base, cwd, think, yolo, use_tools, max_steps, rag=None, auto_recall=True):
        self.base = base.rstrip("/")
        self.cwd = os.path.abspath(cwd)
        self.think = think
        self.yolo = yolo
        self.use_tools = use_tools
        self.max_steps = max_steps
        self.messages = []
        self.tool_defs = []
        self.write_tools = set()
        self.ctx = 16384
        self.ctx_used = None
        self.model = "local"
        self.show_reasoning = False   # 是否在终端实时显示思考过程（默认隐藏）
        self.last_reasoning = ""      # 最近一轮的思考内容，供 /show-think 查看

        # 上下文压缩与 RAG 记忆状态
        self.rag = rag
        self.auto_recall = auto_recall
        self.session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.turn_counter = 0
        self.memory_summary = ""
        self.active_turn_ids = set()
        self._msg_turn_info = {}      # id(user_msg_dict) -> {turn_id, turn_index, archived}

    @property
    def rag_enabled(self):
        return bool(self.rag and self.rag.available)

    # ---------- 服务与工具 ----------

    def probe(self):
        try:
            get_json(self.base + "/health", timeout=5)
        except Exception:
            print(f"{RED}连不上 llama-server: {self.base}{RESET}")
            print("请先启动 llama-server 服务（可参考 README 中的启动命令）。")
            raise SystemExit(1)
        try:
            props = get_json(self.base + "/props")
            self.ctx = props.get("default_generation_settings", {}).get("n_ctx", 16384)
            self.model = os.path.basename(str(props.get("model_path", "local")))
        except Exception:
            pass
        self.fetch_tools()

    def fetch_tools(self):
        self.tool_defs, self.write_tools = [], set()
        if not self.use_tools:
            return
        try:
            tools = get_json(self.base + "/tools")
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"{YELLOW}服务未启用内置工具，进入纯聊天模式（可用 --tools all 启动服务）。{RESET}")
            else:
                print(f"{YELLOW}读取工具列表失败（HTTP {e.code}），进入纯聊天模式。{RESET}")
            self.use_tools = False
            return
        except Exception:
            print(f"{YELLOW}读取工具列表失败，进入纯聊天模式。{RESET}")
            self.use_tools = False
            return
        for t in tools:
            self.tool_defs.append(t["definition"])
            if t.get("permissions", {}).get("write"):
                self.write_tools.add(t["tool"])

        if self.rag_enabled:
            names = {d.get("function", {}).get("name") for d in self.tool_defs}
            if "recall_memory" not in names:
                self.tool_defs.append(RECALL_TOOL_DEF)

    def call_tool(self, name, args):
        try:
            with request("POST", self.base + "/tools",
                         {"tool": name, "params": args},
                         {"x-tool-cwd": self.cwd}) as r:
                res = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", "replace")
            except Exception:
                body = ""
            return f"ERROR: HTTP {e.code} {body[:1000]}"
        except Exception as e:
            return f"ERROR: {e}"
        if isinstance(res, dict) and res.get("error"):
            return "ERROR: " + str(res["error"])
        if isinstance(res, dict) and "plain_text_response" in res:
            return str(res["plain_text_response"])
        if isinstance(res, dict) and "result" in res:
            extra = f" (edits_applied={res['edits_applied']})" if "edits_applied" in res else ""
            return str(res["result"]) + extra
        return json.dumps(res, ensure_ascii=False)

    # ---------- 上下文压缩与 RAG 归档 / 召回 ----------

    def _build_system_content(self, recalled_context=""):
        parts = [SYSTEM_TEMPLATE.format(cwd=self.cwd)]
        if self.rag_enabled:
            parts.append(RAG_SYSTEM_ADDON.strip())
        if self.memory_summary.strip():
            parts.append(
                "<compressed_memory_summary>\n"
                "以下是本会话早先被压缩对话的摘要（完整原文已存入 KVault RAG，需要细节请调用 recall_memory）：\n"
                f"{self.memory_summary.strip()}\n"
                "</compressed_memory_summary>"
            )
        if recalled_context.strip():
            parts.append(
                "<recalled_context>\n"
                "以下是从 KVault RAG 召回的与当前问题相关的早期记忆片段（仅供参考与核对）：\n"
                f"{recalled_context.strip()}\n"
                "</recalled_context>"
            )
        return "\n\n".join(parts)

    def build_messages(self, recalled_context=""):
        return [{"role": "system", "content": self._build_system_content(recalled_context)}] + self.messages

    def _prompt_overhead_tokens(self, recalled_context=""):
        tokens = estimate_tokens(self._build_system_content(recalled_context)) + 16
        if self.use_tools and self.tool_defs:
            tokens += estimate_tokens(json.dumps(self.tool_defs, ensure_ascii=False))
        return tokens

    def _format_turn_for_archive(self, turn):
        """将单个完整回合格式化为适合 RAG 切片检索的 Markdown 文本。"""
        user_text = ""
        tool_calls_by_id = {}
        actions = []
        assistant_texts = []

        for msg in turn:
            role = msg.get("role")
            if role == "user":
                user_text = (msg.get("content") or "").strip()
            elif role == "assistant":
                txt = (msg.get("content") or "").strip()
                if txt:
                    assistant_texts.append(txt)
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    t_name = fn.get("name", "")
                    t_args = fn.get("arguments", "")
                    t_id = tc.get("id", "")
                    tool_calls_by_id[t_id] = (t_name, t_args)
            elif role == "tool":
                t_id = msg.get("tool_call_id", "")
                t_name, t_args = tool_calls_by_id.get(t_id, ("tool", ""))
                # 跳过 recall_memory 自身输出，避免记忆回声循环膨胀
                if t_name == "recall_memory":
                    continue
                t_out = (msg.get("content") or "").strip()
                actions.append((t_name, t_args[:240], t_out[:2000]))

        info = self._msg_turn_info.get(id(turn[0]), {}) if turn else {}
        t_idx = info.get("turn_index", 0)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"## Turn {t_idx} ({ts})", f"**用户**: {user_text}"]
        if actions:
            lines.append("**工具执行**:")
            for t_name, t_args, t_out in actions:
                lines.append(f"- `{t_name}({t_args})`\n  结果: {t_out}")
        final_reply = "\n".join(assistant_texts).strip()
        if final_reply:
            lines.append(f"**助手**: {final_reply}")
        return "\n\n".join(lines), user_text, final_reply

    def _archive_turn_obj(self, turn):
        """把单个回合存入 KVault RAG 并标记已归档。"""
        if not self.rag_enabled or not turn:
            return
        user_msg = turn[0]
        info = self._msg_turn_info.get(id(user_msg))
        if not info:
            self.turn_counter += 1
            info = {
                "turn_id": f"{self.session_id}_t{self.turn_counter:04d}",
                "turn_index": self.turn_counter,
                "archived": False,
            }
            self._msg_turn_info[id(user_msg)] = info
        if info["archived"]:
            return
        text, u_prev, a_prev = self._format_turn_for_archive(turn)
        res = self.rag.call(
            "archive_turn",
            session_id=self.session_id,
            turn_index=info["turn_index"],
            turn_id=info["turn_id"],
            cwd=self.cwd,
            turn_text=text,
            user_preview=u_prev,
            assistant_preview=a_prev,
        )
        if res and res.get("ok"):
            info["archived"] = True

    def _extractive_summary(self, evicted_turns):
        """离线/失败兜底：确定性提取被压缩回合的关键问答摘要。"""
        items = []
        if self.memory_summary.strip():
            items.append(self.memory_summary.strip())
        for turn in evicted_turns:
            info = self._msg_turn_info.get(id(turn[0]), {})
            idx = info.get("turn_index", "?")
            u = (turn[0].get("content") or "").strip().replace("\n", " ")[:120]
            replies = [m.get("content", "").strip() for m in turn if m.get("role") == "assistant" and m.get("content")]
            a = replies[-1].replace("\n", " ")[:160] if replies else ""
            tools = []
            for m in turn:
                for tc in m.get("tool_calls") or []:
                    fn = (tc.get("function") or {}).get("name", "")
                    if fn and fn != "recall_memory":
                        tools.append(fn)
            t_str = f" [工具: {','.join(dict.fromkeys(tools))}]" if tools else ""
            items.append(f"- T{idx}: 问「{u}」{t_str} → 答「{a}」")
        merged = "\n".join(items)
        if len(merged) > SUMMARY_MAX_CHARS:
            merged = "...(更早摘要已存入 RAG)\n" + merged[-(SUMMARY_MAX_CHARS - 20):]
        return merged

    def _summarize_evicted(self, evicted_turns):
        """调用本地模型将旧摘要与即将移除的回合压缩为精简结构化摘要。"""
        snippets = []
        for turn in evicted_turns:
            text, _, _ = self._format_turn_for_archive(turn)
            snippets.append(text[:1200])
        joined = "\n\n---\n\n".join(snippets)
        if len(joined) > 6000:
            joined = joined[:6000] + "\n...(已截断)"

        prompt = (
            "请把【已有摘要】与【待压缩对话】合并成一份精简的中文记忆摘要（严格不超过 380 字）。\n"
            "必须保留：\n"
            "1. 用户的核心目标、偏好与约束；\n"
            "2. 已确认的结论、精确的文件路径/函数名/端口/命令结果；\n"
            "3. 尚未完成的事项。\n"
            "不要废话，直接输出条目。\n\n"
            f"【已有摘要】\n{self.memory_summary.strip() or '(无)'}\n\n"
            f"【待压缩对话】\n{joined}"
        )
        body = {
            "messages": [
                {"role": "system", "content": "你是严谨的对话记忆压缩助手，只输出精炼事实条目。"},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": 450,
            "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            with request("POST", self.base + "/v1/chat/completions", body, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if choices:
                content = (choices[0].get("message") or {}).get("content") or ""
                content = content.strip()
                if content:
                    return content[:SUMMARY_MAX_CHARS]
        except Exception:
            pass
        return self._extractive_summary(evicted_turns)

    def _compact_single_turn_tools(self, turn, budget_tokens):
        """单个回合内若多步工具输出撑爆上下文，将旧 tool 结果归档到 RAG 并替换为指针。"""
        tool_indices = [i for i, m in enumerate(turn) if m.get("role") == "tool"]
        if len(tool_indices) <= 1:
            return False

        info = self._msg_turn_info.get(id(turn[0]), {})
        t_idx = info.get("turn_index", self.turn_counter)
        compacted = 0

        # 保留最新 1 条工具结果，压缩更早的大输出
        for pos in tool_indices[:-1]:
            cur_tokens = sum(estimate_message_tokens(m) for m in turn)
            if cur_tokens <= budget_tokens:
                break
            msg = turn[pos]
            content = msg.get("content") or ""
            if len(content) <= 300 or content.startswith("[历史工具输出已归档"):
                continue
            if self.rag_enabled:
                blob_id = f"{self.session_id}_t{t_idx}_tool_{pos}_{uuid.uuid4().hex[:4]}"
                self.rag.call(
                    "archive_turn",
                    session_id=self.session_id,
                    turn_index=t_idx,
                    turn_id=blob_id,
                    cwd=self.cwd,
                    turn_text=f"## Turn {t_idx} 工具输出归档 ({blob_id})\n\n{content}",
                    user_preview=f"Turn {t_idx} 工具输出",
                    assistant_preview=content[:180],
                )
            preview = content[:160].replace("\n", " ")
            msg["content"] = (
                f"[历史工具输出已归档至 KVault RAG，原始 {len(content)} 字符；"
                f"需要细节时请调用 recall_memory 检索。开头预览: {preview}...]"
            )
            compacted += 1

        if compacted:
            print(f"{DIM}(单轮工具上下文过长，已将 {compacted} 条早期工具输出归档至 RAG){RESET}")
            return True
        return False

    def trim(self, force=False, recalled_context=""):
        """接近上下文上限时，将最早回合归档到 KVault RAG 并压缩为滚动摘要。"""
        if not self.messages:
            return False

        reserve = min(CONTEXT_RESERVE, max(256, self.ctx // 4))
        usable_ctx = max(512, self.ctx - reserve)
        overhead = self._prompt_overhead_tokens(recalled_context)
        msg_tokens = sum(estimate_message_tokens(m) for m in self.messages)
        est_total = overhead + msg_tokens

        trigger_tokens = min(int(self.ctx * 0.72), max(512, usable_ctx - 256))
        target_total = min(int(self.ctx * 0.42), max(400, int(trigger_tokens * 0.65)))
        target_msg_tokens = max(128, target_total - overhead)

        ctx_over = self.ctx_used is not None and self.ctx_used > trigger_tokens
        need = force or (est_total > trigger_tokens) or ctx_over
        if not need:
            return False

        turns = group_turns(self.messages)
        if len(turns) <= 1:
            changed = self._compact_single_turn_tools(turns[0], max(256, usable_ctx - overhead))
            if changed:
                self.ctx_used = None
            return changed

        # 至少保留最近 1 个完整回合，其余最旧回合移出滑窗
        kept = list(turns)
        evicted = []
        while len(kept) > 1:
            kept_tokens = sum(estimate_message_tokens(m) for t in kept for m in t)
            if not force and kept_tokens <= target_msg_tokens:
                break
            if force and len(kept) <= 2 and kept_tokens <= target_msg_tokens:
                break
            evicted.append(kept.pop(0))
            if force and len(kept) == 1:
                break

        # 若实际服务端 ctx_used 超限但估算未超标，仍至少归档最早 1 轮
        if not evicted and ctx_over and len(kept) > 1:
            evicted.append(kept.pop(0))

        if not evicted:
            return False

        archived_count = 0
        for turn in evicted:
            if self.rag_enabled:
                self._archive_turn_obj(turn)
                archived_count += 1
            info = self._msg_turn_info.pop(id(turn[0]), None)
            if info:
                self.active_turn_ids.discard(info["turn_id"])

        if self.rag_enabled:
            self.memory_summary = self._summarize_evicted(evicted)
            print(
                f"{DIM}(上下文压缩: 已归档 {archived_count} 轮至 KVault RAG，"
                f"保留最近 {len(kept)} 轮，摘要 {len(self.memory_summary)} 字){RESET}"
            )
        else:
            print(f"{DIM}(上下文接近上限，已丢弃最早的 {len(evicted)} 轮对话){RESET}")

        self.messages = [m for t in kept for m in t]
        self.ctx_used = None
        return True

    def _auto_recall_for_query(self, user_text):
        """基于用户提问从 KVault RAG 自动召回高相关历史片段（仅用于本轮临时注入）。"""
        if not self.rag_enabled or not self.auto_recall:
            return ""
        st = self.rag.call("status") or {}
        if not st.get("turns"):
            return ""

        res = self.rag.call(
            "search",
            query=user_text,
            top_k=3,
            scope="conversation",
            exclude_turn_ids=list(self.active_turn_ids),
            current_session_id=self.session_id,
            min_vector_score=0.45,
        )
        hits = (res or {}).get("results") or []
        if not hits:
            return ""

        blocks = []
        total_chars = 0
        best_score = 0.0
        for h in hits:
            tag = "本会话" if h.get("is_current_session") else f"历史会话 {h.get('session_id', '')}"
            t_idx = h.get("turn_index", -1)
            sc = h.get("score", 0.0)
            best_score = max(best_score, sc)
            snippet = (h.get("content") or "").strip()
            block = f"- [{tag} · Turn {t_idx} · 匹配度 {sc:.2f}]\n{snippet}"
            if total_chars + len(block) > AUTO_RECALL_MAX_CHARS:
                remain = AUTO_RECALL_MAX_CHARS - total_chars
                if remain > 120:
                    blocks.append(block[:remain].rstrip() + "...")
                break
            blocks.append(block)
            total_chars += len(block)

        if blocks:
            print(f"{DIM}(RAG 自动召回 {len(blocks)} 条相关记忆，最高匹配 {best_score:.2f}){RESET}")
        return "\n\n".join(blocks)

    def _execute_recall_tool(self, args):
        """在本地执行模型发起的 recall_memory 工具调用。"""
        if not self.rag_enabled:
            return "RAG 记忆功能未启用。"
        query = str(args.get("query") or "").strip()
        if not query:
            return "ERROR: query 不能为空。"
        top_k = args.get("top_k", 5)
        scope = str(args.get("scope") or "conversation").strip().lower()
        if scope not in ("conversation", "knowledge", "all"):
            scope = "conversation"

        res = self.rag.call(
            "search",
            query=query,
            top_k=top_k,
            scope=scope,
            exclude_turn_ids=list(self.active_turn_ids),
            current_session_id=self.session_id,
            min_vector_score=0.0,
        )
        if not res:
            return "ERROR: RAG 检索调用失败。"
        if res.get("error"):
            return f"ERROR: {res['error']}"
        hits = res.get("results") or []
        if not hits:
            return f"未在 {scope} 中检索到与「{query}」相关的记录。"

        lines = [f"检索模式: {res.get('mode', 'rag')} | 命中 {len(hits)} 条:"]
        for i, h in enumerate(hits, 1):
            if h.get("source") == "knowledge":
                header = f"[{i}] 知识库文档《{h.get('document_name', '')}》(score={h.get('score', 0):.3f})"
            else:
                sess_tag = "当前会话" if h.get("is_current_session") else f"历史会话 {h.get('session_id', '')}"
                header = (
                    f"[{i}] {sess_tag} Turn {h.get('turn_index', '?')} "
                    f"({h.get('timestamp', '')}, {h.get('hit_source', 'rag')}, score={h.get('score', 0):.3f})"
                )
            lines.append(f"{header}\n{h.get('content', '').strip()}")
        return "\n\n".join(lines)

    # ---------- 对话请求 ----------

    def chat_stream(self, messages):
        body = {
            "messages": messages,
            "stream": True,
            "max_tokens": -1,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": self.think},
        }
        if self.use_tools:
            body.update({"tools": self.tool_defs, "tool_choice": "auto"})

        content, reasoning, calls = "", "", {}
        meta = {"prompt_tokens": None, "completion_tokens": None,
                "predicted_per_second": None, "prompt_per_second": None,
                "ttft": None, "elapsed": 0.0, "generated": 0}
        start = time.monotonic()
        first_at = None
        in_reasoning = False
        thinking_line = False
        live_title = sys.stdout.isatty()
        wait_shown = live_title
        last_ui = 0.0
        if wait_shown:
            print(f"\r{DIM}⏳ 等待响应...{RESET}", end="", flush=True)

        try:
            with request("POST", self.base + "/v1/chat/completions", body) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    usage = chunk.get("usage")
                    if usage:
                        meta["prompt_tokens"] = usage.get("prompt_tokens")
                        meta["completion_tokens"] = usage.get("completion_tokens")
                    timings = chunk.get("timings")
                    if timings:
                        meta["predicted_per_second"] = timings.get("predicted_per_second")
                        meta["prompt_per_second"] = timings.get("prompt_per_second")
                        if meta["completion_tokens"] is None:
                            meta["completion_tokens"] = timings.get("predicted_n")

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}

                    think_piece = delta.get("reasoning_content") or ""
                    text_piece = delta.get("content") or ""
                    tool_pieces = delta.get("tool_calls") or []
                    produced = bool(think_piece or text_piece or tool_pieces)

                    if produced:
                        meta["generated"] += 1
                        if first_at is None:
                            first_at = time.monotonic() - start
                            meta["ttft"] = first_at
                            if wait_shown:
                                print("\x1b[2K\r", end="", flush=True)
                                wait_shown = False

                    if think_piece:
                        reasoning += think_piece
                        if self.show_reasoning:
                            if not in_reasoning:
                                print(f"{DIM}", end="", flush=True)
                                in_reasoning = True
                            print(think_piece, end="", flush=True)
                        elif not thinking_line and live_title:
                            print(f"\r\x1b[2K{DIM}🤔 思考中...{RESET}", end="", flush=True)
                            thinking_line = True

                    if text_piece:
                        if thinking_line:
                            print("\x1b[2K\r", end="", flush=True)
                            thinking_line = False
                        if in_reasoning:
                            print(f"{RESET}", end="", flush=True)
                            print()
                            in_reasoning = False
                        content += text_piece
                        print(text_piece, end="", flush=True)

                    for tc in tool_pieces:
                        slot = calls.setdefault(tc.get("index", 0),
                                                {"id": "", "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""

                    now = time.monotonic()
                    if now - last_ui >= 0.3:
                        last_ui = now
                        if thinking_line:
                            print(f"\r\x1b[2K{DIM}🤔 思考中... {meta['generated']} tok{RESET}",
                                  end="", flush=True)
                        elif live_title and first_at is not None:
                            tps = meta["generated"] / max(now - start - first_at, 0.001)
                            _set_title(f"llama-agent | {tps:.1f} t/s | {meta['generated']} tok")
        finally:
            if wait_shown or thinking_line:
                print("\x1b[2K\r", end="", flush=True)
            if live_title:
                _set_title(_ORIGINAL_TITLE)

        if in_reasoning:
            print(f"{RESET}", end="")
        print()

        meta["elapsed"] = time.monotonic() - start
        meta["reasoning_chars"] = len(reasoning)
        if meta["completion_tokens"] is None:
            meta["completion_tokens"] = meta["generated"]
        if meta["prompt_tokens"] is not None:
            self.ctx_used = meta["prompt_tokens"] + meta["completion_tokens"]
        return content, reasoning, [calls[i] for i in sorted(calls)], meta

    def print_metrics(self, meta):
        """输出上下文占用条与生成速度。"""
        parts = []
        if self.ctx_used is not None:
            ratio = min(self.ctx_used / max(self.ctx, 1), 1.0)
            color = GREEN if ratio < 0.6 else YELLOW if ratio < 0.85 else RED
            width = 18
            filled = int(round(ratio * width))
            bar = "█" * filled + "░" * (width - filled)
            parts.append(f"{DIM}上下文{RESET} {color}{bar}{RESET} "
                         f"{self.ctx_used / 1000:.1f}k/{self.ctx / 1000:.1f}k {color}{ratio:.0%}{RESET}")
        speed = meta.get("predicted_per_second")
        tok = meta.get("completion_tokens") or 0
        secs = meta.get("elapsed") or 0
        if speed is None and secs > 0 and tok:
            speed = tok / secs
        if speed:
            parts.append(f"{DIM}输出{RESET} {YELLOW}{speed:.1f}{RESET} {DIM}t/s · {tok} tok · {secs:.1f}s{RESET}")
        if meta.get("ttft") is not None:
            parts.append(f"{DIM}首字 {meta['ttft']:.2f}s{RESET}")
        if meta.get("prompt_per_second"):
            parts.append(f"{DIM}prefill {meta['prompt_per_second']:.0f} t/s{RESET}")
        if meta.get("reasoning_chars") and not self.show_reasoning:
            parts.append(f"{DIM}思考已隐藏 · /show-think{RESET}")
        if parts:
            print(f"  {DIM}└─{RESET} " + f"  {DIM}│{RESET}  ".join(parts))

    def ctx_tag(self):
        if self.ctx_used is None:
            return f"{DIM}[--]{RESET} "
        ratio = min(self.ctx_used / max(self.ctx, 1), 1.0)
        color = GREEN if ratio < 0.6 else YELLOW if ratio < 0.85 else RED
        return f"{DIM}[{color}{ratio:.0%}{RESET}{DIM}]{RESET} "

    def confirm(self, name, args):
        if name not in self.write_tools or self.yolo:
            return True
        preview = json.dumps(args, ensure_ascii=False)
        if len(preview) > 300:
            preview = preview[:300] + " ..."
        while True:
            try:
                answer = input(f"{YELLOW}允许 {name}？{RESET} {preview}\n[y=允许 / n=拒绝 / a=本次全部允许] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return False
            if answer in ("y", "yes"):
                return True
            if answer in ("a", "all"):
                self.yolo = True
                return True
            if answer in ("n", "no", ""):
                return False

    def _archive_active_turn_if_complete(self, user_msg):
        """本轮对话完成后立即持久化到 KVault RAG（仍在窗口内则检索时按 active_turn_ids 排除）。"""
        if not self.rag_enabled:
            return
        for turn in group_turns(self.messages):
            if turn and turn[0] is user_msg:
                self._archive_turn_obj(turn)
                return

    def run_turn(self, user_text):
        self.turn_counter += 1
        turn_id = f"{self.session_id}_t{self.turn_counter:04d}"
        user_msg = {"role": "user", "content": user_text}
        self._msg_turn_info[id(user_msg)] = {
            "turn_id": turn_id,
            "turn_index": self.turn_counter,
            "archived": False,
        }
        self.active_turn_ids.add(turn_id)

        self.messages.append(user_msg)
        self.trim()
        recalled_context = self._auto_recall_for_query(user_text)
        steps = 0
        reasoning_parts = []
        known_tools = {d["function"]["name"] for d in self.tool_defs}

        try:
            while True:
                # 仅在首次请求时注入自动召回片段；后续 tool-loop 依赖本轮 tool 结果与 recall_memory
                cur_recall = recalled_context if steps == 0 else ""
                if steps > 0:
                    self.trim(recalled_context=cur_recall)
                try:
                    content, reasoning, calls, meta = self.chat_stream(
                        self.build_messages(recalled_context=cur_recall)
                    )
                except urllib.error.HTTPError as e:
                    body = ""
                    try:
                        body = e.read().decode("utf-8", "replace")
                    except Exception:
                        pass
                    print(f"{RED}请求失败 HTTP {e.code}: {body[:500]}{RESET}")
                    if steps == 0 and self.messages and self.messages[-1] is user_msg:
                        self.messages.pop()
                        self._msg_turn_info.pop(id(user_msg), None)
                        self.active_turn_ids.discard(turn_id)
                    return
                except Exception as e:
                    print(f"{RED}请求失败: {e}{RESET}")
                    if steps == 0 and self.messages and self.messages[-1] is user_msg:
                        self.messages.pop()
                        self._msg_turn_info.pop(id(user_msg), None)
                        self.active_turn_ids.discard(turn_id)
                    return

                self.print_metrics(meta)
                if reasoning and reasoning.strip():
                    reasoning_parts.append(reasoning)
                    self.last_reasoning = "\n".join(reasoning_parts)

                assistant = {"role": "assistant", "content": content}
                if not content and not calls and reasoning:
                    if self.show_reasoning:
                        content = reasoning
                        assistant["content"] = content
                    else:
                        print(f"{YELLOW}(模型只输出了思考内容，未生成正文；输入 /show-think 查看){RESET}")
                if calls:
                    assistant["tool_calls"] = [{
                        "id": c["id"] or f"call_{steps}_{i}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                    } for i, c in enumerate(calls)]
                self.messages.append(assistant)
                if not calls:
                    return

                steps += 1
                if steps > self.max_steps:
                    print(f"{RED}达到单轮最多 {self.max_steps} 步工具调用，已停止。{RESET}")
                    return

                for i, c in enumerate(calls):
                    name = c["name"]
                    try:
                        args = json.loads(c["arguments"] or "{}")
                    except json.JSONDecodeError:
                        result = "ERROR: 参数不是合法 JSON，请重新给出 tool call。"
                        self.messages.append({"role": "tool", "tool_call_id": c["id"] or "call", "content": result})
                        continue

                    if name == "recall_memory" and self.rag_enabled:
                        print(f"{CYAN}→ {name}{RESET} {DIM}{json.dumps(args, ensure_ascii=False)[:200]}{RESET}")
                        result = self._execute_recall_tool(args)
                        first = result.strip().splitlines()[0] if result.strip() else "(空)"
                        print(f"{DIM}  {first[:160]}{RESET}")
                    elif name not in known_tools:
                        result = f"ERROR: 未知工具 {name}"
                    elif not self.confirm(name, args):
                        for rest in calls[i:]:
                            self.messages.append({
                                "role": "tool", "tool_call_id": rest["id"] or "call",
                                "content": "未执行：用户取消了本轮操作。"})
                        print(f"{YELLOW}已取消本轮操作。{RESET}")
                        return
                    else:
                        print(f"{CYAN}→ {name}{RESET} {DIM}{json.dumps(args, ensure_ascii=False)[:200]}{RESET}")
                        result = self.call_tool(name, args)
                        first = result.strip().splitlines()[0] if result.strip() else "(空)"
                        print(f"{DIM}  {first[:160]}{RESET}")

                    if len(result) > MAX_TOOL_OUTPUT:
                        result = result[:MAX_TOOL_OUTPUT] + f"\n...（已截断，原始 {len(result)} 字符）"
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": c["id"] or "call",
                        "content": result,
                    })
        finally:
            self._archive_active_turn_if_complete(user_msg)

    # ---------- 交互 ----------

    def _print_rag_status(self):
        if not self.rag_enabled:
            reason = self.rag.error if self.rag else "已禁用"
            print(f"{YELLOW}RAG 记忆未启用: {reason}{RESET}\n")
            return
        st = self.rag.call("status") or {}
        print(f"{GREEN}KVault RAG 记忆状态{RESET}")
        print(f"  知识库目录: {st.get('kvault_dir')}")
        print(f"  记忆存储区: {st.get('memory_dir')}")
        print(f"  检索后端  : {st.get('embed_status')} | jieba={'on' if st.get('jieba_available') else 'off'}")
        print(f"  已归档数据: {st.get('sessions', 0)} 个会话 / {st.get('turns', 0)} 轮 / "
              f"{st.get('chunks', 0)} 片段 / {st.get('vectors', 0)} 向量"
              + (f" (待补向量 {st['pending_embed']})" if st.get("pending_embed") else ""))
        print(f"  主知识库  : {st.get('main_kb_vectors', 0)} 条可用向量")
        print(f"  当前会话  : {self.session_id} | 自动召回={'on' if self.auto_recall else 'off'} | "
              f"摘要={len(self.memory_summary)} 字\n")

    def _handle_rag_command(self, text):
        rest = text[4:].strip()
        if not rest or rest == "status":
            self._print_rag_status()
            return
        if rest.startswith("auto"):
            val = rest[4:].strip().lower()
            self.auto_recall = val != "off" if val else not self.auto_recall
            print(f"RAG 自动召回={'on' if self.auto_recall else 'off'}\n")
            return
        if not self.rag_enabled:
            print(f"{RED}RAG 记忆未启用。{RESET}\n")
            return
        if rest == "sync":
            res = self.rag.call("sync_pending") or {}
            if res.get("ok"):
                print(f"{GREEN}已同步 {res.get('synced', 0)} 轮待嵌入对话 ({res.get('mode')})。{RESET}\n")
            else:
                print(f"{YELLOW}同步未执行: {res.get('reason', res.get('error', 'Embedding 服务不可用'))}{RESET}\n")
            return
        if rest.startswith("search "):
            q = rest[7:].strip()
            if not q:
                print(f"{RED}用法: /rag search <关键词>{RESET}\n")
                return
            res = self.rag.call(
                "search",
                query=q,
                top_k=5,
                scope="all",
                exclude_turn_ids=[],
                current_session_id=self.session_id,
                min_vector_score=0.0,
            ) or {}
            hits = res.get("results") or []
            if not hits:
                print(f"{DIM}(未命中相关记忆){RESET}\n")
                return
            print(f"{GREEN}RAG 检索结果 ({res.get('mode', '')}):{RESET}")
            for i, h in enumerate(hits, 1):
                src = "知识库" if h.get("source") == "knowledge" else f"Turn {h.get('turn_index')}"
                print(f"  {CYAN}[{i}] {src} ({h.get('hit_source')}, score={h.get('score')}){RESET}")
                print(f"  {DIM}{h.get('content', '').replace(chr(10), ' ')[:220]}{RESET}")
            print()
            return
        if rest == "wipe":
            try:
                ans = input(f"{YELLOW}确认清空所有 KVault 对话归档记忆？[y/N] > {RESET}").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if ans in ("y", "yes"):
                self.rag.call("wipe")
                self.memory_summary = ""
                print("已清空 KVault 对话记忆。\n")
            else:
                print("已取消。\n")
            return
        print(f"{RED}未知 /rag 子命令，可用: status | search <词> | auto on|off | sync | wipe{RESET}\n")

    def repl(self):
        mode = f"{len(self.tool_defs)} 个工具" if self.use_tools else "纯聊天"
        print(f"{GREEN}本地 agent{RESET} | 模型: {self.model}")
        print(f"工作目录: {self.cwd} | 上下文: {self.ctx} token | {mode} | thinking={'on' if self.think else 'off'}")
        if self.rag_enabled:
            st = self.rag.init_status or {}
            print(f"RAG 记忆: {GREEN}KVault{RESET} ({st.get('embed_status', 'ok')} | "
                  f"已存 {st.get('turns', 0)} 轮/{st.get('chunks', 0)} 片段) | "
                  f"自动召回={'on' if self.auto_recall else 'off'}")
        elif self.rag is not None:
            print(f"RAG 记忆: {YELLOW}未启用 ({self.rag.error}){RESET}")
        print(f"{DIM}思考过程默认隐藏，/show-think 查看最近一次；输入 /help 查看命令，/exit 退出{RESET}\n")

        while True:
            try:
                text = input(f"{self.ctx_tag()}{CYAN}你> {RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not text:
                continue
            if text in ("/exit", "/quit"):
                return
            if text == "/help":
                print("/clear 清空当前窗口（历史保留在 RAG） | /compact 立即压缩归档上下文")
                print("/memory 查看滚动摘要与近期归档 | /rag [status|search <词>|auto on|off|sync|wipe]")
                print("/cwd PATH 切换工作目录 | /tools 刷新工具列表")
                print("/think on|off 模型思考 | /show-think 查看最近一次思考过程")
                print("/think-view on|off 是否实时显示思考（默认隐藏） | /yolo on|off 免确认 | /exit 退出")
                print(f"{DIM}回复下方条形 = 上下文占用 + 输出速度；提示符 [xx%] = 当前上下文占比{RESET}\n")
                continue
            if text == "/compact":
                if not self.messages:
                    print("(当前没有可压缩的对话)\n")
                elif not self.trim(force=True):
                    print(f"{DIM}(当前仅剩 1 轮且未超限，无需压缩){RESET}\n")
                else:
                    print()
                continue
            if text == "/memory":
                if self.memory_summary.strip():
                    print(f"{CYAN}当前滚动摘要:{RESET}\n{self.memory_summary.strip()}\n")
                else:
                    print(f"{DIM}(当前暂无滚动摘要){RESET}\n")
                if self.rag_enabled:
                    turns = self.rag.call("list_turns", limit=8) or []
                    if turns:
                        print(f"{CYAN}最近归档记忆:{RESET}")
                        for t in turns:
                            vec = "✓向量" if t.get("embedded") else "BM25"
                            print(f"  {DIM}[{t['timestamp']} | {t['session_id']} T{t['turn_index']} | {vec}]{RESET} {t['user_preview'][:70]}")
                        print()
                continue
            if text == "/rag" or text.startswith("/rag "):
                self._handle_rag_command(text)
                continue
            if text == "/show-think":
                if self.last_reasoning.strip():
                    print(f"{DIM}{self.last_reasoning.strip()}{RESET}\n")
                else:
                    print("(没有可显示的思考内容)\n")
                continue
            if text.startswith("/think-view"):
                value = text[11:].strip().lower()
                self.show_reasoning = value != "off" if value else not self.show_reasoning
                print(f"思考显示={'on' if self.show_reasoning else 'hidden'}\n")
                continue
            if text == "/clear":
                for turn in group_turns(self.messages):
                    self._archive_turn_obj(turn)
                self.messages.clear()
                self._msg_turn_info.clear()
                self.active_turn_ids.clear()
                self.memory_summary = ""
                self.ctx_used = None
                self.last_reasoning = ""
                self.turn_counter = 0
                self.session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
                print("已清空当前上下文窗口（对话记录已保存在 RAG 知识库，新会话已启动）。\n")
                continue
            if text.startswith("/cwd"):
                path = text[4:].strip().strip('"')
                if path and os.path.isdir(path):
                    self.cwd = os.path.abspath(path)
                    print(f"工作目录 → {self.cwd}\n")
                else:
                    print(f"{RED}目录不存在: {path or '(空)'}{RESET}\n")
                continue
            if text == "/tools":
                self.fetch_tools()
                print("工具: " + ", ".join(d["function"]["name"] for d in self.tool_defs) + "\n")
                continue
            if text.startswith("/think"):
                value = text[6:].strip().lower()
                self.think = value != "off" if value else not self.think
                print(f"thinking={'on' if self.think else 'off'}\n")
                continue
            if text.startswith("/yolo"):
                value = text[5:].strip().lower()
                self.yolo = value != "off" if value else not self.yolo
                print(f"免确认={'on' if self.yolo else 'off'}\n")
                continue
            if text.startswith("/"):
                print(f"{RED}未知命令，输入 /help 查看。{RESET}\n")
                continue
            self.run_turn(text)
            print()


def main():
    parser = argparse.ArgumentParser(description="llama-server 轻量本地 agent (KVault RAG 上下文压缩)")
    parser.add_argument("--url", default=DEFAULT_URL, help="llama-server 地址")
    parser.add_argument("--cwd", default=os.getcwd(), help="工具工作目录")
    parser.add_argument("--no-tools", action="store_true", help="纯聊天模式")
    parser.add_argument("--no-think", action="store_true", help="关闭思考模式")
    parser.add_argument("--yolo", action="store_true", help="写文件/执行命令不再确认")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="单轮最多工具轮数")
    parser.add_argument("--kvault-dir", default=None, help="KVault 知识库目录（默认自动探测）")
    parser.add_argument("--rag-dir", default=None, help="对话 RAG 存储目录（默认 <kvault>/data/agent_memory）")
    parser.add_argument("--no-rag", action="store_true", help="禁用 KVault RAG 上下文压缩与召回")
    parser.add_argument("--no-auto-recall", action="store_true", help="关闭每轮自动召回，仅保留 recall_memory 工具")
    parser.add_argument("--embed-backend", default="auto", choices=["auto", "llama_cpp_server", "ollama", "bm25"], help="Embedding 后端")
    parser.add_argument("--embed-url", default="", help="Embedding 服务地址（如 http://127.0.0.1:8001）")
    parser.add_argument("--embed-model", default="", help="Embedding 模型名称/别名")
    parser.add_argument("--keep-server", action="store_true", help="退出时不关闭由启动脚本拉起的 llama-server")
    parser.add_argument("--rag-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("prompt", nargs="*", help="可选：一次性提问后退出")
    args = parser.parse_args()

    if args.rag_worker:
        kv_dir = resolve_kvault_dir(args.kvault_dir) or DEFAULT_KVAULT_DIR
        args.kvault_dir = kv_dir
        raise SystemExit(run_rag_worker(args))

    enable_ansi()
    rag = None
    if not args.no_rag:
        kv_dir = resolve_kvault_dir(args.kvault_dir)
        rag = RAGClient(
            kvault_dir=kv_dir,
            rag_dir=args.rag_dir,
            embed_backend=args.embed_backend,
            embed_url=args.embed_url,
            embed_model=args.embed_model,
        )

    agent = Agent(
        args.url, args.cwd, not args.no_think, args.yolo,
        not args.no_tools, args.max_steps,
        rag=rag, auto_recall=not args.no_auto_recall,
    )
    try:
        agent.probe()
        if args.prompt:
            agent.run_turn(" ".join(args.prompt))
        else:
            agent.repl()
    finally:
        if rag is not None:
            rag.close()


if __name__ == "__main__":
    main()
