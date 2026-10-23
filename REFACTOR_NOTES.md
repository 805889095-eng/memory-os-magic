# Memory OS 魔改版 · 改造说明（Zero-Local-Model / Remote-API-Only / Self-Evolving）

> 本文件是本工程唯一新增的说明文档，随源码一起放在仓库根目录。
> 改造对象：`github.com/ClaudioDrews/memory-os`（main 分支，MIT 许可，2026-10-23 拉取的 tar 快照）。
> 改造目标：**删掉全部本地模型，只用远程 API；同时把「记忆自我进化」这条链真正接通，让 agent 越用越准。**

---

## 一、一句话总览

| 维度 | 改前 | 改后 |
|---|---|---|
| 稀疏向量（BM25） | `fastembed` / ONNX Runtime 本地推理，需 build-essential、python3-dev、下载模型 | **纯 Python `bm25_tf.py`**，零模型、零网络、零原生扩展 |
| 稠密向量（embedding） | 本地/远程混合 | **仅远程** OpenAI 兼容 `POST {EMBEDDING_API_BASE}/embeddings`，缺 key 直接报错 |
| 生成（LLM） | 原生 Ollama（`OLLAMA_BASE_URL` 等） | **远程** `POST {LLM_API_BASE}/chat/completions`（OpenAI 兼容） |
| 事实信任分 | 恒为 0.50，**只读不写**，README 承诺的 "automatic feedback loop" 无实现 | **五步闭环**：写入 → 计数 → 自动反馈 → 调信任分 → 重排+衰减，起始 0.55 |
| 事实衰减调度 | 无（`facts` 表永不衰减） | `icarus/evolution.py --decay 30` **每日 03:30 cron** |
| 维护调度 | 1 条（每小时 wiki 摄取） | **6 条**幂等 cron（含反思、衰减、去重、DLQ、事实衰减） |
| 配置缺失 | fail-open 静默降级（"记忆变笨但不报错"） | **缺 key / 缺依赖 → 显式 `RuntimeError`** |
| 阻塞缺陷 | P0 密钥外泄 + 7 类 P1 | **全部修复**（见第五节） |

改动规模：**43 个文件被修改，3 个文件新增**（另加本说明文档），合计新增/改写约 4,700 行（其中 `icarus/hooks.py` 841 行、`scripts/context_enhancer.py` 786 行、`icarus/state.py` 502 行、`icarus/fabric-retrieve.py` 315 行、`docker/worker/main.py` 131 行）。

---

## 二、零本地模型：替换清单

### 2.1 稀疏向量 —— 去模型化（最关键的一处）

原实现：`fastembed` 的 `SparseTextEmbedding(model_name="Qdrant/bm25")` 在进程内跑 ONNX 推理。

**它为什么能安全去掉**：`docker/worker/services/local_qdrant.py` 建集合时，稀疏向量声明为
`SparseVectorParams(modifier=Modifier.IDF)`。也就是说 —— **IDF 是 Qdrant 服务端算的，客户端只需要吐 BM25 的 tf 饱和项**。而 tf 饱和项不依赖任何模型参数，纯算即可。

新实现：`bm25_tf(text, *, k1=1.2, b=0.75, avg_len=256.0) -> {"indices": [...], "values": [...]}`

- 分词：`re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())` —— 英文数字整段保留，**每个汉字单独成一个 token**（无中文分词器，这是取舍，见第八节）
- 索引：`zlib.crc32(token.encode("utf-8"))` —— 稳定 uint32，同一进程内外、写入与查询都能算出同一个 index
- 空串 / 纯标点 → 返回空列表（不报错、不伪造）

**同一份实现放三个位置且逐字节一致**（避免宿主脚本与容器 worker 漂移），sha256 均为
`4b1b686d0f00a350b100da4ad9a631d7c4d47fb34aee56738a7727523aec36a7`：

| 文件 | 用途 |
|---|---|
| `scripts/bm25_tf.py` | 宿主侧（`context_enhancer.py`、`bulk_wiki_ingest.py` 用） |
| `docker/worker/services/bm25_tf.py` | 容器侧（`sparse_embedding.py` 用） |

调用方 `docker/worker/services/sparse_embedding.py` 现在只是**转发**到 `bm25_tf`，保留原函数名与返回结构，`tasks/file_ingestion.py` 无需改动。

### 2.2 稠密向量 —— 仅远程

- `docker/worker/services/embedding.py`：`get_embedding(text)` → `POST {EMBEDDING_API_BASE}/embeddings`，请求体 `{"model", "input", "dimensions"}`；key 解析顺序 `EMBEDDING_API_KEY` → `OPENROUTER_API_KEY`（base 含 openrouter 时）→ `~/.env`；没有 key 就 `raise RuntimeError(...)`。
- `scripts/context_enhancer.py`：同一个口径，`embed_query()` 缺 key 抛 `RuntimeError`；`search_with_fallback()` 的**四级回退链完整保留**（这是原工程最有价值的部分，没有削弱）。
- 删除了 `scripts/context_enhancer.py` 里所有 `fastembed` 与 `subprocess` 调用（原实现会 `subprocess` 起子进程规避 ONNX 的 `ModuleNotFoundError`）。

### 2.3 生成 —— 仅远程 OpenAI 兼容

`docker/worker/services/llm.py` 整体重写：

- `chat_completion(messages, ...)` → `POST {LLM_API_BASE}/chat/completions`
- **保留了 `ollama_chat = chat_completion` 这个别名**（`tasks/*` 里有 `from services.llm import ollama_chat`，改别名会连带炸掉 worker）—— 这是兼容性锚点，**升级时不要删**
- key 回退顺序：`LLM_API_KEY` → `OPENROUTER_API_KEY`（base 含 openrouter）→ `DEEPSEEK_API_KEY`（把 `LLM_API_BASE` 设为 `https://api.deepseek.com/v1`）
- 删掉 `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_API_KEY` 全部读取

### 2.4 连带清理

| 文件 | 改动 |
|---|---|
| `docker/worker/Dockerfile` | 删掉 `build-essential` + `python3-dev`（只为 fastembed 编译而装）→ 镜像更小、构建更快 |
| `docker/worker/requirements.txt`、根 `requirements.txt` | 删 `fastembed`；**所有依赖补上上界**（`qdrant-client>=1.17.0,<1.19`、`arq>=0.28.0,<0.29`、`httpx>=0.27,<1`、`redis<6`、`pydantic<3` 等）—— 防止上游破坏性升级把整条栈静默打死 |
| `.env.example` | 删全部 `OLLAMA_*`，新增远端键，顶部注明"本栈不下载、不运行任何进程内模型" |

### 2.5 残留扫描结果

全仓 `grep -rniE "fastembed|ollama|llama.cpp|vllm|sentence-transformers"` 唯一命中是
`docker/worker/services/llm.py:91: ollama_chat = chat_completion` —— **这是刻意保留的兼容别名，不是本地模型依赖**。

---

## 三、怎么配（远程 API）

最小可用配置（写进 `~/.hermes/.env`，或 `docker/.env`）：

```bash
# ── 稠密向量（必填）──────────────────────────────
EMBEDDING_API_BASE=https://openrouter.ai/api/v1
EMBEDDING_API_KEY=sk-or-v1-xxxxxxxx        # 必填，缺失会直接报错
EMBEDDING_MODEL=qwen/qwen3-embedding-8b    # 或 openai/text-embedding-3-small 等
EMBEDDING_DIMS=4096                        # 必须与 Qdrant 集合维度一致！

# ── 生成（必填）──────────────────────────────────
LLM_API_BASE=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-xxxxxxxx
LLM_MODEL=deepseek/deepseek-chat

# ── 运维 ────────────────────────────────────────
ARQ_MAX_TRIES=3
HERMES_DLQ_PATH=/home/your-user/.hermes/wiki_ingest_failures.json
```

**省钱提示（贴合"白嫖免费又好用"）**：`EMBEDDING_API_BASE` / `LLM_API_BASE` 都是 OpenAI 兼容口径，可直接指向
DeepSeek（`https://api.deepseek.com/v1` + `DEEPSEEK_API_KEY`）、SiliconFlow（`https://api.siliconflow.cn/v1`）、
或任何自建 OpenAI 兼容网关。**稀疏检索已经完全不吃 API**，所以检索这一半的 token 成本是 0。

⚠️ **`EMBEDDING_DIMS` 必须和 Qdrant 集合维度一致**。换 embedding 模型后维度变了，需要重建集合，否则 upsert 直接失败。

---

## 四、自我进化：越用越聪明的五步闭环

这是本次改造的第二条主线。原工程 README 声称 facts 有 "automatic feedback loop"，
但仓库里**没有任何写入 facts 或更新 trust_score 的代码**，`layers/03-fact-store.md` 自己写着
"必须每轮手动调 `fact_feedback`，否则信任分永远停在 0.50"。本次把它接通了。

新增引擎：`icarus/evolution.py`（696 行，纯标准库，可单独 `--selftest`）。

| 步骤 | 位置 | 做什么 |
|---|---|---|
| 1. 写入 | `hooks.py:1535 _writeback_session_facts` → `evolution.record_fact` | 会话结束时抽取的事实镜像进 `facts` 表；`category="session"`、`tags="session:{sid}"`；**<20 字符跳过**（防垃圾）；内容 UNIQUE，重复自动折叠（幂等） |
| 2. 计数 | `hooks.py:1152 evolution.record_retrieval` | **仅当事实真正被拼进 prompt 时**才 +1（不是"查到了就算用"） |
| 3. 自动反馈 | `hooks.py:1184 _auto_feedback` | 助手回复与注入事实做关键词重合度判定（`_FEEDBACK_MIN_CHARS=12`）；**用户什么都不用做**，回复里用到了就自动奖励 |
| 4. 调信任分 | `evolution.apply_feedback` | 有用：`trust = min(1.0, trust + 0.08*w)` 且 `helpful_count += 1`；没用/误导：`trust = max(0.0, trust - 0.12*w)` |
| 5. 重排 + 衰减 | `hooks.py:657 evolution.rank_score` + `evolution.decay_unused` | 注入顺序按 `rank_score = trust × (1 + 0.15·log1p(retrieval_count)) × exp(−ln2·age_days/180)` 重排；长期不用（≥30 天）每轮 −0.02，**下限 0.05**（永不彻底遗忘） |

**可调参数（引擎的"性格"，都在 `icarus/evolution.py:41-51`）**

| 常量 | 值 | 含义 |
|---|---|---|
| `INITIAL_TRUST` | 0.55 | 新事实起始信任分（改前是恒定的 0.50） |
| `HELPFUL_DELTA` | 0.08 | 每次被证实有用涨多少 |
| `HARMFUL_DELTA` | 0.12 | 每次被证实误导掉多少（**惩罚 > 奖励**，保证"错一次比对一次更被记住"） |
| `TRUST_FLOOR` / `TRUST_CEIL` | 0.05 / 1.0 | 信任分硬边界 |
| `DECAY_RATE` / `DECAY_DAYS` | 0.02 / 30 | 衰减步长 / "长期不用"阈值 |
| `RANK_HALF_LIFE_DAYS` | 180 | 重排用的新鲜度半衰期 |
| `RANK_USE_GAIN` | 0.15 | 使用次数的对数增益权重 |

**怎么启用**：不用配置 —— Icarus 插件启用即生效。数据库在 `$HERMES_HOME/memory_store.db`（表 `facts` + `facts_fts`），
表结构与 `setup/setup_db.py` 的 `MEMORY_SCHEMA` 保持一致，新增列 `last_retrieved_at` 用幂等 `ALTER TABLE` 就地补上（老库能直接用）。

**怎么观测 / 干预**（新增两个工具，已在 `icarus/plugin.yaml` 登记，插件工具数 16 → 18）：

- `memory_evolution_stats()` —— 事实总数、平均信任分、被用得最多的前 N 条（含各自 `rank_score`）
- `fact_feedback(fact_id=..., helpful=True/False)` —— 显式纠偏（自动反馈判定不准时用它）

**调度**：事实衰减不是事件驱动，必须有定时器。新增 `icarus/evolution.py --decay 30`，并挂进 `setup.sh` 第 6 条 cron：

```
30 3 * * *   cd $REPO_ROOT && HERMES_HOME=$HERMES_HOME python3 icarus/evolution.py --decay 30   >> $HERMES_HOME/logs/fact_decay.log 2>&1
```

（选 03:30 是为了避开 03:00 的 weekly `decay_scanner` 与 monthly `semantic_dedup`，不打架。）

> 说明：Qdrant 向量库那侧的"衰减/归档"由既有 `scripts/decay_scanner.py` 负责；
> facts 表在 SQLite 里，两者不重叠。本次给 facts 侧补上了它一直缺的那条衰减链。

---

## 五、顺手修掉的阻塞缺陷（P0 / P1）

全部带位置，方便你复核。

### P0

| # | 问题 | 位置 | 修复 |
|---|---|---|---|
| 1 | **API key 明文打进日志** | `scripts/bulk_wiki_ingest.py:54` | 改为 `Embedding key configured: yes/no`，不再回显 |
| 2 | **切换模型时 `.env` 里 key 明文落盘 + 固定备份名反复覆盖** | `icarus/state.py:1131-1195` | 写入占位符 `__SET_FROM_ENV_TOGETHER_API_KEY__` 不再明文；备份改为**时间戳+随机串唯一名** `.env.backup.{stamp}.{hex}`，不再互相覆盖 |

### P1

| # | 问题 | 位置 | 修复 |
|---|---|---|---|
| 3 | 默认主机名是私有拓扑 `redis-maas` / `qdrant-maas`，换个环境直接连不上 | `docker/worker/main.py`、`services/local_qdrant.py`、`tasks/file_ingestion.py`、`tasks/reflection.py` | 默认值统一改 `redis` / `qdrant`（compose 服务名），可用环境变量覆盖 |
| 4 | worker **没有重试、没有死信队列**：任务失败即静默丢弃 | `docker/worker/main.py:49-146` | 新增 `ARQ_MAX_TRIES`（默认 3）+ DLQ 落盘（原子 `tmp` 写 + `os.replace`）；条目字段与 `scripts/dlq_manager.py` 的 `DLQEntry` **逐字段对齐**（`file,error,timestamp,failure_class,reported,retry_count,last_retry,error_hash`）。注：arq 0.28 没有 `on_job_failure` 原生钩子，故用 `_dlq_guard` 装饰器包装任务 |
| 5 | **多 profile 读写错库**：路径一半写死 `~/.hermes`、一半用别的 | `icarus/state.py`、`icarus/hooks.py`、`scripts/*` | 统一 `HERMES_HOME`（脚本侧再支持 `STATE_DB_PATH` / `MAA_ENV_PATH` / `WIKI_STATE_PATH` 覆盖） |
| 6 | **文件名拼接可路径穿越**：`entry_type` 原样进文件名 `{agent}-{entry_type}-{slug}.md` | `icarus/state.py:49-62`、`icarus/tools.py:41` | 白名单正则 `^[a-z0-9_-]{1,64}$`；非法值抛 `ValueError`（工具层先拦，给可读错误）；`state.py:503` 另有输出目录 containment 兜底 |
| 7 | **字段契约不一致**：worker 写 `file_path`，别的模块读 `source_file` | `docker/worker/tasks/reflection.py`、`scripts/test_ingestion.py` | 统一 `source_file` |
| 8 | **死表 / 失效逻辑**：`fabric_fts` 从未落数据；`e.get("_full")` 恒为空导致 token 预算过滤器 100% 失效；title 恒空 | `icarus/fabric-retrieve.py:31-65,107,552,609-616` | 丢弃并重建死表；`_reconstruct_full()` 从落库列重建全文，预算真正生效；`_entry_title()` 三级回退（title → summary → subject）保证不空 |
| 9 | **静默失效**：`except Exception: return []` 把"模块缺失"吞成"知识库是空的"，故障不可见 | `icarus/hooks.py:395-430 _load_context_enhancer` | 改为健壮导入（失败则补 `sys.path` 重试一次）+ `logger.error` 写明原因与补救办法；配置缺失场景一律显式 `RuntimeError` |
| 10 | **抽取 JSON 解析脆弱**：LLM 返回带解释文字或嵌套花括号就整段失败 | `docker/worker/tasks/reflection.py` | 新增括号配平 `_match_brace()`（识别嵌套与字符串内花括号），单次解析失败不再废掉整轮 |
| 11 | **`backfill_decay_metadata.py` 会批量写错 `created_at`**：容器路径 `stat()` 抛 OSError 时被当成"取不到时间"而回填当前时间 | `scripts/backfill_decay_metadata.py` | 改为**跳过并计数**，绝不伪造时间戳 |
| 12 | 调度缺口：元数据回填 / facts 衰减从来没被调度过 | `setup.sh` | Phase 7b 从 1 条扩到 **6 条幂等 cron**（用 `CRON_BEGIN/CRON_END` 标记 + `awk` 剥离旧块，重复跑不叠加；无 crontab 时优雅跳过） |

**顺带清理**：`docker/worker/tasks/reflection.py`、`scripts/test_ingestion.py` 里若干格式/路径问题；
`test_ingestion.py` 原来把测试文件写进宿主临时目录，而 worker 在容器里读 `/wiki` 卷 —— **永远 FileNotFoundError**，已改为写进 `WIKI_ROOT/raw/test/`。

---

## 六、调度全景（`setup.sh` Phase 7b 安装的 6 条）

| 频率 | 任务 | 作用 |
|---|---|---|
| 每小时 | `scripts/wiki_continuous_ingest.py` | SHA-256 diff，新/改文件入 Qdrant |
| 每 5 分钟 | `scripts/reflection_trigger.py` | worker 空闲时触发微反思 |
| 每周日 03:00 | `scripts/decay_scanner.py` | Qdrant 低价值老内容归档 |
| 每月 1 日 03:00 | `scripts/semantic_dedup.py` | 近重复向量合并（cosine > 0.92） |
| 每 2 小时 | `scripts/dlq_manager.py --report` | 死信队列分类与报表 |
| **每日 03:30** | **`icarus/evolution.py --decay 30`** | **facts 表信任分衰减（新增）** |

> 与 README 的口径差异说明：原版 README/setup 文档说这些要"手动 `hermes cron create`"。
> 本版 `setup.sh` 直接装好全部 6 条，**装完即闭环**，不再依赖手工步骤。
> Windows / git-bash 无 `crontab` 时会打印警告并跳过（栈本身照常工作），需自行用系统计划任务。

---

## 七、验证记录（真实执行输出）

环境：Windows + git-bash，Python 3.13.13。

```
$ python -m compileall -q .            →  COMPILE_OK
$ bash -n setup.sh                     →  BASH_SYNTAX_OK
$ python _test_sanitize.py             →  === ALL 24 TESTS PASS ===
$ python _test_collapse.py             →  === ALL COLLAPSE TESTS PASS ===
$ python scripts/collapse_eval.py      →  176 → 140 tokens (−20%)，剪掉 3 条弱/重复，attestation verify=True
$ python icarus/evolution.py --selftest→  === EVOLUTION SELFTEST PASSED (35 checks) ===
$ grep -rniE "fastembed|ollama|llama.cpp|vllm|sentence-transformers" .
                                        →  仅 llm.py:91 的兼容别名（刻意保留）
$ sha256sum scripts/bm25_tf.py docker/worker/services/bm25_tf.py
                                        →  两侧哈希一致 4b1b686d…ec36a7
```

**缺失配置的显式报错实测**：

```
embedding.get_embedding("hi") 无 key → RuntimeError: EMBEDDING_API_KEY (or OPENROUTER_API_KEY) is required for the OpenRouter embedding endpoint https://openrouter.ai/api/v1
llm.chat_completion(...)      无 key → RuntimeError: LLM_API_KEY is required. Fallbacks: OPENROUTER_API_KEY ... or DEEPSEEK_API_KEY.
context_enhancer.embed_query("hi") 无 key → RuntimeError: Dense embedding unavailable: EMBEDDING_API_KEY is not configured ...
context_enhancer.embed_query_sparse("hello world")  → {'indices': [907060870, 980881731], 'values': [1.6832..., 1.6832...]}  # 无网络、无模型
```

**cron 幂等块实测**：预置一个旧 managed block 后再跑，旧块被 `awk` 正确剥离、只剩新 6 条（不叠加）。

---

## 八、遗留风险与取舍（请务必读）

1. **中文分词退化（最重要的质量取舍）**
   为彻底摆脱本地模型，BM25 改用纯 Python 分词：英文数字整段保留，**每个汉字单独成 token**，且**没有词干化（stemming）**。
   后果：`running` / `runs` 在稀疏检索里是两个 token；中文短语会退化成"字级"匹配（召回变宽、精度略降）。
   **影响可控**：稀疏只是混合检索的一半，稠密向量仍由远程模型提供语义，Qdrant 服务端会施加 IDF 抑制高频字。
   **想恢复词干化**：装 `snowballstemmer`（纯 Python 包，仍**不是**本地模型），在 `bm25_tf.py` 分词后加一步 stem，两侧副本同步改并保持哈希一致即可。

2. **事实抽取的质量取决于 LLM 模型**
   自动反馈用"关键词重合"判定助手回复是否用到了某条事实 —— 这是**启发式**，会有误判（尤其改写、同义替换场景）。
   所以保留了 `fact_feedback` 显式纠偏入口。想让闭环更准，把 `ICARUS_EXTRACTION_MODEL` 换成更强的模型。

3. **`icarus/hooks.py` 无法端到端实跑**
   它依赖 Hermes Agent 运行时上下文（`ctx`、会话生命周期、fabric 目录等），本机没有这套运行时。
   已做的验证是：全量 `compileall` + 自带测试 + `evolution.py --selftest`（35 项）+ 逐函数静态核对。
   **首次真实部署时请重点观察** `on_session_start` / `pre_llm_call` / `post_llm_call` / `on_session_end` 四个钩子的日志。

4. **一键安装仍会改你的系统**
   `setup.sh` 会写 `~/.hermes/.env`、`~/.hermes/SOUL.md`，并执行 `crontab -` 与 `hermes gateway restart`。
   不想让它动 crontab，就先注释掉 Phase 7b（第 295-360 行附近）。

5. **`EMBEDDING_DIMS` 与集合维度必须一致**（见第三节警告）。

6. **上游同步建议**：本魔改与上游 main 已有较大分叉（43 个文件被改动）。若上游后续修 bug，建议**按文件挑拣合并**，不要直接 merge。
   特别注意三个兼容锚点不要被上游覆盖掉：
   - `docker/worker/services/llm.py` 的 `ollama_chat = chat_completion` 别名
   - `scripts/bm25_tf.py` 与 `docker/worker/services/bm25_tf.py` 必须**保持逐字节相同**
   - `icarus/evolution.py` 的表结构必须与 `setup/setup_db.py` 的 `MEMORY_SCHEMA` 兼容

---

## 九、运行环境说明（Windows + git-bash）

- **纯 Python 部分**（`scripts/*`、`icarus/*`、宿主侧 `bm25_tf`）：Windows + git-bash 下直接 `python xxx.py` 即可，本次已在
  Windows + git-bash + Python 3.13 实测通过（编译、自带测试、`evolution.py --selftest`、`bash -n setup.sh`）。
- **完整栈**（Redis 队列 + Qdrant + worker）需要 Docker：Windows 上装 Docker Desktop 后
  `cd docker && docker compose up -d`，与 Linux 用法一致。
- **`crontab` 在 git-bash 下不存在**，`setup.sh` 会打印警告并**优雅跳过**调度安装（不会失败、不会中断安装）；
  定时任务请改用 Windows 任务计划程序（`schtasks`），或干脆把 6 条命令写成一个 `.bat`/`.ps1` 由计划程序调用。
  命令模板见第六节表格，逐条替换 `$REPO_ROOT` / `$HERMES_HOME` 即可。
- **路径写法**：脚本内部用 `os.environ.get(...)` 拿路径，Windows 下建议显式设 `HERMES_HOME`
  （如 `C:/Users/你/.hermes`），避免 `~` 展开差异。
- **`.sh` 脚本行尾必须是 LF**：如果你在 Windows 编辑器里改过 `setup.sh`，记得保持 LF（CRLF 会让 git-bash 报 `bad interpreter`）。

---

## 十、怎么推到你自己的仓库

本环境无法直连 GitHub 的 git 协议（`git clone` 空回复），所以工程以本地目录形式交付给你，推送由你执行：

```bash
cd <解压后的 memory-os 目录>
git init
git add -A
git commit -m "fork: zero-local-model, remote-API-only, self-evolving memory OS"
git branch -M main
git remote add origin git@github.com:<你的用户名>/<你的仓库名>.git
git push -u origin main
```

保持 MIT 许可（原工程为 MIT）：建议在 `README.md` 里注明 fork 来源与上游地址，合规且体面。

---

## 十一、文件索引（本次新增/重点改动）

**新增**

- `icarus/evolution.py` — 自我进化引擎（696 行，含 `--selftest` / `--stats` / `--decay N`）
- `scripts/bm25_tf.py`、`docker/worker/services/bm25_tf.py` — 纯 Python BM25 tf 编码器（逐字节相同）
- `REFACTOR_NOTES.md` — 本文件

**重点改动**

- `docker/worker/services/llm.py` — 远程 OpenAI 兼容重写（保留 `ollama_chat` 别名）
- `docker/worker/services/embedding.py` — 仅远程 + 显式报错
- `docker/worker/services/sparse_embedding.py` — 转发到 `bm25_tf`
- `docker/worker/main.py` — 重试策略 + DLQ
- `scripts/context_enhancer.py` — 去 fastembed，四级回退保留
- `icarus/hooks.py` — 五步闭环接线
- `icarus/state.py` — 密钥安全 + 路径穿越防护 + 多 profile 统一
- `icarus/fabric-retrieve.py` — 死表 / 标题 / token 预算修复
- `icarus/tools.py` + `icarus/schemas.py` + `icarus/plugin.yaml` — 新增 `fact_feedback`、`memory_evolution_stats`（工具数 16 → 18）
- `setup.sh` — 6 条幂等 cron
- `.env.example`、`README.md`、`QUICKSTART.md`、`setup/install.md`、`infrastructure/architecture.md`、`layers/03-05-06`、`skills/memory-architecture.md` — 全量改口径为"远端 API + 纯 Python BM25 + 自我进化"
- 根 `requirements.txt`、`docker/worker/requirements.txt` — 删 fastembed + 全部补上界
- `docker/worker/Dockerfile` — 删 build-essential / python3-dev