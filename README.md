# Memory OS Magic — Hermes Agent 自我进化记忆操作系统

> **Forked from [ClaudioDrews/memory-os](https://github.com/ClaudioDrews/memory-os) (MIT License)**  
> **面向实战的深度魔改版：零本地模型 · 纯远程 API · 闭环自我进化 · 国产端点避坑加固**

![Memory OS Banner](assets/banner.jpg)

> **让你的 Hermes Agent 彻底告别“金鱼脑”与“间歇性失忆”。**  
> 七层立体记忆体系 · 极致 Token 注入控制 · 事实信任度自我进化 · 跨会话决策织网 · 权威真理层级（Layer 7）约束。

Memory OS Magic 为 Hermes Agent 赋予了真正的长期心智。它能沉淀你的项目架构、技术栈、关键决策、踩坑教训与偏好，并在最恰当的时机毫秒级精准注入。更重要的是：**通过五步进化闭环，它会随着日常对话越用越聪明、去粗取精、自动淘汰陈旧经验！**

---

## ⚡ 本次魔改核心特质（Magic 版特有）

| 核心维度 | 原版 Memory OS | 🪄 Magic 魔改版 |
|---|---|---|
| **模型部署依赖** | 强依赖本地 Ollama 推理与 C++ `fastembed` 库 | **零本地模型（Zero Local Model）**：纯远程 OpenAI 兼容 API，彻底不跑本地推理 |
| **稀疏向量检索** | fastembed / ONNX 进程内加载 BM25 模型 | **纯 Python `bm25_tf.py` 算法**：0 模型依赖、0 网络开销、Qdrant 服务端计算 IDF |
| **事实自我进化** | `trust_score` 为死代码（只读不写，无反馈循环） | **五步真实闭环**：写入 → 计数 → 命中自动奖励 → 纠偏惩罚 → 衰减重排 |
| **国产端点避坑** | 无条件发送 `dimensions`（SiliconFlow 等报 20015） | **智能适配与自愈熔断**：非动态模型不送参数，遇 20015 自动脱敏秒级重试 |
| **中文生态适配** | 提示词为英文，违背词库为葡语/西语 | **原生中文会话提炼** + 中英文禁忌规则防御词库（禁止、避坑、踩坑等） |
| **API 统一规范** | 环境变量散落在各自逻辑，兼容繁琐 | **统一标准**：一套 `LLM_API_BASE` / `EMBEDDING_API_BASE` 通吃全栈 |

---

## 🧠 核心架构：七层记忆千层饼

```
┌──────────────────────────────────────────────────────────────────┐
│  第 1 层 · 工作区全局文件 (WORKSPACE)                            │
│  MEMORY.md · USER.md · CREATIVE.md                               │
│  → 每一轮对话硬塞进 System Prompt（核心身份与全局设定）           │
├──────────────────────────────────────────────────────────────────┤
│  第 2 层 · 会话时光机 (SESSIONS)                                  │
│  state.db (SQLite + FTS5 全文索引)                               │
│  → 毫秒级检索过去所有历史对话的上下文与执行脉络                  │
├──────────────────────────────────────────────────────────────────┤
│  第 3 层 · 结构化事实与进化池 (STRUCTURED FACTS)                  │
│  memory_store.db (SQLite + 信任度评分 + 实体图谱)                 │
│  → 事实自动沉淀，经过实战检验后动态调整权重与生命周期            │
├──────────────────────────────────────────────────────────────────┤
│  第 4 层 · 跨会话决策织网 (FABRIC / 魔改版 Icarus 插件)           │
│  Markdown + YAML Frontmatter + 4大 Hook 拦截器                    │
│  → 会话结束自动提炼决策、踩坑教训与解决方案                      │
├──────────────────────────────────────────────────────────────────┤
│  第 5 层 · 混合向量数据库 (VECTOR DB / Qdrant)                    │
│  Qdrant 1.17+ (稠密向量 + 纯 Python BM25 稀疏向量)               │
│  → 四级降级链：混合检索 → 稠密向量 → 纯词法检索 → SQLite 回退     │
├──────────────────────────────────────────────────────────────────┤
│  第 6 层 · 自动提炼数字花园 (LLM WIKI)                            │
│  概念库 concepts/ · 实体库 entities/ · 对比库 comparisons/       │
│  → 定时巡检后台自动归纳沉淀，全量向量化持续灌库                  │
├──────────────────────────────────────────────────────────────────┤
│  ⚡ 第 7 层 · 绝对真理权威层 (GROUND TRUTH HIERARCHY)             │
│  SOUL.md · rulebook.md                                           │
│  → 灵魂之笔：强制 Agent 必须将注入记忆作为已知真理，严禁重复验证 │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 为什么说 Layer 7（真理层级）是灵魂所在？

很多记忆插件最尴尬的痛点是：**记忆已经完美注入到了前置提示词里，但模型转头就说“好的，我来用 search_files 查一下…”**。

模型不是瞎，而是不知道这些注入信息的**权限等级**！在原生系统里，官方文档和实时命令输出权威度极高，未标明的注入记忆被当成“无凭无据的建议”。

**Magic 版继承并强化了 Layer 7 权威法则：**
1. **终端实时输出**（最高权威，反映当前系统真实状态）
2. **系统注入记忆 `[facts] / [fabric] / [qdrant]`**（**绝对事实，直接作为已知前提，禁止二次查库浪费 Token！**）
3. **外部官方文档**
4. **大模型预训练通识知识**

---

## 🔄 记忆自我进化闭环：越用越聪明的底层原理

Memory OS Magic 杜绝了“记忆越攒越多变成垃圾堆”的恶疾。存储在 `memory_store.db` 的每条事实都自带生命周期评分：

1. **初始建档（Write）**：新提取的事实初始信任分为 **0.55**。
2. **检索命中（Retrieve）**：每次事实被检索并拼接进 Prompt 时，`retrieval_count` 自动累加。
3. **效果反馈（Feedback）**：当 Agent 的最终回答引用了该条事实的关键词，系统在 `post_llm_call` 中自动触发正向反馈（也可通过 `fact_feedback` 工具显式评价）。
4. **信任调权（Trust Tuning）**：有用事实每次增加 **+0.08** 分（上限 1.0）；误导事实惩罚扣除 **-0.12** 分。
5. **重排与半衰衰减（Re-rank & Decay）**：
   - 排序公式：`rank_score = trust × (1 + 0.15 · log1p(retrieval_count)) × exp(−ln2 · age_days / 180)`
   - 高频、高信度、近期的优质经验自动置顶；
   - 30 天未被调用的僵尸记忆每日自动扣除 **0.02**，触底至 0.05 保底不丢失。

---

## 🛠️ 快速上手与配置指南

### 1. 环境准备
- **Docker & Docker Compose**（用于运行 Qdrant 向量库与 Redis 队列）
- **Python 3.11+**
- **Hermes Agent** 运行环境

### 2. 克隆与配置
```bash
git clone https://github.com/805889095-eng/memory-os-magic.git ~/memory-os
cd ~/memory-os
cp .env.example .env
```

在 `.env` 中填入你的远程 API（支持任意兼容 OpenAI 规范的端点，例如 SiliconFlow、DeepSeek、OpenRouter 等）：

```bash
# ── 向量模型 (纯远程) ──────────────────────────
EMBEDDING_API_BASE=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_API_KEY=sk-your-siliconflow-key
EMBEDDING_DIMS=1024
# 智能避免 20015 错误：auto (默认) / false / true
EMBEDDING_PASS_DIMENSIONS=auto

# ── 对话与提炼模型 (纯远程) ───────────────────
LLM_API_BASE=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-your-deepseek-key

# ── 基础服务密码 ──────────────────────────────
REDIS_PASSWORD=your_secure_password
```

### 3. 一键初始化服务
```bash
# 启动 Docker 组件 (Qdrant + Redis + Worker)
docker compose -f docker/docker-compose.yml up -d

# 初始化 SQLite 本地数据库 (state.db + memory_store.db)
python3 setup/setup_db.py

# 软链安装插件至 Hermes（本地路径安装，更新 Hermes 绝不被覆盖）
ln -sfn $(pwd)/icarus ~/.hermes/plugins/icarus
```

### 4. 运行全链路健康自检
```bash
# 1. 验证五步自我进化逻辑 (35 项断言)
python3 icarus/evolution.py --selftest

# 2. 验证防撑爆折叠与清洗算法 (24 项断言)
python3 _test_collapse.py
python3 _test_sanitize.py

# 3. 运行基础组件冒烟测试
bash setup/smoke_test.sh --quick
```

---

## 📋 常见命令与会话工具

在 Hermes 对话中，你可以直接调用以下原生增强工具：

| 工具名 | 功能说明 | 示例 |
|---|---|---|
| `fabric_recall` | 跨会话沉淀经验与决策召回 | `fabric_recall(query="Docker部署规范")` |
| `fact_store` | 结构化事实库探查与实体检索 | `fact_store(action="probe", entity="Qdrant")` |
| `fact_feedback` | 对某条记忆人工调分打分 | `fact_feedback(fact_id=12, helpful=True)` |
| `memory_evolution_stats` | 查看当前记忆库总体健康度与置顶高频词 | `memory_evolution_stats(top_n=5)` |

---

## 📄 开源许可证

本项目基于 [MIT License](LICENSE) 开源。  
上游项目原作者：[Claudio Drews](https://github.com/ClaudioDrews/memory-os)。  
魔改与工程增强由团队深度定制研发，欢迎提交 Issue 与 PR 共同完善！
