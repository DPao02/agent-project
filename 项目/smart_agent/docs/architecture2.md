# 架构设计文档（当前 Python 实现）

> 更新日期：2026-11-15。本文以 `python-impl` 当前代码为依据，沿用原架构文档的组织方式，区分在线业务、离线实验与尚未实现的能力。原 `architecture.md` 保留不变。

## 1. 系统总体架构

### 1.1 架构概览

系统面向政策咨询、订单查询和工单办理，采用 **Supervisor 编排模式**：Supervisor 负责路由和汇总，知识检索、业务办理、合规审查分别由三个节点处理。

按角色理解，可以说“一个 Supervisor 加三个专业 Agent”。`intent_router.py` 虽保留在目录中，但当前图没有接入独立的意图路由节点。`synthesize` 是 Supervisor 的汇总方法，不是另一个独立 Agent。

```text
                         用户 / Swagger / Postman
                                    |
                                    | HTTP JSON
                                    v
                         FastAPI: POST /api/chat
                                    |
                   用户 + 会话隔离、会话锁、请求重放检查
                                    |
                         Redis 读取对话和任务快照
                                    |
                     重建 Graph State 与工作记忆缓存
                                    |
                                    v
                           Supervisor 路由节点
                          /                   \
                         v                     v
                  KnowledgeRAGAgent      TicketHandlerAgent
                  政策知识问答           订单查询 / 工单办理
                         |                     |
                         v                     v
                 百炼 Embedding          工具注册与调用层
                    + Milvus                   |
                         |                     v
                         |              BusinessService
                         |                     |
                         |                     v
                         |                   MySQL
                          \                   /
                           v                 v
                            合规审查节点
                                  |
                            Supervisor 汇总
                                  |
                    Redis 原子保存对话、任务和回执
                                  |
                              返回 JSON

   业务 REST 接口 ----------------------> 同一个 BusinessService
   /api/tools/call ---------------------> 同一个工具服务
   OpenTelemetry ----------------------> 装饰器覆盖的 Agent 方法
```

FastAPI 当前承担应用接口职责，尚未接入登录、角色权限、通用限流或 SSE 流式输出，不能将这些能力作为现有 API Gateway 功能描述。

### 1.2 编排流程（Supervisor Pattern）

```text
从会话快照恢复 messages、task、context
                  |
                  v
          supervisor_route
                  |
     +------------+-------------------+
     |            |                   |
     v            v                   | 直接进入合规审查
knowledge_rag  ticket_handler          |
     |            |                   |
     +------------+-------------------+
                  v
          compliance_check
                  |
              synthesize
                  |
                 END
```

三个路由分支最终都经过同一个 `compliance_check`，然后汇总，不会重复审查。代码入口为 [agents/supervisor.py](../python-impl/agents/supervisor.py)。

当前一次请求主要选择一个处理分支，再汇合审查和汇总；没有实现多个专业 Agent 并行竞争、循环规划或完整人工审批工作流。

### 1.3 资源初始化与共享方式

[api/main.py](../python-impl/api/main.py) 的 `lifespan()` 负责在开始接收请求前：

1. 初始化追踪配置。
2. 创建 MySQL `Database`，检查四张业务表，构造 `BusinessService`。
3. 创建 `LongTermMemory`，读取政策文档，复用或写入 Milvus 文档块。
4. 将业务服务和知识库注入工具服务，再把工具服务传入图。
5. 将同一个业务服务暴露给业务 REST 路由。

```python
service = BusinessService(database.sessions)
mcp_server = create_default_tools(MCPToolServer(), service, long_term_memory)
graph = create_supervisor_graph(
    working_memory=working_memory,
    short_term_memory=short_term_memory,
    long_term_memory=long_term_memory,
    tool_server=mcp_server,
)
app.state.business_service = service
```

这就是本项目的**依赖注入**：在入口创建好依赖，再通过参数交给使用者。工具和 HTTP 接口因此不会各自维护一份工单存储。关闭服务时释放 Milvus、MySQL 连接池和 Redis 客户端；关闭连接不等于删除数据。

## 2. 核心组件设计

### 2.1 Agent 与业务分层

| 组件                           | 当前职责                                         | 主要输入                       | 输出                           |
| ------------------------------ | ------------------------------------------------ | ------------------------------ | ------------------------------ |
| SupervisorNode                 | 大类意图识别、路由、结果汇总                     | 有界历史、工作上下文、当前任务 | 路由标签、最终回复             |
| KnowledgeRAGAgent              | 指代补全、政策检索、重排与回答                   | 最新问题和最近历史             | 文档回答、来源、改写查询       |
| TicketHandlerAgent             | 结构化提取本轮增量、补字段、确认与取消、调用工具 | 历史、TaskState、用户标识      | 新任务状态、工具结果、业务回执 |
| ComplianceCheckerAgent         | PII/敏感词规则及 LLM 回复审查                    | 各分支的回复文本               | 审查结果与违规说明             |
| MCPToolServer / business_tools | 工具发现、参数适配、调用记录                     | 工具名和参数                   | success/result/error           |
| BusinessService                | 业务校验、归属检查、事务、状态流转               | 经过校验的业务参数             | 数据库中的实际记录             |

分工原则：**模型理解需求，程序决定允许的操作，业务层访问数据库。** 模型不直接生成 SQL，也不决定当前用户身份。

### 2.2 State 与跨轮任务设计

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    session_id: str
    request_id: str
    task: dict | None
    rewritten_query: str
    intent: str
    sub_results: dict[str, Any]
    compliance_passed: bool
    final_response: str
    current_agent: str
    retry_count: int
```

`State` 是节点之间传递的共享数据结构，可以理解为本轮办理的“工作表”，不是一个数据库。`current_agent`、`retry_count` 等字段不代表已经完成完整节点状态追踪或业务自动重试。

[business/task_state.py](../python-impl/business/task_state.py) 定义跨多轮保存的任务：

| 字段                        | 含义                                                    |
| --------------------------- | ------------------------------------------------------- |
| `task_id`                 | 一次办理任务的稳定编号，后续确认消息改变时仍不变        |
| `action`                  | order_query、ticket_query、ticket_create、ticket_update |
| `fields`                  | 已收集的订单号、工单号、原因、类型、优先级、目标状态等  |
| `missing_fields`          | 程序根据操作类型计算的缺失字段                          |
| `stage`                   | collecting、awaiting_confirmation、completed、cancelled |
| `revision`                | 草稿修订版本，不是 MySQL 业务记录的 version             |
| `last_error` / `result` | 最近错误或已完成操作的简要回执                          |

```text
用户提出办理需求
      |
      v
collecting <----- 补充、修正、删除字段
      |
      | 必填字段齐全
      v
awaiting_confirmation ---- 修改字段 ----> 重新计算与展示摘要
      |
      | 明确“确认提交”
      v
工具执行 + MySQL 事务提交
      |
      v
completed

collecting / awaiting_confirmation -- 取消 --> cancelled
```

查询操作在编号齐全后可以直接执行；明确要求列出全部/最近记录时也可查询列表。创建和更新工单必须先预览，再确认。取消草稿不会撤销已经提交的工单，直接业务 REST 接口仍按结构化请求立即执行。

退款和理赔任务需要订单号及真实问题描述；未提及的增量字段不会覆盖已有值。材料不全、模型输出无法解析或为空时只追问，不默认创建工单。

### 2.3 分层记忆与业务数据

```text
工作记忆：Python dict
  保存路由/任务上下文缓存，按用户和会话隔离，进程退出即丢失
                         |
                         | 每轮从 Redis 快照重建
                         v
短期记忆：Redis SessionSnapshot
  messages + task + context + replies
  保存最近对话、未完成任务与最近请求回执，默认滑动过期 1800 秒

图检查点：LangGraph MemorySaver
  记录进程内图执行状态；不是 Redis 会话快照的替代品

长期知识库：Milvus
  共享政策原文、向量、文件与页码；不保存个人画像或业务工单

业务事实：MySQL
  orders / order_items / tickets / ticket_events
  以事务保存，业务查询按 user_id 过滤，不受会话 TTL 影响
```

| 层         | 代码入口                                  | 保存内容                               | 隔离与保留                                                              |
| ---------- | ----------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| 工作记忆   | `memory/working_memory.py`              | `_context` 与有限更新记录 `_store` | 用户+会话；默认30分钟，1000个缓存会话、每会话50条更新；读写时清理过期项 |
| 短期记忆   | `memory/short_term.py`                  | SessionSnapshot JSON                   | 用户+会话；默认20轮，即最多40条消息；最多10条请求回执                   |
| 图检查点   | `agents/supervisor.py`                  | 图状态与执行快照                       | thread_id 按用户+会话生成；MemorySaver 位于进程内                       |
| 长期知识库 | `memory/long_term.py`                   | 向量、原文、来源、元数据、模型标识     | 当前为公共 collection；无会话 TTL，不等于永久不删或已有备份             |
| 业务库     | `business/database.py` / `service.py` | 订单和工单事实                         | 同一用户可跨会话查询自己的业务记录                                      |

#### 用户和会话隔离

[memory/identity.py](../python-impl/memory/identity.py) 对 `[user_id, session_id]` 的 JSON 表示计算 SHA-256，生成统一 scope：

```text
工作缓存 key：scope
Redis 快照 key：smartcs:v2:session:<scope>
Redis 会话锁：smartcs:v2:lock:<scope>
Graph thread_id：scope
```

共享的是服务对象和连接，不是所有人的上下文。不同用户的同名会话、同一用户的不同会话都分开。哈希只生成稳定键，不是登录认证或数据加密；当前 user_id 仍由客户端传入。

#### Redis 的实际读写

```python
# 读取：JSON字符串恢复为经过校验的Python对象
value = await client.get(key)
snapshot = SessionSnapshot.model_validate_json(value)

# 写入：一条SET同时替换整个快照并设置过期时间
await client.set(key, snapshot.model_dump_json(), ex=ttl_seconds)
```

一个快照中一起保存消息、任务、上下文与回执，避免“聊天已保存但任务字段没保存”的半份会话状态。读取本身不刷新 TTL，成功保存新一轮才刷新。API 进入图之前会重建历史、追加当前问题，并用 `Overwrite(value=messages)` 替换旧检查点的消息。

Redis 当前运行配置包含 AOF、`appendfsync everysec`、命名数据卷和 `noeviction`。AOF 是 Redis 的写命令日志，数据卷让容器重建后仍可复用数据；二者不意味着故障时绝对零丢失，也不能防止 TTL 到期。

代码允许通过配置选择内存降级；本地项目此前设置了 `MEMORY_ALLOW_FALLBACK=false`。关闭降级时 Redis 不可用会明确失败。允许降级时，`memory_backend=memory` 表示状态仅留在该进程，不能跨进程或重启恢复。

#### 工作记忆与模型记忆的关系

工作记忆使用普通字典更新，不是把内容训练进模型。每轮 API 从 Redis 读回快照，清理并重建工作缓存，再把最近消息和任务数据放入模型请求。历史保存了不等于模型自动知道；实际发送哪些历史由 `memory/context.py` 的消息数量和字符上限控制。

### 2.4 MySQL 业务数据设计

MySQL 通过 SQLAlchemy ORM 和 PyMySQL 访问。ORM 是“Python对象与数据库表之间的映射工具”，最终执行的仍然是 SQL。

```text
orders (订单)
  order_id / user_id / status / amount / version / created_at / updated_at
       |
       +----< order_items (商品明细)
       |        order_id / product_name / quantity / unit_price
       |
       +----< tickets (工单，可选关联订单)
                 ticket_id / user_id / order_id / type / priority / status
                 summary / details / request_id / request_hash / version
                       |
                       +----< ticket_events (工单流转事件)
                                ticket_id / from_status / to_status / note / created_at
```

图中的 `type` 对应返回结果字段；ORM 表中实际字段名为 `ticket_type`。工单可以不关联订单，但退款/理赔对话任务会要求订单号。

数据库初始化由 `init_database.py` 显式执行；API 启动只检查表是否就绪。`create_all()` 不删除已有表，也不会自动迁移已有表结构。

#### 事务和连接池

```python
# 一次业务操作创建一个Session和事务
with self.sessions.begin() as session:
    session.add(ticket)
    session.flush()
    session.add(TicketEvent(...))
# 正常离开时提交；异常时回滚并关闭Session
return result
```

Engine 的连接池可以共享，每次操作的 Session 不跨请求或线程共享。连接池是可复用数据库连接的集合；Session 是一次业务读写使用的工作上下文。MySQL 中的订单及商品、工单及事件分别在同一事务内保存。金额使用 DECIMAL/Decimal，避免浮点误差。

这里的 SQLAlchemy `flush()` 是把待执行 SQL 发给数据库，不等于事务已提交；Milvus `flush()` 是另一套数据库的操作，不能把两者混为一谈。

#### 防重与并发更新

| 机制          | 代码实现                                                                    | 解决的问题                                    |
| ------------- | --------------------------------------------------------------------------- | --------------------------------------------- |
| HTTP 请求重放 | Redis 最近10条 request_id、消息指纹及响应                                   | 同一消息重试不再执行图；同编号不同消息返回409 |
| 工单创建幂等  | 稳定 task_id 作为业务 request_id；`UniqueConstraint(user_id, request_id)` | 确认请求重试或并发时避免重复工单              |
| 参数一致性    | request_hash                                                                | 同一业务请求号不能用于不同工单内容            |
| 用户归属      | 查询和更新包含 user_id；创建时检查关联订单归属                              | 防止从业务入口误关联其他用户订单              |
| 数据库行锁    | `select(...).with_for_update()`                                           | 同时修改同一记录时串行校验和写入              |
| 版本检查      | expected_version 与当前 version 比较                                        | 拒绝用户基于旧记录提交的修改                  |
| 流转规则      | ORDER_TRANSITIONS / TICKET_TRANSITIONS                                      | 拦截不允许的状态跳转                          |

“幂等”指同一操作重复提交，不产生重复的业务效果。本项目不是仅靠 Python 先查一下：数据库唯一约束负责兜底并发竞争，冲突后会重新查询原工单。

更新工单时，在展示确认摘要前读取数据库版本，确认时仍使用该版本。其他请求在此期间改动记录，会触发版本冲突，要求重新核实，而不是直接覆盖。

## 3. RAG 检索流程

### 3.1 知识库构建

```text
knowledge_base/*.pdf 或 *.txt
           |
           v
pypdf提取文本，保留文件名和PDF页码
           |
           v
递归分块：512字符，重叠128字符
           |
           v
稳定ID：hash(Embedding模型 + 来源 + 块正文)
           |
           +---- 相同ID已存在：复用
           |
           v
百炼 text-embedding-v4（默认1024维）
           |
           v
校验维度和有限数值，归一化向量
           |
           v
Milvus upsert：id + vector + content + source + metadata + embedding_model
```

[memory/corpus.py](../python-impl/memory/corpus.py) 由线上入库和离线评估共用。Milvus schema 检查维度，查询过滤模型标识，防止误用其他练习集合或混合不同模型的向量。

相同内容重复导入会跳过；文档修改后旧块不会自动删除，当前不是完整的知识版本同步系统。后续应实现旧块清理或按知识版本使用集合。

### 3.2 当前在线检索与回答

```text
最新问题 + 本会话最近历史
            |
            v
DeepSeek Query改写：补全省略和指代
            |
            v
百炼Embedding → Milvus COSINE召回Top5
            |
            v
DeepSeek LLM重排 → 目标Top3
            |
            v
检索原文、来源 + 最近历史 + 本轮问题
            |
            v
DeepSeek生成政策回答 → 合规审查 → 汇总
```

当前 LLM 重排只把候选块前200个字符送给模型，存在关键信息被截断、输出索引解析失败后回退等限制。来源以文件名和页码保存并进入提示词，引用由模型生成，尚未接入严格的引用正确性验证器。

Milvus 存的是公共政策知识，不包含单笔订单的实时状态，也没有自动保存用户偏好、私人聊天或工单。问“这个订单发货了吗”应调用订单工具；问“普通商品通常多久发货”才是政策检索。

### 3.3 离线实验与线上策略的区别

`rag/` 和 `rag_eval/` 新增 BM25、应用层 RRF、专用 `qwen3-rerank`、业务词典和改写消融能力。这些方法已在离线预评测中运行，但没有自动替换第3.2节的线上链路。

不能将“评测中验证了某方案”描述成“线上已默认启用全部优化”。详见第7节。

## 4. MCP 风格工具接口

### 4.1 实际调用关系

```text
TicketHandlerAgent
  结构化提取 BusinessDecision
          |
          | Python对象方法调用 call_tool(name, arguments)
          v
MCPToolServer
  注册 / 发现 / 调用 / 计时 / 错误结果
          |
          v
business_tools 参数适配与Pydantic校验
          |
          +---- BusinessService → MySQL
          +---- LongTermMemory → Milvus

GET  /api/tools       → list_tools()
POST /api/tools/call  → call_tool()
```

| 工具             | 当前实现                                             |
| ---------------- | ---------------------------------------------------- |
| order_query      | 查询当前用户订单；直接工具接口未给编号时列出最近记录 |
| ticket_create    | 在 MySQL 创建工单及初始事件，支持请求防重            |
| ticket_query     | 查询当前用户工单                                     |
| ticket_update    | 校验版本和状态后实际更新工单，写入事件               |
| knowledge_search | 调用现有 Milvus 知识检索                             |
| risk_check       | 沿用金额阈值规则，未对接外部风控系统                 |

工具注册的 `inputSchema` 描述参数结构，适配函数再使用 Pydantic 业务模型实际校验。业务节点的结构化输出用于提取操作参数；后续由程序分派工具调用，不是模型自主无限循环执行工具。

### 4.2 协议边界

`MCPToolServer` 提供 JSON-RPC 2.0 风格的 `tools/list`、`tools/call`、`ping` 处理方法，但当前聊天主要使用进程内方法调用，FastAPI 暴露的是自定义工具接口。

尚未接入官方 MCP SDK 的完整初始化握手、能力协商与标准传输服务，也没有把 JSON-RPC helper 自动等同于完整远程 MCP 服务。工具的认证标志不等于已完成权限校验。

## 5. 追踪与合规审查

### 5.1 当前 Span 覆盖

```text
supervisor.route_decision

knowledge_rag.process
  +-- rewrite_query
  +-- retrieve_documents
  +-- rerank_documents
  +-- generate_answer

ticket_handler.process
  +-- analyze_request
  +-- _call（工具执行）

compliance.process
  +-- full_check
        +-- rule_check
        +-- llm_check（需要时）

supervisor.synthesize_response
```

`trace_agent_call()` 为被装饰的方法创建 Span，当前记录 Agent 名、方法名、耗时、成功状态、异常和结果键名。支持控制台导出；OTLP 导出依赖与地址配置正确时可接入 Jaeger。

Span 可以理解为“一次操作的追踪记录”。当前未统一建立覆盖 HTTP 请求、所有 SQL 与 Redis 操作的完整根链路；未自动接通 Token 计量。`AgentMetrics` 有统计接口，但记录逻辑尚未自动接到所有调用，不能把 `/api/metrics` 中全部指标当作已完整采集的生产监控。

### 5.2 两阶段合规机制

```text
业务/知识回复文本
      |
      v
规则检查：手机号、身份证、银行卡、邮箱模式和敏感用语
      |
      +---- 高风险规则命中 → 拒绝展示完整内容
      |
      v
LLM语义审查：不当承诺、表述等
      |
      v
通过 → 正常汇总
不通过 → 限制展示；如已写入业务，按真实回执说明结果
```

合规节点审查的是回复文本，不是退款审批系统，也不是写入前的统一授权网关。它不会撤销已经提交的 MySQL 事务。当前 LLM 审查 JSON 解析失败时存在默认放行逻辑；直接合规分支没有回复内容时也可能跳过有效审查，这些是现阶段待改进点。

没有真正的人工转接执行器，因此拒绝回复不能声称已转人工。系统已依据实际业务回执替代原先凭空生成工单号的表述。

## 6. 技术选型与实现取舍

| 技术                         | 在本项目中的职责                           | 选用原因与边界                                 |
| ---------------------------- | ------------------------------------------ | ---------------------------------------------- |
| Python / LangChain           | 模型消息、客户端、结构化输出               | 便于连接模型与工程组件，不代表自动拥有业务逻辑 |
| LangGraph                    | 显式图状态、分支路由和节点编排             | 使流程可追踪、可测试；当前图不是自主规划循环   |
| FastAPI                      | JSON接口、输入校验、生命周期               | 管理共享依赖；认证和限流尚未补齐               |
| Redis                        | 会话快照、请求回执和会话互斥租约           | 跨API实例共享短期状态；单实例和TTL仍是限制     |
| Milvus                       | 政策语义检索                               | 按语义找文本；不承担订单金额和工单事实管理     |
| SQLAlchemy / PyMySQL / MySQL | 结构化业务记录及事务                       | 支持关联、唯一约束和更新校验                   |
| OpenTelemetry                | 方法级Span                                 | 用于定位慢调用与异常，当前并非完整监控闭环     |
| Docker                       | 运行Redis、既有Milvus依赖，提供API构建文件 | 当前主要验证的是本机API配合容器依赖            |

不使用未经本项目实测的“某数据库固定几毫秒”“支持百亿数据”等数字来说明当前系统能力。选型重点是数据语义、访问方式和失败恢复需求。

### 6.1 async 与同步数据库客户端

LLM 和 Redis 调用使用异步接口；同步 Milvus SDK、SQLAlchemy/PyMySQL 操作通过 `asyncio.to_thread()` 放到线程执行。

`await` 让当前协程等待结果时把执行机会交给其他请求，不是让当前办理跳过尚未完成的数据库操作。`to_thread` 是把同步阻塞函数交给线程，并未把 PyMySQL 变成原生异步驱动。

## 7. 离线检索评估

### 7.1 数据与口径

实现入口为 [rag_eval/__main__.py](../python-impl/rag_eval/__main__.py)。保存原始文件哈希、语料指纹、分块ID、词典指纹与逐条证据，标签绑定语料版本。

当前样本为100条**待人工复核草案**：90条可回答、10条无答案/需要其他业务工具的问题。按意图组切分80条开发样本和20条测试样本，同义问法不跨集合。正式模式拒绝未复核标签，预评测必须显式使用 `--allow-draft`。

| 指标     | 当前计算方式                                             |
| -------- | -------------------------------------------------------- |
| Recall@5 | 前5个结果覆盖的相关块数量 / 标注相关块总数，再按问题平均 |
| MRR@10   | 前10个结果中首个相关块排名的倒数，再按问题平均           |
| Hit@5    | 前5个结果是否至少命中一个相关块                          |
| P95      | 独立无应用结果缓存实验中，95%样本不超过的耗时位置        |

无答案样本不参与 Recall/MRR 分母；本轮未测答案正确率、忠实度或拒答准确率。质量测试可复用模型结果缓存，不能把缓存命中耗时作为线上P95。

### 7.2 已完成的预评测

| 方案               | 全量90条可回答草案 Recall@5 | MRR@10 |
| ------------------ | --------------------------: | -----: |
| BM25               |                      94.44% | 0.8029 |
| 向量检索           |                      92.41% | 0.7728 |
| RRF                |                      96.11% | 0.8661 |
| RRF + qwen3-rerank |                      99.63% | 0.9815 |
| 再加入词典和改写   |                      99.63% | 0.9889 |

这些是文档衍生查询、18个块的小语料预评测，不是100条人工标注结果或真实业务准确率。完整9组实验、开发/测试划分、区间和失败记录位于 `evaluation_artifacts`。

当前在线策略还单独评估了召回5条、最终输出3条；与输出10条的其他方案并非完全相同预算，不将全部差额归因于某一个优化。

延迟测试采用10个意图、3次重复、串行执行并排除预热。实际18候选下，检索链路P95约3.04秒，专用Rerank阶段P95约1.03秒，不含生成、路由与合规。当前语料不足50块，Top50和Top20都最多取得18个候选，不能宣称候选裁剪带来提速。

## 8. 本地运行、接口与文件导航

### 8.1 当前运行形态

```text
Windows本机 uv / Uvicorn：FastAPI与Graph
Windows本机 MySQL服务：业务库
Docker Redis：会话快照与租约
既有Docker Milvus / etcd / MinIO：向量数据库依赖
云端DeepSeek：路由、业务提取、在线改写/重排/生成/审查
云端百炼：Embedding；离线实验另用qwen3-rerank
Jaeger：可选追踪后端
```

```powershell
# 仅恢复本项目Redis服务
cd "D:\Python-Project\项目\smart_agent"
docker compose -p smart-cs up -d redis

# 初次准备MySQL表后运行API
cd .\python-impl
uv run --no-sync python init_database.py
uv run --no-sync python -m api.main
```

MySQL 账号、数据库及 `.env` 应先按 [README.mysql.md](../python-impl/README.mysql.md) 配好。Milvus继续复用已配置的实例，不必重新创建一套数据库。

项目提供 Dockerfile，但 `python-agent` 的 Compose 环境仍含旧 OpenAI 变量；容器内地址和 `.env` 覆盖策略也需单独适配，不能把“存在Dockerfile”描述为已完成全栈一键容器部署。

### 8.2 主要接口

| 接口                                                                    | 作用                           |
| ----------------------------------------------------------------------- | ------------------------------ |
| POST /api/chat                                                          | 多轮咨询和业务办理             |
| GET /api/history/{session_id}?user_id=...                               | 读取对应用户的历史             |
| GET /api/sessions/{session_id}?user_id=...                              | 读取任务阶段和记忆后端         |
| DELETE /api/sessions/{session_id}?user_id=...                           | 清空会话与草稿，不删除业务记录 |
| GET /api/tools、POST /api/tools/call                                    | 工具发现和直接调用             |
| POST /api/orders、GET /api/orders、GET/PATCH /api/orders/{order_id}     | 订单录入、查询与状态更新       |
| POST /api/tickets、GET /api/tickets、GET/PATCH /api/tickets/{ticket_id} | 工单创建、查询与更新           |
| GET /api/tickets/{ticket_id}/events?user_id=...                         | 工单流转记录                   |
| GET /api/metrics、GET /health                                           | 当前统计接口与健康入口         |

聊天每轮保持相同 user_id/session_id，使用新的 request_id；重试同一次请求才复用request_id。不提供session_id会生成新会话，无法自动知道要继续哪个草稿。

### 8.3 阅读代码顺序

```text
api/main.py                   请求入口、快照恢复、共享依赖、快照保存
agents/supervisor.py           图结构与路由
business/task_state.py        任务模型、必填字段规则
agents/ticket_handler.py      补充、修改、确认、取消及工具分派
mcp/business_tools.py         工具参数到业务参数的适配
business/service.py           业务查询和事务
business/database.py          连接池与表结构
memory/identity.py            联合命名空间
memory/working_memory.py      进程内缓存
memory/short_term.py          Redis快照与会话锁
memory/corpus.py              文档分块和稳定ID
memory/long_term.py            Milvus写入、去重和检索
agents/knowledge_rag.py       在线RAG流程
tracing/otel_config.py        方法级Span与统计定义
rag/、rag_eval/               离线检索方案、评测、复核与请求预算
```

## 9. 一致性、恢复与现阶段边界

### 9.1 为什么两种锁都需要

Redis租约约束的是同一用户/会话同时发请求；MySQL行锁约束的是同一条业务记录。不同会话仍可能修改同一张工单，因此会话锁不能替代数据库版本检查。

当前图执行最长120秒，会话租约180秒。内存降级仅使用进程内锁，不提供跨进程互斥保证；单Redis部署也不能宣称具备集群高可用能力。

### 9.2 恢复与故障结果

| 情况                      | 当前行为                                                                     |
| ------------------------- | ---------------------------------------------------------------------------- |
| API进程重启               | 工作缓存和MemorySaver丢失；可从仍有效的Redis快照恢复最近完整保存的对话和草稿 |
| Redis会话TTL到期          | 该快照、草稿和回执不再可读；MySQL业务记录及Milvus政策不随之删除              |
| Redis服务重启             | 是否恢复取决于数据卷、AOF和TTL；不是绝对零丢失                               |
| Redis不可用               | 依据降级配置返回错误或显式memory后端；后者不能跨重启恢复                     |
| MySQL写入事务内失败       | 回滚同一事务内的记录；工具返回失败，直接业务接口按异常类型响应               |
| MySQL提交后模型/Redis失败 | 已提交记录不会自动撤销；稳定任务号防重复创建，更新需查询并核对版本           |
| 清空会话                  | 删除Redis快照和工作缓存；下一轮显式空task和Overwrite防止旧检查点被重新使用   |

恢复的是会话和任务快照，不是任意节点执行到一半的持久化续跑。

### 9.3 不应宣称已经完成的能力

- 登录、角色权限、多租户授权与通用请求限流。
- 支付退款、库存扣减、真实快递发货、人工审批后台或真实人工转接。
- 用户个人长期画像、私有知识库以及所有聊天自动沉淀至Milvus。
- Redis和MySQL之间的分布式事务或端到端严格“恰好执行一次”。
- 完整官方MCP协议服务、全HTTP/SQL/Redis根追踪及自动Token统计。
- 已将所有离线优化默认上线，或已完成100条人工标注的正式评估。

## 10. 面试时如何组织讲解

可以从一条真实业务链开始，而不是先罗列数据库名字：

> 用户说“我要退款”后，API先从Redis恢复会话和未完成任务。Supervisor把它路由给业务节点，模型只提取本轮新增字段，程序合并订单号、原因等信息并计算还缺什么。字段齐全后先给摘要，用户确认才调用工单工具。工具和REST接口共用业务服务，由MySQL事务同时保存工单和流转记录。处理结束，再把最新对话和任务状态保存回Redis。

再解释四种存储为何分开：

> 内存字典是进程内工作缓存，Redis是跨请求和API重启的短期会话快照，Milvus用于按语义检索公共政策，MySQL保存可精确查询和事务更新的业务事实。不是每个用户都新建四套存储，而是会话数据按用户加会话分区，业务数据按用户归属过滤，公共政策共享。

常见追问可以结合具体代码回答：

| 追问                                       | 回答要点                                                                               |
| ------------------------------------------ | -------------------------------------------------------------------------------------- |
| 保存历史为什么还会“失忆”？               | 历史必须重新进入模型输入；本项目还显式保存task字段，避免只依赖消息回溯                 |
| 为什么草稿不直接建工单？                   | 信息可能未齐、仍会修正或取消；确认后才形成业务事实                                     |
| request_id、task_id、ticket_id有什么不同？ | 单次HTTP请求、跨多轮任务、已保存业务工单三个层次                                       |
| 重启会丢什么？                             | 内存和MemorySaver会丢；Redis未过期快照恢复任务，MySQL/Milvus分别持久化                 |
| 为什么不把订单放Milvus？                   | 订单需要精确编号、金额、关联和更新事务，语义相近不能替代业务事实                       |
| 为什么重复确认不会多建？                   | Redis请求重放加上稳定task_id和MySQL唯一约束，说明并发竞态也由数据库兜底                |
| Redis和MySQL谁是真相？                     | 会话草稿以Redis快照为准，已提交业务以MySQL为准；两者用途不同，不是互相复制同一份业务表 |
| 合规失败是不是就没写入？                   | 不是；当前审查在工具执行后，必须根据真实回执说明已经提交的结果                         |

具体接口示例参见 [README.memory.md](../python-impl/README.memory.md) 和 [README.mysql.md](../python-impl/README.mysql.md)。
