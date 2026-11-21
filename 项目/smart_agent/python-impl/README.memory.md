# 多轮任务与分层记忆：当前实现与验证方法

**1. 检查结论与修改内容**

原项目并未完整实现跨多轮业务办理：工单节点能读取图中最近消息，但没有显式任务；RAG主要读取最后一句；API没有把Redis历史重新传入图。工作记忆和Redis历史只按session_id分组，图检查点才带user_id。虚拟环境还缺Redis客户端，短期记忆实际上会退回进程内存。

现在已直接修改原代码，新增显式任务模型、用户/会话联合命名空间、Redis会话快照与恢复、写入前确认、字段增量修正和取消。Redis客户端已通过uv安装并记录依赖，已启动项目compose中的redis服务。

**2. 一次请求的数据流**

```text
user_id + session_id → 联合命名空间
  → 取得会话锁
  → 从Redis加载有界消息、当前任务、最近请求回执
  → 重建图状态，覆盖旧检查点的消息，恢复工作上下文
  → Supervisor根据对话和任务路由
      → 业务节点：提取本轮增量 → 合并字段 → 计算缺失项 → 追问/预览/确认执行
      → RAG节点：结合历史解析“这个”“那运费呢”等省略与指代
  → 合规检查 → 汇总回复
  → 一次保存消息、任务和回执，刷新Redis TTL
  → 释放会话锁
```

Redis是可跨API重启恢复的会话快照来源，工作记忆是它的进程内缓存。LangGraph仍使用MemorySaver记录执行状态，但不把旧检查点当成第二份业务真相。每次通过Overwrite替换messages，并明确写入task，包括空值，避免重复累积历史、清空会话后旧任务复活。

**3. 任务保存哪些信息**

模型在 `business/task_state.py`，主要字段如下：

```json
{
  "task_id": "task-唯一编号",
  "action": "ticket_create",
  "stage": "collecting",
  "fields": {
    "ticket_type": "refund",
    "order_id": "ORD-DEMO-1001",
    "details": null
  },
  "missing_fields": ["details"],
  "revision": 2,
  "last_error": null,
  "result": null
}
```

这里是节选，实际响应还包含其他可选字段。

阶段为：collecting收集中、awaiting_confirmation待确认、completed已完成、cancelled已取消。

模型只提取最新一轮新增或修改的字段，未提及字段返回null，不覆盖旧值。明确删除字段时使用clear_fields，程序重新计算缺失项。退款、理赔需要订单号和真实问题描述；查询单个订单/工单缺编号会追问，不再擅自列出全部记录。只有明确请求列出最近/全部记录时，才使用list_all。

写入工单前必须返回摘要，并等待明确的“确认提交”等确认语。确认和常见取消语由程序直接识别，不让模型仅因字段齐全就执行写入。修改已存在工单时，预览阶段从数据库读取版本，确认时检查该版本，避免覆盖预览之后发生的变更。

新建工单的数据库幂等请求号使用稳定task_id，不是最后一条确认消息的新request_id。网络重试不会为同一任务重复创建工单。会话内还缓存最近10条请求回执，相同request_id和消息会重放结果，不重复追加对话；相同request_id用于不同消息则返回409。

取消只取消未提交任务，并清掉已收集字段；不会撤销已经创建的数据库工单。已提交工单需要单独查询或办理关闭。切换任务可以先取消，或明确说“重新开始”；新任务不会自动继承旧草稿字段。

**4. 用户与会话如何隔离**

对象可以共享，数据通过命名空间隔离，不需要为每个用户新建一套数据库或Redis连接。

| 层 | 隔离方式 | 生命周期与用途 |
|---|---|---|
| 工作记忆 WorkingMemory | 对[user_id, session_id]的JSON计算SHA-256作为key | 进程内缓存路由和任务上下文，有TTL与容量限制；重启后从会话快照重建 |
| 短期记忆 Redis | smartcs:v2:session:同一hash | 原子保存最近20轮对话、任务和回执；默认1800秒滑动过期 |
| 图检查点 MemorySaver | thread_id使用同一hash | 进程内图状态；API重启后不依赖它恢复业务，而从Redis恢复 |
| 长期知识库 Milvus | 当前所有用户共用配置的collection | 退款、物流等公共政策，跨用户/会话共享；没有实现个人偏好或私有文档记忆 |
| 业务记录 MySQL | 查询和更新包含user_id条件 | 同一用户可跨会话查询其订单和工单，不依赖会话TTL |

使用JSON数组再hash，而不是简单user:session拼接，避免(a:b,c)与(a,b:c)这样的分隔符碰撞。不同用户即使使用相同session_id，也不会读取彼此的工作缓存、Redis消息或任务。相同用户不同session_id也相互独立。

Milvus目前是企业公共知识库，故意不按每个用户复制一份。用户偏好、跨会话个人事实、用户私有知识库都尚未实现，也不会把聊天或订单自动写入共享Milvus。

这里的隔离以传入的user_id为标识，不等于身份认证。当前仍是本地项目，登录和角色授权需单独接入，后续应从认证凭据取得user_id。

**5. Redis与降级策略**

当前项目.env增加：

```dotenv
SESSION_TTL_SECONDS=1800
MEMORY_ALLOW_FALLBACK=false
```

false表示Redis不可用时明确失败，避免悄悄丢掉待办理任务。若你自行改成true，初始连接不可用时允许降级内存，响应memory_backend会变成memory；此时不能跨API重启恢复，也不支持多进程共享。降级中已开始的会话会继续使用原内存快照，避免Redis恢复时突然切到空状态；新会话可以重新连接Redis。

Redis会话key是v2命名空间。不会把旧的、只按session_id存储且归属不明的消息自动归到某个用户，以免误读他人历史。

本次已启动smart-cs-redis-1，使用Docker命名卷smart-cs_redis_data；开启AOF，everysec刷盘，内存策略noeviction，避免会话锁被LRU随机淘汰。API重启后可恢复未过期快照；Redis重启的持久性仍受AOF刷盘、数据卷和TTL限制，不是永不过期的永久档案。

同一用户/会话采用Redis租约防止并发覆盖，重复并发请求返回409。图执行限制120秒，锁租约180秒。使用内存后端时采用进程内锁，不能宣称提供跨进程并发保证。

**6. 如何运行**

Redis已经启动，以后需要恢复服务时，在PowerShell执行：

```powershell
cd "D:\Python-Project\项目\smart_agent"
docker compose -p smart-cs up -d redis
cd .\python-impl
uv run --no-sync python -m api.main
```

如果API正在运行，先Ctrl+C完整退出，再启动，确保加载新的依赖与配置。数据库和Milvus仍按原配置运行，本次不需要重新建表或删除数据。

**7. Swagger/Postman多轮验收**

打开http://127.0.0.1:8000/docs，POST /api/chat。每轮保持相同user_id和session_id，使用不同request_id；只有重试同一次请求才复用request_id。

第一轮：

```json
{
  "user_id": "u001",
  "session_id": "refund-demo-01",
  "request_id": "refund-turn-001",
  "message": "我要申请退款"
}
```

预期task.stage=collecting，missing_fields包含订单号和原因。之后保持同一会话，依次修改message和request_id：

| 轮次 | message | 预期 |
|---|---|---|
| 2 | 订单号是 ORD-DEMO-1001 | 保留退款类型，仍询问原因 |
| 3 | 原因是商品不合适 | 字段齐全，awaiting_confirmation，尚未创建工单 |
| 4 | 原因改成尺寸不合适 | 更新details，保留订单号，重新展示摘要 |
| 5 | 确认提交 | MySQL创建一张工单，task.stage=completed |

订单号应替换为实际存在且属于该用户的订单。不要对不存在的订单期待创建成功。

另开一个session测试“我要申请退款 → 补充订单号 → 取消”，预期cancelled且数据库不新增工单；随后重新申请不能继承旧订单号。也可以在等待补充时重启API，再使用原user_id/session_id补充原因，验证Redis恢复。

在未完成任务时询问“退款政策怎么规定？”，再问“那运费呢？”，RAG会接收最近历史来补全查询，原业务任务仍保留，不会因为咨询政策自动提交。

**8. 查看和清空状态**

- GET /api/sessions/refund-demo-01?user_id=u001：查看当前任务、消息条数和实际记忆后端。
- GET /api/history/refund-demo-01?user_id=u001：查看该用户/会话的历史。
- DELETE /api/sessions/refund-demo-01?user_id=u001：清空会话及草稿，不删除MySQL记录。

使用u002查询同名session，应得到空任务/空历史；使用u001但另一个session也应为空。不要遗漏user_id，否则接口会按anonymous查询，与u001是不同命名空间。

响应中的memory_backend=redis表示实际用了Redis；memory表示降级，不要仅凭项目存在Redis类就认为Redis已生效。

**9. 自动化测试与边界**

```powershell
uv run --no-sync python -m unittest discover -s tests -v
```

默认测试不连接外部模型或用户MySQL，使用固定模型输出和临时SQLite验证真实业务代码。包括字段补充/修正/删除、明确确认、取消、请求重放、用户与会话隔离、RAG历史注入、上下文长度限制、TTL以及清空后旧检查点不复活。

真实Redis客户端恢复、互斥锁与API组件重建恢复测试：

```powershell
$env:TEST_REDIS_URL = 'redis://127.0.0.1:6379/0'
uv run --no-sync python -m unittest discover -s tests -p test_memory.py -k Redis -v
```

这些Redis测试仅创建随机会话命名空间并清理自己的key，不清空Redis数据库；业务数据库仍是临时SQLite，模型仍为替身。它们验证状态流和恢复，不代表真实模型的意图识别已达到某个准确率。实际对话仍应通过第7步测试。

会话快照、数据库事务和回复审查不是一个跨系统事务。极端情况下数据库已提交但后续模型或Redis失败，API可能没有完整回复；稳定task_id可防止重复创建，更新版本可避免盲目覆盖，此时应先查询数据库核实。取消草稿不承诺回滚已经提交的业务。
