# Python 版 - 多Agent智能会议助手

> 使用 LangGraph + WhisperX + FastAPI 实现的完整版本

## 技术栈

- **Agent框架**: LangGraph (状态图编排，支持并行)
- **语音转写**: WhisperX + pyannote-audio (说话人识别)
- **LLM**: DeepSeek（默认 deepseek-flash，JSON输出）
- **Web框架**: FastAPI + WebSocket
- **集成**: Jira Cloud API + 飞书 Open API
- **存储**: SQLAlchemy + PyMySQL + MySQL（完整结果），Milvus + 百炼Embedding（语义索引）

## 快速开始

### 当前共享 uv 工作区（Windows）

项目优先读取 `meeting_assistant/.env`，缺少的模型配置可从上层 uv 工作区
`.env` 补充。项目文件覆盖依赖提前加载的同名环境变量；已有工作区
`DEEPSEEK_API_KEY` 时无需重复复制密钥。

```dotenv
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
# 若工作区没有密钥，在meeting_assistant/.env填写：
# DEEPSEEK_API_KEY=你的DeepSeek密钥
JIRA_ENABLED=false
FEISHU_ENABLED=false
```

```powershell
cd "D:\Python-Project\项目\meeting_assistant\python"
uv run --no-sync python -m unittest discover -s src/tests -v
uv run --no-sync python -m src.main
```

启动后打开 `http://127.0.0.1:8000/docs`，执行
`POST /api/v1/meeting/{meeting_id}/demo`，meeting_id填写 `learn-001`。
演示使用内置转写，不需要安装语音模型；摘要、待办、未决事项会实际调用DeepSeek。
Jira/飞书开关为false时不执行外部同步。下面的自动化测试则使用模拟客户端，
不访问模型、Jira或飞书，也不会下载WhisperX。

节点只返回本节点新增字段，`errors` 通过 `Annotated[list[str], add]` 合并；
三个并行分支全部完成后，跟进节点执行一次。

### 真实音频、说话人编号和持久化

当前继续采用“录完上传，再处理”。API启动时复用一套图、Agent和客户端；
语音模型第一次真实转写时加载，后续会议复用。每次调用创建独立State，
共享语音模型的推理使用锁保护。

- `/demo` 明确使用演示模式；`/upload` 明确使用真实模式。
- 真实模式没有音频、缺少模型依赖或转写失败时，保存 `failed` 和错误，不生成演示结果。
- 每个转写片段保留 `speaker_id` 和兼容字段 `speaker`，两者都使用原始说话人编号。
- 摘要的 `speaker_summaries` 按编号列出发言要点，未决事项的原文引用保留编号和时间位置。
- 待办保留原文明示的负责人姓名，例如“李明负责”记录为“李明”；“我来负责”使用发言者编号。未指派负责人时才标为“未明确”，不需要人工填写姓名映射。
- 完整会议存入MySQL的 `ma_meetings`，分段和待办分别存入
  `ma_transcript_segments`、`ma_action_items`；目标为独立的 `meeting_assistant` 库。
- 原始录音保存在 `meeting_assistant/data/audio/<meeting_id>/`；数据库记录文件路径。
- 报告正文也保存到MySQL，下载接口返回实际正文。
- Milvus使用独立集合 `meeting_assistant_v4_1024_v1`，保存文本块、向量、会议/用户编号、
  版本和时间范围。MySQL结果先保存，向量索引失败时记录 `index_status=failed`，可单独重建。

数据库连接统一填写在 `meeting_assistant/.env`：

```dotenv
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=meeting_assistant
MYSQL_USER=meeting_assistant_app
MYSQL_PASSWORD=这里填写对应账号的密码
```

本地学习时也可使用root账号及其对应密码。数据库本身需要先存在；使用有权限的账号
执行下列SQL，应用启动时会自动创建缺失的四张ma_表：

```sql
CREATE DATABASE IF NOT EXISTS meeting_assistant
  CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;
```

如需独立应用账号，可由root执行以下SQL；先把密码占位文字替换为自己的密码：

```sql
CREATE USER IF NOT EXISTS 'meeting_assistant_app'@'127.0.0.1'
  IDENTIFIED BY '这里替换为应用账号密码';
GRANT SELECT, INSERT, UPDATE, CREATE, INDEX, REFERENCES
  ON meeting_assistant.* TO 'meeting_assistant_app'@'127.0.0.1';
```

此前smart_cs库中的验收记录保持原样；应用连接哪个库由上述配置决定。

真实转写需让 `ffmpeg` 命令在PATH中可用。WhisperX提供模型推理，
FFmpeg负责解码上传的音频。可先使用 `/demo` 验证其余流程。

```powershell
# 在共享uv工作区记录并安装语音依赖；安装会包含PyTorch等较大依赖。
cd "D:\Python-Project"
uv add whisperx
# 指定winget源，避免同时查询msstore源。
winget install --exact --id Gyan.FFmpeg --source winget --accept-source-agreements --accept-package-agreements
# 安装后重新打开终端，再验证：
ffmpeg -version
```

当前环境安装的是CPU版PyTorch（2.8.0+cpu）。先用CPU和较小模型运行：

```dotenv
WHISPER_MODEL_SIZE=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_BATCH_SIZE=4
DIARIZATION_ENABLED=false
```

电脑有NVIDIA显卡也需要安装兼容的CUDA版PyTorch，才能改为 `WHISPER_DEVICE=cuda`。

需要说话人分离时，使用已安装的WhisperX 3.8.6默认模型
`pyannote/speaker-diarization-community-1`：

1. 登录 https://huggingface.co/pyannote/speaker-diarization-community-1 ，在模型页接受访问条件。
2. 在 https://huggingface.co/settings/tokens 创建Read权限的访问令牌；使用接受模型条件的同一账号。
3. 编辑会议项目根目录的 `.env`，设置 `DIARIZATION_ENABLED=true`、`HF_TOKEN=hf_你的实际令牌`。
4. 停止并重新启动会议服务，再上传一段至少两人轮流说话的录音。首次运行会下载模型，需能访问Hugging Face。
5. 查看转写接口的segments，确认speaker_id出现不同编号；编号用于区分声音，不代表真实姓名。

服务流程为：FFmpeg解码 → WhisperX转写与时间戳对齐 → pyannote区分说话人 → 将编号附到转写片段。
关闭分离时，无法确定的说话人以 `UNKNOWN` 保存，并在转写warnings中说明。

直接上传完整录音即可生成按编号组织的结果，Swagger中依次操作：

1. `POST /api/v1/meeting/start`：可传会议编号、`user_id`、标题，不需要姓名。
2. `POST /api/v1/meeting/{meeting_id}/upload?user_id=u001`：file字段选择音频文件；默认转写并完成所有分析。
3. `GET /api/v1/meeting/{meeting_id}/report?user_id=u001`：查看完整结果。
4. `GET /api/v1/meeting/{meeting_id}/report/download?user_id=u001`：下载报告。

Postman使用Body → form-data，字段名为file，类型选择File。服务通过FFmpeg解码音频，
可使用WAV、MP3、M4A等常见格式。转写按时间顺序标注编号，摘要再按编号分组。
`SPEAKER_00`只代表本次录音中的一个声音，不表示某个固定姓名或职位，也不跨会议识别人。
`transcribe_only=true`仅供单独查看转写；需要重新生成分析时可调用 `/analyze`，复用已保存转写。
人工姓名映射输入和 `/speakers` 接口已移除。

```json
POST /api/v1/meetings/search
{"user_id": "u001", "query": "哪次会议讨论过服务器采购？", "top_k": 5}
```

检索返回相关文本及会议编号、版本、起止时间。默认只检索真实会议；验证演示数据时
增加 `"mode": "demo"`，显式传 `"mode": null` 可检索两种模式。
精确读取某会议的完整结果仍走MySQL查询接口。
修改百炼密钥后，可执行 `POST /api/v1/meeting/{meeting_id}/reindex?user_id=u001`，
无需重新调用转写和DeepSeek。

存储实机验收（保留CHECK前缀记录，用户为 `__storage_check__`）：

```powershell
cd "D:\Python-Project\项目\meeting_assistant\python"
# 连接真实MySQL和Milvus，使用独立测试集合及本地测试向量，不调用云API。
uv run --no-sync python check_storage.py
# 使用真实百炼Embedding；会产生少量API费用。
uv run --no-sync python check_storage.py --live-embedding
```

若文件配置正确而实际调用401，检查是否由依赖提前加载了工作区.env：
`pymilvus` 导入时会调用 `load_dotenv()`。本项目显式使用 `override=True`
加载会议项目.env，确保实际请求使用该文件中的密钥和地址；修改配置后重启API。

### 独立虚拟环境

```bash
# 1. 安装依赖
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置环境变量
cp ../.env.example ../.env
# 编辑 ../.env 填入你的 API Key

# 3. 启动服务
python -m src.main

# 4. 测试演示模式（无需音频）
curl -X POST http://localhost:8000/api/v1/meeting/demo/demo
```

## Docker 启动

```bash
docker-compose up -d
```

## 项目结构

```
python/
├── src/
│   ├── agents/                  # 5个Agent实现
│   │   ├── transcription_agent.py   # 转写Agent (WhisperX)
│   │   ├── summary_agent.py         # 摘要Agent (LLM)
│   │   ├── action_agent.py          # 待办Agent (LLM + Jira/飞书)
│   │   ├── unresolved_agent.py      # 未决事项Agent (LLM提取+原文校验)
│   │   └── followup_agent.py        # 跟进Agent (汇聚+推送)
│   ├── graph/
│   │   └── meeting_graph.py         # LangGraph 编排核心
│   ├── integrations/
│   │   ├── deepseek_client.py       # LLMClient读取DeepSeek配置
│   │   ├── jira_client.py           # Jira Cloud 集成
│   │   └── feishu_client.py         # 飞书 Open API 集成
│   ├── models/
│   │   └── schemas.py               # Pydantic 数据模型
│   ├── websocket/
│   │   └── server.py                # FastAPI + WebSocket 服务
│   ├── tests/                       # 离线图、模型适配和HTTP测试
│   ├── config.py                    # 明确路径加载配置
│   └── main.py                      # 入口
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## API 文档

启动后访问: http://localhost:8000/docs

## 未决事项Agent与完整测试流程

当前处理图：

```text
录音上传 → Transcription
                ↓
        Summary / Action / Unresolved（并行）
                ↓
            Follow-up → 保存MySQL → 更新Milvus索引
```

摘要记录讨论结果，待办记录明确安排的任务，未决事项记录会议结束时仍待明确的问题。
`UnresolvedAgent` 使用现有DeepSeek客户端，分类包含 `decision`（待决策）、
`disagreement`（待达成一致）、`missing_information`（待补充信息）、`dependency`（待确认依赖）。
每项附 `topic`、`description`、`priority`、`impact`、`suggested_next_step` 和 `evidence`。
引用中的 `segment_index` 从0开始，`quote`必须出现在该片段，编号和秒数由程序从实际转写补齐。
后续建议保留在未决事项中；会议明确安排的任务仍由Action Agent提取并保存。

### 1. 自动化测试

```powershell
cd "D:\Python-Project\项目\meeting_assistant\python"
uv run --no-sync python -B -m unittest discover -s src/tests -v
```

检查输出最后为 `OK`。测试覆盖三个分支并行及汇聚、原文校验、去重、空结果、
单分支失败、HTTP/WebSocket、旧会议重新分析、待办和提醒回归，不调用付费模型或真实消息平台。

### 2. MySQL与Milvus验收

保持已有MySQL、Milvus服务运行，执行：

```powershell
uv run --no-sync python -B check_storage.py
```

脚本使用独立CHECK会议和测试向量集合，验证报告下载、未决事项保存、原文时间、
Milvus中 `kind=unresolved` 的索引及最新版本过滤，重开MySQL连接后仍能恢复结果。
默认使用模拟LLM和本地测试向量；`--live-embedding` 才调用百炼真实Embedding。

### 3. 启动服务，用内置转写测试真实DeepSeek

```powershell
uv run --no-sync python -m src.main
```

打开 `http://127.0.0.1:8000/docs`，调用：

```http
POST /api/v1/meeting/unresolved-demo-001/demo?user_id=u001
```

此步骤实际调用DeepSeek和百炼Embedding。查看 `status`、`errors`、`index_status`、`unresolved.issues`。
演示中有招聘编制未审批、服务器供应商报价不完整的问题；提交方式在后文已确定为邮件，
应排除“提交方式未确定”。返回的每条引用都能在transcript.segments中找到。

### 4. 查看未决事项、报告与语义检索

```http
GET /api/v1/meeting/unresolved-demo-001/unresolved?user_id=u001
GET /api/v1/meeting/unresolved-demo-001/report/download?user_id=u001
POST /api/v1/meetings/search
```

检索请求体：

```json
{
  "query": "服务器供应商还有哪些问题没有确定？",
  "user_id": "u001",
  "meeting_id": "unresolved-demo-001",
  "mode": "demo",
  "top_k": 5
}
```

未决事项接口返回结构化结果，下载报告展示“未决事项”和原文秒数。
检索返回文本来源 `kind`、起止时间及版本；demo会议查询须显式使用 `mode=demo`。
停止并重启服务后再次GET，结果仍应存在，不会因为GET重新调用模型。

### 5. 使用真实录音测试

先确认 `ffmpeg -version` 有输出、WhisperX已安装。当前CPU环境使用 `WHISPER_DEVICE=cpu`；
需要区分说话人时，保持 `DIARIZATION_ENABLED=true`，并配置有模型访问权限的 `HF_TOKEN`。
首次真实转写会下载语音模型。

可以录制一段包含以下内容的短音频：

```text
甲：A和B两家服务器供应商还没选定，两份报价都没拿到，今天无法确定。
乙：李明明天下午整理报价。
甲：报告是发邮件还是通过系统提交？
乙：已经确定统一发邮件。
甲：采购预算上限还需要财务审批，今天无法确定。
```

在Swagger调用 `POST /api/v1/meeting/unresolved-audio-001/upload?user_id=u001`，
file选择音频，`transcribe_only=false`。随后查询transcript、actions、unresolved和报告。
预期供应商选择和预算审批出现在未决事项中；整理报价出现在待办中；已解决的提交方式不再作为未决问题。
只有流程成功且确实没有未决问题时返回 `issues: []`；该分支失败时 `unresolved=null`，
完整结果中有errors及partial状态。

### 6. 复用已有转写重新分析

```http
POST /api/v1/meeting/unresolved-audio-001/analyze?user_id=u001
```

检查版本增加、报告更新、原文引用保留；无需再次上传或执行语音转写。
升级前的旧会议也使用这个接口生成未决事项，MySQL无需删表重建。

## 自动待办与会后提醒

会议分析后直接保存待办，不增加人工确认步骤。待办的 `action_id` 用于后续更新，
`status` 支持 `pending`（待处理）、`in_progress`（进行中）、`completed`（已完成）、
`cancelled`（已取消）。同一会议重新分析时，任务描述和负责人相同的待办复用ID及完成状态。

完整流程为：Action Agent提取待办 → MySQL保存待办并生成提醒计划 → 后台任务每分钟扫描 →
检查任务仍未完成且到达提醒时间 → 发送并保存结果。提醒使用固定业务规则，不额外调用LLM。

- `ma_action_items.data` 保存截止日期、状态、完成时间及原有待办信息。
- `ma_action_reminders` 保存计划时间、发送状态、尝试次数、发送时间、通道与发送结果。
- 默认北京时间截止前3天、前1天的09:00各提醒一次。例如11月20日截止，11月17日、19日各提醒一次。
- 只补发当天错过09:00的提醒，已过日期的计划标为 `skipped`；已完成/取消的待办停止后续提醒。
- 修改截止日期会取消旧的未发送计划并生成新计划；重复扫描、重启不会重发已发送的计划。
- 未配置飞书时生成 `in_app` 站内提醒，通过提醒接口读取；真实会议启用飞书Webhook后推送到对应群。
- 演示会议使用站内提醒。网络超时导致发送结果不确定时标为 `uncertain`，保留记录供核对。

默认配置已在代码中生效，可在会议项目 `.env` 调整：

```dotenv
REMINDERS_ENABLED=true
REMINDER_POLL_SECONDS=60
REMINDER_HOUR=9
```

如需飞书群提醒，配置 `FEISHU_ENABLED=true` 和真实的 `FEISHU_WEBHOOK_URL`。
服务启动时自动创建缺失的提醒表，并为现有会议当前版本的待办补齐计划。

Swagger测试步骤：

1. 上传会议录音，或调用 `POST /api/v1/meeting/{meeting_id}/demo?user_id=u001`。
2. 调用 `GET /api/v1/meeting/{meeting_id}/actions?user_id=u001`，复制待办的 `action_id`。
3. 调用 `PATCH /api/v1/meeting/{meeting_id}/actions/{action_id}?user_id=u001`，例如：

   ```json
   {"deadline": "2026-11-20", "status": "in_progress"}
   ```

4. 调用 `GET /api/v1/meeting/{meeting_id}/reminders?user_id=u001`，查看计划及发送记录。
5. 用同一PATCH接口发送 `{"status":"completed"}`，再次查看待办和提醒，确认完成时间已保存、未发送提醒已取消。

日期需按测试当天调整；想当天观察提醒，可设截止日期为3天后，并在当天09:00后测试。
报告Markdown是生成时的会议纪要，后续的办理状态通过actions接口查询。

运行离线测试，无需等待真实日期、不发送真实消息：

```powershell
cd "D:\Python-Project\项目\meeting_assistant\python"
uv run python -B -m unittest discover -s src/tests -v
```
