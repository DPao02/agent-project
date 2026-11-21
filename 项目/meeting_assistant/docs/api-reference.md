# API 参考文档

## REST API

### 创建会议
```
POST /api/v1/meeting/start

Response:
{
  "meeting_id": "abc123",
  "websocket_url": "ws://localhost:8000/ws/meeting/abc123",
  "status": "created"
}
```

### 运行演示
```
POST /api/v1/meeting/{meeting_id}/demo

Response: 完整的会议处理结果（转写+摘要+待办+未决事项+跟进）
```

### 上传音频
```
POST /api/v1/meeting/{meeting_id}/upload
Content-Type: multipart/form-data
Body: file=@meeting.wav
```

### 获取转写结果
```
GET /api/v1/meeting/{meeting_id}/transcript
```

### 获取会议纪要
```
GET /api/v1/meeting/{meeting_id}/summary
```

### 获取待办事项
```
GET /api/v1/meeting/{meeting_id}/actions
```

### 获取未决事项
```
GET /api/v1/meeting/{meeting_id}/unresolved?user_id=u001
```

返回 `meeting_id` 和 `issues`。每项包含 `issue_id`、`topic`、`description`、
`category`、`priority`、`impact`、`suggested_next_step`、`evidence`。
引用中的 `segment_index` 从0开始，`speaker_id`、`start`、`end` 从实际转写片段读取，时间单位为秒。
`issues: []` 表示没有未决问题；分析失败时该接口返回409，完整报告的errors给出原因。
旧会议调用 `POST /api/v1/meeting/{meeting_id}/analyze?user_id=u001`，复用保存的转写生成新结果。

### 获取完整报告
```
GET /api/v1/meeting/{meeting_id}/report
```

## WebSocket API

### 连接
```
ws://localhost:8000/ws/meeting/{meeting_id}
```

### 发送消息
```json
{"type": "start"}    // 开始录制
{"type": "stop"}     // 停止录制，触发处理
{"type": "demo"}     // 运行演示模式
{"type": "ping"}     // 心跳
```

### 接收消息
```json
{"type": "connected", "meeting_id": "..."}
{"type": "recording", "buffer_size": 1024}
{"type": "processing", "message": "处理中..."}
{"type": "transcript", "data": {...}}
{"type": "summary", "data": {...}}
{"type": "actions", "data": {...}}
{"type": "unresolved", "data": {...}}
{"type": "completed", "meeting_id": "..."}
```
