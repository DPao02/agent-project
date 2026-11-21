"""
Transcription Agent（转写Agent）
- 接收音频数据，使用 WhisperX 进行语音转文字
- 使用 pyannote-audio 进行说话人识别（Speaker Diarization）
- 输出带说话人标签和时间戳的转写文本
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from pathlib import Path

from loguru import logger

from ..models.schemas import (
    MeetingStatus,
    MeetingMode,
    TranscriptResult,
    TranscriptSegment,
)


class TranscriptionConfig:
    """转写配置"""

    def __init__(
        self,
        model_size: str = "large-v2",
        device: str = "cpu",
        compute_type: str = "float32",
        language: str = "zh",
        hf_token: str = "",
        batch_size: int = 16,
    ):
        self.model_size = os.getenv("WHISPER_MODEL_SIZE", model_size)
        self.device = os.getenv("WHISPER_DEVICE", device)
        self.compute_type = os.getenv("WHISPER_COMPUTE_TYPE", compute_type)
        self.language = os.getenv("WHISPER_LANGUAGE", language)
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.batch_size = int(os.getenv("WHISPER_BATCH_SIZE", str(batch_size)))
        self.diarization_enabled = os.getenv("DIARIZATION_ENABLED", "false").lower() == "true"


class TranscriptionAgent:
    """
    转写Agent - Pipeline的第一个节点

    架构说明:
    1. 接收音频字节数据（来自 WebSocket 或文件上传）
    2. 使用 WhisperX 进行批量转写（比原版 Whisper 快 70x）
    3. wav2vec2 强制对齐获取精确时间戳
    4. pyannote-audio 进行说话人识别
    5. 合并结果，输出 TranscriptResult

    面试考点:
    - 为什么用 WhisperX 而不是原版 Whisper？（速度 + 时间戳精度）
    - VAD 预处理有什么作用？（降低幻觉，过滤静音段）
    - 说话人识别的原理？（speaker embedding + 聚类）
    """

    def __init__(self, config: TranscriptionConfig | None = None):
        self.config = config or TranscriptionConfig()
        self._model = None
        self._align_model = None
        self._diarize_pipeline = None
        self._initialized = False
        # 模型共享，输入不共享。锁覆盖加载和推理，保护模型内部可变状态。
        self._inference_lock = threading.Lock()

    def _lazy_init(self):
        """
        懒加载模型 —— 避免在导入时就加载大模型。
        生产中模型应在服务启动时预热。
        """
        if self._initialized:
            return

        try:
            import whisperx

            logger.info(
                f"Loading WhisperX model: {self.config.model_size} "
                f"on {self.config.device}"
            )
            self._model = whisperx.load_model(
                self.config.model_size,
                self.config.device,
                compute_type=self.config.compute_type,
            )
            self._initialized = True
            logger.info("WhisperX model loaded successfully")
        except ImportError:
            raise RuntimeError("真实转写需要WhisperX，请先安装语音依赖；不会改用演示内容") from None

    async def process(self, state: dict) -> dict:
        """
        LangGraph 节点函数 —— 执行语音转写

        Args:
            state: MeetingState 字典，包含 audio_data 字段

        Returns:
            本节点的增量更新，包含 status、transcript、transcript_text 和新增错误
        """
        meeting_id = state.get("meeting_id", "unknown")
        logger.info(f"[TranscriptionAgent] Processing meeting: {meeting_id}")

        try:
            mode = MeetingMode(state.get("mode", MeetingMode.REAL))
            if state.get("transcript") is not None:
                # 重新分析时复用本会议已持久化的转写，不再次跑ASR。
                transcript = state["transcript"].model_copy(deep=True)
                if transcript.meeting_id != meeting_id:
                    raise ValueError("转写结果不属于当前会议")
            elif mode == MeetingMode.DEMO:
                transcript = self._generate_demo_transcript(meeting_id)
            else:
                audio_data = state.get("audio_data", b"")
                if not audio_data:
                    raise ValueError("真实模式没有收到音频数据")
                transcript = await self._transcribe(audio_data, meeting_id)
            transcript = self.use_speaker_ids(transcript)
            logger.info(
                f"[TranscriptionAgent] Transcription complete: "
                f"{len(transcript.segments)} segments"
            )
        except Exception as e:
            logger.error(f"[TranscriptionAgent] Error: {e}")
            return {
                "status": MeetingStatus.FAILED,
                "transcript": None,
                "transcript_text": "",
                "errors": [f"TranscriptionAgent: {str(e)}"],
            }

        return {
            "status": MeetingStatus.TRANSCRIBED,
            "transcript": transcript,
            "transcript_text": self._format_transcript_text(transcript),
            "errors": [],
        }

    async def _transcribe(
        self, audio_data: bytes, meeting_id: str
    ) -> TranscriptResult:
        """同步模型推理放进线程；不会阻塞FastAPI的事件循环。"""
        return await asyncio.to_thread(self._transcribe_locked, audio_data, meeting_id)

    def _transcribe_locked(self, audio_data: bytes, meeting_id: str) -> TranscriptResult:
        with self._inference_lock:
            if self.config.diarization_enabled and not self.config.hf_token:
                raise ValueError("开启说话人分离需要HF_TOKEN及对应Hugging Face模型访问权限")
            self._lazy_init()
            return self._transcribe_sync(audio_data, meeting_id)

    def _transcribe_sync(self, audio_data: bytes, meeting_id: str) -> TranscriptResult:
        import whisperx

        # Windows下先关闭文件句柄，再让FFmpeg读取；完成后仅删除本次临时文件。
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(audio_data)
            audio_path = Path(tmp.name)
        try:
            audio = whisperx.load_audio(str(audio_path))
            # Step 1: WhisperX 转写
            result = self._model.transcribe(
                audio,
                batch_size=self.config.batch_size,
                language=self.config.language,
            )

            # Step 2: 时间戳对齐
            if self._align_model is None:
                model_a, metadata = whisperx.load_align_model(
                    language_code=self.config.language,
                    device=self.config.device,
                )
                self._align_model = (model_a, metadata)

            aligned = whisperx.align(
                result["segments"],
                self._align_model[0],
                self._align_model[1],
                audio,
                self.config.device,
            )

            # Step 3: 说话人识别
            if self.config.diarization_enabled and not self.config.hf_token:
                raise ValueError("开启说话人分离需要HF_TOKEN及对应Hugging Face模型访问权限")
            if self.config.diarization_enabled and self._diarize_pipeline is None:
                from whisperx.diarize import DiarizationPipeline
                self._diarize_pipeline = DiarizationPipeline(
                    token=self.config.hf_token,
                    device=self.config.device,
                )

            if self._diarize_pipeline:
                diarize_result = self._diarize_pipeline(audio)
                final = whisperx.assign_word_speakers(
                    diarize_result, aligned
                )
            else:
                final = aligned
        finally:
            audio_path.unlink(missing_ok=True)

        segments = []
        for seg in final.get("segments", []):
            segments.append(
                TranscriptSegment(
                    speaker_id=seg.get("speaker", "UNKNOWN"),
                    speaker=seg.get("speaker", "UNKNOWN"),
                    text=seg.get("text", "").strip(),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    confidence=seg.get("confidence", 0.0),
                )
            )

        duration = segments[-1].end if segments else 0.0
        if not segments or not any(s.text for s in segments):
            raise ValueError("音频中没有识别到有效语音")
        full_text = " ".join(s.text for s in segments)

        return TranscriptResult(
            meeting_id=meeting_id,
            segments=segments,
            language=self.config.language,
            duration_seconds=duration,
            full_text=full_text,
            diarization_available=self.config.diarization_enabled,
            warnings=[] if self.config.diarization_enabled else ["未启用说话人分离，UNKNOWN表示说话人尚未确定"],
        )

    @staticmethod
    def _generate_demo_transcript(meeting_id: str) -> TranscriptResult:
        """生成演示转写结果（无音频时使用）"""
        demo_segments = [
            TranscriptSegment(
                speaker="张总",
                text="好的，我们开始今天的Q3预算评审会议。首先请李明汇报一下目前的预算执行情况。",
                start=0.0, end=8.5, confidence=0.96,
            ),
            TranscriptSegment(
                speaker="李明",
                text="好的张总。截至目前，Q2预算执行率为87%，其中研发投入占比最大，达到42%。",
                start=9.0, end=16.2, confidence=0.95,
            ),
            TranscriptSegment(
                speaker="李明",
                text="Q3我们计划将预算上调15%，主要增加在AI基础设施和人才招聘方面。",
                start=16.5, end=23.1, confidence=0.94,
            ),
            TranscriptSegment(
                speaker="王芳",
                text="关于人才招聘，我建议我们重点招聘3名高级算法工程师，预算大概在每人年薪80万左右。",
                start=23.5, end=31.0, confidence=0.93,
            ),
            TranscriptSegment(
                speaker="张总",
                text="可以。李明你来负责整理Q3的详细预算方案，下周五之前提交给我审批。",
                start=31.5, end=38.2, confidence=0.97,
            ),
            TranscriptSegment(
                speaker="张总",
                text="王芳负责拟定招聘JD，本周三前完成。另外，赵伟跟进一下服务器采购的事情。",
                start=38.5, end=46.0, confidence=0.95,
            ),
            TranscriptSegment(
                speaker="赵伟",
                text="收到，我这边已经在对比几家供应商了，预计下周一可以给出采购方案。",
                start=46.5, end=52.8, confidence=0.94,
            ),
            TranscriptSegment(
                speaker="王芳",
                text="不过，三名算法工程师的招聘编制还没有审批，人力部门也没有确认是否能按计划到岗，这一点需要继续明确。",
                start=53.0, end=61.0, confidence=0.95,
            ),
            TranscriptSegment(
                speaker="赵伟",
                text="服务器供应商还没有定下来，A和B两家的完整报价都没有拿到，今天无法确定采购哪家。",
                start=61.5, end=70.2, confidence=0.95,
            ),
            TranscriptSegment(
                speaker="李明",
                text="预算方案的提交方式是否还没确定？需要发邮件还是通过系统提交？",
                start=70.5, end=74.5, confidence=0.94,
            ),
            TranscriptSegment(
                speaker="张总",
                text="提交方式已经确定，统一通过邮件提交，不需要再讨论。招聘编制和供应商选择留到下一次确认。",
                start=75.0, end=81.0, confidence=0.97,
            ),
            TranscriptSegment(
                speaker="张总",
                text="好的，那今天的会议就到这里。各位辛苦了，请大家按时完成各自的任务。",
                start=81.5, end=88.0, confidence=0.96,
            ),
        ]

        # 演示编号固定，但只在显式demo模式中生成。
        ids = {name: f"SPEAKER_{i:02d}" for i, name in enumerate(dict.fromkeys(s.speaker for s in demo_segments))}
        for segment in demo_segments:
            segment.speaker_id = ids[segment.speaker]
            segment.speaker = segment.speaker_id
        full_text = "\n".join(
            f"[{s.speaker}] {s.text}" for s in demo_segments
        )

        return TranscriptResult(
            meeting_id=meeting_id,
            segments=demo_segments,
            language="zh",
            duration_seconds=88.0,
            full_text=full_text,
            diarization_available=True,
        )

    @staticmethod
    def use_speaker_ids(transcript: TranscriptResult) -> TranscriptResult:
        """所有下游统一使用原始编号，不猜测姓名、职务或跨会议身份。"""
        result = transcript.model_copy(deep=True)
        for segment in result.segments:
            segment.speaker = segment.speaker_id
        result.full_text = "\n".join(f"[{s.speaker_id}] {s.text}" for s in result.segments)
        return result

    @staticmethod
    def _format_transcript_text(transcript: TranscriptResult) -> str:
        """将转写结果格式化为纯文本（供后续Agent使用）"""
        lines = []
        for seg in transcript.segments:
            ts = f"[{seg.start:.1f}s-{seg.end:.1f}s]"
            lines.append(f"{ts} {seg.speaker}: {seg.text}")
        return "\n".join(lines)
