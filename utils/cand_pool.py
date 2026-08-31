"""
候选池通用构造：把 `routing.candidates` 的 dict 行解析为排序后的候选对象。

供 TTS 候选池（tts/executor.py::TTSPool）与（可选的）ASR 候选池共用，
消除『解析 candidates + 按 priority 排序』这一份在各池间重复的逻辑。
"""

from dataclasses import dataclass, field


@dataclass
class Candidate:
    """单个候选：对应 registry 合成/识别槽的一个引擎及其启用/优先级/专属参数。"""

    id: str
    engine: str
    enabled: bool = True
    priority: int = 100
    params: dict = field(default_factory=dict)


def build_candidates(rows: list[dict]) -> list[Candidate]:
    """把 candidates 的 dict 行解析成 Candidate 并按 priority 升序排序。

    缺 id 或 engine 的行被忽略（与 asr/executor.py::ASRPool 的既有过滤一致）。
    """
    cands = [
        Candidate(
            id=c.get("id"),
            engine=c.get("engine"),
            enabled=c.get("enabled", True),
            priority=c.get("priority", 100),
            params=c.get("params") or {},
        )
        for c in (rows or [])
        if c.get("id") and c.get("engine")
    ]
    cands.sort(key=lambda c: c.priority)
    return cands