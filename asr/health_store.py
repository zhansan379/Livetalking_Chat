"""
ASR 三态熔断器：共享实现位于 utils/health_store.py，此处仅为保持向后兼容的再导出。

`asr/executor.py` 以 `from .health_store import HealthStore` 引用，因此这里保留同名导出，
无需改动 executor 即可继续工作。
"""

from utils.health_store import HealthStore, CircuitState, CandidateHealth  # noqa: F401