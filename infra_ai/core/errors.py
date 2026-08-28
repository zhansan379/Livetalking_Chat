"""
结构化错误分类。

根据 HTTP 状态码和异常类型将 LLM 调用错误分为 7 类，
供熔断器和路由器判断是否应重试/切换候选。

用法:
    from infra_ai.core.errors import ModelClientErrorType, classify_error
    err_type = classify_error(exception)
    if err_type.should_retry():
        fallback()
"""

from enum import Enum


class ModelClientErrorType(Enum):
    """LLM 调用错误类型。"""

    # 可重试（临时性错误）
    RATE_LIMITED = "RATE_LIMITED"      # 429 Too Many Requests
    SERVER_ERROR = "SERVER_ERROR"      # 5xx 服务端错误
    NETWORK_ERROR = "NETWORK_ERROR"    # 连接超时/DNS/SSL

    # 不可重试（需人工介入）
    UNAUTHORIZED = "UNAUTHORIZED"       # 401/403 鉴权失败
    CLIENT_ERROR = "CLIENT_ERROR"       # 4xx（非 429）请求参数错误
    INVALID_RESPONSE = "INVALID_RESPONSE"  # 响应格式异常
    PROVIDER_ERROR = "PROVIDER_ERROR"   # Provider 特有错误

    # 未分类
    UNKNOWN = "UNKNOWN"

    def should_retry(self) -> bool:
        """是否应重试/切换候选模型。"""
        return self in (
            ModelClientErrorType.RATE_LIMITED,
            ModelClientErrorType.SERVER_ERROR,
            ModelClientErrorType.NETWORK_ERROR,
        )

    def is_fatal(self) -> bool:
        """是否致命错误（不应重试）。"""
        return self in (
            ModelClientErrorType.UNAUTHORIZED,
            ModelClientErrorType.CLIENT_ERROR,
            ModelClientErrorType.PROVIDER_ERROR,
        )


def classify_error(error: Exception) -> ModelClientErrorType:
    """
    根据异常类型/消息分类为 ModelClientErrorType。

    识别逻辑:
        - HTTP 状态码 → 401/403=UNAUTHORIZED, 429=RATE_LIMITED,
          4xx=CLIENT_ERROR, 5xx=SERVER_ERROR
        - 网络异常 → NETWORK_ERROR
        - JSON 解析失败 → INVALID_RESPONSE
        - 其他 → UNKNOWN
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # HTTP 状态码匹配
    if hasattr(error, 'status_code'):
        code = getattr(error, 'status_code', 0)
        if code in (401, 403):
            return ModelClientErrorType.UNAUTHORIZED
        if code == 429:
            return ModelClientErrorType.RATE_LIMITED
        if 400 <= code < 500:
            return ModelClientErrorType.CLIENT_ERROR
        if code >= 500:
            return ModelClientErrorType.SERVER_ERROR

    # API 返回体中的状态码
    if "429" in error_str or "rate" in error_str and "limit" in error_str:
        return ModelClientErrorType.RATE_LIMITED
    if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
        return ModelClientErrorType.UNAUTHORIZED
    if "500" in error_str or "503" in error_str or "server error" in error_str:
        return ModelClientErrorType.SERVER_ERROR

    # 网络层错误
    if any(kw in error_type for kw in ("timeout", "connection", "socket", "dns", "ssl")):
        return ModelClientErrorType.NETWORK_ERROR
    if any(kw in error_str for kw in ("connection refused", "timeout", "name resolution", "tls")):
        return ModelClientErrorType.NETWORK_ERROR

    # 响应解析错误
    if any(kw in error_type for kw in ("json", "decode", "parse", "keyerror", "typeerror", "indexerror", "valueerror")):
        return ModelClientErrorType.INVALID_RESPONSE

    return ModelClientErrorType.UNKNOWN
