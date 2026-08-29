###############################################################################
#  工具循环：TOOL_REGISTRY 注册表 + web_search 执行 + while 工具循环
###############################################################################
#  设计要点（与 agent/agent.py 的接口约定）：
#  - 一个工具 = 一条注册表 entry，同时管住「schema（给模型引用）」和「handler（执行）」，
#    避免两者不同步。新增工具只需加一个 entry + 在 agent_config.yaml 补 tools.<name>.*。
#  - run_tool_loop 的唯一职责：把模型的 tool_calls 解析成一句最终文本答案。
#    它不做「思考循环」——OpenAI/Qwen 函数调用里，模型想用工具时返回的是 tool_calls 而非答案，
#    必须把执行结果以 {"role":"tool"} 回填再问一次，直到模型给出纯文本答案（对应 s01 的
#    while stop_reason == "tool_use"）。
#  - 契约：拿到真答案返回 str，触顶/异常返回 None（交给调用方决定说辞），绝不伪造模型回复。
###############################################################################

import asyncio
import json
import time
import urllib.parse
import urllib.request

from utils.logger import logger

# 观测：随 obs 包可用与否优雅降级（观测失败不影响工具循环）
try:
    from obs import emit, round_span
except Exception:  # noqa: BLE001 - obs 缺失时用空实现，工具循环照常运行
    import contextlib

    class _DummyRound:
        n_tool_calls = 0

    @contextlib.asynccontextmanager
    async def _round_span(_idx):
        yield _DummyRound()

    def _emit(_ev):
        return None

    round_span, emit = _round_span, _emit


def _trunc(text, n=200):
    text = str(text or "")
    return text if len(text) <= n else text[:n] + "…"


# ─── 网络搜索执行（DuckDuckGo）─────────────────────────────────────────────
async def web_search(args: dict, cfg) -> str:
    """执行一次网络搜索，返回若干条「标题+链接+摘要」拼成的文本。"""
    query = (args or {}).get("query", "")
    if not query:
        return "（搜索关键词为空）"
    max_sources = getattr(cfg, "tool_web_search_max_sources", 5)

    def _run():
        # ddgs 是新包名；兼容仍是 duckduckgo_search 的老安装
        try:
            from ddgs import DDGS
        except ImportError:  # noqa: F841
            from duckduckgo_search import DDGS
        return list(DDGS().text(query, max_results=max_sources))

    try:
        rows = await asyncio.to_thread(_run)
    except Exception as e:  # noqa: BLE001 - 搜索失败要让模型自行处理，不能中断对话
        logger.warning("web_search failed: %s", e)
        return f"（网络搜索失败：{e}）"

    lines = []
    for row in rows or []:
        title = (row.get("title") or "").strip()
        href = (row.get("href") or "").strip()
        body = (row.get("body") or "").strip()
        if title:
            lines.append(f"标题：{title}")
        if href:
            lines.append(f"链接：{href}")
        if body:
            lines.append(f"摘要：{body}")
        lines.append("---")

    text = "\n".join(lines).strip()
    if not text:
        return "（无搜索结果）"
    return text


# ─── 天气查询（Open-Meteo，免费免 key 的实时天气）───────────────────────
#  Open-Meteo 返回结构化 JSON，比通用 web_search 的「历史网页摘要」可靠得多；
#  适合「今天/现在某地天气」这类实时正交事实。天气问题优先走本工具。
_WEATHER_FORECAST_BASE = "https://api.open-meteo.com"
_WEATHER_GEOCODING_BASE = "https://geocoding-api.open-meteo.com"
_HTTP_TIMEOUT = 15

# 常见城市直接给坐标，省一次 geocoding 调用（可被 agent_config.yaml weather.coords 覆盖/补充）
_DEFAULT_WEATHER_COORDS: dict[str, tuple[float, float]] = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "山西": (37.8570, 112.5624),  # 近似太原，省份查询给省会兜底
    "太原": (37.8706, 112.5489),
    "太原市": (37.8706, 112.5489),
    "河北": (38.0428, 114.5149),
    "石家庄": (38.0424, 114.5149),
    "陕西": (34.3416, 108.9398),
    "西安": (34.3416, 108.9398),
    "山东": (36.6683, 117.0202),
    "济南": (36.6512, 117.1201),
    "河南": (34.7466, 113.6254),
    "郑州": (34.7466, 113.6254),
    "四川": (30.5728, 104.0668),
    "成都": (30.5728, 104.0668),
    "广东": (23.1291, 113.2644),
    "武汉": (30.5928, 114.3055),
    "杭州": (30.2741, 120.1551),
    "南京": (32.0603, 118.7969),
    "天津": (39.3434, 117.3616),
    "重庆": (29.4316, 106.9123),
}

# WMO 天气码 → 中文天气现象（标准 0-99）
_WMO_CODES: dict[int, str] = {
    0: "晴", 1: "基本晴", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "毛毛雨（轻）", 53: "毛毛雨（中）", 55: "毛毛雨（大）",
    56: "冻毛毛雨（轻）", 57: "冻毛毛雨（大）",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨（轻）", 67: "冻雨（重）",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "米雪",
    80: "阵雨（轻）", 81: "阵雨（中）", 82: "阵雨（强）",
    85: "阵雪（轻）", 86: "阵雪（大）",
    95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴伴冰雹",
}


def _http_get_json(url: str) -> dict:
    """极简 GET + JSON 解析（标准库 urllib，无新依赖）。"""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LiveTalking/1.0 (weather tool)"},
    )
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def _resolve_city(raw_city: str, cfg) -> tuple[float, float, str]:
    """解析城市 → (lat, lon, 展示名)。先查内置坐标，未命中走 geocoding。"""
    coords = dict(getattr(cfg, "weather_coords", {}) or {})
    for name, pt in _DEFAULT_WEATHER_COORDS.items():
        coords.setdefault(name, pt)

    if raw_city in coords:
        lat, lon = coords[raw_city]
        return lat, lon, raw_city

    gbase = getattr(cfg, "weather_geocoding_base_url", None) or _WEATHER_GEOCODING_BASE
    url = (
        f"{gbase}/v1/search?name={urllib.parse.quote(raw_city)}"
        f"&count=1&language=zh&format=json"
    )
    loc = _http_get_json(url).get("results") or []
    if not loc:
        raise ValueError(f"未解析到 {raw_city} 的地理坐标")
    first = loc[0]
    name = first.get("name") or raw_city
    admin1 = first.get("admin1") or ""
    display = name if name == raw_city else f"{admin1 or ''} {name}".strip()
    return float(first["latitude"]), float(first["longitude"]), display


async def get_weather(args: dict, cfg) -> str:
    """查询指定城市的实时天气，返回一段中文描述。"""
    raw_city = (args or {}).get("city", "").strip()
    if not raw_city:
        return "（天气查询需要提供城市名，例如：北京、上海、太原、山西）"

    def _run() -> str:
        lat, lon, display = _resolve_city(raw_city, cfg)
        fbase = getattr(cfg, "weather_forecast_base_url", None) or _WEATHER_FORECAST_BASE
        url = (
            f"{fbase}/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,apparent_temperature,relative_humidity_2m,"
            "precipitation,snowfall,weather_code,wind_speed_10m,is_day,cloud_cover"
            "&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
        )
        data = _http_get_json(url)
        cur = data.get("current") or {}
        daily = data.get("daily") or {}
        today = daily.get("temperature_2m_max") or []
        today = today[0] if today else None

        code = int(cur.get("weather_code", 0) or 0)
        wmo = _WMO_CODES.get(code, f"代码{code}")
        temp = cur.get("temperature_2m")
        feels = cur.get("apparent_temperature")
        humid = cur.get("relative_humidity_2m")

        parts = [f"{display}实时天气"]
        if temp is not None:
            line = f"当前{temp:g}℃"
            if feels is not None:
                line += f"（体感{feels:g}℃）"
            if today is not None:
                line += f"，今日最高{today} / 最低{daily['temperature_2m_min'][0]}℃"
            parts.append(line)
        parts.append(f"{'白天' if cur.get('is_day') else '夜晚'}，天气{wmo}")
        if humid is not None:
            parts.append(f"湿度{humid}%")
        if cur.get("cloud_cover") is not None:
            parts.append(f"云量{cur['cloud_cover']}%")
        if cur.get("precipitation"):
            parts.append(f"降水{cur['precipitation']}mm")
        if cur.get("snowfall"):
            parts.append(f"降雪{cur['snowfall']}cm")
        if cur.get("wind_speed_10m"):
            parts.append(f"{cur['wind_speed_10m']}km/h")
        return "，".join(parts) + "。"

    try:
        return await asyncio.to_thread(_run)
    except Exception as e:  # noqa: BLE001 - 天气失败要让模型自行处理，不中断对话
        logger.warning("get_weather failed: %s", e)
        return f"（天气查询失败：{e}）"


# ─── 工具注册表：一个工具 = 一条 (schema + handler) ─────────────────────────
TOOL_REGISTRY: dict[str, dict] = {
    "web_search": {
        "description": (
            "联网搜索互联网获取实时/最新信息，返回若干条标题、链接和摘要。"
            "当用户问到时效性强、或需要联网确认/查询的事实性问题时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
        "handler": web_search,
    },
    "weather": {
        "description": (
            "查询某个城市的实时天气（当前温度、体感、天气现象、湿度、风、"
            "今日最高/最低温）。当用户问到今天/现在某地的天气，或天气变化、"
            "是否下雨、冷不冷时调用。优先调用本工具，而不是 web_search。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，中文即可，如：北京、上海、太原、山西"},
            },
            "required": ["city"],
        },
        "handler": get_weather,
    },
    # 以后新增工具：在这里加一个 entry，并在 agent_config.yaml 里加 tools.<name>.enabled
}


def list_enabled_tools(cfg) -> list[str]:
    """返回配置里已启用的工具名（约定：配置字段 tool_<name>_enabled）。"""
    return [
        name for name in TOOL_REGISTRY
        if getattr(cfg, f"tool_{name}_enabled", False)
    ]


def build_tools(enabled: list[str]) -> list[dict]:
    """由注册表把启用的工具名转成 OpenAI function calling 的 tools 列表。"""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_REGISTRY[name]["description"],
                "parameters": TOOL_REGISTRY[name]["parameters"],
            },
        }
        for name in enabled
    ]


# ─── 工具循环 ──────────────────────────────────────────────────────────────
def _assistant_turn(resp, tool_calls) -> dict:
    """把原生 assistant message（含 tool_calls）转成可继续喂给下一次调用的 dict。"""
    return {
        "role": "assistant",
        "content": getattr(resp, "content", None) or None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ],
    }


async def run_tool_loop(agent_messages: list, tools: list[dict], cfg) -> str | None:
    """
    while 工具循环：把模型可能打出的 tool_calls 解析成一句最终文本答案。

    :param agent_messages: 已含 system + 历史 + 当前 user 消息的上下文
    :param tools: OpenAI function calling 的 tools 列表（由 build_tools 生成）
    :param cfg: AgentConfig（提供 tool_max_rounds 与各工具参数）
    :return: 模型最终文本答案；循环触顶返回 None（不伪造回复，交给调用方说辞）
    """
    from infra_ai import async_call_llm_with_tools

    msgs = list(agent_messages)
    for idx in range(cfg.tool_max_rounds):
        async with round_span(idx) as rd:
            try:
                resp = await async_call_llm_with_tools(msgs, tools)
            except Exception as e:  # noqa: BLE001 - LLM 调用失败：让上层走降级话术
                logger.exception("run_tool_loop LLM call failed: %s", e)
                return None

            tool_calls = getattr(resp, "tool_calls", None)
            if not tool_calls:
                # 模型决定不再调工具 → 这就是最终答案，循环唯一出口；本轮不调工具
                return (getattr(resp, "content", None) or "").strip()

            rd.n_tool_calls = len(tool_calls)
            msgs.append(_assistant_turn(resp, tool_calls))
            for tc in tool_calls:
                handler = TOOL_REGISTRY.get(tc.function.name, {}).get("handler")
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                t0 = time.monotonic()
                tool_name = tc.function.name
                if handler is None:
                    result = f"（未知工具：{tool_name}）"
                    ok, err = True, None
                else:
                    try:
                        result = await handler(args, cfg) or "（工具无输出）"
                        ok, err = True, None
                    except Exception as e:  # noqa: BLE001 - 单个工具失败不中断循环
                        logger.exception("tool %s handler failed: %s", tool_name, e)
                        result = f"（工具 <{tool_name}> 执行失败：{e}）"
                        ok, err = False, str(e)
                emit({
                    "type": "tool_call",
                    "round": idx,
                    "tool": tool_name,
                    "args": _trunc(json.dumps(args, ensure_ascii=False), 200) if args else "{}",
                    "result_snippet": _trunc(result, 200),
                    "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
                    "success": ok,
                    "error": err,
                })
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    logger.error("tool loop hit %d rounds without a text answer", cfg.tool_max_rounds)
    return None