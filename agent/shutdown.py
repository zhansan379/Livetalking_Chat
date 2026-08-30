###############################################################################
#  系统关机工具：shutdown_pc（工具层通用工具，与 weather/files 平级，非能力）
#
#  把当前这台电脑关机（Windows: shutdown /s /f /t N；Linux/macOS: shutdown -h）。
#  属破坏性动作，工具体可传 force（是否不强制）不改也行；这里用 delay 秒缓冲，
#  handler 只在调用方确实下发时执行，不做二次确认（确认交给人/模型侧）。
###############################################################################

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

from utils.logger import logger


# ── 排定关机状态记录 ─────────────────────────────────────────────────────
# shutdown_pc 下发的关机是 Windows `shutdown /s /t N`，系统侧无回读手段，无法得知
# 还剩多久、这次排的是绝对点还是相对缓冲。为支撑「查看已配置的关机任务」，把每次
# 下发的元信息（目标时间 / 相对秒数 / 来源 action）落盘，供 action=status 回读。
_SHUTDOWN_STATE_FILE = os.path.join("data", "shutdown_state.json")


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_shutdown_state() -> dict:
    try:
        with open(_SHUTDOWN_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 - 无文件或读坏一律视为无排定
        return {}


def _save_shutdown_state(state: dict) -> None:
    # 原子写，避免并发读写坏文件；失败只告警，不影响关机本身。
    tmp = f"{_SHUTDOWN_STATE_FILE}.{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(_SHUTDOWN_STATE_FILE), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _SHUTDOWN_STATE_FILE)
    except Exception as e:  # noqa: BLE001
        logger.warning("shutdown state save failed: %s", e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _parse_shutdown_at(text) -> datetime | None:
    """把用户给的绝对时间点解析为本地 datetime。

    支持 HH:MM / HH:MM:SS（默认今天，已过则推到明天）以及
    YYYY-MM-DD HH:MM[:SS]（日期分隔符兼容 '/' 与 'T'）。解析失败返回 None。
    """
    text = (text or "").strip()
    if not text:
        return None
    text = text.replace("/", "-").replace("T", " ")
    parts = text.split()
    try:
        if len(parts) == 1:
            hm = parts[0]
            if hm.count(":") == 1:
                t = datetime.strptime(hm, "%H:%M")
            elif hm.count(":") == 2:
                t = datetime.strptime(hm, "%H:%M:%S")
            else:
                return None
            now = datetime.now()
            dt = now.replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=0)
            return dt + timedelta(days=1) if dt <= now else dt
        if len(parts) == 2:
            d, t = parts
            for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(f"{d} {t}", fmt)
                except ValueError:
                    continue
    except ValueError:
        return None
    return None


async def shutdown_pc(args: dict, cfg, ctx=None) -> str:
    """关闭/取消/查看关机。action=shutdown（默认）关机；cancel 取消；status 查看已配置的关机任务。"""
    args = args or {}
    action = (args.get("action") or "shutdown").strip().lower()
    win = sys.platform.startswith("win")
    if action == "status":
        # 回读本 agent 记录的排定关机（系统 `shutdown /s` 无回读手段，只能查自己的落盘）
        st = _load_shutdown_state()
        if not st or not st.get("target"):
            return "（当前没有已配置的关机任务。）"
        kind = {"at": "绝对时间点", "delay": "相对缓冲"}.get(st.get("kind"), "立即")
        line = f"（已配置关机：{kind}，目标 {st.get('target')}"
        target = None
        try:
            target = datetime.strptime(st["target"], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            pass
        if target:
            left = (target - datetime.now()).total_seconds()
            line += f"，约剩 {int(left)} 秒" if left > 0 else "，时间已到"
        if st.get("at"):
            line += f"，于 {st.get('at')} 下发"
        line += "。如需取消可再说『取消关机』）"
        return line
    if action == "cancel":
        # 取消已排定的关机：Windows shutdown /a；Linux/macOS shutdown -c
        cmd = ["shutdown", "/a"] if win else ["shutdown", "-c"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except OSError as e:
            logger.warning("shutdown_cancel spawn failed: %s", e)
            return "（取消失败：无法启动取消命令）"
        except subprocess.TimeoutExpired:
            return "（取消失败：取消指令下发超时）"
        if r.returncode == 0:
            _save_shutdown_state({})  # 取消成功后清掉本地记录
            return "（已成功取消本次关机。）"
        msg = (r.stderr or r.stdout or "").strip()
        # 没有排定中的关机时，Windows 会返回错误，属正常，如实告知即可
        logger.info("shutdown_cancel rc=%s: %s", r.returncode, msg or "(无排定关机)")
        return "（当前没有待取消的关机，或取消未生效。）"

    delay = int(args.get("delay_seconds", 0) or 0)
    at = args.get("at")
    if at:
        at_dt = _parse_shutdown_at(at)
        if at_dt is None:
            return f"（无法解析关机时间『{at}』，请用 HH:MM 或 YYYY-MM-DD HH:MM 格式，如 23:00 或 2026-08-30 23:00）"
        now = datetime.now()
        if at_dt < now.replace(microsecond=0):
            return f"（关机时间『{at}』早于当前时间，未下发关机。请检查时间点）"
        delay = int((at_dt - now).total_seconds())
    else:
        delay = max(0, min(delay, 3600))  # 相对缓冲上限 1 小时，防误配超长
    try:
        if win:
            # /s 关机 /f 强制结束运行中的应用 /t 延迟秒数（0=立即）
            cmd = ["shutdown", "/s", "/f", "/t", str(delay)]
        else:
            # +N 分钟后挂起（延迟用分钟粒度）；buff = 秒→分钟（向上取整，至少 1 分钟）
            minutes = max(1, -(-delay // 60)) if delay else 0
            cmd = ["shutdown", "-h", "+%d" % minutes] if delay else ["shutdown", "-h", "now"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except OSError as e:
        logger.warning("shutdown_pc spawn failed: %s", e)
        return "（关机失败：无法启动关机命令）"
    except subprocess.TimeoutExpired:
        return "（关机指令下发超时，未取消本次关机）"
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()
        logger.warning("shutdown_pc failed rc=%s: %s", r.returncode, msg)
        return f"（关机失败：{msg or '返回非零'}）"
    # 记录本次排定，供 action=status 回读；立即关机也算，撞上即失效
    if at:
        when = at_dt.strftime("%Y-%m-%d %H:%M:%S")
        _save_shutdown_state({"kind": "at", "target": when, "at": at})
    elif delay:
        _save_shutdown_state({
            "kind": "delay",
            "target": (datetime.now() + timedelta(seconds=delay)).strftime("%Y-%m-%d %H:%M:%S"),
            "at": _now_iso(),
        })
    else:
        _save_shutdown_state({"kind": "now", "target": _now_iso(), "at": _now_iso()})

    if delay:
        # Windows 无法简便回读剩余时间；给一句朝向的提示
        tip = "（如需取消，可再让我『取消关机』）" if win else ""
        if at:
            when = at_dt.strftime("%Y-%m-%d %H:%M:%S")
            return f"（已发出关机指令，将在 {when}（约 {delay} 秒后）关闭。{tip}）"
        return f"（已发出关机指令，将在大约 {delay} 秒后关闭。{tip}）"
    return "（已发出立即关机指令，电脑即将关闭。）"


def _shutdown_tool_specs() -> list[dict]:
    """关机工具定义（供 TOOL_REGISTRY 注册）。"""
    return [
        {
            "name": "shutdown_pc",
            "description": (
                "管理当前电脑的关机/取消关机。只有当用户明确表达『关机 / 关电脑 / 关闭电脑』或其反悔（"
                "『取消关机 / 别关机了 / 我还想用』）等真实意图时才调用；不要因为『休息一下』『先开着』"
                "『困了』这类模糊表达就擅自关机。\n"
                "关机（action 留空或 shutdown）三种定时方式，任选其一：\n"
                "  - 立即关机：什么都不填，delay_seconds 默认 0。\n"
                "  - 相对缓冲：用户给了相对时长（如『5 分钟后关机』『等我一会再关』），把分钟换算成秒填"
                "delay_seconds，方便用户后悔时能取消。\n"
                "  - 绝对时间点：用户给了具体时间（如『今晚 11 点关』『23:00 关机』『明天早上 8 点关』），"
                "换算成 HH:MM（当天已过则自动顺延到明天）或完整 YYYY-MM-DD HH:MM 填 at；不要预先把绝对时间"
                "算成 delay_seconds，交给本工具换算。\n"
                "取消关机：当用户明确表达反悔、想继续用时，action 填 cancel（其余参数忽略）。\n"
                "查看已配置的关机任务：当用户问『我排的关机啥时候』『现在有排关机吗』『看看关机任务』之类时，"
                "action 填 status，回读上次下发的关机时间与剩余时长（只有经本工具下发过才会查到，清空/取消后为空）。\n"
                "调用后如实把结果（已发出关机指令 / 已取消 / 已配置的具体时间 / 失败原因）告诉用户。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["shutdown", "cancel", "status"],
                        "description": "shutdown=关机（默认）；cancel=取消已排定的关机；status=查看已配置的关机任务。",
                    },
                    "delay_seconds": {
                        "type": "integer",
                        "description": "相对缓冲：多少秒后关机，默认 0（立即）。仅关机动作使用，最大 3600。",
                    },
                    "at": {
                        "type": "string",
                        "description": "绝对时间点：HH:MM（当天，已过自动顺延明天）或 YYYY-MM-DD HH:MM。仅关机动作使用。",
                    },
                },
                "required": [],
            },
            "handler": shutdown_pc,
        },
    ]