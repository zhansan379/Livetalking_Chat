###############################################################################
#  系统关机工具：shutdown_pc（工具层通用工具，与 weather/files 平级，非能力）
#
#  把当前这台电脑关机（Windows: shutdown /s /f /t N；Linux/macOS: shutdown -h）。
#  属破坏性动作，工具体可传 force（是否不强制）不改也行；这里用 delay 秒缓冲，
#  handler 只在调用方确实下发时执行，不做二次确认（确认交给人/模型侧）。
###############################################################################

import subprocess
import sys

from utils.logger import logger


async def shutdown_pc(args: dict, cfg, ctx=None) -> str:
    """关闭当前电脑。delay_seconds>0 时做延迟关机，可在该秒数内用 shutdown /a 取消。"""
    delay = int((args or {}).get("delay_seconds", 0) or 0)
    delay = max(0, min(delay, 3600))  # 缓冲上限 1 小时，防误配超长
    win = sys.platform.startswith("win")
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
    if delay:
        # Windows 无法简便回读剩余时间；给一句朝向的提示
        tip = "（如需取消，请立即运行 shutdown /a）" if win else ""
        return f"（已发出关机指令，将在大约 {delay} 秒后关闭。{tip}）"
    return "（已发出立即关机指令，电脑即将关闭。）"


def _shutdown_tool_specs() -> list[dict]:
    """关机工具定义（供 TOOL_REGISTRY 注册）。"""
    return [
        {
            "name": "shutdown_pc",
            "description": (
                "关闭当前这台电脑。只有当用户明确表达『关机 / 关电脑 / 关闭电脑』等关机的真实意图时"
                "才调用；不要因为『休息一下』『先开着』『困了』这类模糊表达就擅自关机。\n"
                "delay_seconds 默认 0（立即关机）；如果用户给了缓冲（如『5 分钟后关机』『等我一会再关』），"
                "把分钟换算成秒填进去，方便用户后悔时能取消。\n"
                "调用后如实把结果（已发出关机指令 / 失败原因）告诉用户。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "delay_seconds": {
                        "type": "integer",
                        "description": "可选：延迟多少秒后关机，默认 0（立即）。用户给了缓冲计时的才填，最大 3600。",
                    },
                },
                "required": [],
            },
            "handler": shutdown_pc,
        },
    ]