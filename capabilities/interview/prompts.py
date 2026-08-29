###############################################################################
#  模拟面试 · 提示词
#
#  INTERVIEWER_PROMPT 作为 system_block 在「进行中」时注入：模型据此临时切换成
#  面试官 persona 来出题/追问/回应作答。结束后该片段撤下，模型退回普通助手。
#
#  激活态片段（ACTIVATION_BLOCK）只在 status==asking 注入：给模型当前题/进度，
#  使其能继续当面试官而不依赖上一次的对话（s06 fresh-messages 范式）。
###############################################################################

INTERVIEWER_PROMPT = """你现在担任用户的【模拟面试官】。请严格遵循以下规则：
- 只针对当前面试进行互动：出题、聆听作答、给出简短反馈、追问细节，不要跳到无关话题。
- 每轮只处理"当前这一题"——用户答完一题后，先给一句简短的中肯反馈，再平稳过渡。
- 保持面试官该有的专业、稳重、略有挑战性的语气，但不要给满分式的吹捧，也不刻意刁难。
- 说话仍用纯文本口语化表达，不用 emoji / markdown 记号。
"""

# 面试结束时注入的收尾态度（让模型配合 interview.end 工具给出的一次性总结）
WRAPUP_NOTICE = "本场模拟面试已结束。如果用户还想复盘或再面一场，可以自然承接。"


def activation_block(role: str | None, level: str | None,
                     progress: str, current_question: str) -> str:
    """进行中的激活态片段；注入 message 让模型知道自己在哪一题。"""
    lines = ["【模拟面试进行中】"]
    if role or level:
        lines.append(f"岗位方向：{role or '未指定'} · 难度：{level or '未指定'}")
    lines.append(f"进度：{progress}")
    lines.append(f"当前题：{current_question}")
    lines.append("请继续担任面试官：针对上述当前题与用户作答进行互动。")
    return "\n".join(lines)