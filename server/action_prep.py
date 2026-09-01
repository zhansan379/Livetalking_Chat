###############################################################################
#  表情底座生成服务层 — 供 web 接口（POST /api/avatar/action/task）与离线脚本
#  （scripts/prep_emotions.py）共用。把「动作/表情样例视频」转成说话时的
#  表情基地序列 data/actions/<action_id>/，并显式记录「该动作绑定到哪个头像」。
#
#  核心仍是各模型 genavatar.py 的 generate_avatar（save_path='data/actions'）；
#  本模块负责与说话形象相关的对齐：人脸裁剪尺寸(wav2lip)、帧数循环补齐
#  （否则 set_emotion 会因帧数不足杀死渲染线程），并写绑定 manifest。
###############################################################################

import os
import glob
import json
import time
import shutil
import pickle
import logging

from PIL import Image

from utils.logger import logger

SAVE_PATH = "data/actions"
ACTION_INFO = "action_info.json"


# ================= 与说话形象对齐的纯 helper（原 scripts/prep_emotions.py 迁入） =================

def _resolve_wav2lip_img_size(avatar_id):
    """取说话形象 face_imgs 的边长，作为表情底座的人脸裁剪尺寸。

    wav2lip 推理直接拿 face_imgs 喂模型、不做缩放，所以底座的人脸尺寸必须与
    说话形象一致（本机形象 rem 为 256×256；之前用 96 生成会导致模型下采样到
    2×2 时 4×4 卷积直接崩溃）。找不到则回退 256。
    """
    root = "data/avatars"
    cands = (
        ([avatar_id] if avatar_id else [])
        + sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "*")))
    )
    for aid in cands:
        fs = glob.glob(os.path.join(root, aid, "face_imgs", "*"))
        if not fs:
            continue
        try:
            return Image.open(fs[0]).size[0]
        except Exception:  # noqa: BLE001
            continue
    logger.warning("找不到说话形象 face_imgs，wav2lip 人脸裁剪尺寸回退 256。")
    return 256


def _neutral_frame_count(avatar_id):
    """返回说话形象的中性基地帧数（曝光：推理/渲染线程都按它定索引长度）。"""
    root = "data/avatars"
    cands = ([avatar_id] if avatar_id else []) + sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(root, "*"))
    )
    for aid in cands:
        d = os.path.join(root, aid)
        cp = os.path.join(d, "coords.pkl")
        if os.path.isfile(cp):
            try:
                return len(pickle.load(open(cp, "rb")))
            except Exception:  # noqa: BLE001
                continue
        n = len(glob.glob(os.path.join(d, "full_imgs", "*.*")))
        if n:
            return n
    return 240


def _pad_cycle_length(dest, target):
    """把表情底座逐帧循环补齐到 target 帧，保证与中性形象帧数一致。

    推理线程在启动时只取一次 length 就据此给队列编 idx，渲染线程再按同一 idx 取
    帧；若表情底座帧数 < 中性，切换后会出现 list index out of range 并杀死渲染线程。
    """
    # 1) coords.pkl（wav2lip/musetalk/ultralight 通用，与帧一一对应）
    cp = os.path.join(dest, "coords.pkl")
    if os.path.isfile(cp):
        coords = pickle.load(open(cp, "rb"))
        if coords and len(coords) < target:
            coords = [coords[i % len(coords)] for i in range(target)]
            pickle.dump(coords, open(cp, "wb"))
            logger.info("%s: coords.pkl %d -> %d 帧", dest, len(coords), target)
    # 2) 逐帧图像目录（full_imgs/face_imgs/mask）：按已有编号宽度补充循环拷贝
    for sub in ("full_imgs", "face_imgs", "mask"):
        d = os.path.join(dest, sub)
        fs = sorted(glob.glob(os.path.join(d, "*.*")))
        if not fs:
            continue
        n = len(fs)
        if n >= target:
            continue
        ext = os.path.splitext(os.path.basename(fs[0]))[1]
        width = max(len(os.path.splitext(os.path.basename(f))[0]) for f in fs)
        for i in range(n, target):
            shutil.copyfile(fs[i % n], os.path.join(d, f"{i:0{width}d}{ext}"))
        logger.info("%s: %s %d -> %d 帧", dest, sub, n, target)
    # 3) musetalk latents.pt（若存在，沿 dim0 循环补到 target）
    for lp in glob.glob(os.path.join(dest, "*.pt")):
        import torch
        t = torch.load(lp)
        if isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] < target:
            repeats = (target + t.shape[0] - 1) // t.shape[0]
            t = t.repeat(repeats, *([1] * (t.ndim - 1)))[:target]
            torch.save(t, lp)
            logger.info("%s: %s %d -> %d", dest, os.path.basename(lp), -1, target)


# ================= 绑定 manifest =================

def read_actions():
    """列出 data/actions/ 下所有动作及其绑定信息（供 GET /api/avatar/actions）。"""
    items = []
    if not os.path.isdir(SAVE_PATH):
        return items
    for aid in sorted(os.listdir(SAVE_PATH)):
        d = os.path.join(SAVE_PATH, aid)
        if not os.path.isdir(d):
            continue
        info = {}
        p = os.path.join(d, ACTION_INFO)
        if os.path.isfile(p):
            try:
                info = json.load(open(p, "r", encoding="utf-8"))
            except Exception:  # noqa: BLE001
                info = {}
        items.append({
            "action_id": aid,
            "bind_avatar": info.get("bind_avatar", ""),
            "model": info.get("model", ""),
            "source_video": info.get("source_video", ""),
            "has_coords": os.path.isfile(os.path.join(d, "coords.pkl")),
        })
    return items


# ================= 运行时派生查询（config.yaml 的 emotion 块已迁此） =================
# 动作/表情配置的唯一事实来源 = data/actions/<emotion>/action_info.json：
#   enabled（per-action 开关，缺省 True）、bind_avatar（空=全局）、source_video（供 prep 再生）。
# 渲染层 set_emotion 与 LLM 探测统一经这组 helper 派生候选，不再读 yaml。

def _read_manifest(action_id):
    """读单个动作的 manifest dict；缺失/损坏→{}（容忍 legacy 目录与临时损坏）。"""
    p = os.path.join(SAVE_PATH, action_id, ACTION_INFO)
    if not safe_action_token(action_id) or not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            info = json.load(f)
        return info if isinstance(info, dict) else {}
    except Exception:  # noqa: BLE001 - manifest 损坏按全局放行（容忍遗留）
        return {}


def _is_valid_emotion_dir(action_id):
    """该动作是否有可渲染底座：目录存在且 coords.pkl 在（wav2lip/musetalk 都会写）。"""
    return (
        safe_action_token(action_id)
        and os.path.isfile(os.path.join(SAVE_PATH, action_id, "coords.pkl"))
    )


def emotion_system_active(avatar_id=None):
    """总结经验系统是否启用：是否存在 ≥1 个对当前形象可用的表情底座。"""
    return bool(list_usable_emotions(avatar_id))


def list_usable_emotions(avatar_id=None):
    """返回当前说话形象可用的候选表情名（排过序，供 LLM 探测 + 渲染切换共用）。

    过滤规则（按当前形象）：
      - 有效 = data/actions/<em>/ 有底座 coords.pkl；
      - enabled = manifest.enabled 缺省 True；
      - 绑定 = manifest.bind_avatar 空（=全局，放行）；非空且 ≠ avatar_id → 剔除。
    """
    names = []
    if not os.path.isdir(SAVE_PATH):
        return names
    for aid in sorted(os.listdir(SAVE_PATH)):
        if not _is_valid_emotion_dir(aid):
            continue
        info = _read_manifest(aid)
        if not info.get("enabled", True):
            continue
        bind = (info.get("bind_avatar") or "").strip()
        if avatar_id and bind and bind != avatar_id:
            continue
        names.append(aid)
    return names


# ================= 管理（详情 / 预览 / 删除 / 重命名） =================

def safe_action_token(name):
    """动作 id 必须是单一路径分量，防路径穿越（../ 或含分隔符一律拒绝）。"""
    return (
        bool(name)
        and name not in (".", "..")
        and "/" not in name
        and "\\" not in name
    )


def get_action_detail(action_id):
    """返回单个动作的详细元数据；目录不存在返回 None（供 GET .../detail）。"""
    d = os.path.join(SAVE_PATH, action_id)
    if not safe_action_token(action_id) or not os.path.isdir(d):
        return None
    info = {}
    p = os.path.join(d, ACTION_INFO)
    if os.path.isfile(p):
        try:
            info = json.load(open(p, "r", encoding="utf-8"))
        except Exception:  # noqa: BLE001
            info = {}
    full_imgs = sorted(glob.glob(os.path.join(d, "full_imgs", "*.*")))
    frame_size = None
    if full_imgs:
        try:
            frame_size = Image.open(full_imgs[0]).size  # (w, h)
        except Exception:  # noqa: BLE001
            frame_size = None
    return {
        "action_id": action_id,
        "bind_avatar": info.get("bind_avatar", ""),
        "model": info.get("model", ""),
        "source_video": info.get("source_video", ""),
        "created_at": info.get("created_at", ""),
        "frame_count": len(full_imgs),
        "frame_size": frame_size,
        "has_coords": os.path.isfile(os.path.join(d, "coords.pkl")),
    }


def get_action_preview_bytes(action_id, max_side=360):
    """取动作首帧等比缩略为 PNG bytes（供 GET .../preview）；失败返回 None。"""
    d = os.path.join(SAVE_PATH, action_id)
    if not safe_action_token(action_id):
        return None
    full_imgs = sorted(glob.glob(os.path.join(d, "full_imgs", "*.*")))
    if not full_imgs:
        return None
    try:
        img = Image.open(full_imgs[0])
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            # Pillow 10 移除了 Image.LANCZOS 常量，用 Resampling 枚举（兼容旧版回退）
            resample = getattr(Image, "Resampling", Image).LANCZOS
            img = img.resize((int(w * scale), int(h * scale)), resample)
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        logger.exception("action preview failed: %s", action_id)
        return None


def delete_action(action_id):
    """删除 data/actions/<action_id>/（仅删目录，不改 config.yaml）。"""
    if not safe_action_token(action_id):
        return False, "非法的动作 id"
    d = os.path.join(SAVE_PATH, action_id)
    if not os.path.isdir(d):
        return False, f"动作 {action_id} 不存在"
    shutil.rmtree(d)
    logger.info("删除动作：%s", action_id)
    return True, "已删除"


def rename_action(action_id, new_id):
    """把 data/actions/<action_id>/ 改名为 <new_id>，并同步 manifest 里的 action_id。"""
    if not safe_action_token(action_id):
        return False, "非法的原动作 id"
    if not safe_action_token(new_id):
        return False, "新动作 id 非法国段（不能含 / \\ 或为空）"
    old = os.path.join(SAVE_PATH, action_id)
    new = os.path.join(SAVE_PATH, new_id)
    if not os.path.isdir(old):
        return False, f"动作 {action_id} 不存在"
    if os.path.isdir(new):
        return False, f"已存在动作 {new_id}"
    os.rename(old, new)
    # 同步 manifest 里的 action_id 字段
    p = os.path.join(new, ACTION_INFO)
    if os.path.isfile(p):
        try:
            info = json.load(open(p, "r", encoding="utf-8"))
            if info.get("action_id") != new_id:
                info["action_id"] = new_id
                json.dump(info, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            logger.exception("rename: 同步 manifest 失败 %s", new_id)
    logger.info("重命名动作：%s -> %s", action_id, new_id)
    return True, new_id


# ================= 生成入口 =================

def generate_action(model, action_id, video_path, bind_avatar_id,
                    device="cuda", progress_callback=None):
    """上传的样例视频 → data/actions/<action_id>/ 绑定到 bind_avatar_id 的表情底座。

    steps：清空旧目标 → 对齐人脸尺寸(wav2lip) → genavatar
    （save_path='data/actions') → 帧数补齐 → 写 action_info.json 绑定 manifest。
    进度与异常由 task_manager 的 progress_callback 上报，异常向上抛（任务记 failed）。
    """
    if model not in ("wav2lip", "musetalk"):
        raise ValueError(f"当前 model={model} 暂不支持出表情底座（支持 wav2lip/musetalk）")

    def _report(p):
        if progress_callback:
            progress_callback(int(p))

    # 1) 视频必须存在
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"样例视频不存在: {video_path}")

    dest = os.path.join(SAVE_PATH, action_id)
    _report(5)
    # 2) 清空旧目标，避免不同帧数覆盖残留旧帧
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)

    # 3) 调 genavatar 生成逐帧基地序列
    _report(10)
    if model == "wav2lip":
        import avatars.wav2lip.genavatar as G
        G.device = device
        G.generate_avatar(
            video_path=video_path, avatar_id=action_id, save_path=SAVE_PATH,
            img_size=_resolve_wav2lip_img_size(bind_avatar_id),
            pads=[0, 10, 0, 0], nosmooth=False, face_det_batch_size=4,
            progress_callback=lambda p: _report(10 + int(p * 0.8)),
        )
    else:  # musetalk
        from avatars.musetalk.genavatar import generate_avatar
        generate_avatar(
            video_path=video_path, avatar_id=action_id, save_path=SAVE_PATH,
            bbox_shift=0, extra_margin=10, parsing_mode="jaw", version="v15",
            progress_callback=lambda p: _report(10 + int(p * 0.8)),
        )
    _report(90)

    # 5) 帧数循环补齐到与中性形象一致（短视频如 sad 会被补长），防渲染线程越界
    _pad_cycle_length(dest, _neutral_frame_count(bind_avatar_id))
    _report(95)

    # 6) 写绑定 manifest（供运行时 set_emotion 强制校验）
    with open(os.path.join(dest, ACTION_INFO), "w", encoding="utf-8") as f:
        json.dump({
            "action_id": action_id,
            "bind_avatar": bind_avatar_id,
            "enabled": True,   # per-action 开关；false=该表情不作为候选
            "model": model,
            "source_video": video_path,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)
    _report(100)
    logger.info("动作 %s 生成完成，绑定头像 %s（model=%s）", action_id, bind_avatar_id, model)