import os
import uuid
from aiohttp import web
from server.task_manager import task_manager
from server.responses import json_ok, json_error
from server.action_prep import (
    read_actions,
    get_action_detail,
    get_action_preview_bytes,
    delete_action,
    rename_action,
)
from utils.logger import logger

async def create_avatar_task(request):
    """
    POST /api/avatar/task
    Parameters: model, avatar_id, video_file (upload), video_path (local), ...
    """
    try:
        if request.content_type == 'multipart/form-data':
            reader = await request.multipart()
            params = {}
            video_path = None

            while True:
                part = await reader.next()
                if part is None:
                    break

                if part.name == 'video_file':
                    filename = part.filename
                    temp_dir = os.path.abspath('./data/tmp')
                    os.makedirs(temp_dir, exist_ok=True)
                    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
                    with open(video_path, 'wb') as f:
                        while True:
                            chunk = await part.read_chunk()
                            if not chunk:
                                break
                            f.write(chunk)
                else:
                    value = await part.text()
                    params[part.name] = value

            if video_path:
                params['video_path'] = video_path
        else:
            params = await request.json()

        model_type = params.get('model')
        avatar_id = params.get('avatar_id')

        if not model_type or not avatar_id:
            return json_error("model and avatar_id are required")

        if 'video_path' not in params:
            return json_error("video_file or video_path is required")

        data_path = './data/avatars'

        video_path = params['video_path']
        if not os.path.isabs(video_path):
            video_path = os.path.join(data_path, video_path)

        save_path = data_path

        task_params = {
            "video_path": video_path,
            "save_path": save_path,
            "img_size": int(params.get('img_size', 256)),
            "nosmooth": params.get('nosmooth', 'false').lower() == 'true' if isinstance(params.get('nosmooth'), str) else params.get('nosmooth', False),
            "bbox_shift": int(params.get('bbox_shift', 0)),
            "extra_margin": int(params.get('extra_margin', 10)),
            "parsing_mode": params.get('parsing_mode', 'jaw'),
            "version": params.get('version', 'v15'),
            "face_det_batch_size": int(params.get('face_det_batch_size', 16))
        }

        pads_str = params.get('pads', "0 10 0 0")
        if isinstance(pads_str, str):
            task_params['pads'] = [int(x) for x in pads_str.split()]
        else:
            task_params['pads'] = pads_str

        task_id_input = params.get('task_id')
        notify_url = params.get('notifyurl')

        task_id = task_manager.add_task(model_type, avatar_id, task_params, task_id=task_id_input, notify_url=notify_url)
        return json_ok(data={"task_id": task_id})

    except Exception as e:
        logger.exception("create_avatar_task error:")
        return json_error(str(e))

async def get_avatar_task_status(request):
    """
    GET /api/avatar/task/{task_id}
    """
    task_id = request.match_info.get('task_id')
    task = task_manager.get_task(task_id)
    if not task:
        return json_error("Task not found", code=404)

    return json_ok(data=task.to_dict())

async def list_avatar_tasks(request):
    """
    GET /api/avatar/tasks
    """
    tasks = task_manager.list_tasks()
    return json_ok(data={"tasks": tasks})

async def delete_avatar_task(request):
    """
    DELETE /api/avatar/task/{task_id}
    """
    task_id = request.match_info.get('task_id')
    success, msg = task_manager.delete_task(task_id)
    if not success:
        return json_error(msg)
    return json_ok(data={"msg": msg})

async def create_action_task(request):
    """
    POST /api/avatar/action/task
    Parameters: model, action_id (表情名), target_avatar (绑定目标头像), video_file (upload)
    上传样例视频 → 生成 data/actions/<action_id>/ 并写入绑定 manifest（action_info.json）。
    """
    try:
        if request.content_type == 'multipart/form-data':
            reader = await request.multipart()
            params = {}
            video_path = None

            while True:
                part = await reader.next()
                if part is None:
                    break

                if part.name == 'video_file':
                    filename = part.filename
                    temp_dir = os.path.abspath('./data/tmp')
                    os.makedirs(temp_dir, exist_ok=True)
                    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
                    with open(video_path, 'wb') as f:
                        while True:
                            chunk = await part.read_chunk()
                            if not chunk:
                                break
                            f.write(chunk)
                else:
                    value = await part.text()
                    params[part.name] = value

            if video_path:
                params['video_path'] = video_path
        else:
            params = await request.json()

        model_type = params.get('model')
        action_id = params.get('action_id')
        bind_avatar = params.get('target_avatar')

        if not model_type or not action_id or not bind_avatar:
            return json_error("model, action_id and target_avatar are required")

        if model_type not in ('wav2lip', 'musetalk'):
            return json_error("动作生成仅支持 wav2lip / musetalk")

        if 'video_path' not in params:
            return json_error("video_file or video_path is required")

        video_path = params['video_path']

        task_params = {
            "video_path": video_path,
            "bind_avatar": bind_avatar,
            "device": params.get('device', 'cuda'),
        }

        task_id = task_manager.add_task(model_type, action_id, task_params,
                                       kind='action')
        return json_ok(data={"task_id": task_id})

    except Exception as e:
        logger.exception("create_action_task error:")
        return json_error(str(e))

async def list_actions(request):
    """
    GET /api/avatar/actions  — 列出所有动作及其绑定头像。

    在此把「正在生成的动作」标注出来（generating / gen_bind / gen_progress）：
    生成中底座 coords.pkl 与 manifest 都还没写，若只按目录判会误报「底座缺失」。
    """
    actions = read_actions()
    inflight = task_manager.active_action_tasks()
    for a in actions:
        t = inflight.get(a["action_id"])
        if not t:
            continue
        a["generating"] = True
        a["gen_status"] = t["status"]
        a["gen_progress"] = t["progress"]
        a["gen_bind"] = t["bind_avatar"]
    return json_ok(data={"actions": actions})

async def list_avatars(request):
    """
    GET /api/avatar/avatars — 列出 data/avatars/ 下已有头像（供「绑定到」下拉）。
    """
    avatars = []
    root = './data/avatars'
    if os.path.isdir(root):
        avatars = sorted(
            name for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )
    return json_ok(data={"avatars": avatars})

async def get_action_detail_handler(request):
    """
    GET /api/avatar/action/{action_id}/detail — 单个动作的详细元数据。
    """
    action_id = request.match_info.get('action_id')
    detail = get_action_detail(action_id)
    if not detail:
        return json_error(f"动作 {action_id} 不存在", code=404)
    return json_ok(data=detail)

async def get_action_preview_handler(request):
    """
    GET /api/avatar/action/{action_id}/preview — 动作首帧缩略 PNG。
    """
    action_id = request.match_info.get('action_id')
    png = get_action_preview_bytes(action_id)
    if not png:
        return web.Response(status=404, text="preview not found")
    return web.Response(body=png, content_type='image/png')

async def delete_action_handler(request):
    """
    DELETE /api/avatar/action/{action_id} — 删除动作目录（仅删目录，不改 config.yaml）。
    """
    action_id = request.match_info.get('action_id')
    success, msg = delete_action(action_id)
    if not success:
        return json_error(msg)
    return json_ok(data={"msg": msg})

async def rename_action_handler(request):
    """
    POST /api/avatar/action/{action_id}/rename  body={new_id} — 重命名动作 ID。
    """
    action_id = request.match_info.get('action_id')
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        return json_error("JSON body required")
    new_id = (data.get('new_id') or '').strip()
    if not new_id:
        return json_error("new_id is required")
    success, msg = rename_action(action_id, new_id)
    if not success:
        return json_error(msg)
    return json_ok(data={"action_id": new_id})

def setup_avatar_routes(app):
    app.router.add_post("/api/avatar/task", create_avatar_task)
    app.router.add_post("/api/avatar/action/task", create_action_task)
    app.router.add_get("/api/avatar/task/{task_id}", get_avatar_task_status)
    app.router.add_delete("/api/avatar/task/{task_id}", delete_avatar_task)
    app.router.add_get("/api/avatar/tasks", list_avatar_tasks)
    app.router.add_get("/api/avatar/actions", list_actions)
    app.router.add_get("/api/avatar/avatars", list_avatars)
    app.router.add_get("/api/avatar/action/{action_id}/detail", get_action_detail_handler)
    app.router.add_get("/api/avatar/action/{action_id}/preview", get_action_preview_handler)
    app.router.add_delete("/api/avatar/action/{action_id}", delete_action_handler)
    app.router.add_post("/api/avatar/action/{action_id}/rename", rename_action_handler)
