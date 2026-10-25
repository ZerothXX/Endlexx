import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from filelock import FileLock
from PIL import Image, ImageFilter

from .assets import asset_path, store_image
from .config import ROOT, settings
from .db import Session, Task, Trace
from .providers import generate, parse_object, runtime_call


class NeedsClarification(ValueError):
    pass


class TaskCancelled(RuntimeError):
    pass


def propose_damage(image: Image.Image) -> list[dict]:
    """Pixel proposals are hypotheses, never permission to edit an image."""
    import cv2

    canvas = image.copy()
    canvas.thumbnail((1280, 1280))
    array = np.asarray(canvas.convert("RGB"))
    height, width = array.shape[:2]
    proposals = []
    for polarity, binary in [("light", array.min(axis=2) >= 245), ("dark", array.max(axis=2) <= 10)]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=4)
        for left, top, rw, rh, area in stats[1:count]:
            fraction = area / (width * height)
            if (
                left <= 1
                or top <= 1
                or left + rw >= width - 1
                or top + rh >= height - 1
                or not 0.0006 <= fraction <= 0.3
                or min(rw, rh) < 6
                or area / (rw * rh) < 0.72
            ):
                continue
            proposals.append(
                {
                    "box": [
                        round(left / width * 1000),
                        round(top / height * 1000),
                        round((left + rw) / width * 1000),
                        round((top + rh) / height * 1000),
                    ],
                    "polarity": polarity,
                    "area": int(area),
                    "rectangularity": round(float(area / (rw * rh)), 3),
                }
            )
    proposals.sort(key=lambda item: (item["rectangularity"], item["area"]), reverse=True)
    return [dict(item, id=i) for i, item in enumerate(proposals[:12])]


def check_cancel(task_id: str):
    with Session() as db:
        task = db.get(Task, task_id)
        if task and task.cancel_requested:
            raise TaskCancelled("任务已取消，原图保留")


def run_process(args: list[str], cwd: Path, task_id: str, log: Path, heartbeat=None):
    import os

    env = dict(
        os.environ,
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        HF_HOME=str(settings().data_dir / "hf-cache"),
    )
    started = time.monotonic()
    notified = started
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            while process.poll() is None:
                check_cancel(task_id)
                if time.monotonic() - started > settings().task_timeout:
                    raise TimeoutError("图像任务超过执行时限")
                if heartbeat and time.monotonic() - notified >= 15:
                    heartbeat(int(time.monotonic() - started))
                    notified = time.monotonic()
                time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(f"图像执行器退出码 {process.returncode}，可通过任务日志检查原因")
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False
                    )
                else:
                    process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def automatic_mask(image: Image.Image, diagnosis: dict) -> Image.Image:
    """Conservative flat-hole baseline, restricted by visual diagnosis.

    A VLM bounding box is NOT itself a mask. Only enclosed nearly uniform connected
    regions inside supported defect boxes are accepted. Other defects abstain.
    """
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    output = np.zeros((height, width), dtype=np.uint8)
    for candidate in diagnosis.get("regions", [])[:6]:
        if candidate.get("kind") not in {"hole", "flat_occlusion"}:
            continue
        box = candidate.get("box", [])
        if len(box) != 4 or not all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in box):
            continue
        left, top, right, bottom = [int(box[i] / 1000 * (width if i % 2 == 0 else height)) for i in range(4)]
        if right <= left or bottom <= top or (right - left) * (bottom - top) > width * height * 0.4:
            continue
        roi = array[top:bottom, left:right]
        # Quantized dominant flat colour is supported only if all local channels are stable.
        quantized = (roi // 8).reshape(-1, 3)
        colours, counts = np.unique(quantized, axis=0, return_counts=True)
        colour = colours[np.argmax(counts)].astype(float) * 8 + 3.5
        flat = np.max(np.abs(array.astype(float) - colour), axis=2) <= 9
        # Flood only from the interior of the proposed box; components touching image edges
        # are likely background. PIL floodfill avoids an OpenCV dependency in the API env.
        from PIL import ImageDraw

        raw = Image.fromarray((flat * 255).astype(np.uint8)).copy()
        seed_roi = flat[top:bottom, left:right]
        ys, xs = np.where(seed_roi)
        if not len(xs):
            continue
        centre = np.array([(bottom - top) / 2, (right - left) / 2])
        idx = np.argmin((ys - centre[0]) ** 2 + (xs - centre[1]) ** 2)
        seed = (int(xs[idx] + left), int(ys[idx] + top))
        ImageDraw.floodfill(raw, seed, 128, thresh=0)
        component = np.asarray(raw) == 128
        if component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any():
            continue
        count = int(component.sum())
        if not 16 <= count <= width * height * 0.3:
            continue
        overlap = int(component[top:bottom, left:right].sum()) / count
        if overlap < 0.75:
            continue
        output[component] = 255
    if not output.any():
        raise NeedsClarification(
            "暂时无法可靠定位可修复的局部缺损。请用一句话描述主要损坏的位置或外观，我会重新分析。"
        )
    mask = Image.fromarray(output).filter(ImageFilter.MaxFilter(3))
    if (np.asarray(mask) > 0).mean() > 0.35:
        raise NeedsClarification(
            "疑似缺损范围过大，目前无法可靠区分破损和原图内容，请补充主要需要修复的位置。"
        )
    return mask


def repair_image(task_id: str, asset_id: str, query: str, snapshot: dict, update) -> dict:
    cfg = settings()
    path = asset_path(asset_id)
    image = Image.open(path).convert("RGB")
    update("分析图像与缺损位置", 30)
    candidates = propose_damage(image)
    prompt = (
        "你是图像破损诊断模块。图片中检测到一些候选均匀色区域，但多数可能是正常画面，不能直接修复。"
        "任务是补全被遮挡或抹去的图像内容。人工白色矩形、黑色色块、涂抹擦除也属于可修复的局部缺损，"
        "不要求是真实物理损坏；当色块遮断主体的纹理或线条、且用户要求补全时应选择。"
        "正常背景、留白、衣服和配饰不能选择。"
        "坐标为0-1000归一化，box顺序为左上右下。只返回候选id，不要生成坐标。"
        '输出严格JSON：{"supported":true,"reason":"可见依据","confirmed_ids":[]}。'
        "把确定需要补全的候选编号放入confirmed_ids，不能确定则空列表。"
        "模糊、噪声、超分需求supported=false。\n候选：" + json.dumps(candidates) + "\n用户需求：" + query
    )
    diagnosis = parse_object(generate(snapshot, [{"role": "user", "content": prompt}], [path], 650, True))
    chosen = {item for item in diagnosis.get("confirmed_ids", []) if type(item) is int}
    diagnosis["regions"] = [
        dict(candidate, kind="hole") for candidate in candidates if candidate["id"] in chosen
    ]
    diagnosis["candidates"] = candidates
    with Session() as db:
        db.add(Trace(task_id=task_id, node="repair_diagnosis", data=diagnosis))
        db.commit()
    if not diagnosis.get("supported", True):
        raise NeedsClarification(
            "当前修复工具支持局部缺损补全，尚不支持本次需求：" + str(diagnosis.get("reason", ""))
        )
    check_cancel(task_id)
    update("自动定位缺损并校验区域", 45)
    try:
        mask = automatic_mask(image, diagnosis)
    except NeedsClarification:
        # One bounded re-observation; never ask the user to paint a mask.
        rechecked = parse_object(
            generate(
                snapshot,
                [
                    {
                        "role": "user",
                        "content": prompt
                        + "\n第一次区域定位未通过像素校验，请复查候选id；只选择真正被白色或黑色色块抹去的局部破洞。",
                    }
                ],
                [path],
                650,
                True,
            )
        )
        chosen = {item for item in rechecked.get("confirmed_ids", []) if type(item) is int}
        diagnosis = rechecked | {
            "regions": [
                dict(candidate, kind="hole") for candidate in candidates if candidate["id"] in chosen
            ],
            "candidates": candidates,
        }
        if not rechecked.get("supported", True):
            raise NeedsClarification("复查未确认可安全修复的局部缺损，请补充主要损坏位置的文字描述。")
        mask = automatic_mask(image, diagnosis)
    mask_asset = store_image(mask, "自动缺损区域.png", asset_id, "mask")
    folder = cfg.data_dir.resolve() / "tasks" / task_id
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"repaired_{asset_id}.png"
    report = folder / "repair_metrics.json"
    update("LaMa 正在补全局部缺损", 60)
    runtime_call("/unload")
    with FileLock(str(cfg.gpu_lock), timeout=cfg.task_timeout):
        check_cancel(task_id)
        run_process(
            [
                str(cfg.model_python),
                str(ROOT / "scripts/image_inference.py"),
                "--input",
                str(path),
                "--mask",
                str(asset_path(mask_asset["id"])),
                "--weights",
                str(cfg.lama_path.resolve()),
                "--output",
                str(output),
                "--report",
                str(report),
            ],
            ROOT,
            task_id,
            folder / "repair.log",
        )
    update("检查保护区与保存结果", 85)
    metrics = json.loads(report.read_text(encoding="utf-8"))
    if metrics["protected_pixels_changed"] != 0:
        raise RuntimeError("结果未通过原图保护区校验，未交付该结果")
    result = store_image(output, "修复结果.png", asset_id)
    return {
        "tool": "repair",
        "asset": result,
        "before": asset_id,
        "diagnosis": diagnosis,
        "mask_asset_id": mask_asset["id"],
        "metrics": metrics,
    }


def transfer_image(
    task_id: str,
    asset_id: str,
    package: str | None,
    update,
    runtime: str | None = None,
    expected_manifest: str | None = None,
) -> dict:
    import re

    cfg = settings()
    runtime = runtime or cfg.transfer_runtime
    if runtime not in {"balanced", "research"}:
        raise ValueError("未知迁移运行策略")
    if not package or not re.fullmatch(r"[\w-]{1,48}", package):
        raise NeedsClarification("当前角色尚未绑定可用的 Endless 角色包，请在角色档案中选择已发布的角色包。")
    manifest = cfg.endless_root / "output/characters" / package / "manifest.json"
    if not manifest.is_file():
        raise NeedsClarification("角色包尚未发布，请先在 Endless 完成训练和发布。")
    manifest_bytes = manifest.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_manifest and expected_manifest != manifest_hash:
        raise NeedsClarification("排队期间角色包版本发生变化，请重新提交以使用新版本。")
    package_data = json.loads(manifest_bytes)
    if not isinstance(package_data.get("weights"), list) or not package_data["weights"]:
        raise NeedsClarification("角色包缺少权重清单，请先在 Endless 重新发布。")
    character_root = manifest.parent.parent.resolve()
    for weight in package_data.get("weights", []):
        weight_path = (character_root / weight["file"]).resolve()
        if not weight_path.is_relative_to(character_root) or not weight_path.is_file():
            raise NeedsClarification("角色权重文件缺失或路径无效，请重新发布角色包。")
        if weight.get("sha256"):
            with weight_path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != weight["sha256"]:
                raise NeedsClarification("角色权重与发布清单不一致，请先在 Endless 重新发布。")
    folder = cfg.data_dir.resolve() / "tasks" / task_id
    folder.mkdir(parents=True, exist_ok=True)
    update("调用 Endless 角色迁移", 50)
    runtime_call("/unload")
    source = asset_path(asset_id)
    output = cfg.endless_root / "output/result" / f"{package}_result" / f"{package}_{source.stem}_result.png"
    with FileLock(str(cfg.gpu_lock), timeout=cfg.task_timeout):
        previous_stamp = output.stat().st_mtime_ns if output.is_file() else None
        run_process(
            [
                str(cfg.model_python),
                str(ROOT / "scripts/endless_inference.py"),
                "--root",
                str(cfg.endless_root.resolve()),
                "--runtime",
                runtime,
                "--character",
                package,
                "--input",
                str(source),
            ],
            cfg.endless_root.resolve(),
            task_id,
            folder / "transfer.log",
            heartbeat=lambda elapsed: update(
                f"Endless 迁移执行中，已用 {elapsed // 60} 分 {elapsed % 60} 秒", 50
            ),
        )
        if not output.is_file() or output.stat().st_mtime_ns == previous_stamp:
            raise RuntimeError("Endless 没有返回本次任务的新结果文件，原有结果不会冒充本次输出")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_hash:
        raise RuntimeError("迁移执行期间角色包发生变化，未交付不明版本的结果")
    result = store_image(output, f"{package} 迁移结果.png", asset_id)
    return {
        "tool": "transfer",
        "asset": result,
        "before": asset_id,
        "package": package,
        "runtime": runtime,
        "package_version": package_data.get("created_at"),
        "manifest_sha256": manifest_hash,
        "weights": [
            {"role": w.get("role"), "sha256": w.get("sha256")} for w in package_data.get("weights", [])
        ],
    }
