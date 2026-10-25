# -*- coding: utf-8 -*-
"""把"标志承载计划"**确定性地**落到目标主体上（规格 §6 的执行侧）。

与 `vision/mark_carry.py` 的分工：那边是像素操作，这边负责
"选哪些标志、挑哪张素材、放在哪个区域、用哪种承载方式、留下什么审计"。

关键设计（每条都有实测依据，见 `mark_carry.py` 顶部说明）：
* 只用**有真实像素证据**的素材；没有素材的标志不做，并如实报告原因。
* 位置完全来自**目标主体自己的掩版/框**，不复制原图的身体坐标（§6 禁止）。
* 承载方式按标志类别选择：emblem 居中适配透视、pattern 平铺、trim 沿边、
  wearable 缩小放到主体下部、palette 交给扩散（本模块不处理）。
* 每种承载都记录"用了哪个素材、放在哪、铺了几次"，便于复核与回归测试。
"""
import os

import numpy as np
import cv2
from PIL import Image

import config
import image_utils
from character import marks as marks_module
from vision import mark_carry
from vision import mark_style


def refine_asset_shape(rgba):
    """对**形状分离之后仍被判为不合格**的素材做一次"只保留标志本体"的抢救。

    为什么需要（实测）：`白色樱花图案` 的三张候选裁剪全部被下面的闸门挡掉——
    前景填充率 0.11（判为零散碎块）、50x67 过小、前景 0.68/0.85（判为插画）。
    结果是这只杯子上**一个樱花都没有**（日志写着"没有可用素材"），
    而樱花正是用户在非人物效果里最先点名要看的东西。

    根因不在闸门（闸门挡掉插画是对的），而在"形状分离"这一步：
    细线描图案在颜色判据下会被打散成零散碎块，于是要么填充率过低、
    要么为了保住它们而把背景一起留下（前景占比过高）。

    这里补一条**只用素材自身统计量**的抢救路径：
      1. 取 alpha 的连通块，只保留"面积够大 **且** 外接框填充率够高"的成块前景；
      2. 按保留下来的连通块**重新外接裁剪**——四周零散点被丢掉，外接框因此更紧，
         填充率与前景占比都会回升到闸门之内。

    返回 (新的 rgba 或 None, 说明)。`None` 表示抢救无效，调用方应继续判为不合格。
    """
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        return None, "素材为空或没有 alpha 通道"
    import cv2
    alpha = rgba[:, :, 3]
    binary = (alpha > 96).astype(np.uint8)
    if int(binary.sum()) < 24:
        return None, "前景像素过少，抢救无依据"
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return None, "没有可分离的连通块"
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    keep = np.zeros_like(binary, dtype=bool)
    for index, area in enumerate(areas, start=1):
        if area < max(24, largest * 0.25):
            continue
        box_w = int(stats[index, cv2.CC_STAT_WIDTH])
        box_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if float(area) / max(1.0, float(box_w * box_h)) < 0.25:
            continue                  # 细长、散乱的一缕 -> 不是成块的标志
        keep |= (labels == index)
    if int(keep.sum()) < 24:
        return None, "没有成块的连通块（判为细碎线条，不是标志本体）"
    rows = np.where(keep.any(axis=1))[0]
    cols = np.where(keep.any(axis=0))[0]
    pad = 2
    top = max(0, int(rows[0]) - pad)
    bottom = min(rgba.shape[0], int(rows[-1]) + 1 + pad)
    left = max(0, int(cols[0]) - pad)
    right = min(rgba.shape[1], int(cols[-1]) + 1 + pad)
    cropped = rgba[top:bottom, left:right].copy()
    cropped[:, :, 3] = np.where(keep[top:bottom, left:right],
                                cropped[:, :, 3], 0).astype(np.uint8)
    height, width = cropped.shape[:2]
    coverage = float((cropped[:, :, 3] > 128).mean())
    fill = _foreground_bbox_fill(cropped[:, :, 3].astype(np.float32) / 255.0)
    return cropped, (f"抢救出成块前景：{width}x{height}，前景占比 {coverage:.2f}，"
                     f"外接框填充率 {fill if fill is None else round(fill, 2)}")


def asset_purity(rgba):
    """判断一张素材"像不像一个标志"，返回 (是否可用, 说明)。

    为什么需要这道闸（本轮实测的**最严重异常**）：
    `mark_refs.txt` 指向的其实是**整张插画**（用户说明里这些是"保存在 images 内的
    单独象征标志"，也就是该标志的**样例图**，不是抠好的贴图）。
    直接把参考图当作承载素材，会把**整幅角色立绘**平铺到器物上——
    实测出图上出现了九宫格的完整角色插画，这是最明显的"异常"。
    旧的"背景透传"形状分离对插画也无效（`ref_006.png` 抠完仍剩 68% 前景）。

    判据只有图像自身的统计量，不依赖语义：
      * 前景占比过高 -> 内容几乎铺满整张图，那是**插画**而不是孤立标志；
      * 素材长宽比极端（很扁/很长）-> 多半是带状局部，不适合当纹样单元。
    另外**尺寸过小**的素材也不要（放大后全是模糊）。
    """
    if rgba is None:
        return False, "素材为空"
    height, width = rgba.shape[:2]
    if min(height, width) < 16:
        return False, f"素材过小 {width}x{height}"
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    coverage = float((alpha > 0.5).mean())
    limit = float(getattr(config, "MARK_ASSET_MAX_FOREGROUND", 0.55))
    if coverage > limit:
        return False, (f"前景占比 {coverage:.2f} > {limit:.2f}，"
                       f"内容铺满整幅，判为**插画**而不是孤立标志")
    # **前景过长条**：实测逐图裁剪有时会圈到"一缕头发 / 一条衣边"，
    # 这类素材铺上去就是几条乱线（出图上出现一排弯线，完全看不出是标志）。
    # 真正的纹样/徽记接近成块，长宽比不会极端。
    ratio_limit = float(getattr(config, "MARK_ASSET_MAX_ASPECT", 2.2))
    ratio = max(width, height) / max(1, min(width, height))
    if ratio > ratio_limit:
        return False, (f"前景长宽比 {ratio:.1f} > {ratio_limit:.1f}，"
                       f"判为条状局部（一缕头发/一条衣边），不是标志")
    # 前景像素本身的形状也要成块：用前景外接框的填充率判断
    crop_limit = float(getattr(config, "MARK_ASSET_MIN_FILL", 0.30))
    filled = _foreground_bbox_fill(alpha)
    if filled is not None and filled < crop_limit:
        return False, (f"前景填充率 {filled:.2f} < {crop_limit:.2f}，"
                       f"判为零散碎块，不是标志")
    return True, f"前景占比 {coverage:.2f} 长宽比 {ratio:.1f}"


def _foreground_bbox_fill(alpha, threshold=0.5):
    """前景像素在其**外接框**内的填充率；没有前景时返回 None。"""
    binary = alpha > threshold
    if not binary.any():
        return None
    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]
    box = binary[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    if box.size == 0:
        return None
    return float(box.mean())


def _asset_for(folder, mark, observations):
    """为该标志挑一张**可用的**真实素材，返回 (路径, 说明)。

    素材来源（按可行性排序，**不再无条件优先参考图**）：

      1. **逐图真实裁剪**：从角色图里按定位框裁出的该标志实例。
         这是"标志在角色身上的真实像素"，也是本项目已有证据层最直接的产物。
      2. `mark_refs.txt` 指向的参考图：**只有通过 `asset_purity` 闸门时**才用。
         实测这些是整张插画，未经检验就使用会把整幅立绘平铺到器物上（见 `asset_purity`）。

    两个来源都会先过 `asset_purity`，都不合格就如实返回"没有可用素材"——
    **宁可不做，也不用插画顶替**（这是"异常"的直接来源）。
    """
    library = marks_module.load_library(folder) or {}
    candidates = []          # [(路径, 说明, 优先级)]

    # 0) `mark_refs.txt` 指向的参考图**优先**，但**必须通过闸门**。
    #
    # 为什么参考图优先（本轮定案，三轮反复后的结论）：
    #   * 用户的语义里，`mark_refs.txt` 就是这个标志的**样例图**
    #     （"保存在 images 内的单独象征标志"），`021.png` 是白底 + 黑色樱花，
    #     `022.png` 是金色徽记——干净、平面、正是"一眼可辨"要的素材；
    #   * 但同一个文件里也有**整张插画**（`006.jpg` 是角色立绘），
    #     未经检验使用会把整幅立绘平铺到器物上（实测出现过九宫格立绘）。
    # 因此：**优先 + 过闸**。合格的参考图（实测黑樱花 0.51、金色图案 0.26）直接用；
    # 不合格的（立绘 0.68/0.69、扇子整图）退回逐图裁剪。
    for link in (library.get("links") or []):
        if link.get("mark_id") != mark.get("mark_id"):
            continue
        for reference in (library.get("references") or []):
            if reference.get("rel_path") != link.get("rel_path"):
                continue
            path = reference.get("crop_path") or reference.get("ref_crop_path")
            if not path:
                stem = os.path.splitext(os.path.basename(
                    str(reference.get("resolved_rel_path") or "")))[0]
                candidate = os.path.join(config.OUTPUT_DIR, "marks", folder,
                                         f"ref_{stem}.png")
                path = candidate if os.path.isfile(candidate) else None
            if path:
                full = path if os.path.isabs(path) else os.path.join(
                    config.PROJECT_ROOT, path)
                if os.path.isfile(full):
                    candidates.append(
                        (full, f"mark_refs 参考图（{reference.get('description')}）", 0))
            break
        break

    # 1) 逐图真实裁剪：从角色图里按定位框裁出的该标志实例
    for row in observations or []:
        if row.get("mark_id") != mark.get("mark_id"):
            continue
        crop = row.get("crop_path")
        if not crop:
            continue
        path = os.path.join(config.PROJECT_ROOT, crop)
        if not os.path.isfile(path):
            continue
        candidates.append((path, f"逐图裁剪（{row.get('rel_path')}）", 1))

    rejected = []
    for path, note, _priority in candidates:
        try:
            asset, load_note = mark_carry.load_asset(path)
        except Exception as exc:      # noqa: BLE001
            rejected.append(f"{os.path.basename(path)}: 读取失败 {type(exc).__name__}")
            continue
        if asset is None:
            rejected.append(f"{os.path.basename(path)}: {load_note}")
            continue
        # 形状分离：把非标志的背景透掉（对干净图案有效）
        if getattr(config, "MARK_SHAPE_ENABLED", True):
            try:
                from vision import mark_shape
                shaped, _shape_audit = mark_shape.extract_shape(path, sam=None)
                if shaped is not None:
                    asset = shaped
            except Exception:      # noqa: BLE001
                pass
        ok, purity = asset_purity(asset)
        if not ok:
            # 抢救：只保留成块的标志本体、按连通块重新外接裁剪，再判一次。
            # 这一条是为"细线描图案被颜色判据打散"准备的（实测：白色樱花图案
            # 多张候选被打散成零散碎块）。抢救**不放松闸门**：
            # 插画（前景占比过高）救不回来，只是把"碎块"问题修好。
            salvaged, salvage_note = refine_asset_shape(asset)
            if salvaged is not None:
                retry_ok, retry_purity = asset_purity(salvaged)
                if retry_ok:
                    asset, ok, purity = salvaged, True, f"{retry_purity}；{salvage_note}"
                else:
                    purity = f"{purity}；抢救后仍不合格（{retry_purity}）"
            else:
                purity = f"{purity}；抢救无效（{salvage_note}）"
        if ok:
            # **能不能"印"上去**是另一道判据（与"像不像标志"互补）：
            # 形状必须真的是轮廓，不能是一整块近似矩形的东西——否则贴出来就是
            # 一块带直角边的方片（实测：`crop_009` 在杯子上变成三块方形贴片）。
            usable, usable_note = _asset_printable(asset)
            if not usable:
                rejected.append(f"{os.path.basename(path)}: {purity}；{usable_note}")
                continue
            # **优先用"印得清楚"的素材**（实测缺陷的直接对策）：
            # `crop_009`（50x67）虽然通过了全部闸门，但它在杯子上只能放到 60 像素左右，
            # 而且内容本身是"平坦蓝底 + 一块金属斜边"——印出来就是方形贴片。
            # 同一个角色库里有一枚 1349x1248 的干净樱花徽记，明显更适合。
            # 因此：素材小到"只能勉强印"的时候，先在库里找一份更好的；
            # 找不到才用它自己（不因为小就否决，那会让杯子上什么都没有）。
            short_side_now = int(min(asset.shape[:2]))
            prefer_limit = int(float(getattr(config, "MARK_SUBSTITUTE_MIN_SIDE", 160)) * 0.6)
            if short_side_now < prefer_limit:
                substitute = _substitute_asset(folder, mark, observations,
                                               min_side=prefer_limit)
                if substitute:
                    sub_path, sub_note, sub_cap = substitute
                    print(f"[carry] 标志「{mark.get('name')}」自己的素材偏小"
                          f"（{short_side_now}px），改用库内更清晰的素材: "
                          f"{os.path.basename(sub_path)}")
                    return sub_path, sub_note, sub_cap
            # **分辨率下限不能按绝对像素一刀切**（实测改了）：
            # 原判据是"素材短边 < MARK_ASSET_MIN_SIDE_PX(96) 就否决"，
            # 于是 50x67 的素材被判"放大到糊"直接丢掉。
            # 问题不在素材小，而在**承载时把它放得太大**：
            # 50 像素的素材放到 170 像素才是 3.4 倍放大，那当然糊；
            # 放到 60 像素只有 1.2 倍，清晰可辨。
            # 因此改成**按比值判定**：只否决"必然要放大超过上限"的素材，
            # 并把允许的显示尺寸如实带出去（由调用方按它限制单元大小）。
            height, width = asset.shape[:2]
            short_side = int(min(height, width))
            max_up = float(getattr(config, "MARK_MAX_UPSCALE", 12.0))
            if short_side <= 0:
                rejected.append(f"{os.path.basename(path)}: 素材尺寸非法")
                continue
            scale_cap = min(1.0, float(short_side) * max_up /
                            max(1.0, float(getattr(config, "MARK_ASSET_MIN_SIDE_PX", 96))))
            note = f"{note}；{purity}；{usable_note}"
            if scale_cap < 1.0:
                note += f"（尺寸上限按 {scale_cap:.2f} 收窄，避免放大发糊）"
            return path, note, scale_cap
        rejected.append(f"{os.path.basename(path)}: {purity}")
    if rejected:
        print(f"[carry] 标志「{mark.get('name')}」没有可用素材，已跳过:")
        for item in rejected[:4]:
            print(f"[carry]   - {item}")
    # 该标志自己的素材全不可用时，**从同一个角色库里另找一份"印得上去"的素材**。
    #
    # 为什么允许这样做（实测依据）：`白色樱花图案` 的四张候选全部不可用——
    # 三张是插画、一张（`crop_009`）是"平坦蓝底 + 金属斜边"，印上去就是方形贴片；
    # 而同一个角色的 `ref_021.png` 恰好是一枚**干净的樱花徽记**（白底黑花，
    # 形状分离正确），只是它被 `mark_refs.txt` 归到了另一条标志名下。
    # 用户的要求是"非人物上要一眼看得出是这个角色的标志"，
    # 因此"因为归档归属而让杯子上一个樱花都没有"是更差的结果。
    # 这里**仍然只使用有真实像素证据的素材**（不生成、不用通用花纹），
    # 并且把替换关系如实写进审计，便于事后核对。
    substitute = _substitute_asset(folder, mark, observations)
    if substitute:
        path, substitute_note, scale_cap = substitute
        print(f"[carry] 标志「{mark.get('name')}」改用库内可印素材: "
              f"{os.path.basename(path)}（{substitute_note}）")
        return path, substitute_note, scale_cap
    return None, "没有可用素材（已排除插画/过小/极端长宽比；不做，不用通用花纹顶替）", 1.0


def _asset_printable(asset):
    """融合侧"能不能印上去"的判据（见 `vision.surface_fusion.asset_usable`）。"""
    try:
        from vision import surface_fusion
        return surface_fusion.asset_usable(asset)
    except Exception as exc:              # noqa: BLE001
        return True, f"可印性判据不可用（{type(exc).__name__}）"


def _substitute_asset(folder, mark, observations, min_side=None):
    """在**同一个角色的标志库**里另找一份能印的素材，返回 (路径, 说明, 尺寸上限)。

    候选来源两类，都要过同一组闸门（像不像标志 + 能不能印 + 够不够清晰）：
      * 该角色**其它标志**的独立标志参考图（`mark_refs.txt` 指向的那些，通常最干净）；
      * 该角色**其它标志**的逐图真实裁剪。
    只使用有真实像素证据的素材，不生成、不用通用花纹。
    """
    if not getattr(config, "MARK_SUBSTITUTE_ASSET", True):
        return None
    need = int(getattr(config, "MARK_SUBSTITUTE_MIN_SIDE", 160)) \
        if min_side is None else int(min_side)
    library = marks_module.load_library(folder) or {}
    candidates = []          # 参考图优先（通常是干净的单独标志图）
    for reference in (library.get("references") or []):
        stem = os.path.splitext(os.path.basename(
            str(reference.get("resolved_rel_path") or "")))[0]
        if not stem:
            continue
        for candidate in (reference.get("crop_path"), reference.get("ref_crop_path"),
                          os.path.join(config.OUTPUT_DIR, "marks", folder,
                                       f"ref_{stem}.png")):
            if not candidate:
                continue
            full = candidate if os.path.isabs(candidate) else \
                os.path.join(config.PROJECT_ROOT, candidate)
            if os.path.isfile(full):
                candidates.append(full)
                break
    for row in observations or []:
        if row.get("mark_id") == mark.get("mark_id"):
            continue
        crop = row.get("crop_path")
        if not crop:
            continue
        full = os.path.join(config.PROJECT_ROOT, crop)
        if os.path.isfile(full):
            candidates.append(full)
    tried = set()
    for path in candidates:
        if path in tried:
            continue
        tried.add(path)
        try:
            from vision import mark_shape
            shaped, _audit = mark_shape.extract_shape(path, sam=None)
        except Exception:                 # noqa: BLE001
            continue
        if shaped is None:
            continue
        ok, purity = asset_purity(shaped)
        if not ok:
            continue
        usable, usable_note = _asset_printable(shaped)
        if not usable:
            continue
        height, width = shaped.shape[:2]
        if int(min(height, width)) < need:
            continue
        owner = ""
        for other in (library.get("marks") or []):
            for row in (observations or []):
                if row.get("mark_id") == other.get("mark_id") and \
                        str(row.get("crop_path") or "").endswith(os.path.basename(path)):
                    owner = f"来自标志「{other.get('name')}」"
        if not owner:
            owner = "来自角色独立标志参考图"
        return (path,
                f"代用素材（{owner}；原标志自己的素材不可印或过小）：{purity}；{usable_note}",
                1.0)
    return None


def _place_emblem(canvas, mark, asset, region_mask, slot=0, slots=1, scale_cap=1.0):
    """徽记：适配到主体可见面。

    `slot/slots`：多个徽记要**分散摆放**，不能都落在区域中心——
    实测把两个徽记都放在中心时，它们互相叠在一起，而且后铺的花纹正好压在中心，
    结果"先放徽记"这个顺序调整等于白做（杯子中央的金色图案与扇子糊成一团）。

    `scale_cap`：素材**允许的显示尺寸上限**（来自 `_asset_for` 的放大倍率判定）。
    小素材被放得太大才是"糊"的真正原因，因此这里按上限收窄，
    而不是像旧判据那样在小素材上直接否决整个标志。
    """
    rows = np.where(region_mask.any(axis=1))[0]
    cols = np.where(region_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return canvas, None
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    # 占可见面短边的一半左右：太小看不出，太大会糊满整个主体
    ratio = float(config.MARK_CARRY_EMBLEM_RATIO) * float(scale_cap)
    side = max(16, int(min(right - left, bottom - top) * ratio))
    # 纵向分散：单个居中；多个按等分线摆放（0.5 -> 0.33/0.67 -> 0.25/0.5/0.75 …）
    fraction = 0.5 if slots <= 1 else (slot + 1) / (slots + 1)
    centre_x = (left + right) // 2
    centre_y = top + int((bottom - top) * fraction)
    box = (centre_x - side // 2, centre_y - side // 2,
           centre_x + side // 2, centre_y + side // 2)
    resized, position = mark_carry.fit_to_box(asset, box, canvas.shape[1::-1])
    canvas, fusion = _apply_patch(canvas, resized, position, region_mask)
    return canvas, {
        "mode": "emblem", "slot": f"{slot + 1}/{slots}",
        "box": [int(v) for v in box],
        "size": [int(resized.shape[1]), int(resized.shape[0])],
        "scale_cap": round(float(scale_cap), 3),
        "fusion": fusion}


def _place_wearable(canvas, mark, asset, region_mask, scale_cap=1.0):
    """可佩戴件：缩小放在主体**下部**（像挂在一件器物上），并保持比例。"""
    rows = np.where(region_mask.any(axis=1))[0]
    cols = np.where(region_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return canvas, None
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    width = int((right - left) * float(config.MARK_CARRY_WEARABLE_RATIO)
                * float(scale_cap))
    width = max(16, width)
    centre_x = (left + right) // 2
    centre_y = bottom - int((bottom - top) * 0.28)
    box = (centre_x - width // 2, centre_y - width // 2,
           centre_x + width // 2, centre_y + width // 2)
    resized, position = mark_carry.fit_to_box(asset, box, canvas.shape[1::-1])
    canvas, fusion = _apply_patch(canvas, resized, position, region_mask)
    return canvas, {
        "mode": "wearable", "box": [int(v) for v in box],
        "size": [int(resized.shape[1]), int(resized.shape[0])],
        "scale_cap": round(float(scale_cap), 3),
        "fusion": fusion}


def _apply_patch(canvas, rgba, position, region_mask):
    """把一块素材落到画布上：**优先走承印融合**，不可用时退回 alpha 合成。

    为什么必须优先走融合（用户判据"不是简单粘贴"）：
    `mark_carry.paste_rgba` 是 alpha 合成——**只能改颜色、改不了光照关系**，
    素材自带原服装的受光、器物自身的纹理被整块盖掉、边界留一圈硬框。
    `vision/surface_fusion` 把"图案提供颜色与花纹、器物提供明暗与材质"这件事
    用确定性运算做掉，实测（同素材同足迹对照）：
      * 明暗跟随斜率 0.40 → **1.00**（跟随器物受光）；
      * 边界硬框 14.7 → **2.8**（不再是一块贴纸）；
      * 器物纹理保留 10.1 → **8.3**（更接近 1）。
    融合拒绝执行时（素材没有可用的形状 alpha）**如实回退并记录原因**，
    不静默降级、也不假装用了融合。
    """
    if not getattr(config, "MARK_FUSION_ENABLED", True):
        return mark_carry.paste_rgba(canvas, rgba, position), {"applied": False,
                                                               "reason": "MARK_FUSION_ENABLED=False"}
    try:
        from vision import surface_fusion
    except Exception as exc:                      # noqa: BLE001
        return mark_carry.paste_rgba(canvas, rgba, position), {
            "applied": False, "reason": f"融合模块不可用: {type(exc).__name__}"}
    x, y = position
    height, width = rgba.shape[:2]
    box = (x, y, x + width, y + height)
    fused, audit = surface_fusion.fuse_onto_surface(rgba, canvas, box,
                                                    region_mask=region_mask)
    if audit.get("applied"):
        return fused, {k: v for k, v in audit.items() if k != "applied"}
    return mark_carry.paste_rgba(canvas, rgba, position), {
        "applied": False, "reason": audit.get("reason")}


def _lab(rgb_array):
    import cv2
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB).astype(np.float32)


def _rim_row(image, left, right, search_bottom):
    """粗定位"器物上沿"（杯口/罐口）所在的行：横向强边最强的行。

    为什么要单独找它（实测）：可承载区域是按"亮 + 低饱和 + 平滑"判据求出的，
    而**杯口内壁与杯中液体**同样满足这些条件，于是区域上沿会越过器物口沿
    （实测区域 y 从 268 起、而杯口上沿在 y≈309），贴出来的图案压在杯口与咖啡上。
    器物的口沿在图像上是**横向强边**，用这一条就能定位，不需要任何语义模型。
    """
    grey = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    grey = grey.astype(np.float32)
    gy = np.abs(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
    left = max(0, int(left))
    right = min(grey.shape[1], int(right))
    if right <= left:
        return None
    band = gy[:, left:right].mean(axis=1)
    limit = min(max(4, int(search_bottom)), band.shape[0])
    if limit < 4:
        return None
    row = int(np.argmax(band[:limit]))
    # 太弱的"边"不是口沿（可能是噪点），不采信
    if float(band[row]) < float(np.percentile(band[:limit], 80)):
        return None
    return row


def _exclude_contents(image, mask):
    """把"器物内容物"（杯中的咖啡/水）从主体掩版里去掉，返回 (新掩版, 说明)。

    为什么要单独做（实测）：杯口内壁与杯中液体在亮度/饱和度/平滑度上与器物本体
    几乎一致，SAM 会把它们和器物一起圈成同一个连通块，于是配色迁移**把咖啡染成了
    紫蓝色**——而需求是"只对主体做加法"，咖啡不是器物的表面。

    做法：器物口沿是图像上的**横向强边**（`_rim_row`），排除口沿上方即可。
    带安全阀：若削掉的面积过多，判为口沿误检并保留原掩版。
    """
    if not bool(getattr(config, "MARK_STYLE_EXCLUDE_CONTENTS", True)):
        return mask, ""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return mask, ""
    search_bottom = int(rows[0] + (rows[-1] - rows[0]) * 0.60)
    rim = _rim_row(image, cols[0], cols[-1] + 1, search_bottom=search_bottom)
    if rim is None:
        return mask, ""
    margin = int(max(2.0, (rows[-1] - rows[0]) * float(
        getattr(config, "MARK_CARRY_RIM_MARGIN", 0.04))))
    clear = int(rim + margin)
    if not (rows[0] < clear < rows[-1]):
        return mask, ""
    before = int(mask.sum())
    trimmed = mask.copy()
    trimmed[:clear, :] = False
    remaining = int(trimmed.sum())
    floor = float(getattr(config, "MARK_CARRY_RIM_MIN_KEEP", 0.35))
    if remaining >= before * floor:
        return trimmed, (f"；已排除口沿之上的内容物 {before - remaining} 像素"
                         f"（口沿 y≈{rim}）")
    return mask, (f"；口沿候选 y≈{rim} 会削掉 "
                  f"{100 * (1 - remaining / max(1, before)):.0f}%，判为误检，保留原区域")


def subject_surface_region(image, hint_box=None, min_pixels=400):
    """从**图像自身统计量**求出"主体的可见表面"区域，返回 (bool ndarray, 说明)。

    为什么要自己算（实测踩过）：`locate_regions` 给出的物体框是**归一化**坐标，
    一旦检测是在另一个尺寸/另一帧上做的（缓存复用、缩放时机不同），
    换算到当前画布就会整体偏移——实测把图案贴到了杯子的**左右两侧背景**上
    （x 386~716 的框，画面里的杯子却在 x≈260~430）。
    与其依赖可能过期的坐标，不如直接在**当前这张画布**上找主体：

      * 主体（陶瓷杯/浅色物体）通常是**亮且低饱和**的一大块；
      * 背景（桌面/水面/布料）通常更暗或更花；
      * 取"亮 + 低饱和"像素的最大连通块，再做闭运算补洞。

    `hint_box` 只作为**搜索范围**（像素坐标），用来避免把远处的亮区也算进来；
    它不决定最终边界。返回的说明会写清用了什么判据、覆盖多少像素。
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)

    # 判据（只用图像自身的分位数，避免写死阈值）：
    #   1. **够亮**：主体是画面里亮的那一撮（实测 60 分位会把亮背景也算进来，
    #      于是"最大连通块"落在背景上；取 88 分位才稳定落在物体上）；
    #   2. **低饱和**：陶瓷/塑料/金属表面比彩色背景更灰；
    #   3. **够平滑**：这一条是拉开"物体表面 vs 粗糙背景"的关键——
    #      实测陶杯杯身是平滑的（局部方差极小），而桌面/布料/水面都有明显颗粒纹理。
    #      只靠亮+低饱和会漏掉杯身中段（它比杯口稍暗），加上平滑度就补回来了。
    v_cut = float(np.percentile(value, float(getattr(config, "MARK_CARRY_SUBJECT_VALUE_PCT", 88.0))))
    s_cut = float(np.percentile(saturation, 55))
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_mean = cv2.blur(grey, (9, 9))
    local_var = cv2.blur(grey * grey, (9, 9)) - local_mean * local_mean
    var_cut = float(np.percentile(local_var, float(
        getattr(config, "MARK_CARRY_SUBJECT_SMOOTH_PCT", 45.0))))
    candidate = ((value >= v_cut) & (saturation <= s_cut) & (local_var <= var_cut)).astype(np.uint8)

    if hint_box:
        left, top, right, bottom = [int(round(v)) for v in hint_box]
        limited = np.zeros_like(candidate)
        # 搜索范围比提示框放大一圈：检测框本身可能偏小
        pad_x = int((right - left) * 0.35)
        pad_y = int((bottom - top) * 0.35)
        search_top = max(0, top - pad_y)
        search_bottom = min(height, bottom + pad_y)
        search_left = max(0, left - pad_x)
        search_right = min(width, right + pad_x)
        limited[search_top:search_bottom, search_left:search_right] = 1
        candidate = candidate * limited
    else:
        search_top, search_bottom = 0, height
        search_left, search_right = 0, width

    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 1:
        return None, "图上找不到符合条件的浅色主体块"
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == largest)
    pixels = int(region.sum())
    if pixels < min_pixels:
        return None, f"最大浅色块仅 {pixels} 像素（<{min_pixels}）"
    rows = np.where(region.any(axis=1))[0]
    cols = np.where(region.any(axis=0))[0]
    # 用"确定属于主体"的像素做种子，再求**可贴范围（支撑区）**。
    #
    # 为什么不能直接拿种子当范围：判据只能标出主体上**特征最明显**的那部分
    # （实测陶杯只标出杯身左半边 11583 像素——杯身中段偏暗、右半边偏灰，
    # 都不满足"够亮"），直接用它贴图会让图案挤在一侧。
    #
    # 做法：在种子外接框外扩一圈，在这个放大框内**只按平滑度**筛
    # （去掉"够亮"这一条，让稍暗的杯身中段也能进来），
    # 背景因为颗粒纹理更重而被排除。这样支撑区覆盖整个主体表面，又不会糊到桌面上。
    seed_pixels = int(region.sum())
    x0, x1 = int(cols[0]), int(cols[-1])
    y0, y1 = int(rows[0]), int(rows[-1])
    grow = float(getattr(config, "MARK_CARRY_GROW", 0.40))
    gx = int((x1 - x0) * grow)
    gy = int((y1 - y0) * grow)
    sx0 = max(search_left, x0 - gx)
    sx1 = min(search_right, x1 + gx + 1)
    sy0 = max(search_top, y0 - gy)
    sy1 = min(search_bottom, y1 + gy + 1)
    support = np.zeros((height, width), dtype=bool)
    support[sy0:sy1, sx0:sx1] = True
    smooth_cut = float(np.percentile(local_var, float(
        getattr(config, "MARK_CARRY_SUPPORT_SMOOTH_PCT", 62.0))))
    support &= (local_var <= smooth_cut)
    support |= region          # 判据确认的表面一定包含在内
    # 只保留最大连通块：避免零散碎块把图案撒得到处都是
    count2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8)
    if count2 > 1:
        keep = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
        support = (labels2 == keep)
    region = support
    pixels = int(support.sum())
    # 去掉器物**口沿之上**的部分：杯口内壁与杯中液体同样"亮 + 低饱和 + 平滑"，
    # 会被判据一起选进来，贴出来的图案于是压在口沿与液体上（实测）。
    rim_note = ""
    before = pixels
    rim = _rim_row(image, sx0, sx1, search_bottom=sy0 + int((sy1 - sy0) * 0.60))
    if rim is not None:
        clear = int(rim + max(2.0, (sy1 - sy0) * float(
            getattr(config, "MARK_CARRY_RIM_MARGIN", 0.04))))
        if sy0 < clear < sy1:
            candidate = region.copy()
            candidate[:clear, :] = False
            remaining = int(candidate.sum())
            # 安全阀：若"去掉口沿之上"把区域削掉太多，说明找到的不是口沿
            # （实测踩过：把**杯底**那条强边当成了口沿，区域被整块清空）。
            floor = float(getattr(config, "MARK_CARRY_RIM_MIN_KEEP", 0.35))
            if remaining >= before * floor:
                removed = before - remaining
                region = candidate
                pixels = remaining
                rim_note = f"；已去掉口沿之上 {removed} 像素（口沿 y≈{rim}）"
            else:
                rim_note = (f"；口沿候选 y≈{rim} 会削掉 "
                            f"{100 * (1 - remaining / max(1, before)):.0f}% 区域，判为误检，保留原区域")
    note = (f"图像统计量定位主体表面: 判据像素 {seed_pixels}"
            f"（亮≥{v_cut:.0f} 低饱和≤{s_cut:.0f} 平滑≤{var_cut:.0f}），"
            f"外扩 {grow:.0%} 后按平滑度≤{smooth_cut:.0f} 填充 -> {pixels} 像素 "
            f"x {sx0}~{sx1 - 1} y {sy0}~{sy1 - 1}{rim_note}")
    return region, note


def subject_appearance_region(image, hint_box=None, min_pixels=400):
    """定位**整个主体可见表面**（用于配色迁移）。

    ## 优先用 SAM（规格 §2 明确要求用项目内的分割能力）

    先用亮度判据取**种子点**（一定落在主体上），再把种子点交给 SAM，
    由 SAM 给出主体的**真实轮廓**。

    ## 为什么不再靠手写阈值（实测三轮都失败，必须记录）

    配色迁移要覆盖"整个主体"且**不能溢出到背景**。手写判据试过三种，各自的问题：

    1. `亮 + 低饱和 + 极平滑`（沿用可贴区域的判据）：
       在陶杯上只标出**杯身左半边一小块**，配色迁移变成"一条竖色带"；
    2. `亮（整幅 88 分位）+ 外接框填充`：
       种子正好是杯子，但向外填充时把**杯外被照亮的桌面**拉进来
       （掩版从 x 258~581 胀到 x 178~661）；
    3. 再加"膨胀邻域约束 + 轮廓填充"：
       轮廓把"主体 + 邻近亮桌面"整体填成一个外接形状，
       杯子周围出现**一圈蓝色光晕**（背景被染色）。

    共同原因是：**这张图里桌面与杯子在颜色/饱和度/平滑度上都很接近**
    （实测主体 V 中位 200、背景 142；饱和度都是个位数；平滑度 2.24 vs 3.85），
    阈值法没有稳定的可分性。SAM 给的是**对象的轮廓**，
    不受"背景恰好也亮"影响，因此这里改用 SAM。
    拿不到 SAM 时才退回亮度法，并在说明里写明退化原因。

    返回 (掩版, 说明)；找不到时返回 (None, 原因)。
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2].astype(np.float32)

    # 搜索范围（避免把远处的亮区算进来）
    if hint_box:
        left, top, right, bottom = [int(round(v)) for v in hint_box]
        pad_x = int((right - left) * 0.30)
        pad_y = int((bottom - top) * 0.30)
    else:
        left, top, right, bottom = 0, 0, width, height
        pad_x = pad_y = 0
    sy0 = max(0, top - pad_y)
    sy1 = min(height, bottom + pad_y)
    sx0 = max(0, left - pad_x)
    sx1 = min(width, right + pad_x)
    window = np.zeros((height, width), dtype=bool)
    window[sy0:sy1, sx0:sx1] = True

    # 种子：搜索框内取亮度高分位，**同时**不低于整幅分位的比例（全局下限挡住桌面）
    v_pct = float(getattr(config, "MARK_STYLE_VALUE_PCT", 88.0))
    v_cut_global = float(np.percentile(value, v_pct))
    v_cut_local = float(np.percentile(value[window], v_pct))
    v_cut = max(v_cut_local, v_cut_global * float(
        getattr(config, "MARK_STYLE_GLOBAL_FLOOR", 1.0)))
    seed_bool = (value >= v_cut) & window
    if not seed_bool.any():
        return None, "搜索范围内没有足够亮的种子像素"

    # ---- SAM 主体分割：把种子像素作为提示点 ----
    sam = None
    try:
        from vision import segmentation
        sam = segmentation.load_sam2()
        if sam is not None:
            # 取种子最大连通块的质心与几个分散点作为提示（单点容易落到高光上）
            seed_u8 = seed_bool.astype(np.uint8)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                seed_u8, connectivity=8)
            if count > 1:
                order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
                # **逐点单独提示，最后取与"主体种子"重叠最好的那个掩版。**
                # 为什么不是"多点一起送"（实测踩过）：把 3 个点一起送入时，
                # SAM 返回的单个掩版把**主体与背景合并**成一块
                # （外接框 x 0~720 y 190~651，覆盖约 34% 画面，
                # 于是杯子周围出现一圈被染色的背景）。
                # 逐点单独提示后，每个点各自给出候选，再按"与种子像素的重叠"
                # 挑出真正属于主体的那一个——种子一定在主体上，因此这个判据可靠。
                best, best_score = None, 0.0
                # 注意：`sam_point_maskbox` 期望**扁平的 [(x, y), ...]**，
                # 它自己在内部包成 [batch][object][point][2]（曾多包一层导致
                # "Got 5 levels" 而整条 SAM 路径失效）。
                for index in order[:3]:
                    candidate = segmentation.sam_point_maskbox(
                        image, sam, [(float(centroids[index][0]),
                                      float(centroids[index][1]))])
                    if candidate is None:
                        continue
                    array = np.asarray(candidate.convert("L")) > 127
                    # 打分：候选掩版覆盖了多少"确定的种子像素"
                    score = float((array & seed_bool).sum()) / max(1, int(seed_bool.sum()))
                    # 惩罚过大掩版（把整幅都圈进来时重叠率也会高）
                    penalty = float(array.mean())
                    score -= penalty * float(getattr(
                        config, "MARK_STYLE_COVERAGE_PENALTY", 0.25))
                    if score > best_score:
                        best, best_score = array, score
                if best is not None:
                    pixels = int(best.sum())
                    if pixels >= min_pixels:
                        best, trim_note = _exclude_contents(image, best)
                        pixels = int(best.sum())
                        rows = np.where(best.any(axis=1))[0]
                        cols = np.where(best.any(axis=0))[0]
                        return best, (
                            f"SAM 主体分割（种子亮≥{v_cut:.0f}，"
                            f"逐点提示取与种子重叠最佳者，得分 {best_score:.3f}）"
                            f" -> {pixels} 像素 x {cols[0]}~{cols[-1]} "
                            f"y {rows[0]}~{rows[-1]}{trim_note}")
    except Exception as exc:      # noqa: BLE001
        fallback_reason = f"SAM 不可用（{type(exc).__name__}: {exc}）"
    else:
        fallback_reason = "SAM 未返回可用掩版"
    finally:
        if sam is not None:
            try:
                from vision import segmentation as _seg
                _seg.unload_sam(sam)
            except Exception:      # noqa: BLE001
                pass

    # ---- 退回亮度法（可用但会在"背景也亮"时溢出，说明里注明）----
    seed = seed_bool.astype(np.uint8)
    kernel = np.ones((11, 11), np.uint8)
    closed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel, iterations=3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if count <= 1:
        return None, f"找不到符合条件的主体块（{fallback_reason}）"
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = cv2.dilate((labels == largest).astype(np.uint8),
                      np.ones((int(getattr(config, "MARK_STYLE_DILATE_PX", 26)),) * 2,
                              np.uint8)).astype(bool)
    mask &= (value >= v_cut * float(
        getattr(config, "MARK_STYLE_SUPPORT_VALUE_RATIO", 0.80)))
    mask |= (labels == largest)
    pixels = int(mask.sum())
    if pixels < min_pixels:
        return None, f"主体块仅 {pixels} 像素（<{min_pixels}）"
    mask, trim_note = _exclude_contents(image, mask)
    pixels = int(mask.sum())
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask, (f"退化为亮度法（{fallback_reason}）: 亮≥{v_cut:.0f} + "
                  f"膨胀邻域内的亮度填充 -> {pixels} 像素 "
                  f"x {cols[0]}~{cols[-1]} y {rows[0]}~{rows[-1]}{trim_note}；"
                  f"注意此法在'背景也亮'的构图上可能溢出到背景")


def character_palette(folder, observations=None, limit=0):
    """从**角色自己的图像证据**里算出主色（带缓存）。

    素材来源：该角色的逐图真实裁剪（55 张，全部是角色本身的画面）。
    为什么要缓存成 json（实测理由）：k-means 每次要读几十张图；更重要的是
    **结果必须稳定**——同一次对照里两次出图用的调色板不同，就无法判断差异来自哪里。
    """
    import json as _json
    cache = os.path.join(config.OUTPUT_DIR, "semantic", f"{folder}_palette.json")
    if os.path.isfile(cache) and not limit:
        try:
            with open(cache, "r", encoding="utf-8") as stream:
                data = _json.load(stream)
            if data.get("palette"):
                return data["palette"], data.get("source", "缓存")
        except (OSError, ValueError):
            pass
    observations = observations if observations is not None else \
        marks_module.load_observations(folder)
    paths = []
    for row in observations:
        crop = row.get("crop_path")
        if not crop:
            continue
        full = os.path.join(config.PROJECT_ROOT, crop)
        if os.path.isfile(full):
            paths.append(full)
    if not paths:
        return [], "没有可用的角色裁剪素材"
    palette = mark_style.extract_palette(paths, k=int(
        getattr(config, "MARK_STYLE_PALETTE_K", 5)))
    if palette and not limit:
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as stream:
                _json.dump({"palette": palette, "source": f"{len(paths)} 张角色裁剪"},
                           stream, ensure_ascii=False, indent=2)
        except OSError:
            pass
    return palette, f"{len(paths)} 张角色裁剪"


def _unit_scale(assignment, layout_info):
    """由**量出来的角色占比**决定平铺单元的缩放系数。

    用户指出（2026-10-06）："金色图案在角色服饰中虽然较大但不会说占较大视觉形成视觉主体"。
    实测该标志在角色画面上只占 0.31% 面积（`mark_layout` 量出），
    却因为固定单元尺寸被放到近 200 像素见方，成了整只杯子的视觉主体——
    正是这句话描述的问题。因此用 coverage 反推缩放：

      * coverage 越小（角色上是小点缀）-> 单元越小、重复越多；
      * coverage 越大（角色上是主纹样）-> 允许较大的单元。

    映射用开方（面积 -> 边长），并夹在 `MARK_CARRY_UNIT_SCALE_RANGE` 内，
    避免极端值把单元缩到看不清或放到占满。
    """
    coverage = None
    if isinstance(layout_info, dict):
        coverage = layout_info.get("coverage")
    if not coverage:
        return None
    low, high = getattr(config, "MARK_CARRY_UNIT_SCALE_RANGE", (0.35, 1.15))
    ref = float(getattr(config, "MARK_CARRY_UNIT_SCALE_REF", 0.06))
    try:
        scale = (float(coverage) / ref) ** 0.5
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return round(max(float(low), min(float(high), scale)), 3)


def apply_carry_plan(folder, image, subject_type="object", region_mask=None,
                     observations=None, library=None, limit=None):
    """按承载计划把标志贴到 `image` 上，返回 (结果图, 审计列表)。

    `region_mask`：目标主体可承载的区域（布尔 ndarray，与图像同尺寸）。
    没有掩版时**不做**——不知道能贴哪儿就贴，等于把图案画到背景上（实测踩过）。
    """
    import config as _config
    if not getattr(_config, "MARK_CARRY_COMPOSITE", True):
        return image, [{"skipped": "配置关闭 MARK_CARRY_COMPOSITE"}]
    if region_mask is None or not region_mask.any():
        return image, [{"skipped": "没有可承载区域（不猜位置）"}]

    library = library or marks_module.load_library(folder)
    if not library:
        return image, [{"skipped": "没有标志库"}]
    observations = observations if observations is not None else \
        marks_module.load_observations(folder)
    plan = marks_module.carry_plan(folder, subject_type,
                                   limit=limit or _config.MARK_CARRY_LIMIT,
                                   observations=observations, library=library)

    canvas = np.asarray(image.convert("RGB")).copy()
    audits = []
    # ---- 第一步：把**角色自己的颜色关系**迁移到目标表面 ----
    # 这一步是针对用户反馈的核心改动："看起来就只是简单的单独标志图的粘贴，
    # 根本没有学习到这种角色的特征并运用上"。
    # 只迁移色度、**完全保留明度通道**，因此器物的圆润、高光、阴影与材质纹理都在，
    # 变化的是"它属于哪个角色的配色体系"。
    palette, palette_source = character_palette(folder, observations)
    if getattr(config, "MARK_STYLE_ENABLED", True) and palette:
        # **配色迁移用"整个主体"的掩版，不用"可贴区域"**（实测踩过）：
        # 可贴区域的判据是"极平坦"，在陶杯上只覆盖杯身左半边一小块，
        # 用它做配色迁移会得到"一条竖色带"，而不是"整只杯子变成角色的"。
        style_mask = region_mask
        appearance_note = "沿用可贴区域"
        if getattr(config, "MARK_STYLE_FULL_SUBJECT", True):
            appearance, appearance_note = subject_appearance_region(image)
            if appearance is not None:
                style_mask = appearance | region_mask
        canvas, style_audit = mark_style.restyle_surface(
            canvas, style_mask, palette)
        style_audit["palette_source"] = palette_source
        style_audit["palette_summary"] = mark_style.palette_summary(palette)
        style_audit["region_source"] = appearance_note
        audits.append({"region": "style_transfer", **style_audit})
        if style_audit.get("applied"):
            print(f"[carry] 角色配色已迁移到主体表面: {style_audit['palette_summary']}"
                  f"（调色板来源 {palette_source}；{appearance_note}）")
    elif not palette:
        audits.append({"region": "style_transfer", "applied": False,
                       "reason": palette_source})

    # ---- 第二步：按**标志在角色身上的性质与空间用法**决定怎么落笔 ----
    #
    # 用户指出的核心问题（2026-10-06）：
    #   "需分清纹样图案这种花纹式的和配饰装饰这种实质性的，
    #    如果有需要使用配饰装饰来作为添加效果在非人物上只能用配饰装饰形式的特征标志"
    # 实测这个库里**扇子被判成 pattern 并平铺到杯子上**——扇子是角色手里拿的**实物**，
    # 不能当表面花纹刷。因此这里强制两条规则：
    #   * `nature == "accessory"`（实物）：**禁止任何表面平铺/印花**，
    #     只能以"实物形态"出现（放在器物边上/挂上，不做 surface pattern）；
    #   * `nature == "pattern"`（花纹）且 `role == "scattered"`（角色上是零星散布）：
    #     应当**重复铺开**而不是单张大图（樱花在角色服装上就是零星飘落的）。
    from character import mark_layout as _layout
    layout_items = {item.get("mark_id"): item
                    for item in _layout.layout_plan(library, observations, limit=0)}
    for item in plan.get("assignments") or []:
        info = layout_items.get((item.get("mark") or {}).get("mark_id")) or {}
        item["nature"] = info.get("nature")
        item["role"] = info.get("role")
        item["layout_note"] = info.get("note")
        # **把 coverage 一起挂到承载条目上**（方向修整1 §5 P0 第 1 条）。
        # 实测缺陷：这里原先只挂了 nature/role/layout_note，
        # 而下游 `_unit_scale` 要用的 `coverage` 没有被带过去，
        # 只能靠另一份 `info_layout` 再查一次——数据流断了一环，
        # 一旦某条标志不在 layout_plan 里（例如被 descriptive 过滤掉），
        # 缩放就会静默拿到空值、退回统一尺寸。挂在本条目上就不存在这个断点。
        item["coverage"] = info.get("coverage")
        item["spread"] = info.get("spread")
    # 实物配饰从"表面承载"里剔除，并如实说明原因（不静默丢弃）
    kept, dropped = [], []
    for item in plan.get("assignments") or []:
        mark = item.get("mark") or {}
        if item.get("nature") == "accessory":
            dropped.append({"region": "mark_carry", "name": mark.get("name"),
                            "carry": item.get("carry"),
                            "skipped": "实物配饰不做表面平铺"
                                       "（用户要求：配饰只能用配饰装饰形式）",
                            "nature": "accessory", "role": item.get("role")})
            continue
        kept.append(item)
    plan["assignments"] = kept
    audits.extend(dropped)
    for item in dropped:
        print(f"[carry] 跳过实物配饰的表面平铺: {item.get('name')}"
              f"（它是实物，不能当花纹刷在器物表面）")
    # 花纹类：`scattered` 的用平铺重复，`focal/accent` 的按单件放置
    for item in kept:
        if item.get("nature") == "pattern" and item.get("role") == "scattered" \
                and item.get("carry") in {"emblem", "detail"}:
            # 零星散布的花纹改用平铺承载（这正是"樱花零星飘落"的空间用法）
            item["carry"] = "pattern"
            item["carry_reason"] = "角色上是零星散布 -> 改为重复铺开"

    # 落笔顺序：显眼件（徽记/可佩戴件）先放，底纹（花纹/边饰）后铺。
    # 实测问题：按承载计划顺序落笔时，先铺满的花纹会让后放的徽记压在花纹格子上
    # （杯子中央的金色图案叠在一堆扇子上，两者都看不清）。
    preferred = list(getattr(config, "MARK_CARRY_LAYOUT",
                             ("emblem", "wearable", "trim", "pattern", "detail")))
    assignments = sorted(
        plan["assignments"],
        key=lambda item: (preferred.index(item.get("carry"))
                          if item.get("carry") in preferred else len(preferred)))
    emblem_total = sum(1 for item in assignments if item.get("carry") == "emblem")
    emblem_slot = 0
    # 已被"显眼件"占用的位置：铺底纹时必须避开，否则纹样会压住徽记
    blocked = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=bool)
    info_layout = {item.get("mark_id"): item for item in (kept or [])}
    for item in assignments:
        mark = item.get("mark") or {}
        mark_class = str(item.get("carry") or "detail")
        path, note, scale_cap = _asset_for(folder, mark, observations)
        if not path:
            audits.append({"mark_id": mark.get("mark_id"), "name": mark.get("name"),
                           "skipped": note})
            continue
        asset, asset_note = mark_carry.load_asset(path)
        if asset is None:
            audits.append({"mark_id": mark.get("mark_id"), "name": mark.get("name"),
                           "skipped": f"素材不可用: {asset_note}"})
            continue
        asset_note = f"{asset_note}；{note}"
        # ---- 形状分离：只保留标志本身的轮廓 ----
        # 直接贴裁剪矩形时，每个重复单元都是一块带背景的方形照片，
        # 观感像"贴了一排小照片"而不是"器物上的花纹"（实测）。
        # 这里把主导背景色透掉、只留标志轮廓；把背景也贴上去会在深色器物上糊出白方块。
        if getattr(config, "MARK_SHAPE_ENABLED", True):
            try:
                from vision import mark_shape
                shaped, shape_audit = mark_shape.extract_shape(path, sam=None)
                if shaped is not None:
                    asset = shaped
                    asset_note = (f"{asset_note}；形状分离={shape_audit.get('method')} "
                                  f"覆盖={shape_audit.get('coverage')}")
                else:
                    # 形状分离失败时**不给矩形素材兜底**：矩形素材会被承印融合直接拒绝
                    # （`surface_fusion.shape_alpha` 只接受真正的形状 alpha），
                    # 如实记录原因比悄悄贴上一块方照片更有用。
                    salvaged, salvage_note = refine_asset_shape(asset)
                    if salvaged is not None:
                        asset = salvaged
                        asset_note = f"{asset_note}；形状分离未通过，改用抢救形状（{salvage_note}）"
                    else:
                        asset_note = f"{asset_note}；形状分离失败且抢救无效（{salvage_note}）"
            except Exception as exc:      # noqa: BLE001
                asset_note = f"{asset_note}；形状分离失败({type(exc).__name__})"
        if mark_class == "emblem":
            # 用**同一套角色调色板**给标志重新着色：让它与器物被渲染出的配色同源，
            # 读起来是"这件器物的纹样"，而不是"贴上去的一张图"（用户反馈的直接回应）。
            if getattr(config, "MARK_STYLE_ENABLED", True) and palette:
                asset, restyle_audit = mark_style.restyle_asset(asset, palette)
                if restyle_audit.get("applied"):
                    asset_note = f"{asset_note}；按角色调色板重着色"
            canvas, info = _place_emblem(canvas, mark, asset, region_mask,
                                         slot=emblem_slot, slots=emblem_total,
                                         scale_cap=scale_cap)
            emblem_slot += 1
        elif mark_class == "wearable":
            canvas, info = _place_wearable(canvas, mark, asset, region_mask,
                                           scale_cap=scale_cap)
        elif mark_class == "trim":
            canvas, count = mark_carry.band_rgba(canvas, asset, region_mask)
            info = {"mode": "trim", "placed": count} if count else None
        elif mark_class == "pattern":
            # **不再做 6×N 的网格平铺**（本轮用户评价"效果很差"的直接原因之一）。
            #
            # 实测问题：两个标志各铺 6 次，器物表面 94% 像素被改写，
            # 观感像"包装纸"或"重新刷过的杯子"，而不是"这件器物属于这个角色"。
            # 需求是"加法"且"看得出是这个角色的标志"，那就要**看得清、不糊满**。
            #
            # 改为：按"该标志在角色身上的角色"决定放几次——
            #   scattered（零星散布，如樱花）：放 2~3 处，大小略有差异；
            #   focal/accent：放 1 处，作为主件或点缀。
            # 并且单元尺寸按区域短边的比例给出，不再用"铺满几列"反推。
            count_wanted = int(getattr(config, "MARK_CARRY_SCATTERED_COUNT", 3)) \
                if item.get("role") == "scattered" else 1
            canvas, count = mark_carry.place_scattered(
                canvas, asset, region_mask, count=count_wanted,
                max_ratio=float(getattr(config, "MARK_CARRY_MARK_MAX_RATIO", 0.34))
                * float(scale_cap),
                block_mask=blocked, slot=emblem_slot,
                slots=emblem_total + count_wanted)
            emblem_slot += count
            info = {"mode": "scattered", "placed": count} if count else None
            if info:
                # 把**实际落笔的每个框**带出去：后续的曲面承印要避开这些位置，
                # 否则后印的底纹会把刚放好的标志整块盖掉（实测：出图上只剩喷绘底纹，
                # 一枚标志都看不见，而日志里写着"铺放 3 次"）。
                info["boxes"] = [list(map(int, b))
                                 for b in getattr(mark_carry.place_scattered,
                                                  "last_boxes", [])]
        else:
            # detail：不适合当图案铺，按徽记方式居中贴一次（仍用真实像素）
            canvas, info = _place_emblem(canvas, mark, asset, region_mask,
                                         scale_cap=scale_cap)
        if not info:
            # 把"为什么铺不下"的具体分解带上（方向修整1 §5 P0 第 2 条）：
            # 只报一句"铺不下"无法判断是尺寸、区域形状还是被显眼件挡住。
            detail = getattr(mark_carry.tile_rgba, "last_reject", None)
            audits.append({"mark_id": mark.get("mark_id"), "name": mark.get("name"),
                           "carry": mark_class,
                           "unit_scale": _unit_scale(item, item),
                           "coverage": item.get("coverage"), "role": item.get("role"),
                           "nature": item.get("nature"),
                           "reject_detail": detail,
                           "skipped": "该承载方式在可见区域内铺不下"
                                      + (f"（候选格 {detail.get('grid')}："
                                         f"区域外 {detail.get('outside_region')}、"
                                         f"被占用 {detail.get('blocked')}、"
                                         f"贴边切掉 {detail.get('offgrid')}；"
                                         f"单元 {detail.get('unit')} 区域 {detail.get('span')}）"
                                         if isinstance(detail, dict) else "")})
            continue
        # 记录"已占用的位置"：后续底纹要避开它
        if mark_class in ("emblem", "wearable", "detail") and info.get("box"):
            blocked = mark_carry.mark_blocked(blocked, info["box"],
                                              canvas.shape[1::-1])
        audits.append({
            "mark_id": mark.get("mark_id"), "name": mark.get("name"),
            "carry": mark_class, "token": marks_module.mark_token(
                mark, _config.get_trigger(folder)),
            "asset": os.path.relpath(path, _config.PROJECT_ROOT),
            "asset_note": asset_note, "asset_source": note, **info})
    # 把"标志已经放好、底纹还没印"的这一版画布留一份快照。
    #
    # 为什么需要（实测问题）：曲面承印随后会把**整个可承载区域**覆盖成生成的底纹，
    # 成品上因此一枚标志都看不见，只剩一片喷绘图案。职责划分应当是
    # **标志是视觉主体、底纹是衬底**，所以承印之后要把标志落笔区还原成这一版。
    # 快照与"还原哪些框"一起交给调用方（`test.py`），由它决定是否还原。
    apply_carry_plan.last_canvas = canvas.copy()
    apply_carry_plan.last_boxes = [
        list(map(int, box)) for item in audits
        if item.get("region") != "style_transfer"
        for box in ((item.get("boxes") or []) + ([item["box"]] if item.get("box") else []))]
    return Image.fromarray(canvas, mode="RGB"), audits


def describe(audits) -> str:
    """把承载审计渲染成可读日志。

    注意：审计列表里**不只有标志条目**，还有"配色迁移"这类非标志条目
    （它没有 `name`/`carry`/`size` 字段）。早期这里无条件套用
    `f"{item.get('name'):<18}"`，遇到 `None` 直接抛
    `TypeError: unsupported format string passed to NoneType.__format__`，
    整个承载步骤被外层 `except` 吞掉——**配色与标志全都白做了**（实测踩过）。
    因此每一项都按字段存在性分别渲染。
    """
    lines = ["[carry] 确定性标志承载:"]
    for item in audits or []:
        if not isinstance(item, dict):
            continue
        if item.get("skipped"):
            name = str(item.get("name") or "?")
            lines.append(f"  - {name[:16]:<18} 跳过: {item['skipped']}")
            continue
        if item.get("region") == "style_transfer":
            if item.get("applied"):
                lines.append(f"  - {'角色配色迁移':<18} {'surface':<9} "
                             f"{item.get('palette_summary')}  "
                             f"{item.get('region_source')}")
            else:
                lines.append(f"  - {'角色配色迁移':<18} {'surface':<9} "
                             f"未生效: {item.get('reason')}")
            continue
        if "name" not in item and "carry" not in item:
            continue
        name = str(item.get("name") or "?")
        carry = str(item.get("carry") or "?")
        detail = item.get("placed")
        # `size` 可能是 None（跳过或未放置时），用原文而不是格式化占位
        size = item.get("size")
        extra = f"铺放 {detail} 次" if detail else f"尺寸 {size}"
        lines.append(f"  - {name[:16]:<18} {carry:<9} {extra}  {item.get('asset')}")
    return "\n".join(lines)
