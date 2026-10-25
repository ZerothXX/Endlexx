# -*- coding: utf-8 -*-
"""vision/metrics.py —— 出图的**客观评判指标**（纯 numpy / OpenCV，不加载任何模型）。

## 为什么必须有这个模块

用户的验收判据是"人物像不像角色、非人物上的标志是不是被自然运用"，而这类判断
此前只能靠"看一眼"。实测代价很大：多轮改动是在**肉眼无法分辨**的情况下被当成
"有效"的（例如适配器强度根本没生效、支路被闸门关掉、承载被后续步骤整块还原，
都出现过"日志显示做了、出图逐字节没变"）。

因此本模块把"好不好"翻译成**可复现的数字**，供自动化对照使用；
数值只描述可测量的性质，**不替代人工盲评**（规格 §8 明确要求不能只让模型自评）。

## 指标分组

**A. 结构保持（源图该留下的东西有没有留下）**
  * `edge_recall/edge_f1`：源图边缘在成品里还剩多少（Canny 重合）；
  * `sharpness_ratio`：成品/源图的高频能量比。**塌陷检测**：把画面分块，
    统计"高频能量不到源图一半"的块占比 `collapsed_tile_ratio`——
    这是"脸糊成一团"最直接的可测信号，不依赖人脸检测器。

**B. 承载质量（非人物的"加法"是不是贴纸）**
  * `follow_cos`：**逐小块**最小二乘拟合"成品的明暗是否跟随器物的明暗"
    （斜率 `gamma` 按器物波动幅度加权平均）。平贴的图案自带一套明暗、斜率接近 0；
    真正"长在器物上"的装饰跟随器物受光，斜率接近 1。这是本模块区分"贴纸"与"融合"的主判据。
  * `follow_resid`：同一拟合的残差（按**器物**的波动幅度归一）。平贴接近 1，融合接近 0。
  * `grad_cos`：受光**坡向**（亮度梯度向量场）的余弦相似度，作为独立旁证。
  * `boundary_edge_excess`：改写区域边界的梯度是否显著高于画面典型边缘。
    硬边贴图会在边界留下一圈"框"，是"后加的印记"最典型的特征。
  * `texture_retention`：改写区域是否保留了器物自身的高频颗粒/釉面纹理。
    纯平贴会**抹掉**器物纹理（比值偏离 1 都是信号：过低=抹平，过高=外来纹理压上去）。

**C. 画面质量（成品本身有没有坏）**
  * `clipping_low/high`：死黑/死白像素比例；
  * `laplacian_energy`：整体清晰度；
  * `banding`：低频带状的伪影强度（"淡带"、"分界线"）。

**D. 身份（可选，依赖 CLIP/DINO 是否可用）**
  * `embedding_identity`：成品主体裁剪与角色参考图的图像嵌入余弦相似度。
    取不到嵌入模型时**明确返回 None 并写原因**，不用代理指标冒充。

用法：
    from vision import metrics
    report = metrics.evaluate(source_image, result_image, region_mask=mask,
                              reference_images=[...], character="ayaka")
    print(metrics.format_report(report))
"""
import os

import cv2
import numpy as np

# 分块塌陷检测的块数与阈值
TILE_GRID = 6
COLLAPSE_RATIO = 0.5          # 高频能量低于源图该比例的块判为"塌陷"
MIN_TILE_STD = 1e-3


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _rgb(image):
    """接受 PIL.Image / ndarray / 路径，统一成 uint8 RGB ndarray。"""
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        elif array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
        return np.ascontiguousarray(array.astype(np.uint8))
    if isinstance(image, str):
        bgr = cv2.imread(image, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _grey(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _aligned(source, result):
    """把 result 缩放到 source 的尺寸（跨分辨率比较是不公平的）。"""
    if source.shape[:2] == result.shape[:2]:
        return result
    return cv2.resize(result, (source.shape[1], source.shape[0]),
                      interpolation=cv2.INTER_AREA)


def _low_frequency(grey, sigma):
    """低频分量：高斯模糊。sigma 以像素计，代表"关注的尺度"。"""
    size = int(max(3, round(sigma * 4)) | 1)
    return cv2.GaussianBlur(grey.astype(np.float32), (size, size), float(sigma))


def _high_frequency(grey):
    """高频分量：原图减去低通（保留边缘与颗粒）。"""
    return grey.astype(np.float32) - _low_frequency(grey, 2.0)


def _changed_mask(source, result, threshold=8):
    """改写区域：与源图差异超过阈值的像素。取不到掩版时的退化判据。"""
    delta = np.abs(result.astype(np.int16) - source.astype(np.int16)).max(axis=2)
    return delta > threshold


def _largest_component(mask):
    """只保留最大连通块（噪声点会把统计量带偏）。"""
    if mask is None or not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def _clean_region(mask, close=5, min_area_ratio=0.02):
    """把"改写区域"清理成可用于统计的块。

    **必须做闭运算**（实测踩过）：逐像素阈值得到的改写掩版是碎的——
    素材自己的浅色像素与器物同色、"没有超过阈值"，于是掩版里散落着大量孤立小点。
    直接用 `_largest_component` 会把真实的图案区域切成一块块碎片，只留下其中一片，
    统计量随之失去代表性。闭运算先把碎块连成片，再取主要连通块。
    """
    if mask is None or not mask.any():
        return mask
    array = mask.astype(np.uint8)
    if close and close > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close), int(close)))
        array = cv2.morphologyEx(array, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=8)
    if count <= 1:
        return array > 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros_like(array, dtype=bool)
    threshold = max(16.0, float(areas.max()) * float(min_area_ratio))
    for index, area in enumerate(areas, start=1):
        if area >= threshold:
            keep |= (labels == index)
    return keep


def _erode_band(mask, width):
    """区域的内侧带（离边界 `width` 像素以内），用于衡量区域内部性质。"""
    kernel = np.ones((max(1, int(width) * 2 + 1),) * 2, np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    return eroded


def _boundary_band(mask, width):
    """区域的边界带：外扩减去内缩。"""
    kernel = np.ones((max(1, int(width) * 2 + 1),) * 2, np.uint8)
    outer = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    inner = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    return outer & ~inner


def _corr(a, b):
    """皮尔逊相关系数；任一侧没有方差时返回 None（不做无意义的数）。"""
    if a is None or b is None or a.size < 32:
        return None
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _round(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# A. 结构保持
# ---------------------------------------------------------------------------
def structure_scores(source, result, low=None, high=None):
    """源图结构在成品里的保留程度（边缘重合 + 分块清晰度比）。"""
    grey_s = _grey(source)
    grey_r = _grey(result)
    canny_low = 80 if low is None else int(low)
    canny_high = 180 if high is None else int(high)
    edges_s = cv2.Canny(grey_s, canny_low, canny_high) > 0
    edges_r = cv2.Canny(grey_r, canny_low, canny_high) > 0
    # 边缘有一定的位移容忍：扩散重绘会让边缘移动 1~2 像素
    kernel = np.ones((3, 3), np.uint8)
    grown = cv2.dilate(edges_s.astype(np.uint8), kernel, iterations=1) > 0
    hit = float((edges_r & grown).sum())
    total_s = float(edges_s.sum())
    total_r = float(edges_r.sum())
    recall = hit / total_s if total_s > 0 else None
    precision = hit / total_r if total_r > 0 else None
    f1 = None
    if recall and precision and (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)

    # 分块高频能量比：找出"被抹平"的块
    collapse = _tile_collapse(grey_s, grey_r)
    return {
        "edge_recall": _round(recall),
        "edge_precision": _round(precision),
        "edge_f1": _round(f1),
        "sharpness_ratio": _round(collapse["global_ratio"]),
        "collapsed_tile_ratio": _round(collapse["collapsed_ratio"]),
        "tile_ratios": collapse["ratios"],
    }


def _tile_collapse(grey_s, grey_r, grid=TILE_GRID):
    """分块比较高频能量：哪些块在成品里被抹平了。

    为什么用分块而不是全图：全图比值会被大片未改写的背景稀释，
    实测"脸糊了但整体比值 0.9"这种情况完全可能，因此必须**定位到块**。
    """
    high_s = _high_frequency(grey_s)
    high_r = _high_frequency(grey_r)
    height, width = grey_s.shape[:2]
    ratios = []
    rows = np.linspace(0, height, grid + 1).astype(int)
    cols = np.linspace(0, width, grid + 1).astype(int)
    for i in range(grid):
        for j in range(grid):
            patch_s = high_s[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
            patch_r = high_r[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
            if patch_s.size < 32:
                continue
            energy_s = float(patch_s.std())
            energy_r = float(patch_r.std())
            if energy_s < MIN_TILE_STD:
                continue          # 源图本来就没有细节的块不参与
            ratios.append(energy_r / energy_s)
    if not ratios:
        return {"global_ratio": None, "collapsed_ratio": None, "ratios": []}
    array = np.asarray(ratios, dtype=np.float64)
    return {
        "global_ratio": float(high_r.std() / max(1e-6, high_s.std())),
        "collapsed_ratio": float((array < COLLAPSE_RATIO).mean()),
        "ratios": [_round(v, 3) for v in array],
    }


# ---------------------------------------------------------------------------
# B. 承载质量（贴纸感 / 融合度）
# ---------------------------------------------------------------------------
def carry_scores(source, result, mask, base=None, smooth_sigma=None, tile=32):
    """改写区域是"长在器物上"还是"贴上去的一块"。

    参数
    ----
    source : 器物**原图**（未承载的源图）——提供器物自身的明暗与纹理基准。
    result : 成品。
    mask   : 改写区域（bool ndarray）。None 时退回"与源图差异超过阈值"的判据。
    base   : 可选，"承载之前"的画面（已经过扩散重绘的器物）。
             不传时用 source 当基准（承载直接发生在源图上时这是对的）。

    ## 为什么"受光跟随"要按小块算，而不是整块算相关系数

    实测（`histories/scripts/_check_metrics.py`）教训：整块低频相关系数**分不开**
    "融合"与"平贴"。原因是标志自身也携带明暗结构（白底 + 深色图案），
    它的低频方差与器物的方差混在一起，相关系数被两者共同主导，
    平贴时也能得到接近 0 甚至正的值。

    真正有区分度的是：**把尺度局限在"器物受光变化"的那一档**（远小于标志本身），
    逐小块比较"成品与器物在该块内的明暗变化方向"。做法是对每一块
    （默认 32 像素）把两者的低频**标准化**（去掉各自的整体明暗与幅度），
    再比较：

      * `follow_cos`   ：标准化后的余弦相似度（方向一致率），平贴接近 0，融合接近 1；
      * `follow_resid` ：标准化后的平均绝对差，平贴接近 1，融合接近 0；
      * `grad_cos`     ：横向/纵向亮度梯度的余弦相似度（受光的坡向）。

    标准化是必要的：平贴的图案在明暗**幅度**上可能与器物差很多，
    但我们关心的是"它跟不跟着器物的受光走"，而不是"它亮多少"。
    """
    if mask is None:
        mask = _changed_mask(source, result)
    else:
        # **只统计真正被改写的地方**（实测踩过）：素材的形状 alpha 是有洞的
        # （徽记之间的空隙、花瓣之间），洞里的像素**没有被改写**、仍然完美地
        # 跟随器物受光。若把洞一起算进去，"平贴"也能拿到 0.61 的 `follow_cos`——
        # 指标测的是那些没被改的像素，结论完全失真。因此区域必须与"实际变化处"取交集。
        mask = np.asarray(mask).astype(bool) & _changed_mask(source, result)
    mask = _clean_region(mask)
    if mask is None or mask.sum() < 64:
        return {"available": False,
                "reason": f"改写区域过小（{0 if mask is None else int(mask.sum())} 像素）"}
    base = source if base is None else base

    grey_b = _grey(base).astype(np.float32)
    grey_r = _grey(result).astype(np.float32)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    span = int(min(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1))

    # 关注的尺度：由**器物自身的明暗变化**决定，而不是由区域大小决定。
    # 取"区域内该尺度上的明暗标准差 >= 阈值"的**最小**尺度——那才是这块表面真正
    # 存在受光起伏的尺度。尺度取太大（例如区域短边的 1/8）时，模糊核会比区域还宽，
    # 两侧的起伏被抹成常数、相关系数失去意义（实测踩过：器物低频 std 只有 0.12，
    # 于是"跟不跟随"根本无从判断）。
    if smooth_sigma is not None:
        sigma, low_b = float(smooth_sigma), _low_frequency(grey_b, float(smooth_sigma))
    else:
        sigma, low_b = _estimate_shading_scale(grey_b, mask, span)
    low_r = _low_frequency(grey_r, sigma)
    # 只轻微内缩：素材是"有洞的形状"，大幅腐蚀会把真实内容区域整块吃掉
    inside = _erode_band(mask, 2)
    if inside.sum() < 64:
        inside = mask

    follow_cos, follow_resid, tiles_used = _tiled_follow(
        low_b, low_r, inside, max(16, int(round(sigma * 2))))

    # 横向/纵向亮度梯度的余弦相似度（受光的坡向）
    gx_b, gy_b = _gradients(low_b)
    gx_r, gy_r = _gradients(low_r)
    grad_cos = _vector_cos(gx_b, gy_b, gx_r, gy_r, inside)

    # 边界"框"：改写区域边界的梯度是否远高于画面典型梯度
    band = _boundary_band(mask, max(1, span // 40))
    full_high = np.abs(_high_frequency(_grey(result)))
    edge_excess = None
    if band.sum() >= 32:
        edge_excess = float(np.median(full_high[band]) /
                            max(1e-6, np.median(full_high)))

    # 器物纹理是否还在（高频能量比）
    texture_retention = None
    high_b = _high_frequency(grey_b)
    high_r = _high_frequency(grey_r)
    if inside.sum() >= 32:
        texture_retention = float(high_r[inside].std() /
                                  max(1e-6, high_b[inside].std()))

    return {
        "available": True,
        "pixels": int(mask.sum()),
        "span": span,
        "follow_cos": _round(follow_cos),
        "follow_resid": _round(follow_resid),
        "grad_cos": _round(grad_cos),
        "boundary_edge_excess": _round(edge_excess),
        "texture_retention": _round(texture_retention),
        "solidity": _round(_solidity(mask)),
        "follow_tiles": tiles_used,
    }


def _estimate_shading_scale(grey_base, mask, span, threshold=1.5):
    """选出"这块表面上真正存在受光起伏"的尺度（高斯 sigma，像素）。

    做法：从**小**尺度往大试（`span/16` 起到 `span/2`），取第一个
    "区域内低频标准差 >= threshold" 的尺度。小尺度上若已有足够起伏，
    说明器物表面本身有局部明暗（高光、起伏），那才是"融合得像不像"该看的尺度；
    小尺度上全都平坦时（大面积均匀表面）才退到更大的尺度。

    返回 (sigma, 该尺度下的低频场)。全都平坦时返回最大尺度。
    """
    best = None
    for divisor in (16.0, 12.0, 8.0, 6.0, 4.0, 2.0):
        sigma = float(np.clip(span / divisor, 4.0, 240.0))
        low = _low_frequency(grey_base, sigma)
        values = low[mask]
        if values.size < 32:
            continue
        if best is None:
            best = (sigma, low)
        if float(values.std()) >= float(threshold):
            return sigma, low
    if best is None:
        low = _low_frequency(grey_base, 8.0)
        return 8.0, low
    return best


def _tiled_follow(low_base, low_result, inside, tile=32):
    """逐小块比较"成品是否跟着器物受光走"。

    ## 归一化必须用**器物**的尺度，不能用成品自己的尺度（实测踩过）

    第一版把两者各自标准化（除以各自的自身标准差）再算余弦。看起来合理，实际失败：
    平贴的图案在其内部**几乎恒定**，自身标准差接近于 0，于是那点数值噪声被放大成
    满量程的波动，余弦接近 0 甚至为负；而"融合"这一档因为确实跟随器物，
    两者都被同样的方式标准化，余弦反而不高。实测差异只有 0.12（-0.30 → -0.19），
    不足以作判据。

    正确做法是最小二乘意义下的**跟随**：
        out_tile ≈ bias + gamma * (base_tile - mean(base_tile))
    在每块里解出 `gamma`（用器件尺度归一，因此平贴的 gamma 接近 0），
    再看**残差**：平贴的残差接近器物自身的波动幅度（≈1），融合的残差接近 0。
    `follow_cos` 报告的是逐块 `gamma` 的**加权平均**（按器物的波动幅度加权），
    因此"跟随上受光"这件事在数字上直接可读。
    """
    height, width = low_base.shape[:2]
    gammas, weights, residuals = [], [], []
    for top in range(0, height - tile + 1, tile):
        for left in range(0, width - tile + 1, tile):
            block = inside[top:top + tile, left:left + tile]
            if block.mean() < 0.6:            # 只统计基本落在区域内的块
                continue
            base_patch = low_base[top:top + tile, left:left + tile].ravel().astype(np.float64)
            result_patch = low_result[top:top + tile, left:left + tile].ravel().astype(np.float64)
            base_std = float(base_patch.std())
            if base_std < 0.5:
                continue                      # 器物本身没有可比较的明暗变化
            a = base_patch - base_patch.mean()
            b = result_patch - result_patch.mean()
            gamma = float(np.dot(a, b) / max(1e-9, np.dot(a, a)))
            # 残差按**器物**的波动幅度归一
            residual = float(np.abs(b - gamma * a).mean() / base_std)
            gammas.append(gamma)
            weights.append(base_std)
            residuals.append(residual)
    if not gammas:
        return None, None, 0
    weights = np.asarray(weights, dtype=np.float64)
    gammas = np.asarray(gammas, dtype=np.float64)
    follow = float((gammas * weights).sum() / max(1e-9, weights.sum()))
    return follow, float(np.mean(residuals)), len(gammas)


def _gradients(grey):
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def _vector_cos(gx_a, gy_a, gx_b, gy_b, mask):
    """把 (gx, gy) 当向量场，算区域内的余弦相似度（受光坡向一致率）。"""
    if mask is None or mask.sum() < 32:
        return None
    a = np.stack([gx_a[mask], gy_a[mask]], axis=1).ravel()
    b = np.stack([gx_b[mask], gy_b[mask]], axis=1).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-6:
        return None
    return float(np.dot(a, b) / denominator)


def _solidity(mask):
    """区域面积 / 外接框面积。矩形贴片的实心度接近 1，自然分布的纹样显著更低。"""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    box = float((rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1))
    return float(mask.sum()) / max(1.0, box)


# ---------------------------------------------------------------------------
# C. 画面质量
# ---------------------------------------------------------------------------
def quality_scores(result, source=None):
    """整体画面质量：清晰度、死黑死白、带状伪影。"""
    grey = _grey(result)
    lap = cv2.Laplacian(grey, cv2.CV_32F)
    low = _low_frequency(grey, 12.0)
    report = {
        "laplacian_energy": _round(float(lap.var())),
        "clipping_low": _round(float((grey <= 2).mean())),
        "clipping_high": _round(float((grey >= 253).mean())),
        "mean_luminance": _round(float(grey.mean()), 2),
        # 带状伪影：低频在水平/垂直方向的二阶差分的中位绝对值
        "banding": _round(float(np.median(np.abs(np.diff(low, axis=0)))) or 0.0, 3),
    }
    if source is not None:
        delta = result.astype(np.float32) - source.astype(np.float32)
        report["mean_abs_diff"] = _round(float(np.abs(delta).mean()), 3)
        report["changed_ratio"] = _round(float(_changed_mask(source, result).mean()))
        # 明度整体位移：>0 说明成品被提亮，<0 被压暗
        report["luminance_shift"] = _round(
            float(_grey(result).mean() - _grey(source).mean()), 2)
    return report


def colour_palette_scores(image, palette, mask=None, top=6):
    """成品（或指定区域）的显著颜色与角色调色板的重合度。

    `palette` 允许是 `["#3a6e9b", (58, 110, 155), ...]` 这类混合输入。
    """
    rgb = _rgb(image)
    if rgb is None:
        return {"available": False, "reason": "无图像"}
    targets = [_parse_colour(item) for item in (palette or [])]
    targets = [item for item in targets if item is not None]
    if not targets:
        return {"available": False, "reason": "没有可解析的角色配色"}
    pixels = rgb[mask] if (mask is not None and np.asarray(mask).any()) else rgb.reshape(-1, 3)
    if pixels.size < 64:
        return {"available": False, "reason": "像素过少"}
    observed = _dominant_colours(pixels, count=top)
    distances = []
    for colour, weight in observed:
        best = min(_lab_distance(colour, target) for target in targets)
        distances.append((best, weight))
    weighted = sum(d * w for d, w in distances) / max(1e-6, sum(w for _, w in distances))
    return {
        "available": True,
        "palette_match_distance": _round(weighted, 2),
        "observed": ["#%02x%02x%02x" % tuple(int(v) for v in c) for c, _ in observed],
    }


def _parse_colour(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lstrip("#")
        if len(text) == 6:
            try:
                return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return None
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return tuple(int(round(float(v))) for v in value[:3])
    return None


def _dominant_colours(pixels, count=6, bins=4):
    """按量化直方图取显著颜色（不用 kmeans，保证确定性与速度）。"""
    array = pixels.reshape(-1, 3).astype(np.uint8)
    quant = (array // (256 // bins)).astype(np.int32)
    keys = quant[:, 0] * bins * bins + quant[:, 1] * bins + quant[:, 2]
    unique, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)[:count]
    out = []
    for index in order:
        mask = keys == unique[index]
        colour = array[mask].mean(axis=0)
        out.append((tuple(float(v) for v in colour), float(counts[index])))
    return out


def _lab_distance(colour_a, colour_b):
    a = np.uint8([[list(colour_a)]])
    b = np.uint8([[list(colour_b)]])
    lab_a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    lab_b = cv2.cvtColor(b, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    return float(np.linalg.norm(lab_a - lab_b))


# ---------------------------------------------------------------------------
# D. 身份（可选，依赖图像嵌入模型）
# ---------------------------------------------------------------------------
def embedding_identity(subject_crop, references):
    """主体裁剪与角色参考图的嵌入余弦相似度。

    本机没有可用的图像嵌入模型时**返回 None 并写明原因**——
    宁可给出"这项没测"，也不用"颜色距离"之类的代理指标冒充身份相似度。
    """
    try:
        embedder = _load_embedder()
    except Exception as exc:                       # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if embedder is None:
        return {"available": False,
                "reason": _EMBEDDER_CACHE.get("reason") or "本机没有可用的图像嵌入模型"}
    subject = _rgb(subject_crop)
    if subject is None:
        return {"available": False, "reason": "主体裁剪无效"}
    scores = []
    for reference in (references or []):
        image = _rgb(reference)
        if image is None:
            continue
        scores.append(embedder(subject, image))
    if not scores:
        return {"available": False, "reason": "没有可用的角色参考图"}
    return {"available": True, "max_cosine": _round(max(scores)),
            "mean_cosine": _round(float(np.mean(scores))), "count": len(scores)}


_EMBEDDER_CACHE = {}


def _load_embedder():
    """装一个**本机离线**可用的 CLIP 图像编码器；装不上返回 None（不联网、不下载）。

    ## 为什么用项目自带的那个编码器，而不是 `openai/clip-vit-*`

    实测（`histories/scripts/_probe_similarity_capability.py`）：
    本机的 HF 缓存里**没有任何图像编码器权重**（整个缓存 6.6 MB，只有 tokenizer 与 config），
    `openai/clip-vit-base-patch32` 取不到，而原来的 `except Exception: fn = None`
    把这件事**静默吞掉**——于是报告里 `identity` 一项永远显示"不可用"，
    看起来像"这项没测"，其实是"代码根本没找到模型"。

    而 `models/ip-adapter/models/image_encoder` 是一份**完整的 CLIP ViT-H/14** 视觉编码器
    （2.53 GB、632M 参数、输出 1024 维），随项目一起分发、完全离线可用，
    实测自比余弦 1.000000、不同图之间 0.69~0.98，区分度正常。
    它没有 `preprocessor_config.json`，因此这里**手工构造** `CLIPImageProcessor`。
    """
    if "fn" in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE["fn"]
    if os.environ.get("ENDLESS_METRICS_EMBEDDING", "auto") == "off":
        _EMBEDDER_CACHE["fn"] = None
        return None
    fn = None
    reason = ""
    try:
        import config
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        encoder_dir = os.path.join(config.MODEL_DIR, "ip-adapter", "models", "image_encoder")
        if not os.path.isdir(encoder_dir):
            reason = f"本机没有离线图像编码器: {encoder_dir}"
            raise FileNotFoundError(reason)
        # 目录里没有 preprocessor_config.json，按 CLIP 的标准预处理参数手工构造
        processor = CLIPImageProcessor(
            do_resize=True, size={"shortest_edge": 224}, do_center_crop=True,
            crop_size={"height": 224, "width": 224},
            do_normalize=True, image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711], do_convert_rgb=True)
        model = CLIPVisionModelWithProjection.from_pretrained(
            encoder_dir, local_files_only=True).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _embed(first, second):
            with torch.no_grad():
                inputs = processor(images=[first, second], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model.to(device)
                out = model(**inputs).image_embeds
                out = out / out.norm(dim=-1, keepdim=True)
                return float((out[0] * out[1]).sum())

        fn = _embed
    except Exception as exc:                       # noqa: BLE001
        reason = reason or f"{type(exc).__name__}: {exc}"
        fn = None
    _EMBEDDER_CACHE["fn"] = fn
    _EMBEDDER_CACHE["reason"] = reason
    return fn


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def subject_box(mask):
    """掩版的外接框 (left, top, right, bottom)；空掩版返回 None。"""
    if mask is None or not np.asarray(mask).any():
        return None
    array = np.asarray(mask).astype(bool)
    rows = np.where(array.any(axis=1))[0]
    cols = np.where(array.any(axis=0))[0]
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def crop_box(image, box, pad=0.0):
    """按框裁剪（可外扩），返回 PIL 图；框无效返回 None。"""
    if box is None:
        return None
    rgb = _rgb(image)
    height, width = rgb.shape[:2]
    left, top, right, bottom = box
    pad_x = int((right - left) * pad)
    pad_y = int((bottom - top) * pad)
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)
    if right - left < 8 or bottom - top < 8:
        return None
    from PIL import Image
    return Image.fromarray(rgb[top:bottom, left:right])


def evaluate(source, result, region_mask=None, base=None, palette=None,
             reference_images=None, embed=True):
    """把上面各组合并成一份报告（全部字段缺失时写 None，不编数字）。"""
    src = _rgb(source)
    res = _rgb(result)
    if src is None or res is None:
        return {"available": False, "reason": "源图或成品无法读取"}
    res = _aligned(src, res)
    mask = None if region_mask is None else np.asarray(region_mask).astype(bool)
    if mask is not None and mask.shape[:2] != src.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (src.shape[1], src.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
    report = {
        "available": True,
        "size": [int(src.shape[1]), int(src.shape[0])],
        "structure": structure_scores(src, res),
        "quality": quality_scores(res, src),
    }
    report["carry"] = carry_scores(src, res, mask, base=base)
    if palette:
        report["palette"] = colour_palette_scores(res, palette, mask=mask)
    if embed:
        box = subject_box(mask) if mask is not None else None
        crop = crop_box(res, box, pad=0.05) if box else None
        report["identity"] = embedding_identity(crop, reference_images)
    return report


# ---------------------------------------------------------------------------
# 报告与汇总
# ---------------------------------------------------------------------------
# 每项的含义与方向（越大越好 / 越小越好），供判定与打印使用
DIRECTIONS = {
    "edge_recall": ("高", 0.55),
    "edge_f1": ("高", 0.50),
    "sharpness_ratio": ("接近1", 0.60),
    "collapsed_tile_ratio": ("低", 0.25),
    "follow_cos": ("高", 0.50),
    "follow_resid": ("低", 0.75),
    "grad_cos": ("高", 0.40),
    "boundary_edge_excess": ("低", 2.0),
    "clipping_high": ("低", 0.01),
    "palette_match_distance": ("低", 90.0),
    "max_cosine": ("高", 0.75),
}

# 判定阈值集中在这里，方便根据基线实测再标定（不在指标里写死结论）
def verdict(report):
    """把报告翻译成 {指标: (值, 门槛, 是否达标)}；缺失的指标不参与判定。"""
    flat = {}
    for group in ("structure", "quality", "carry", "palette", "identity"):
        section = report.get(group) or {}
        for key, value in section.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                flat.setdefault(key, value)
    result = {}
    for key, (direction, threshold) in DIRECTIONS.items():
        if key not in flat:
            continue
        value = flat[key]
        if key in ("sharpness_ratio", "texture_retention"):
            ok = value >= threshold
        elif direction == "低":
            ok = value <= threshold
        else:
            ok = value >= threshold
        result[key] = {"value": value, "threshold": threshold, "ok": bool(ok)}
    return result


def format_report(report, indent="  "):
    """把报告打成人类可读的多行文本（数值与判定并列）。"""
    if not report.get("available"):
        return f"{indent}评测不可用: {report.get('reason')}"
    lines = []
    checks = verdict(report)
    group_names = {"structure": "结构保持", "quality": "画面质量",
                   "carry": "承载融合", "palette": "角色配色", "identity": "身份相似"}
    for group, title in group_names.items():
        section = report.get(group) or {}
        if not section:
            continue
        shown = []
        for key, value in section.items():
            if value is None or isinstance(value, (list, dict)):
                continue
            mark = ""
            if key in checks:
                mark = " ✓" if checks[key]["ok"] else " ✗"
            shown.append(f"{key}={value}{mark}")
        if shown:
            lines.append(f"{indent}{title}: " + ", ".join(shown))
    failures = [k for k, v in checks.items() if not v["ok"]]
    lines.append(f"{indent}未达标项: {failures if failures else '（无）'}")
    return "\n".join(lines)


def summarise(reports):
    """把多张图的报告汇总成 {指标: 平均值}（None 不计入）。"""
    buckets = {}
    for report in reports:
        if not report or not report.get("available"):
            continue
        for group in ("structure", "quality", "carry", "palette", "identity"):
            for key, value in (report.get(group) or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    buckets.setdefault(key, []).append(float(value))
    return {key: _round(float(np.mean(values)), 4)
            for key, values in buckets.items() if values}
