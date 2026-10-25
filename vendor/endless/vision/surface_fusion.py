# -*- coding: utf-8 -*-
"""vision/surface_fusion.py —— 把标志**印在器物表面上**（而不是贴在画面上）。

## 要解决的问题（用户原话与实测依据）

用户对非人物效果的判定是"简单的单独标志图的粘贴""并没有体现模型学习了角色的视觉特征
并自然得运用"。此前 `vision/mark_carry.py` 的做法是 alpha 合成：把素材按 alpha 叠到器物
像素上。**alpha 合成只能改颜色，改不了光照关系**，因此无论素材多干净、边缘多羽化，
观感仍然是"贴上去的一块"：

* 素材自带原服装的受光，与器物的受光方向、明暗分布无关；
* 器物表面的颗粒/釉面纹理被整块盖掉（`texture_retention` 实测 5.8~18.5，远偏离 1）；
* 边界留下一圈"框"（`boundary_edge_excess` 实测 12.0，融合后应降到 2 以内）；
* 逐小块比较"是否跟随器物受光"的 `follow_cos` 实测 -0.20（融合后应 > 0.5）。

## 做法：图像调和（image harmonization）的确定性版本

按顺序做四件事，每一步都对应上面的一条症状：

1. **去底**：素材背景透掉（复用 `mark_shape` 的判据），得到形状 alpha；
2. **去光**：把素材自身的低频明暗**减掉**，只留"图案本身的色与花纹层次"。
   素材的明暗是原服装的受光，带到器物上就是"照片贴纸"的直接原因；
3. **重光**：把器物的**低频明暗场**乘回去——纹样因此跟随器物受光（`follow_cos`）；
   同时在**柱面**上把标记横向收窄、上下边弯成弧（`warp_cylinder`），
   避免出现"直的矩形块"；
4. **合材**：把器物的**高频颗粒**按比例混进标记内部（`texture_retention` 回到 1 附近），
   并只沿轮廓羽化，不做矩形硬边（`boundary_edge_excess` 下降）。

第 2、3 步合起来就是"图案提供颜色与花纹，器物提供明暗"，是本模块的核心；
它是确定性的，因此结果可复现、可逐项核对，不引入扩散模型"随机画错"的风险。

## 与扩散的关系

本模块**不排斥**扩散：`test.py` 仍可在融合后对标志邻域做一次低强度的 img2img
（"收边"），把高光与微边缘统一起来。但扩散是可选项，不是唯一手段——
实测"让扩散直接画到器物表面"是画不出来的（见 `_probe_diffusion_paint.py`）。

## 不做什么

* 不改器物轮廓外一个像素（调用方保证）；
* 不"发明"素材里没有的图案（只在有真实证据素材时工作）；
* 不假设角色或题材（判据全部来自目标器物与素材自身的像素统计）。
"""
import numpy as np
import cv2

import config


# ---------------------------------------------------------------------------
# 1) 去底：形状 alpha
# ---------------------------------------------------------------------------
def shape_alpha(rgba, min_coverage=0.02):
    """取素材的形状 alpha（0~255 uint8）；拿不到可靠形状时返回 None。

    `rgba` 的第 4 通道若已经是"分离过的形状"（`mark_shape.extract_shape` 的输出），
    就直接用它；若是 `mark_carry.load_asset` 那种"只有矩形羽化"的 alpha，
    则**不能**当成形状（否则会贴上一整块方形），此时返回 None，
    由调用方决定是否改用 `mark_shape.extract_shape` 重新分离。
    """
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        return None
    alpha = rgba[:, :, 3].astype(np.float32)
    if alpha.size == 0:
        return None
    solid = alpha > 200
    coverage = float(solid.mean())
    if coverage < float(min_coverage):
        return None
    # 判据：把 alpha 当二值图看，若"实心区"几乎等于整幅，说明它只是矩形羽化，
    # 不是形状分离的结果。
    if coverage > 0.985:
        return None
    return np.clip(alpha, 0, 255).astype(np.uint8)


def _trim_to_shape(rgb, alpha, pad=2):
    """按 alpha 的外接框裁掉四周空白（避免"素材四周的透明边"影响缩放比例）。"""
    rows = np.where(alpha.max(axis=1) > 8)[0]
    cols = np.where(alpha.max(axis=0) > 8)[0]
    if rows.size == 0 or cols.size == 0:
        return rgb, alpha
    top = max(0, int(rows[0]) - pad)
    bottom = min(alpha.shape[0], int(rows[-1]) + 1 + pad)
    left = max(0, int(cols[0]) - pad)
    right = min(alpha.shape[1], int(cols[-1]) + 1 + pad)
    return rgb[top:bottom, left:right], alpha[top:bottom, left:right]


# ---------------------------------------------------------------------------
# 1b) 背景抠除（素材自带背景时）
# ---------------------------------------------------------------------------
def _border_colour(rgb, band=0.12):
    """取素材四边内侧一圈的**平均配色**与离散度，作为"背景色"的估计。"""
    height, width = rgb.shape[:2]
    pad = max(1, int(min(height, width) * float(band)))
    ring = np.concatenate([
        rgb[:pad, :, :].reshape(-1, 3), rgb[-pad:, :, :].reshape(-1, 3),
        rgb[:, :pad, :].reshape(-1, 3), rgb[:, -pad:, :].reshape(-1, 3)])
    values = ring.astype(np.float32)
    return values.mean(axis=0), float(values.std(axis=0).mean()), pad


def auto_matte(rgba, tolerance=None, ramp=None):
    """按"与边框背景色的色度距离"自动生成形状 alpha（背景透掉，标志留下）。

    为什么必须有这一条（实测，本轮）：`crop_009`（所谓的白色樱花裁剪）实际内容是
    **一片平坦的蓝色 + 一块粉紫金属斜边**，而 `mark_shape` 的背景判据按 LAB 距离
    只透掉了 73% 的像素，剩下的**平坦蓝背景被当成标志**一起贴了上去——
    出图上是三块带直角边的方形贴片，一眼就是"贴上去的一张图"。
    素材本身有背景并不是罕见情况，不能假设上游的形状分离一定干净。

    判据只用素材自身的像素统计（与 `mark_shape` 一致，不引入语义）：
      1. 取四边内侧一圈的平均色作为背景色估计；
      2. **测地距离**（在"与边框差异小于容差"的掩版上做距离变换）而不是欧氏距离——
         徽记内部的颜色往往与背景相近，欧氏距离会把它整块剔掉，只剩一圈描边；
         测地距离则要求"从边缘一路连通到背景"才判为背景；
      3. 边框本身离散度很大时说明没有可辨识的平坦背景，直接返回 None（不硬抠）。

    返回 0~255 的 float32 alpha，或 None（无法可靠抠除）。
    """
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
        return None
    import cv2
    rgb = rgba[:, :, :3]
    background, spread, pad = _border_colour(rgb)
    limit = float(getattr(config, "MARK_FUSION_BG_SPREAD", 26.0)) \
        if tolerance is None else float(tolerance) * 2.0
    if spread > limit:
        return None                      # 边框颜色本身就很杂 -> 没有平坦背景可抠
    tolerance = float(getattr(config, "MARK_FUSION_MATTE_TOLERANCE", 22.0)) \
        if tolerance is None else float(tolerance)
    ramp = float(getattr(config, "MARK_FUSION_MATTE_RAMP", 10.0)) \
        if ramp is None else float(ramp)
    distance = np.sqrt(((rgb.astype(np.float32) - background) ** 2).sum(axis=2))
    similar = (distance < tolerance).astype(np.uint8)
    # 测地距离：从边框出发、在"与背景相似"的像素上做距离变换
    if similar.any():
        geodesic = cv2.distanceTransform(1 - similar, cv2.DIST_L2, 3)
    else:
        geodesic = np.full(similar.shape, 1e6, np.float32)
    alpha = np.clip((geodesic - 1.0) / max(1.0, ramp), 0.0, 1.0)
    # 边框本身必须被判为背景，否则说明这一步没有起作用
    ring = np.concatenate([alpha[:pad, :].ravel(), alpha[-pad:, :].ravel(),
                           alpha[:, :pad].ravel(), alpha[:, -pad:].ravel()])
    coverage = float((alpha > 0.5).mean())
    if ring.mean() > 0.25 or coverage < 0.02 or coverage > 0.95:
        return None
    return alpha


def asset_usable(rgba, min_side=None, max_blockiness=None, max_fill=None,
                 min_detail=None):
    """判断一份素材能不能被**印**到器物上；返回 (是否可用, 说明)。

    与 `carry_compose.asset_purity` 的分工（两者互补，判据不重复）：
      * `asset_purity` 管"这像不像一个标志"（插画 / 条状局部 / 零散碎块）；
      * 本函数管"这份素材能不能被印上去"——形状必须是**可分离的轮廓**，
        不能是一整块近似矩形的东西。

    ## 为什么必须有这条判据（实测缺陷）

    `crop_009`（所谓的白色樱花裁剪）实际内容是"一片平坦的蓝色 + 一块粉紫金属斜边"，
    它的 alpha 在**自身外接框内**的实心度 0.58、覆盖率 0.27——
    `carry_compose` 的闸门全部通过（覆盖率不高、长宽比正常、不是碎块），
    于是它被当成"白色樱花图案"印到杯子上，出图上是**三块带直角边的方形贴片**，
    一眼就是"贴上去的一张图"（`output/_probe/rect_zoom.png`）。

    外接框实心度是关键判据：真正的徽记在它自己的框里不会填得这么满。
    另外要求形状边界有足够的"轮廓像素占比"（`min_detail`），
    一整块纯色（连边界都没有起伏）也判为不可用。
    """
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
        return False, "素材为空或没有 alpha"
    alpha = shape_alpha(rgba)
    if alpha is None:
        return False, "没有可用的形状 alpha（只有矩形羽化）"
    height, width = rgba.shape[:2]
    if min_side is None:
        min_side = int(getattr(config, "MARK_FUSION_MIN_ASSET_SIDE", 48))
    if min(height, width) < int(min_side):
        return False, f"素材过小 {width}x{height}（< {int(min_side)}px），放大后必然糊"
    blockiness = _blockiness(alpha)
    fill = _fill_ratio(alpha)
    max_blockiness = float(getattr(config, "MARK_FUSION_MAX_BLOCKINESS", 0.62)) \
        if max_blockiness is None else float(max_blockiness)
    max_fill = float(getattr(config, "MARK_FUSION_MAX_FILL", 0.90)) \
        if max_fill is None else float(max_fill)
    if blockiness is not None and blockiness > max_blockiness and fill > max_fill:
        return False, (f"形状近乎整块矩形（实心度 {blockiness:.2f}、覆盖率 {fill:.2f}）")
    # 边界起伏：形状边界上的像素应当有明显的明暗变化，否则是"一整块纯色"
    import cv2
    binary = (alpha > 127).astype(np.uint8)
    contour = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT,
                               np.ones((3, 3), np.uint8)) > 0
    if int(contour.sum()) >= 24:
        grey = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        variation = float(grey[contour].std())
        limit = float(getattr(config, "MARK_FUSION_MIN_CONTOUR_DETAIL", 4.0)) \
            if min_detail is None else float(min_detail)
        if variation < limit:
            return False, (f"形状边界几乎没有起伏（标准差 {variation:.1f}），"
                           f"判为一整块纯色，不是图案")
    return True, (f"可用：{width}x{height}，实心度 "
                  f"{'n/a' if blockiness is None else round(blockiness, 2)}，"
                  f"覆盖率 {fill:.2f}")


def _fill_ratio(alpha, threshold=0.5):
    """形状 alpha 的覆盖率（占整幅素材的比例）。"""
    if alpha is None:
        return 0.0
    return float((np.asarray(alpha) > (threshold * 255.0)).mean())


def _blockiness(alpha, threshold=0.5):
    """形状在其外接框内的**实心度**：接近 1 说明这是一整块矩形，不是形状。"""
    binary = np.asarray(alpha) > (threshold * 255.0)
    if not binary.any():
        return None
    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]
    box = binary[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    if box.size == 0:
        return None
    return float(box.mean())


# ---------------------------------------------------------------------------
# 2) 器物表面的低频明暗场
# ---------------------------------------------------------------------------
def _surface_relative_shading(destination, box, sigma):
    """器物表面的**相对受光场**（均值 1.0），返回 (框内切片, 参考明度)。

    ## 必须在**整幅**画布上算，不能在放置框的小切片上算（实测踩过）

    第一版把 `destination[top:bottom, left:right]` 切片后直接模糊，结果
    `受光场 std = 0.001`——**几乎是常数**。原因是高斯核尺寸按 sigma 放大到 400 像素，
    比切片本身还大，边界复制把整块拉成了平的；于是"重光"这一步等于没做，
    出图上纹样当然不跟随器物受光（实测 `follow_cos` 只有 -0.36）。
    整幅画布上有真实的梯度信息，模糊后再裁出框内才有意义。
    """
    grey = cv2.cvtColor(destination, cv2.COLOR_RGB2GRAY).astype(np.float32)
    size = int(max(5, round(sigma * 4)) | 1)
    low = cv2.GaussianBlur(grey, (size, size), float(sigma))
    left, top, right, bottom = box
    patch = low[top:bottom, left:right]
    reference = max(1.0, float(np.median(patch)))
    return patch / reference, reference


def _surface_texture(destination, box, sigma, radius):
    """器物自身的高频细节，**归一化到自身低频的尺度**上；返回 (框内切片, 明度场)。

    为什么必须归一化（实测）：直接混入原始高频时，幅度与"器物为什么这么亮"无关，
    实测 `texture_retention` 反而是 18（外来高频压过器物纹理），比平贴更差。
    这里把细节按"器物自身低频"缩放，等于"这块表面在这个明度下应有的颗粒量"，
    混入后局部高频能量才回到器物自己的水平。
    """
    left, top, right, bottom = box
    grey = cv2.cvtColor(destination, cv2.COLOR_RGB2GRAY).astype(np.float32)
    low_size = int(max(5, round(sigma * 4)) | 1)
    low = cv2.GaussianBlur(grey, (low_size, low_size), float(sigma))
    high_size = int(max(3, int(radius) * 2 + 1) | 1)
    high = grey - cv2.GaussianBlur(grey, (high_size, high_size), float(radius))
    level = np.maximum(8.0, low)
    return high[top:bottom, left:right] / level[top:bottom, left:right], \
        level[top:bottom, left:right]


def relight(flat_rgb, detail, shading, target_low, detail_amount, texture,
            texture_amount):
    """把器物明暗乘回图案，并混入器物纹理。

    `shading`      ：与素材同尺寸的器物相对受光场（均值 1）；
    `target_low`   ：器物在该处的低频明度——决定图案整体明暗的**量级**；
    `detail`       ：素材自身高频（花纹层次），按 `detail_amount` 保留；
    `texture`      ：器物高频细节（已按器物低频归一），按 `texture_amount` 混入。
    """
    out = flat_rgb.astype(np.float32)
    grey = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    flat_mean = max(1.0, float(np.median(grey)))
    # 图案的"平光"明度先对齐到器物明度，再乘器物相对受光
    out = out * (float(np.median(target_low)) / flat_mean)
    out = out * shading[:, :, None]
    # 图案自身的高频按"相对自身明度"的比例混回（乘法，避免改变色相）
    if detail_amount > 0 and detail is not None:
        relative = detail / np.maximum(30.0, grey)
        out = out * (1.0 + float(detail_amount) * relative)[:, :, None]
    # 器物纹理：按器物自身低频的量级换算成绝对像素，再加回
    if texture_amount > 0 and texture is not None:
        out = out + (texture * float(texture_amount) * target_low)[:, :, None]
    return np.clip(out, 0, 255)


# ---------------------------------------------------------------------------
# 3) 柱面适配
# ---------------------------------------------------------------------------
def warp_cylinder(patch_rgb, patch_alpha=None, strength=0.0):
    """把平面素材按**柱面**预畸变：中间拉伸、两侧压缩，上下边弯成弧。

    为什么必须有这一步（实测）：直接贴上去是一块**上下边水平、左右边垂直**的矩形，
    在杯身上一眼就是"贴纸"。真实印在柱面上的图案，
    两侧因为远离视线中心而被压缩，上下边沿柱面弯曲。

    做法（确定性、不需要三维重建）：把横向采样位置按
        x_source = 0.5 + (x - 0.5) * (1 - 0.65 * strength * (2x - 1)^2)
    重采样——**中间几乎不压缩、越靠边压缩越多**（真实的柱面在视线中心处
    「看起来」没有被压缩，压缩集中在两侧）；再按 `strength` 给每列加一个与
    "离中心距离平方"成正比的纵向位移，让上下边沿柱面起弧。
    `strength=0` 时原样返回（平面器物不受影响）。

    注意 `0.45` 这个系数：第一版用 0.45 时，60 像素宽的图案最边缘也只挪动 2 像素，
    在整数网格上**重采样后与原图逐像素相同**（实测：`maxdiff=0`，
    等于柱面适配根本没生效）。现在用 0.65，边缘压缩量约为 `strength` 本身的比例。
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-3 or patch_rgb is None or patch_rgb.size == 0:
        return patch_rgb, patch_alpha
    height, width = patch_rgb.shape[:2]
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32),
                         np.arange(height, dtype=np.float32))
    # 横向压缩：越靠边压缩越多，中心几乎不动
    offset = (xx / max(1.0, width - 1)) * 2.0 - 1.0            # -1..1
    source_x = (0.5 + (offset * (1.0 - 0.65 * strength * offset ** 2)) * 0.5) \
        * (width - 1)
    # 纵向弯曲：上下边沿柱面拱起
    bow = strength * 0.12 * height
    source_y = yy + bow * (offset ** 2) - bow * 0.5
    map_x = np.clip(source_x, 0, width - 1).astype(np.float32)
    map_y = np.clip(source_y, 0, height - 1).astype(np.float32)
    warped = cv2.remap(patch_rgb, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    warped_alpha = None
    if patch_alpha is not None:
        warped_alpha = cv2.remap(patch_alpha, map_x, map_y, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, warped_alpha


# ---------------------------------------------------------------------------
# 4) 去光 + 重光
# ---------------------------------------------------------------------------
def _background_manifold(patch_rgb, radius):
    """估计"如果这块素材上没有图案，底材的明暗该是多少"。

    这是**去光**的关键，也是实测改过来的做法。最初用高斯低通减掉素材明暗，
    结果失败：白底徽记里的深色图案会**拉低**模糊值，于是"去光"之后的底子偏暗，
    重光后整个标志比器物暗一截，`follow_cos` 仍然是负的（实测 -0.22）。

    正确做法是**非对称**估计：
      * 灰度做**闭运算**（先膨胀后腐蚀）→ 填掉深色图案，保留亮底的高值；
      * 灰度做**开运算**（先腐蚀后膨胀）→ 抹掉亮色图案，保留暗底的低值；
      * 两者取平均 → 得到"图案不在时"的底材明暗。
    对"白底 + 深色图案"的徽记，这个估计是白底；对"深底 + 亮图案"同样成立。
    """
    grey = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    size = int(max(3, int(radius) * 2 + 1) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)
    manifold = 0.5 * (closed + opened)
    # 再轻微平滑，避免形态学的块状边界进入重光结果（否则会出现阶梯状明暗）
    smooth_size = int(max(5, int(radius) * 2 + 1) | 1)
    return cv2.GaussianBlur(manifold, (smooth_size, smooth_size), max(1.0, radius * 0.5))


def _delight(patch_rgb, shading_sigma):
    """减掉素材自身的低频明暗，只留图案的色与花纹层次。

    返回 (base_rgb, detail, manifold)：
      * `base_rgb`：素材颜色被"平光化"（按底材明暗归一），保留色相与饱和度关系；
      * `detail`  ：素材自身的高频（描边、花瓣层次）；
      * `manifold`：底材明暗估计，供调用方核对去光是否合理。
    """
    rgb = patch_rgb.astype(np.float32)
    radius = int(np.clip(shading_sigma * 0.6, 3.0, 60.0))
    manifold = _background_manifold(patch_rgb, radius)
    reference = max(1.0, float(np.median(manifold)))
    gain = reference / np.maximum(1.0, manifold)
    flat = np.clip(rgb * gain[:, :, None], 0, 255)
    grey = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = grey - manifold
    return flat, detail, manifold


def relight(flat_rgb, detail, shading, target_low, detail_amount, texture,
            texture_amount, conformity=None):
    """把**器物的光**接到图案上，并混入器物纹理。

    参数
    ----
    `flat_rgb`  ：去光后的图案（颜色与花纹层次保留）；
    `shading`   ：器物相对受光场（均值 1，逐像素）；
    `target_low` ：器物的低频明度场（逐像素）；
    `detail`    ：素材自身高频（花纹层次），按 `detail_amount` 保留；
    `texture`   ：器物高频细节（已按器物低频归一），按 `texture_amount` 混入；
    `conformity`：图案整体明度**向器物看齐**的比例（1=完全看齐，0=完全保留素材本色）。

    ## 这一步做错过两次，都记在这里（实测）

    **第一版**：`图案 × (器物均值/图案均值) × 器物相对受光`。结果图案变成近乎常数——
    去光之后图案明度已经是平的，再乘一个均值恒定的系数仍然是平的；
    同足迹斜率只有 **0.19**，比平贴（0.96）还差。教训：**去光过头会把图案变成死色块**。

    **第二版（本版）**：把"跟随器物受光"与"整体明度看齐"分成两件事：
      * **跟随**是乘法：`图案 × 器物相对受光`——亮处的图案更亮、暗处更暗，
        这是"印在曲面上"最本质的一条，且**不会抹掉图案自身的明暗结构**；
      * **看齐**是插值：`图案 + conformity × (器物明度 − 图案) × 图案明度比例`——
        让浅色器物上的图案不至于"更白"，同时保留图案的对比与色相；
      * **层次**与**材质**分别加回，保证图案可辨认、且长在同一个材质上。

    由此得到的行为：`conformity=0` 时图案的明暗完全来自自己（但会被器物受光调制），
    `conformity=1` 时图案的明度分布与器物一致——两种极端都可测，参数可标定。
    """
    flat = flat_rgb.astype(np.float32)
    grey = cv2.cvtColor(flat.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    flat_mean = max(1.0, float(np.median(grey)))
    surface = np.asarray(target_low, dtype=np.float32)
    relative = np.asarray(shading, dtype=np.float32)

    k = float(getattr(config, "MARK_FUSION_CONFORMITY", 0.45)) \
        if conformity is None else float(conformity)
    k = float(np.clip(k, 0.0, 1.0))

    # 1) 图案整体明度向器物看齐（保留图案自己的明暗层次与色相）
    ratio = (k * flat / flat_mean)
    out = flat + (surface[:, :, None] - flat) * ratio
    # 2) 跟随器物受光（乘法，这才是"印在曲面上"的关键）
    out = out * relative[:, :, None]
    # 3) 图案自身的花纹层次按同一比例加回（暗处对比更小才自然）
    if detail_amount > 0 and detail is not None:
        out = out + (detail * float(detail_amount))[:, :, None] * \
            (surface / flat_mean * relative)[:, :, None]
    # 4) 器物纹理（按器物低频换算成绝对像素）
    if texture_amount > 0 and texture is not None:
        out = out + (texture * float(texture_amount) * surface)[:, :, None]
    return np.clip(out, 0, 255)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def prepare_patch(rgba, canvas, box, region_mask=None, cylinder=None, audit=None):
    """算出一块"已经按器物重光、合材"的图案补片，返回 (补片, 权重, 审计)。

    与 `fuse_onto_surface` 的区别只是**不写回画布**。拆出来有两个用途：
      1. 让"承印"与"平贴"能在**同一份补片、同一份权重**上做对照
         （`histories/scripts/_check_fusion.py` 的同足迹比较）——
         否则两种做法的足迹形状不同，指标不可比（实测踩过）；
      2. 出图诊断时可以单独看"补片长什么样"，不必从成品反推。

    `weight` 是 0~1 的落笔权重（形状 alpha × 区域限制 × 羽化）。
    """
    audit = dict(audit or {})
    if rgba is None or canvas is None:
        return {"ok": False, "reason": "缺少素材或画布", "audit": audit}
    alpha = shape_alpha(rgba)
    matte_note = ""
    if alpha is not None:
        # 形状"太方"时不要直接用（实测缺陷）：`crop_009` 的 alpha 在外接框内实心度很高，
        # 贴出来就是一块带直角边的方形贴片，一眼就是"贴上去的一张图"。
        #
        # **两个条件同时成立才判为矩形块**（实测教训，必须记下来）：
        #   1. `blockiness` 高——它在自身外接框里是实心的；
        #   2. `_fill_ratio` 也高——它同时**几乎占满整幅素材**。
        # 只看 blockiness 会误伤干净的大素材：`ref_021.png`（1349x1248 的黑色樱花徽记）
        # 的 blockiness 是 0.64，但那是因为一张大图里花朵本来就占得密，
        # 它的形状分离其实是**正确**的（alpha 覆盖率 0.35，背景已透掉）。
        # 它正是"白色樱花图案"最需要的干净素材，绝不能因为一条判据被丢掉。
        blockiness = _blockiness(alpha)
        fill = _fill_ratio(alpha)
        limit = float(getattr(config, "MARK_FUSION_MAX_BLOCKINESS", 0.62))
        fill_limit = float(getattr(config, "MARK_FUSION_MAX_FILL", 0.90))
        if blockiness is not None and blockiness > limit and fill > fill_limit:
            rematte = auto_matte(rgba)
            if rematte is not None:
                alpha = np.clip(rematte * 255.0, 0, 255).astype(np.uint8)
                matte_note = (f"形状实心度 {blockiness:.2f}、覆盖率 {fill:.2f}，"
                              f"判为整块矩形；改用背景抠除"
                              f"（抠后覆盖 {_fill_ratio(alpha):.2f}）")
            else:
                return {"ok": False,
                        "reason": (f"素材形状近乎整块矩形（实心度 {blockiness:.2f}、"
                                   f"覆盖率 {fill:.2f}），且无法按背景色抠除；"
                                   f"拒绝贴出方形贴片"),
                        "audit": {**audit, "blockiness": round(blockiness, 3),
                                  "fill": round(fill, 3)}}
    if alpha is None:
        return {"ok": False,
                "reason": "素材没有可用的形状 alpha（只有矩形羽化），拒绝做矩形硬贴",
                "audit": audit}
    left, top, right, bottom = [int(round(v)) for v in box]
    left = max(0, left)
    top = max(0, top)
    right = min(canvas.shape[1], right)
    bottom = min(canvas.shape[0], bottom)
    if right - left < 12 or bottom - top < 12:
        return {"ok": False, "reason": "放置框过小", "audit": audit}
    width, height = right - left, bottom - top

    rgb = rgba[:, :, :3]
    rgb, alpha = _trim_to_shape(rgb, alpha)
    if rgb.size == 0:
        return {"ok": False, "reason": "素材裁剪后为空", "audit": audit}

    # 素材缩放到放置框（保持形状自身的宽高比）
    patch = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    patch_alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)

    # 关注的尺度：图案的 1/4 边长——比它小的算"花纹层次"，比它大的算"受光"
    sigma = float(np.clip(min(width, height) / 4.0, 3.0, 40.0))

    # 1) 去光（拿掉素材自身受光）
    flat, detail, manifold = _delight(patch, sigma)

    # 2) 柱面预畸变
    if cylinder is None:
        cylinder = float(getattr(config, "MARK_CARRY_CYLINDER", 0.0))
    flat, patch_alpha = warp_cylinder(flat, patch_alpha, cylinder)
    detail = cv2.resize(detail, (flat.shape[1], flat.shape[0]),
                        interpolation=cv2.INTER_LINEAR)

    # 3) 取器物的受光场与纹理（在**整幅画布**上算，再裁到框内）
    surface = canvas[top:bottom, left:right]
    sub_mask = None
    if region_mask is not None:
        sub_mask = np.asarray(region_mask)[top:bottom, left:right].astype(bool)
        if sub_mask.shape[:2] != surface.shape[:2]:
            sub_mask = None
    # 受光场与明度场**用不同的尺度**（实测踩过：同一尺度会让"跟随"失去意义）：
    #   * 相对受光要在"器物明暗起伏"的尺度上取，才能体现高光与背光 → sigma 较小；
    #   * 低频明度场要在"比图案更大"的尺度上取，才是"这块地方有多亮"而不是图案自身
    #     的明暗 → sigma 为图案尺寸的一半。
    shading_sigma = float(np.clip(min(width, height) / 8.0, 6.0, 60.0))
    level_sigma = float(np.clip(min(width, height) / 2.0, 12.0, 120.0))
    shading_only, _sref = _surface_relative_shading(
        canvas, (left, top, right, bottom), shading_sigma)
    level_small, reference = _surface_relative_shading(
        canvas, (left, top, right, bottom), level_sigma)
    level_small = level_small * reference                 # 还原成绝对明度
    texture_small, _lv = _surface_texture(canvas, (left, top, right, bottom),
                                          level_sigma, 1.6)
    shading = cv2.resize(shading_only, (flat.shape[1], flat.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    texture = cv2.resize(texture_small, (flat.shape[1], flat.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    level = cv2.resize(level_small, (flat.shape[1], flat.shape[0]),
                       interpolation=cv2.INTER_LINEAR)

    # 4) 重光 + 合材
    relit = relight(flat, detail, shading, level,
                    float(getattr(config, "MARK_FUSION_DETAIL", 0.75)),
                    texture, float(getattr(config, "MARK_FUSION_TEXTURE", 0.60)))

    # 5) 落笔权重：形状 alpha -> 收紧 -> 羽化 -> 限制在可承载区域内
    #
    # **必须先收紧再羽化**（实测缺陷）：形状分离输出的 alpha 在背景处往往不是 0，
    # 而是 0.2~0.3 的"淡影"（判据是距离阈值，背景色附近总会残留一点）。
    # 直接拿它当权重，背景会以 20%~30% 的比例混进成品——实测现象是
    # **黑色樱花周围出现一块淡蓝色方片**，一眼就是"贴上去的一张图"，
    # 而且因为底纹更亮，这块方片反而更显眼。
    # 做法：先把低于阈值的部分压到 0（背景彻底让开）、高于阈值的抬到 1，
    # 再用 `gamma` 把中间地带拉向两端，最后才羽化（羽化是给**轮廓**用的）。
    weight = patch_alpha.astype(np.float32) / 255.0
    bias = float(getattr(config, "MARK_FUSION_ALPHA_BIAS", 0.35))
    gamma = float(getattr(config, "MARK_FUSION_ALPHA_GAMMA", 1.6))
    if bias > 0:
        weight = np.clip((weight - bias) / max(1e-6, 1.0 - bias), 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-6:
        weight = np.power(weight, gamma)
    feather = float(getattr(config, "MARK_FUSION_FEATHER", 1.8))
    if feather > 0:
        weight = cv2.GaussianBlur(weight, (0, 0), feather)
    if sub_mask is not None and sub_mask.any():
        region_weight = cv2.resize(sub_mask.astype(np.float32),
                                   (flat.shape[1], flat.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        weight = weight * np.clip(region_weight, 0.0, 1.0)
    weight = np.clip(weight, 0.0, 1.0)

    audit.update({
        "ok": True,
        "box": [left, top, right, bottom],
        "coverage": round(float((weight > 0.5).mean()), 4),
        "sigma": round(sigma, 2),
        "shading_sigma": round(shading_sigma, 2),
        "cylinder": round(float(cylinder), 3),
        "detail": float(getattr(config, "MARK_FUSION_DETAIL", 0.75)),
        "texture": float(getattr(config, "MARK_FUSION_TEXTURE", 0.60)),
        "surface_luminance": round(float(reference), 2),
    })
    if matte_note:
        audit["matte"] = matte_note
    return {"ok": True, "relit": relit, "weight": weight,
            "manifold": manifold, "shading": shading, "level": level,
            "audit": audit}


def fuse_onto_surface(rgba, canvas, box, region_mask=None, cylinder=None,
                      audit=None):
    """把标志 `rgba` **印**到 `canvas` 的 `box` 上，返回 (新画布, 审计)。

    参数
    ----
    rgba   : 素材（H,W,4 uint8）。第 4 通道必须是**形状 alpha**
             （用 `shape_alpha()` 校验；不合格时本函数**拒绝执行**并说明原因，
             不会退回矩形硬贴）。
    canvas : 目标画布（H,W,3 uint8 RGB）。
    box    : 放置框 (left, top, right, bottom)，像素坐标。
    region_mask : 器物可承载区域（bool）；越界的部分不落笔。
    cylinder    : 柱面强度 0~1；None 时读 `config.MARK_CARRY_CYLINDER`。
    """
    bundle = prepare_patch(rgba, canvas, box, region_mask=region_mask,
                           cylinder=cylinder, audit=audit)
    if not bundle.get("ok"):
        return canvas, {**(bundle.get("audit") or {}), "applied": False,
                        "reason": bundle.get("reason")}
    left, top, right, bottom = bundle["audit"]["box"]
    weight = bundle["weight"]
    out = canvas.copy()
    target = out[top:bottom, left:right].astype(np.float32)
    blended = target * (1.0 - weight[:, :, None]) + \
        bundle["relit"].astype(np.float32) * weight[:, :, None]
    out[top:bottom, left:right] = np.clip(blended, 0, 255).astype(np.uint8)
    record = dict(bundle["audit"])
    record["applied"] = True
    return out, record


def follow_scores(patch_rgb, canvas, box, weight_mask):
    """衡量"这块图案的明暗是否跟随器物"——**同足迹可比**的最小二乘量。

    返回 `{follow_gamma, follow_resid, pixels}`：
      * `follow_gamma`：逐块拟合 `图案明暗 ≈ a + gamma × 器物明暗` 的斜率
        （按器物波动幅度加权）。**gamma ≈ 1 表示完全跟随器物受光**；
        平贴的图案自带一套明暗，gamma ≈ 0。
      * `follow_resid`：拟合残差（按器物波动幅度归一）。融合接近 0，平贴接近 1。

    与 `vision/metrics.carry_scores` 的关系：那边在**整幅成品**上测，
    用于出图验收；这里在**同一块补片、同一份权重**上测，用于算法自检。
    两者互补：验收看成品，自检排除"足迹不同"的干扰。
    """
    left, top, right, bottom = [int(v) for v in box]
    sub = np.asarray(canvas[top:bottom, left:right])
    grey_base = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grey_patch = cv2.cvtColor(np.asarray(patch_rgb).astype(np.uint8),
                              cv2.COLOR_RGB2GRAY).astype(np.float32)
    if grey_patch.shape != grey_base.shape:
        grey_patch = cv2.resize(grey_patch, (grey_base.shape[1], grey_base.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
    mask = np.asarray(weight_mask).astype(bool)
    if mask.shape != grey_base.shape:
        mask = cv2.resize(mask.astype(np.uint8), (grey_base.shape[1], grey_base.shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
    if mask.sum() < 64:
        return {"follow_gamma": None, "follow_resid": None, "pixels": int(mask.sum())}
    span = int(min(grey_base.shape))
    sigma, low_b = _estimate_scale(grey_base, mask, span)
    low_p = cv2.GaussianBlur(grey_patch, (0, 0), sigma)
    tile = max(16, int(round(sigma * 2)))
    gammas, weights, residuals = [], [], []
    height, width = grey_base.shape
    for y in range(0, max(1, height - tile + 1), tile):
        for x in range(0, max(1, width - tile + 1), tile):
            block = mask[y:y + tile, x:x + tile]
            if block.size == 0 or block.mean() < 0.6:
                continue
            a = low_b[y:y + tile, x:x + tile].ravel().astype(np.float64)
            b = low_p[y:y + tile, x:x + tile].ravel().astype(np.float64)
            if a.std() < 0.5:
                continue
            a = a - a.mean()
            b = b - b.mean()
            gamma = float(np.dot(a, b) / max(1e-9, np.dot(a, a)))
            gammas.append(gamma)
            weights.append(float(a.std()))
            residuals.append(float(np.abs(b - gamma * a).mean() / max(1e-9, a.std())))
    if not gammas:
        return {"follow_gamma": None, "follow_resid": None, "pixels": int(mask.sum())}
    weights = np.asarray(weights, dtype=np.float64)
    gammas = np.asarray(gammas, dtype=np.float64)
    return {
        "follow_gamma": round(float((gammas * weights).sum() / weights.sum()), 4),
        "follow_resid": round(float(np.mean(residuals)), 4),
        "pixels": int(mask.sum()),
    }


def _estimate_scale(grey, mask, span, threshold=1.5):
    """取"该表面上真正存在明暗起伏"的最小尺度（与 metrics 里的判据一致）。"""
    best = None
    for divisor in (16.0, 12.0, 8.0, 6.0, 4.0, 2.0):
        sigma = float(np.clip(span / divisor, 4.0, 240.0))
        size = int(max(5, round(sigma * 4)) | 1)
        low = cv2.GaussianBlur(grey, (size, size), sigma)
        values = low[mask]
        if values.size < 32:
            continue
        if best is None:
            best = (sigma, low)
        if float(values.std()) >= float(threshold):
            return sigma, low
    if best is None:
        size = 33
        return 8.0, cv2.GaussianBlur(grey, (size, size), 8.0)
    return best


def fuse_placements(canvas, placements, region_mask=None, audit=None):
    """按顺序把多个 `(rgba, box)` 融合到画布上，返回 (画布, 审计列表)。

    与 `mark_carry` 的循环承载保持同样的调用顺序（显眼件先放、底纹后铺），
    这样"落笔顺序"的既有设计不被破坏。
    """
    records = list(audit or [])
    for rgba, box in placements:
        canvas, record = fuse_onto_surface(rgba, canvas, box, region_mask=region_mask)
        records.append(record)
    return canvas, records
