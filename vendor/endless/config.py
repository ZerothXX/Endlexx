# -*- coding: utf-8 -*-
"""config.py —— 全项目唯一参数来源（合同文件）

所有路径、模型开关、训练参数、推理参数、VLM 参数均集中于此；各模块一律
`import config` 读取，禁止在业务代码里硬编码路径或超参数。

对应 docs/角色视觉语义LoRA训练与迁移方案_20260929.txt：
- 单一 SDXL 底模：角色身份与视觉概念学习在同一底模上完成，权重因此可与底模、
  与其他角色自由切换（此前"动漫底模 + 写实底模"的双底模分支已归档）。
- 权重是唯一的学习产物：JSON 档案、缓存与 embedding 只是条件，不替代训练所得参数。
- 已废弃的实验开关全部删除；历史实验脚本保留在 histories/ 供查阅。
"""

import os

# =========================
# 基础路径（基于本文件位置推导，保证任意工作目录下一致）
# =========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")      # 各角色原始图片
MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")       # 预训练模型
INPUT_DIR   = os.path.join(PROJECT_ROOT, "input")        # 推理输入
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "output")       # 所有输出

TRAIN_MODELS_DIR = os.path.join(OUTPUT_DIR, "train_models")   # 旧目录（正式权重现在进角色包）
CHARACTERS_DIR   = os.path.join(OUTPUT_DIR, "characters")     # 角色包（权重 + manifest + 语义）
CURVES_DIR       = os.path.join(OUTPUT_DIR, "curves")         # 训练曲线与指标
SEMANTIC_DIR     = os.path.join(OUTPUT_DIR, "semantic")       # 角色语义档案
RESULT_DIR       = os.path.join(OUTPUT_DIR, "result")         # 推理结果
LOGS_DIR         = os.path.join(OUTPUT_DIR, "logs")           # 日志
MANIFEST_DIR     = os.path.join(OUTPUT_DIR, "manifests")      # 固定训练清单（不可变）
PROFILE_DIR      = os.path.join(OUTPUT_DIR, "profiles")       # 数据版本与来源映射
REGISTRY_DB_PATH = os.path.join(OUTPUT_DIR, "registry", "characters.db")   # SQLite 后端

# =========================
# 模型路径（单一底模）
# =========================
# SDXL 底模：角色身份与概念学习、以及正式推理全部使用这一份权重。
BASE_MODEL_PATH = os.path.join(
    MODEL_DIR, "Illustrious-xl-early-release-v0", "Illustrious-XL-v0.1.safetensors")
BASE_VAE_PATH = os.path.join(MODEL_DIR, "sdxl-vae-fp16-fix", "sdxl.vae.safetensors")
# 结构条件：保留源图形态/比例/动作/构图（方案 §7「保留源图结构与表现方式」）
CONTROLNET_PATH = os.path.join(MODEL_DIR, "controlnet-canny-sdxl-1.0")
# 主体分割（非人物分支必须用主体蒙版限定编辑范围）
SAM_PATH = os.path.join(MODEL_DIR, "sam2")
# 视觉语言模型：本项目唯一使用的 VLM（Qwen3-VL-4B，需 transformers>=4.57 的 Qwen3VL 支持）
VLM_PATH = os.path.join(MODEL_DIR, "Qwen3-VL-4B-Instruct")
# 受控参考增强用的 IP-Adapter；仅在该增强启用时加载
IP_ADAPTER_PATH = os.path.join(MODEL_DIR, "ip-adapter")
# 本机目录是 HF 仓库快照：image_encoder 在 models/image_encoder 下（不是根目录的
# image_encoder/），加载时必须显式告知，否则 diffusers 会在根目录找 model.safetensors 并报错。
REFERENCE_IMAGE_ENCODER = "models/image_encoder"

# =========================
# 角色（多角色只靠目录与角色包区分，不为任何角色写专用分支）
# =========================
# 角色包形态（方案 §6）：
#     output/characters/<角色>/
#         manifest.json                     角色 ID、触发词、各权重文件、适用底模、训练来源
#         semantics.json                    自动提取的特征、区域、证据与置信度
#         <角色>_identity.safetensors       身份权重（角色本身）
#         <角色>_concept.safetensors        概念权重（配饰/图案/花纹），按需生成
# 切换角色即切换这个文件夹：权重、触发词、语义与条件一起换。
CHARACTER_ID = "37"          # 命令行未指定时的默认角色；运行时会显示菜单并覆盖
TRIGGER_WORD = f"ch{CHARACTER_ID}"   # 触发词按角色目录名派生，更换角色自动更换
# 训练阶段：all=身份+概念（默认）；identity=只训身份；concept=只训概念分支。
# CLI 的 --stage 会覆盖它；单独重训概念分支时用具更大容量的身份档位配置。
TRAIN_STAGE = "all"


def get_trigger(folder: str) -> str:
    """触发词：ch<角色目录名>。角色身份与视觉概念都由它承载。"""
    return f"ch{folder}"


CHARACTER_PROFILES = {}   # 人工覆盖的角色档案：{folder: {...}}；为空时由 VLM 自动生成
MIN_DATASET_IMAGES = 10   # 有效原始图片下限；不足即停止训练（局部裁剪/合成样本不计入）
TRAIN_QUALITY_TAGS = "masterpiece, best quality"   # 训练与推理共用，避免分布偏差
# 迁移目标：极致的 cosplay —— 以还原角色为准，因此允许并追求把源图主体的脸换成角色脸。
# 源图需要保留的是：形态、比例、动作、朝向、构图、穿戴关系与表现媒介（画风）。
TRANSFER_GOAL = "cosplay_identity"

# =========================
# 训练参数（RTX 4060 8GB，实测标定）
# =========================
# 实测：768 分辨率 + rank32 时显存 5.7GB、约 8s/步；512 分辨率 + rank16 时 5.6GB、约 2s/步。
RESOLUTION = 512                    # 训练分辨率（OOM 时按 768 -> 512 降级）
TRAIN_BUCKETS = [                   # 整图训练的宽高比桶（64 的倍数，面积 ≤ TRAIN_MAX_AREA）
    (512, 512), (512, 384), (384, 512), (448, 448),
]
TRAIN_MAX_LONG = 512                # 宽高比桶的长边上限
TRAIN_MAX_AREA = 512 * 512          # 宽高比桶的面积上限
TRAIN_BATCH_SIZE = 1                # prompt 编码按单样本缓存，batch>1 需先改造广播逻辑
GRADIENT_ACCUMULATION_STEPS = 1     # 8GB 实测 accum>1 会溢出到共享内存，得不偿失
MIXED_PRECISION = "fp16"
LORA_RANK = 16                      # rank16 约 93MB/权重；rank32 约 186MB
LORA_ALPHA = 16
LORA_TARGET_MODULES = None          # None=自动扫描 UNet 注意力线性层；也可显式给列表
LEARNING_RATE = 1e-4
LR_SCHEDULER = "constant"           # 角色绑定用恒定 LR；cosine 在后半段等于停学
LR_WARMUP_STEPS = 50
MAX_TRAIN_STEPS = 700               # 完整流程步数。用户 2026-10-07 明确要求"按完整流程步数 700 步运行"，
                                    # 上限仍为 800；网页端的短程训练另有 80 步的覆盖（web/worker.py）。
NUM_EPOCHS = None                   # None=只按 MAX_TRAIN_STEPS 控制步数，不按 epoch 控制
# checkpoint 策略：小数据集下几百步已进入过拟合区间，因此只保留"最优"一份权重，
# 不产出 *_final 与大量中间文件（用户要求：以 best 解决选择问题）。
SAVE_INTERMEDIATE_CHECKPOINTS = False   # True 时按 SAVE_STEPS 另存中间权重用于对照
SAVE_STEPS = 100                    # 中间 checkpoint 间隔（仅在上面的开关为 True 时生效）
KEEP_BEST_ONLY = True               # 训练结束只把最优权重发布为角色包权重
LOG_EVERY_N_STEPS = 10
EVAL_LOSS_INTERVAL = 20             # 固定时间步评估 loss 的间隔（0=关闭）
EVAL_SAMPLE_COUNT = 4               # 固定诊断集大小；样例一经确定不得因效果差而替换（方案 §5）
MAX_GRAD_NORM = 1.0
SEED = 42
TRAIN_IMAGE_LIMIT = 0               # >0 时只用前 N 张（冒烟测试）
DATALOADER_NUM_WORKERS = 0          # Windows 下 DataLoader 子进程易出问题，主进程加载
USE_8BIT_ADAM = True                # bitsandbytes 不可用时自动降级为 torch.optim.AdamW
USE_GRADIENT_CHECKPOINTING = True   # 8GB 训练的关键项
TRAIN_CACHE_TEXT_ENCODERS = True    # 冻结的文本编码器只编码一次后移出显存
# ---- 两个分支各学什么（用户重申，不要混为一谈）----
# 主（身份）LoRA：角色本身 —— 面部、发色、配色、服装结构。它决定"这是哪个角色"。
# 概念 LoRA：配饰、图案、花纹、边饰等细粒度视觉概念，用来补主 LoRA 学不到的细节。
DETAIL_CROP_RATIO = 0.20            # 主 LoRA 阶段局部裁剪样本占比（概念阶段单独产出裁剪）
DETAIL_CROP_MIN_SIDE = 256          # 局部裁剪的**工作尺寸**下限（放大后的边长）。
                                    # 注意：这**不是**原图像素门槛——
                                    # 角色图里的配饰/结绳/纹样本来只有几十像素，
                                    # 能不能用由证据层的 `MARK_MIN_SOURCE_PIXELS` +
                                    # `MARK_MAX_UPSCALE` 判定，这里只描述工作尺寸。
                                    # 实测踩过两次：`crops_from_marks` 与
                                    # `build_concept_entries` 都曾拿它去卡**原图**像素，
                                    # 把 60 条已通过证据层判定的裁剪否掉了 52 与 56 条。
# ---- 区域加权 loss ----
# 角色最醒目的发色与服装颜色在整图 loss 里只占很小一块像素，容易被当成次要信息。
# 打开后，发区/服装区像素在 loss 里权重更高（权重图内部均值为 1，loss 量级不变）。
REGION_WEIGHTED_LOSS = True
REGION_LOSS_WEIGHT = 3.0            # 被点名区域的像素权重倍数

# ---- 概念分支（配饰/图案/花纹）----
# 概念分支**只用角色真实图片的局部裁剪**训练（不是程序化合成图），产物是角色包里的
# 独立权重文件 <角色>_concept.safetensors。
# 非人物的权重分工（实测标定）：概念权重单独使用、身份权重归零；
# 两个 LoRA 叠加会把主体涂成一团纯色（实测：猫变成蓝色块）。
CONCEPT_TRAIN = True                # 训练时是否在身份分支之后自动训练概念分支
CONCEPT_TRAIN_STEPS = 700           # 概念分支步数（与身份分支同级；同样按"完整流程 700 步"）
CONCEPT_NON_HUMAN_SCALE = 0.55      # 非人物：概念权重强度（做"加法"，不做替换）
                                    # 实测：0.35 时"加法"太弱（猫颈部只有一点暖色，
                                    # 看不出是角色标志）；0.55 在不破坏主体花纹的前提下
                                    # 让配色与图案真正显出来。
# 附着区域的生成参数（实测标定）：概念权重单独使用时强度要够，
# 否则"加法"弱到看不出是角色标志；结构条件要放弱，否则主体花纹会把配饰的形状压掉。
ATTACH_REFINE_STRENGTH = 0.80       # 附着区域的重绘强度
ATTACH_LORA_SCALE = 1.60            # 附着区域的概念权重强度（局部加，强度可以更高；
                                    # 实测 1.35 偏弱、配饰几乎看不出角色配色）
ATTACH_CONTROL_SCALE = 0.0          # 附着区域的结构条件（**关闭**：实测 0.24 时猫颈部
                                    # 只出现一层"灰扑扑的蒙版感"，因为源图的结构条件一直在
                                    # 把猫原本的灰色花纹按回配饰上。配饰是**新增**的东西，
                                    # 它的形状不该来自源图，因此这一遍完全交给权重与提示词；
                                    # 区域之外的像素本来就逐像素保留，不受影响。
CONCEPT_HUMAN_SCALE = 0.0           # 人物：不启用概念权重（身份已由主权重承担）
NON_HUMAN_IDENTITY_SCALE = 0.0      # 非人物：身份权重强度（与概念权重叠加会糊成一团）
CONCEPT_MIN_CROPS = 6               # 概念样本少于该数量则跳过概念分支

# ---- 非人物"加法"的附着区域 ----
# 用户要求：非人物不做身份替换，只做"加法"——给猫加一条角色配色的项圈（带角色的图案花纹），
# 再给头部添一个标志配饰。实测：把概念权重作用在整个主体上是"整体换肤"（猫被整只染色），
# 所以必须把概念条件限制在**可附着**的两块区域里，主体其余部分逐像素保留。
ATTACH_NON_HUMAN = False
# 非人物"加法"里的**动物附着配饰**（给猫加项圈等）**默认关闭**，原因见下；
# 机制完整保留，置 True 即可启用。
#
# ⚠ 本开关**只管动物附着配饰**，不再兼管物体表面图案：
# 物体表面图案由 `OBJECT_SURFACE` 单独控制。早期 `test.py` 只判本开关，
# 于是本开关被置 False 时**物体表面支路也从来没运行过**——
# cat/cup 成品与源图逐**字节**相同（max diff = 0），而 `OBJECT_SURFACE` 是 True、
# 承载计划也有分配，整条"加法"链路配置齐全却什么都没做（实测踩过，代价很大）。
# 两条支路现在各有自己的闸门，日志与 `attachment` 审计都会如实记录是否执行。
#
# 关闭本开关的实测记录：换成"视觉模型直接定位颈部"之后位置对了
# （覆盖率 0.9% → 6.6% → 10.2%），但**渲染结果看不出可辨认的配饰**：
# 那一片只呈现为一层浅浅的暖色/灰色。
# 已排除的原因：概念权重（0.35 / 0.55 / 1.35 / 1.6）、重绘强度（0.80）、
# 结构条件（0.24 → 0.0，让配饰形状不来自源图）、掩版厚度（纵向放大 1.6 倍）。
# 剩下的可能原因：概念权重学的是"角色裁片长什么样"，没有"这是一条戴在脖子上的项圈"
# 这种位置/语义概念；单靠提示词很难让它在一条薄带里成形。
# → 在找到"把概念条件真正变成可辨配件"的方法之前，保持关闭：
#   出图与源图一致（猫完整保留）比"多一层看不出是什么的色块"更符合用户要求。
# 定位经验（四轮实测）：附着区域必须**锚定头部**，不能按全身外接框的比例去猜
# （头小身大的卧姿主体上会依次落到耳间 / 眼睛上 / 背上）；也不能做成 1~3% 宽的细横带
# （扩散模型只会填一块平色）。头部由视觉模型框 + SAM 细化给出，各区域从头部派生：
# 头顶配饰：实测在动物上是**不可行**的区域，默认关闭。
# 原因：动物头部顶端往往只是两只耳朵（中间是背景），SAM 掩版在该处只有两条细窄像素，
# 任何团块落上去都会裁成空集或退化成平色条。颈圈与胸饰没有这个问题（实测能生成出
# 带刺绣与图案的衣物），因此非人物的"加法"由这两块承担。
ATTACH_HEAD_ORNAMENT = False        # 是否尝试头顶配饰区域
ATTACH_ORNAMENT_WIDTH = 0.42        # 头顶配饰团块的宽度（占头宽比例）
ATTACH_ORNAMENT_HEIGHT = 0.55       # 头顶配饰团块的高度（占头高比例）
ATTACH_COLLAR_BAND = 0.55           # 颈圈带高度（占头高比例，紧贴头下方沿）
                                    # 实测：0.40 在猫身上只有 0.9% 画面（细带 → 模型只填平色）；
                                    # 提到 0.55 并配合视觉模型的颈部框后达到 6.6%，
                                    # 生成出可辨认的项圈/饰带。
ATTACH_NECK_EXPAND = 1.6            # 颈部框的纵向放大倍数（以框中心为基准）。
                                    # 实测：视觉模型给的颈部框偏小，带子太薄时扩散模型
                                    # 只会填一块浅色，给不出配饰形状。
ATTACH_CHEST_BAND = 0.80            # 胸饰带高度（占头高比例，接在颈圈之下）
ATTACH_SIDE_INSET = 0.10            # 横向各收比例（避免把毛边/描边带上）
ATTACH_HEAD_JUMP_RATIO = 1.8        # 行宽超过"头宽 × 此比例"即判定为头/身分界
ATTACH_HEAD_FALLBACK = 0.28         # 找不到分界时头部高度退化为主体高度的此比例
ATTACH_HEAD_MAX_FRACTION = 0.45     # 头部高度上限（占主体高度比例），防止把躯干算成头

# ---- 物体（非生物）的"表面图案"支路 ----
# 与动物的附着点机制分开：杯子这类物体没有可附着的脖颈/头顶，
# 用户要的是"把角色配色、配饰、图案、花纹用到杯身的图案上"——即在**可见表面内做图案**。
# 实测的坑：SAM 按视觉模型框分割物体时会把整张桌面一起算成主体（覆盖率 39%），
# 因此这条支路不用 SAM 掩版，直接用视觉模型给出的物体框作为表面范围（框外逐像素保留源图）。
OBJECT_SURFACE = True               # 物体是否启用表面图案支路
# 物体表面范围：**默认不用分割掩版**，用视觉模型的物体框。
# 实测（本轮重测）：杯子那张图里 SAM 按 IoU 或按边缘密度给出的都是**整张桌面**
# （覆盖率 63%，外接框覆盖全画面）——灰色杯身与灰色石纹台面在颜色/纹理上太接近，
# "物体 vs 背景"的语义提示分不开它们。掩版里恰好是"桌子白、杯子黑"，
# 拿它当表面范围会把图案画到桌面上（实测出图正是如此）。
# 改用物体框：框内做图案、框外逐像素保留源图，出图虽然只覆盖框内，但**不会画错地方**。
# 掩版支路保留（`OBJECT_SURFACE_USE_SAM_MASK=True` 可启用），待有更可靠的分割再打开。
OBJECT_SURFACE_USE_SAM_MASK = False  # 保留开关以便将来验证更好的物体分割方案
OBJECT_SURFACE_MAX_COVERAGE = 0.60  # 物体表面掩版的覆盖率上限：超过就认为把背景算进来了
OBJECT_SURFACE_STRENGTH = 0.62      # 表面图案的重绘强度（要能真正画出图案）
OBJECT_SURFACE_LORA_SCALE = 1.45    # 概念权重强度（物体靠概念权重画图案，身份权重为 0）
OBJECT_SURFACE_CONTROL_SCALE = 0.40 # 结构条件：保留物体的形状与光影，同时允许图案成形
SURFACE_MARK_TOKEN_BUDGET = 60      # 表面图案提示词里留给"角色标志描述"的 token 额度。
                                    # 实测：单条标志描述约 22 token，早期给 30 只装得下 1 条，
                                    # 于是"角色标志"在物体上只剩一个图案，看不出是这个角色。
                                    # 60 能装 2~3 条，同时给必需语义约束（不要人、保留形状）留出余量。
SUBJECT_MASK_MAX_COVERAGE = 0.72    # 掩版覆盖率上限：超过就认为把背景算进来了（实测杯子 39%→近满幅）
SUBJECT_MASK_MIN_COVERAGE = 0.03    # 掩版覆盖率下限：低于此认为没框住东西
SUBJECT_COMPONENT_KEEP_RATIO = 0.18 # 连通块保留比例：面积达到最大块此比例的块都保留
                                    # （一个物体会被 SAM 拆成杯口圈/杯身/把手，只留最大块会漏掉杯身）
# 附着区域（动物加法）的生成参数见上方 "非人物加法的附着区域" 一节，此处不重复定义——
# 同一份配置出现两处赋值时，后一处会静默覆盖前一处（实测踩过）。

# =========================
# 统一角色标志库（docs/统一角色标志与自动证据提取_技术调整说明_20261004.md §3、§4）
# =========================
# 用户的两份文本文件只是同一种"角色标志"的两种输入方式，进入系统后统一为一个库：
#     dataset/<角色>/marks.txt       每行一个标志描述，搜索整个图集（角色级通用象征标志）
#     dataset/<角色>/mark_refs.txt   每行 "相对图片路径<TAB>描述"（单独标志参考图）
# 分级语义由用户人工标注约定：不带部位指示的行=角色级通用象征标志（贯穿整个设计）；
# 带部位指示的行=人物各部位的观察记录。两者同等对待，只是默认重要性倾向不同。
MARK_TRAIN = True                   # 训练时是否接入统一标志证据（False=退回旧的 visual_features 路径）
MARK_PROMPT_VERSION = "v1"          # 标志识别/结构化提示词版本；提示词一变就进缓存键
MARK_RESTRUCTURE = False            # True 时忽略"已结构化"标记，对每条标志重跑结构化。
                                    # 提示词本身改过之后必须能重跑：早期提示词只给 JSON 模板、
                                    # 模型返回了一堆空字段，14 条标志被标成"已结构化"却什么都没写进去，
                                    # 于是中文原文被直接塞进英文 CLIP 条件（等于没有条件）。
MARK_REGIONS = ("whole", "head", "face", "eyes", "neck", "chest", "torso", "waist",
                "back", "arms", "hands", "legs", "feet", "surface")
MARK_MERGE_SIMILARITY = 0.45        # 两份文件里判为"同一标志"的描述重合度阈值。
                                    # 相似名称不足以判定重复：还需要部位不冲突（规格 §4）。
MARK_EMPHASIS_MAX_SOURCES = 3       # 重复输入的强调上限（有界规则：基础重要性 + 跨入口增量，截断）
MARK_CONDITION_MAX_CHARS = 200      # 标志条件在**训练条件**里的字符预算。
                                    # 裁剪样本的提示词只有触发词 + 质量词 + 标志描述，
                                    # 因此这里可以宽松一些（实测结构化描述单条可到 272 字符）。
MARK_PROMPT_TOKEN_BUDGET = 14       # 整图提示词里留给标志点名的 token 额度。
                                    # 实测标定两次，结论是**14 比 22 好**：
                                    #   14 -> 点名 1~2 条，出图发色与面部都更干净；
                                    #   22 -> 能点名 3 条（73/77 token），但同一 seed 下
                                    #         人物面部明显变糊、整体更灰（人工对照确认）。
                                    # 原因推测：标志点名挤占了"换脸要求 / 源图媒介 /
                                    # 姿态 / 质量词"这些**保住源图**的条目的权重，
                                    # 而用户明确要求"保留原图画风"。
                                    # 两条真实标志已能进条件（`purple Three circles…`,
                                    # `gold fan`），这是"标志可辨"与"画质"之间目前最好的平衡。
                                    # 注意装不下时按"条"丢弃，**不切碎描述**。
MARK_PROMPT_MAX_CHARS = 60          # 标志条件在**人物推理提示词**里的预算。
                                    # 比训练侧紧得多：推理提示词还要装身份条件、源图媒介、构图、
                                    # 姿态与质量词，而"保留原图画风"是用户的硬要求——
                                    # 实测结构化标志描述很长（单条 272 字符 / 60+ token），
                                    # 直接拼进去会让正向词到 90~122 token，`fit_prompt`
                                    # 为了容纳它先把 `pixel art` 这类媒介词整个丢掉。
                                    # 宁可只写最重要的那半条标志，也不拿画风去换。
MARK_IN_PROMPT = True               # 是否把标志条件写进人物推理提示词
MARK_MIN_CROP_SIDE = 256            # 标志裁剪样本的最小边（像素，经等比放大后的工作尺寸）。
                                    # 角色图里的配饰、结绳、纹样本来就只占几十像素，因此这里
                                    # **不按原图像素数否决**，而是按放大后的清晰度判断。
MARK_MIN_SOURCE_PIXELS = 12         # 原始像素的硬下限：低于此说明框其实没框住标志，
                                    # 放大只会把噪声当证据（规格 §5.3：放大只帮助阅读已有像素）。
                                    # 实测标定：角色腰间的小结绳是 55~96 像素，最小的樱花徽记约 26 像素；
                                    # 12 像素已经接近"看不清"的边界。
                                    # 取 12 而不是 16 是为了让下面那条放大上限**可达**：
                                    # 若下限是 16，则 256/16=16，而要求上限定在 ≤16，
                                    # 两道门会互相顶死（实测踩过：一道永远不触发，测试直接暴露）。
MARK_MAX_UPSCALE = 12.0             # 单个裁剪允许的最大放大倍数；超过说明框太小、不可信。
                                    # 实测标定（2026-10-05，ayaka 22 张图）：
                                    #   腰间结绳 96×55 像素 -> 放大 5.3 倍；
                                    #   项圈/挂饰 50~70 像素 -> 放大 5~7 倍；
                                    #   樱花图案等小徽记最低到 ~26 像素 -> 放大 ~10 倍。
                                    # 上限定 6.0 会把 56 条真实裁剪里的 30 条**在训练时抛错**
                                    # （"细节裁剪需要放大 9.3 倍，超过上限"），训练直接起不来。
                                    # 角色图里的配饰本来就只有几十像素，放大到工作尺寸是必要的
                                    # （整图也是从 1242×1656 缩到 512，同一量纲）。
                                    #
                                    # 两道门必须自洽（由 tests/test_marks.py 的
                                    # test_upscale_guard_is_reachable 守着）：
                                    #   MARK_MAX_UPSCALE > MARK_MIN_CROP_SIDE / MARK_MIN_SOURCE_PIXELS
                                    # 当前 12.0 > 256/12 = 21.33？**不成立**——这里的正确形式是
                                    # 要求"被上限拒绝的尺寸"仍高于像素下限：
                                    #   256/MARK_MAX_UPSCALE >= MARK_MIN_SOURCE_PIXELS
                                    #   -> 256/12 = 21.3 >= 12 ✓
                                    # 于是三档都有意义：
                                    #   <12 像素          -> 框没框住东西；
                                    #   12~21 像素        -> 需要放大 >12 倍，靠插值造细节；
                                    #   >=22 像素         -> 允许。
MARK_CONTEXT_PAD = 0.18             # 标志裁剪时在框外保留的上下文比例（规格 §6：局部裁剪保留少量上下文）
MARK_REF_CROP_PAD = 0.06            # 单独标志参考图裁剪时的留白比例
MARK_MAX_CROPS_PER_MARK = 4         # 每个标志最多保留几张真实裁剪样本（按图片均衡，防大面积服装淹没小徽记）
MARK_LOCATE_STRICT = True           # True=只接受模型明确定位到的框；拿不到就记缺口，不编造位置
MARK_LOCATE_MIN_SIDE = 0.02         # 定位框的最小边长（归一化），低于此视为不可用
MARK_LOCATE_UPSCALE = 2.0           # 送入 VLM 前把画布放大几倍。
                                    # 规格 §5.3：放大只帮助阅读**已有像素**，不声称恢复不存在的纹样。
                                    # 小徽记在原图 768 短边下只有几十像素，不放大时模型给的是"大概这一带"的宽框。
MARK_LOCATE_SHARPEN = 0.6           # 放大后的反锐化强度（0=不锐化），让内部结构可辨
MARK_REGION_OVERLAP_MIN = 0.25      # 位置可信度下限：定位框落在声明部位内的面积占比低于此值，
                                    # 记为 unknown（模型对泛化描述会给"大概在这一带"的宽框，
                                    # 那样的证据学到的其实是背景）。人体区域只做搜索范围，
                                    # 不充当标志定位结果本身（规格 §2 第 4 行）。
MARK_BOX_COLLISION_IOU = 0.6        # 同一张图上两条标志的框 IoU 超过此值即判为"撞框"，
                                    # 组内全部记 unknown。实测（2026-10-05）：模型会把
                                    # 同一个框回报给多条不同标志（腰结 / 背后蝴蝶结 / 扇子
                                    # 同时拿到 0.655,0.454,1.0,0.649）。不挑赢家——
                                    # 挑一个等于把猜测写进训练监督（规格 §5.4）。

# =========================
# 非人物"标志承载计划"（规格 §6：标志如何落到目标主体）
# =========================
# 同一个徽记可以出现在人物胸前、猫的项圈饰片或杯子的可见表面；它仍是同一个标志，
# 承载关系不同。这里按标志类别给出承载方式，不做"把胸前坐标复制到画面中心"这种事。
MARK_CARRY_EXPLICIT_PREFIX = False  # 承载条件里是否保留 `this real repeating pattern:` 这类前缀。
                                    # **默认不保留**，因为训练时用的是
                                    # `chayaka_02, light purple three concentric circles…`——
                                    # 一旦推理时在 token 与描述之间插进一长串没训练过的前缀，
                                    # 条件就与训练分布不一致（实测：杯子上的图案因此只呈现
                                    # 一层很淡的渐变），而且每条前缀约 7 token，
                                    # 两条标志就是 14 token，正好把 `no human face` 这类
                                    # **必需语义约束**挤出 77 token 预算。
                                    # 需要更"像在描述承载方式"时可以打开，但要重新训练对齐。
# ---- 确定性承载（把标志真实像素贴到目标表面）----
# 为什么不再指望"概念权重把标志画出来"（实测 2026-10-05）：把承载条件调到与训练同形、
# 概念强度 1.45/2.5、重绘强度 0.62/0.80，杯子图**改写 81~85%、平均差异 19.95~29.19**
# （支路确实在重绘），但人工核对**依旧是素白杯子、没有任何标志图案**。
# 概念权重学的是"角色裁片长什么样"，缺少"把图案画到物体表面"的承载语义。
# 而用户的判据是"一眼可辨为该角色独有标志"——最可靠的实现是把**用户参考图里
# 标志的真实像素**按目标表面适配后贴上去：形状色彩 100% 来自真实素材。
# 扩散仍负责配色氛围（这条实测有效），两者分工。
# ---- 标志形状分离（从裁剪里只取"标志本身"）----
# 实测依据：直接贴裁剪矩形时，每个重复单元都是一块**带背景的方形照片**
# （黑底 + 衣服图案），观感像"贴了一排小照片"而不是"器物上的花纹"。
# 因此需要把标志的轮廓取出来，只保留轮廓内的像素。
MARK_SHAPE_ENABLED = True           # 承载时是否做形状分离（只保留标志轮廓）。
                                    # 直接贴裁剪矩形会像"贴了一排小照片"；分离后才是"器物上的花纹"。
MARK_SHAPE_USE_SAM = True           # 优先用 SAM 分割标志（分不出来时自动退回颜色判据）
MARK_SHAPE_MIN_COVERAGE = 0.12      # 标志在裁剪里至少应占的比例（低于此判为"碎片"候选）
MARK_SHAPE_MAX_COVERAGE = 0.92      # 掩版覆盖率上限：超过就认为"没分出来"，退回颜色判据
MARK_SHAPE_COLOUR_RATIO = 0.45      # 颜色回退判据的距离阈值（相对四边背景色的 95 分位）
MARK_SHAPE_FEATHER = 1.6            # 轮廓边缘羽化（硬切边同样是"贴纸感"的来源）
MARK_SHAPE_BG_TOLERANCE = 12.0      # "背景透传"的容差：与主导背景色的 LAB 距离小于它的像素判为背景。
                                    # 实测必要性：白樱花图案是"白底 + 白花瓣 + 棕描边"，
                                    # 光看颜色没有边界，而把白底一起贴上去会在深色器物上糊出白方块。
MARK_SHAPE_BG_RAMP = 2.2            # 容差到"完全前景"的过渡倍数（避免锯齿）
MARK_CARRY_COMPOSITE = True         # 是否启用确定性承载（真实像素贴合）
MARK_CARRY_LIMIT = 3                # 最多承载几个标志（多了会互相糊在一起）
MARK_CARRY_LAYOUT = ("emblem", "wearable", "trim", "pattern", "detail", "material")
                                    # 承载**落笔顺序**：先放"要显眼的"（徽记/可佩戴件），
                                    # 再铺"做底的"（花纹/边饰）。
                                    # 实测问题：按承载计划顺序落笔时，先铺满的花纹会让后放的徽记
                                    # **压在花纹格子上**（杯子中央的金色图案叠在一堆扇子上）。
                                    # 显眼件先放、底纹后铺，徽记就能保持完整可辨。
MARK_CARRY_EMBLEM_RATIO = 0.38      # 徽记占"目标可见面短边"的比例
MARK_CARRY_WEARABLE_RATIO = 0.42    # 可佩戴件占可见面宽度的比例
MARK_CARRY_PATTERN_SPACING = 0.40   # 重复纹样的单元间距（占单元边长比例）
MARK_CARRY_PATTERN_COLUMNS = 2.0    # 纹样在区域宽度里铺几列。实测 3 列时单元太小、
                                    # 整片像"贴纸墙"，而且格数一多更容易压住徽记。
MARK_CARRY_PATTERN_BLOCK_RATIO = 0.25  # 单元内被"显眼件"占用的比例超过它就跳过该格
MARK_CARRY_RIM_MARGIN = 0.04        # 去掉口沿之上内容时，口沿下方再留一点余量（占区域高度比例）。
MARK_CARRY_RIM_MIN_KEEP = 0.35      # 安全阀：去掉"口沿之上"后至少保留这个比例的区域，
                                    # 否则判为口沿误检（实测踩过：把杯底强边当口沿，区域被清空）。
                                    # 实测：杯口内壁与咖啡同样"亮+低饱和+平滑"，
                                    # 会被承载区域判据一起选进来，图案于是压在杯口与液体上。
MARK_CARRY_MIN_REGION_PIXELS = 400  # 可承载区域的最小像素数（太小就不贴，避免糊成一团）
# ---- 在**当前画布**上用图像统计量定位"主体表面" ----
# 为什么不用检测框直接当范围：归一化框可能来自另一尺寸/另一帧的检测（缓存复用、缩放时机不同），
# 换算到当前画布会整体偏移——实测把图案贴到了杯子**左右两侧的背景**上。
# 因此改为在画布上就地判定，检测框只作为**搜索范围**。
MARK_CARRY_SUBJECT_VALUE_PCT = 88.0  # "够亮"判据的分位点（画面里最亮的那一撮）。
                                     # 实测 60 分位会把亮背景（湖面/浅色桌面）也算进来，
                                     # 最大连通块落到背景上；88 分位才稳定落在主体上。
MARK_CARRY_SUBJECT_SMOOTH_PCT = 45.0 # "够平滑"判据的分位点（局部方差上限）。
                                     # 这一条是拉开"物体表面 vs 粗糙背景"的关键：
                                     # 陶杯杯身平滑（局部方差极小），桌面/布料/水面都有颗粒纹理。
MARK_CARRY_GROW = 0.40               # 判据种子外接框的外扩比例。
                                     # 实测判据只标出杯身左半边（11583 像素，中段偏暗、
                                     # 右侧偏灰都不满足"够亮"），不外扩图案会挤在一侧。
MARK_CARRY_SUPPORT_SMOOTH_PCT = 62.0 # 支撑区填充时的平滑度上限分位点：
                                     # 外扩框内**只按平滑度**筛（去掉"够亮"），
                                     # 让稍暗的主体中段也能进来，同时排除颗粒纹理的背景。
MARK_CARRY_BOX_INSET = 0.12         # 物体框内缩比例：物体框通常把主体外沿也算进去，
                                    # 直接贴会把图案压到边缘甚至背景上，因此向内收一圈。
                                    # 只在"图像统计量定位失败"时作为兜底使用。
MARK_CARRY_USE_SUBJECT_MASK = False # 承载位置用**物体框**还是像素级掩版。默认用框。
                                    # 实测（2026-10-05）：按掩版放会把图案贴到杯子**旁边**——
                                    # 该掩版与图像不同尺寸/未对齐（掩版 264985 像素、
                                    # 缩放到画布后换算出的位置落在杯身左侧之外），
                                    # 而物体框的位置与画面上杯子的位置一致（人工核对标注图
                                    # `output/_probe/carry/cup_regions.png`）。
                                    # 位置宁可保守地用框内缩一圈，也不要"贴到背景上"。
                                    # 实测（2026-10-05）：按掩版放会把图案贴到杯子**旁边**——
                                    # 该掩版与图像不同尺寸/未对齐（掩版 264985 像素、
                                    # 缩放到画布后换算出的位置落在杯身左侧之外），
                                    # 而物体框的位置与画面上杯子的位置一致（人工核对标注图
                                    # `output/_probe/carry/cup_regions.png`）。
                                    # 位置宁可保守地用框内缩一圈，也不要"贴到背景上"。
MARK_CARRY_PLAN = {
    "palette": "map to an editable host area, keep material, structure and lighting",
    "emblem": "adapt the real reference shape to the host surface perspective",
    "pattern": "repeat the real unit direction and spacing along the host surface",
    "trim": "lay the repeating unit along the host boundary or hem",
    "wearable": "choose a wearable/attachable host part, constrain scale, count and contact shadow",
    "material": "transfer colour and texture only, keep the host shape",
    "detail": "adapt a small real detail to a visible host area",
}

# =========================
# 推理参数
# =========================
INPUT_IMAGE = INPUT_DIR
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 7.0
# 推理分辨率：短边 768（用户同意提高分辨率换取细节；代价是单图推理时间明显增加）。
# 细节损失与分辨率直接相关——640 时生成结果的高频细节只剩源图的 18% 左右。
INFERENCE_RESOLUTION = 768          # 短边；长边上限见 INFERENCE_MAX_LONG
INFERENCE_MAX_LONG = 1152
ATTENTION_SLICING = True            # 注意力分片，压低峰值显存
INFERENCE_SEED = 1034015501         # 固定种子便于复核；-1 为随机
NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, blurry, watermark, text"
HUMAN_NEGATIVE_PROMPT = (
    "deformed face, distorted eyes, malformed fingers, fused fingers, extra fingers, "
    "missing fingers, extra arms, duplicate limbs, floating hands")
NON_HUMAN_NEGATIVE_PROMPT = "girl, 1girl, human, person, woman, anime girl"
ANIMAL_NEGATIVE_PROMPT = ("anthropomorphic, humanoid body, human clothing, malformed paws, "
                          "extra legs, extra ears, duplicate tail, deformed animal face")
OBJECT_NEGATIVE_PROMPT = ("plain undecorated object, blank surface, missing character motif, "
                          "tiny indistinct decoration, faint low-contrast pattern")
NEGATIVE_TOKEN_BUDGET = 77          # 负向词与正向词共用 CLIP 的 77 token 上限。
                                    # 实测（2026-10-05，用与训练相同的两个 tokenizer 计数）：
                                    # 动物的负向词到 80/77，末几条被静默截断，
                                    # 而被截断的恰好是"不许画坏"那一类。现在按优先级拼到预算内，
                                    # 超出的条目**丢弃并打印**，不交给 CLIP 静默截断。
CANNY_LOW = 50
CANNY_HIGH = 150

# 结构保留：img2img 提供姿态/构图锚点，ControlNet 保持轮廓与纹理
# 启用初值配色改写后，颜色由初值保证，整图强度降到"只重绘需要变的部分"，
# 这样源图的对比度与纹理不会被整体重画（实测：强度越高画面越淡、涂抹感越强）
IMG2IMG_STRENGTH = 0.62             # 通用起点（unknown 主体）
CONTROLNET_SCALE = 0.40
CONTROL_GUIDANCE_END = 0.80
HUMAN_IMG2IMG_STRENGTH = 0.58
HUMAN_CONTROLNET_SCALE = 0.52
HUMAN_CONTROL_END = 0.75
ANIMAL_IMG2IMG_STRENGTH = 0.60
ANIMAL_CONTROLNET_SCALE = 0.50
ANIMAL_CONTROL_END = 0.85
OBJECT_IMG2IMG_STRENGTH = 0.66
OBJECT_CONTROLNET_SCALE = 0.52
OBJECT_CONTROL_END = 0.85
MASK_FEATHER = 6                    # 非人物主体蒙版合成回原图时的羽化像素
                                    # （主体外逐像素保持输入是这一支路的固定行为）

# ---- 局部强化重绘 ----
# 整图 img2img 的强度受限于"不能破坏姿态与构图"，因此对头发与服装区域各做一次
# 更高强度的重绘，再按掩版羽化合成回去：背景、姿态、构图仍来自源图。
REGION_REFINE = True
REGION_REFINE_NON_HUMAN = False
# 动物/物品**默认不做局部强化重绘**。实测（本轮）：非人物的区域掩版来自视觉模型的框，
# 在那个范围内做更高强度的重绘，会把主体自身的花纹覆盖掉一块
# （猫的头顶出现一块与周围毛发不连续的灰斑，边界正是框的形状）。
# 用户的非人物要求是"只做加法、主体保持原样"，而这个机制本质上是**替换**主体的一部分，
# 与要求冲突；因此默认关闭。整图那一遍（强度 0.60、概念权重 0.55）仍然会做，
# 它才是"在不破坏主体的前提下施加角色配色"的那条路。
REGION_REFINE_TARGETS = ("head", "body")   # 人物需要局部强化的区域（非人物只用 body）
REGION_REFINE_STRENGTH = 0.74       # 局部重绘强度（颜色已由初值保证）
REGION_REFINE_LORA_SCALE = 1.45     # 局部重绘时的角色权重强度
                                    # （回归测试要求它不低于整图那一遍的 HUMAN_LORA_SCALE=1.40，
                                    #   否则"局部强化"反而比整图更弱）
REGION_REFINE_CONTROL_SCALE = 0.45  # 局部结构条件（换内容的同时找回源图纹理）
REGION_REFINE_PAD = 0.08            # 区域框按边长外扩比例
REGION_REFINE_FEATHER = 10          # 掩版羽化像素（内部实心、只在外侧渐变）
# SAM 掩版的形态学修补：掩版边界常漏掉贴边的深色发丝/衣褶，会在掩版内留下源图斑块
REGION_MASK_CLOSE = 15              # 闭运算核尺寸（像素，0=关闭）：填掉掩版内的洞
REGION_MASK_DILATE = 5              # 外扩核尺寸：只补贴边发丝，不外溢到皮肤/背景
REGION_REFINE_REQUIRE_SAM_MASK = True# 局部强化重绘**只接受 SAM 细化出来的像素级掩版**。实测：SAM 细化失败时退回的矩形掩版，
# 合成时会把人像身上切出一条**方块状接缝**（human2 的手臂与肩上出现过明显的直角块）。
# 宁可那一块不做强化（整图那一遍仍然覆盖它），也不要留下硬边。
REGION_REFINE_FRESH_INIT = True     # 局部强化的**重绘初值改用源图**，而不是在上一次的结果上再生成。
# 实测问题（贯穿多轮的"脸糊、画面发灰"）：原实现把累积结果同时当作"重绘初值"与"合成底"，
# 于是头部被重绘两次（整图一遍 + 头部一遍），噪声与模糊逐轮累积——
# 每一步单独看都符合规格，合起来却是一张糊掉的图。改成"每一块都从源图起画、
# 再逐块合成"之后，质量不再随区域数量下降；区域外仍然逐像素等于当前结果。
# 置 False 可退回旧行为以便对照。

# ---- 初值配色改写（解决"源图深色区域锁住颜色"这一实测卡点）----
# 底模能力探针证明：不加载权重、不给源图时，底模能画出角色的发色与服装配色；
# 卡点在于 img2img 从源图出发时，源图最显眼的颜色区域是强信号，重绘强度压不过它。
# 因此先把被点名区域（发区/服装区）的**初值**按角色配色改写（只改色度，
# 保留源图明暗、纹理与高光），再按原参数扩散。区域与目标颜色都由档案自动决定。
# ---- 初值配色锚定（默认关闭，见下方实测结论）----
# 实测结论（2026-10-03，权重真正生效后重测）：
#   把区域初值染向角色配色，效果**不如不染**：
#     * 区域掩版里同时含头发、脸、衣服，任何"整块"改变都会把暖色皮肤一起推向冷色，
#       实测初值图 LAB b 通道整体偏移 -18.4 —— 用户看到的"人物整体偏蓝"就是它；
#     * 用"暗部门控"只染暗部时，又会连**背景阴影**一起染（阴影分不清是头发还是背景）；
#     * 而它想达到的目的（发色迁移）并没有达成：权重 + 提示词在扩散阶段本来就会处理颜色，
#       实测开启锚定与关闭锚定的出图，发色没有可见差别，但蓝色偏移与涂抹感明显增加。
# 已解决（2026-10-04）：门控不精确的问题用"取最大连通块 + 补洞"解决了——
#   头发连成一整片，而脸上的阴影是若干小块；实测丢弃的散块只占 0.95%，
#   脸上与手臂上的色块随之消失（`_anchor_check.py` 可复核：头发 b 位移 +58.5，
#   脸/皮肤完全不变）。
# 因此默认开启；若某张图出现异常可改回 "off"。
COLOR_HINT = True
COLOR_ANCHOR_MODE = "color"         # off | luminance | color（三种都实测过，见上方结论）
# ---- 配色锚定的掩版合理性判据 ----
# 这几条挡住的是"整幅上半身"这类**退化掩版**（视觉模型定位失败时退回的假设），
# 不是"头发本来就铺得很开/垂得很长"的正常掩版。
#
# 实测（human2，2026-10-05）：长直发被风吹开，头发掩版宽 70%、面积 22%、
# bbox 从 y=196 一直垂到画面底部。旧判据只看**宽度**就把它判成"不像头发区"跳过，
# 于是**发色锚定从来没有生效过**——出图头发一直是源图的深棕色，
# 看起来像"权重完全没学到角色"，真正原因却在这条几何判据上。
#
# 换成"宽度+面积+触底"组合判据后又踩一次：这张图的头发确实垂到画幅底部，
# 触底判据同样误杀。**宽度、面积、触底都只是代理指标**，
# 它们区分不了"一大片头发"与"头发+躯干混在一起"。
#
# 真正有区分度的是**材质同质性**：头发是一整片同色材质，
# 退化掩版必然同时包含头发、皮肤与衣物。因此主判据改为掩版内的
# **LAB 颜色离散度**（均值化后的通道标准差），并保留一条覆盖率兜底。
COLOR_ANCHOR_MAX_MASK_SPREAD = 34.0     # 掩版内 LAB 三通道标准差均值上限。
                                        # 头发实测约 20~28；混入皮肤+衣物后会明显更高。
COLOR_ANCHOR_MAX_MASK_COVERAGE = 0.60   # 覆盖率兜底：接近整幅画面就不是"某个区域"
COLOR_ANCHOR_MAX_MASK_FRACTION = 0.55   # 保留但**不再单独否决**：只作为审计记录里的参考值
# 允许做配色锚定的区域：**只锚头发**。
# 服装区与"整幅重绘"叠加会把衣服整体刷成角色色，而用户要的是"头发与服装结构像角色"，
# 不是把整幅画面调色；实测把服装一起锚定之后，线稿样例的裙子与皮肤都被涂成角色色
# （那张图的头/身掩版几乎覆盖整个人物）。
# 另外，只有**模型真正定位到**的区域才会被锚定：头部定位失败时的退化框
# （整幅上半部）同样会把整个上半身刷色，宁可不锚。
COLOR_ANCHOR_REGIONS = ("head", "hair", "face")
# ---- 角色颜色参照（让模型自己把角色画一遍，再从那张图取颜色）----
# 角色档案里存的是颜色**名字**，而"最还原角色的形象"要的是具体色值。让模型在纯文生图
# 条件下把角色画出来，是它对"这个角色长什么样"的最佳表达；从这张图里取色比颜色名更准，
# 而且完全自适应（换角色同样成立）。默认关闭：先由对照实验确认它确实带来改善再打开。
COLOR_REFERENCE_FROM_T2I = False
COLOR_REFERENCE_STEPS = 20          # 参照图的采样步数（只用于取色，不需要很精细）
COLOR_REFERENCE_SIZE = 768          # 参照图边长（SDXL 原生分辨率）
# ---- 配色锚定（结构保持的线性变换，取代早期的"整块替换色度"）----
# 旧做法把区域内每个像素的色度直接换成目标色度，实测造成三个可见缺陷：
#   "人物整体偏蓝"（整片区域同一色度）、"颜色糅杂涂抹"（色相层次被抹平）、
#   "轮廓蓝色层"（暖色皮肤与青色边界相减出现灰带）。
# 新做法只把区域的颜色分布**整体搬家并适度缩放**，保留源图的色相与明暗层次：
#   src' = (src - mean) * keep * ratio + mean + (target_c - mean) * ratio
# 其中 keep 越大，源图自身的颜色差异保留得越多（1.0=完全不动，0.0=旧版的平色）。
COLOR_ANCHOR_CHROMA = 0.95          # 色度锚定强度（0=不动；1=把目标色像素的色度搬到目标色）
COLOR_ANCHOR_LUMINANCE = 0.5        # 发区明度锚定强度（深色源发只换色度会得到"暗蓝"）
                                    # 实测标定（2026-10-05 对照）：0.9 时整头被推亮成惨白；
                                    # 0.5 时发色是深蓝且有发丝层次，是观感最好的一档。
COLOR_ANCHOR_CHROMA_KEEP = 0.45     # 目标色在区域色度里的占比权重：0.45 表示目标色占 55%，
                                    # 其余 45% 保留区域自身均值 → 不会变成一块平色
COLOR_ANCHOR_LUMINANCE_KEEP = 1.0   # 明度**层次**保留比例（1.0 = 区域原有明暗次序完整保留）。
                                    # 注意这里是"层次"而不是"位置"：位置一定会朝目标色移动，
                                    # 这个系数只决定移动后还保留多少原始对比度。
COLOR_ANCHOR_MAX_LUMINANCE_SHIFT = 25.0
                                    # 单次锚定允许的明度位移上限（LAB L 单位）。**实测必需**：
                                    # 源发 L 中位 16、目标发色 L=122，不设上限时一次把头发推到 122，
                                    # 出图整头变成**惨白的浅灰**——
                                    # strength=0.58 的扩散会把"已经很亮"的初值当成既成事实，
                                    # 不再把它压暗回来。发色锚定的职责是**给出色相**，
                                    # 明度交给扩散与提示词。
                                    # 对照标定（2026-10-05，见 output/_compare/sweep_color_anchor）：
                                    #   位移 55 + 强度 0.9 -> 整体偏灰白；
                                    #   位移 25 + 强度 0.5 -> 深蓝且有发丝层次（目前最好）；
                                    #   完全关闭           -> 发色仍是源图棕色，看不出是哪个角色。
COLOR_ANCHOR_HUE_RADIUS = 26.0      # 色相选择半径（LAB 色度平面上的距离）：
                                    # 距离目标色簇中心小于该值的像素才被上色，皮肤等暖色像素
                                    # 距离远、自动被排除。这是"不再把脸染蓝"的关键参数。
                                    # 调大=影响更多像素（可能染到皮肤），调小=只染最接近的像素。
COLOR_ANCHOR_MAX_GAP = 42.0         # 自适应门槛：区域自身色度与角色目标色度的距离超过该值时，
                                    # （实测：深棕头发 → 浅蓝，距离 70）自动切换到
                                    # "暗部门控 + 纯色相位移"模式：只给区域里较暗的那部分像素
                                    # （深色头发 vs 更亮的脸/皮肤）改色相，亮部（皮肤）一律不动。
                                    # 距离近（同色系）时用色相置信度锚定，把颜色精确钉到目标色。
COLOR_ANCHOR_MIN_HAIR_CHROMA = 14.0  # 发色锚点的**最低彩度**：低于它的"发色名"不可信
# 实测（2026-10-13，本轮最关键的一处）：档案里的发色是视觉模型给的**颜色名**，
# 而名字会随运行漂移——同一个角色的浅蓝发，历史档案写 `light blue`
# （锚定后出图头发变蓝、正确），本次重建写成 **`silver`**
# （目标色 [192,192,200]、彩度仅 4），"锚定"把浅蓝发角色染成了**中性灰**，
# 出图反而比不锚定更不像角色。判据因此从"名字"改为"颜色的彩度"：
# 彩度不足（silver/gray/white/black 一类中性色）时，
# 改用该角色配色里**最像发色**的一项（名字命中下面这几个词的优先，
# 否则取"彩度最高且偏亮"的那项）——两者都是角色自己档案里的数据。
COLOR_ANCHOR_HAIR_PREFERRED = ("blue", "cyan", "aqua", "lavender", "violet",
                               "purple", "lilac", "teal", "mint")
COLOR_ANCHOR_DARK_PERCENTILE = 40.0 # 暗部门控的分位点：区域内明度低于该分位数的像素视为                                    # "主体自身颜色"（头发），高于它的（脸/皮肤/背景）不动。
                                    # 调高=覆盖更多像素（可能染到皮肤），调低=只染最暗的部分。
COLOR_ANCHOR_GATE_RAMP = 14.0       # 门控过渡带宽度（LAB 明度单位）：让"改与不改"之间平滑过渡，
                                    # 避免在额头/脸颊处出现一条硬边。
COLOR_ANCHOR_SKIN_PERCENTILE = 22.0 # 皮肤排除门控：区域 a 通道（越红越大）低于该分位数的像素
                                    # 才视为"低饱和的主体色"（头发）。实测人脸暗部 a≈133~137、
                                    # 头发 a≈128~130，用区域内自身的分位数当门槛即可分开，
                                    # 不需要任何肤色常量，换角色同样成立。
COLOR_ANCHOR_SKIN_RAMP = 4.0        # 皮肤门控的过渡带（LAB a 单位）。
COLOR_ANCHOR_KEEP_THRESHOLD = 0.5
# "偏暗 × 低饱和"门控的判定阈值，用于取最大连通块。
# 实测必要性：门控通过的不只是头发，还有**脸上与颈部的阴影**——它们同样是
# "偏暗 + 低饱和"，颜色上与深色头发无法区分；但头发连成一整片、阴影是若干小块，
# 因此取最大连通块（并补洞）即可把它们分开。不做这一步时，出图上会出现
# 一块块孤立的深色斑（额头/颈部/手臂上的蓝色色块）。
COLOR_ANCHOR_BLOB_GROW = 0.030
# 最大连通块向外扩张的比例（占画面短边）。细发丝与边缘不属于最大连通块，
# 不扩张的话头发上会留下"一半蓝一半褐"的碎片边界（实测）。
COLOR_ANCHOR_GROW_THRESHOLD = 0.02
# 扩张之后重新取门控的阈值：取低值（而不是 0.5），让发丝这类"门控偏弱"的像素
# 也能被上色，但扩张出来的背景像素仍然会被区域掩版挡住。
COLOR_ANCHOR_FEATHER = 5            # 锚定边界羽化像素（内部实心、只在外侧渐变）
# ---- 锚定覆盖范围的扩展（实测：不做扩展只染到掩版的 1/3）----
# 第三轮实测：门控最终只选中头部掩版内 **33.7%** 的像素（占整幅 7.4%），
# 出图里头发出现"一块蓝 + 一片褐"的硬边拼接（人工核对确认）。
# 扩展必须**有界**：头部掩版里同时含脸（实测左半 L 中位 109，是皮肤），
# 无条件按掩版扩张会把脸染蓝——那正是 19 轮里"人物整体偏蓝"的成因。
# 判据只用种子自身统计量：足够暗（头发比皮肤暗）+ 不偏红（皮肤 a 通道偏红）。
COLOR_ANCHOR_EXPAND_L_PERCENTILE = 90.0  # 扩展的明度上限：种子 L 的该分位数
COLOR_ANCHOR_EXPAND_L_RAMP = 8.0         # 明度门控过渡带（LAB L 单位）
COLOR_ANCHOR_EXPAND_A_MARGIN = 2.0       # 允许比种子 a 中位数高出的余量（皮肤 a 更高）
COLOR_ANCHOR_EXPAND_A_RAMP = 3.0         # a 门控过渡带
COLOR_ANCHOR_TINT_FEATHER = 5            # tint 掩版羽化半径（硬边是"平色块"观感的一半来源）
COLOR_ANCHOR_EXPAND_REACH = 0.02         # 允许扩展越过区域掩版的比例（占画面短边）。
                                         # 实测必要性：SAM 按头部框细化出的掩版只覆盖右侧那团头发，
                                         # **左侧长发完全在掩版外**，于是同一头头发"右蓝左褐"。
                                         # 颜色判据（暗 + 不偏红）已排除皮肤与白衣，
                                         # 因此这一步不会引入肤色；0.02 ≈ 1210 图上的 24 像素邻域。
# 兼容旧字段名：旧代码/旧记录里的这两个名字仍可读取，语义映射到新参数
COLOR_HINT_STRENGTH = COLOR_ANCHOR_CHROMA
COLOR_HINT_FEATHER = True
COLOR_HINT_HAIR_LUMINANCE = COLOR_ANCHOR_LUMINANCE

# ---- 源图细节保留（针对实测的"涂抹感"根因）----
# 实测：整图 img2img（即使强度只有 0.58）会把源图的高频细节几乎抹平——
# 照片样例的拉普拉斯方差从 119.1 掉到 5.5（只剩 5%），线稿从 1054 掉到 1.9。
# 也就是说"涂抹感"的主因不是某一档强度，而是**整幅重画**本身。人物样例的源图细节
# （皮肤毛孔、发丝、布料纹理、线稿的干净线条）必须保住，因此：
#   改区域 -> 用扩散重绘；不需要改的地方 -> 逐像素保留源图。
PRESERVE_SOURCE_OUTSIDE_REGIONS = True
PRESERVE_MAX_COLOR_DISTANCE = 18.0  # 判定"这块本来就不需要改"的颜色距离阈值（0~441）
LUMINANCE_MATCH_TOLERANCE = 6.0     # 区域内平均明度与源图的容许差（超过就整体平移回来）
                                    # 抑制"整图重绘顺手提亮/压暗"（实测：猫额头被提亮 15）

# ---- 线条密集型输入（线稿/草图/像素画）单独收紧 ----
# 线稿的"内容"就是线条本身；实测整图 img2img 会把细节抹掉约 95%（1054 → 1.9），
# 线稿首当其冲。判据只用源图自身的高频占比，不含任何角色或题材假设。
LINEART_EDGE_THRESHOLD = 60.0       # 判为"边缘像素"的拉普拉斯绝对阈值
LINEART_EDGE_RATIO = 0.055          # 边缘像素占比达到此值即视为线条密集型输入
LINEART_IMG2IMG_STRENGTH = 0.36     # 线条类输入的整图强度（低于照片，保住线条）
                                    # 实测：0.42 仍然过强——线稿整幅被冲淡（纸色与线条都变浅），
                                    # 因为线稿的"内容"就是线条，任何整体重绘都在稀释它。
LINEART_CONTROLNET_SCALE = 0.78     # 线条类输入的结构条件（高于照片，锁住线条）
LINEART_REGION_STRENGTH = 0.45      # 线条类输入的区域强度
LINEART_REGION_CONTROL_SCALE = 0.70 # 线条类输入的区域结构条件
LINEART_PRESERVE_DISTANCE = 8.0     # 线条类输入的源图保留阈值（更紧，尽量不动线条；
                                    # 实测 10 仍然放行了过多"被改动的像素"，线稿被涂白）
LINEART_REGION_REFINE = False
# 线条类输入**不做局部强化重绘**：实测那一遍（强度 0.74）会把线稿的线条抹掉、
# 只剩一块发白的色块。线稿只需要改颜色，不需要更强的重绘。

# ---- 画质：解决"重绘后画面发糊、分界不清、颜色糅杂" ----
# 实测根因有三条，各自对应一个开关：
#   1) 整图 img2img 会重画不该动的像素（远处/背景也糊）→ LIMIT_CHANGE_TO_REGIONS
#   2) 扩散输出天然比源图软（分界处"糊—锐"突变）→ REGION_SHARPEN
#   3) 掩版羽化过宽会把边界摊成模糊带 → EDGE_CLEAN_*
LIMIT_CHANGE_TO_REGIONS = True      # 只有"相对源图确有变化"的像素才采用生成结果
LIMIT_CHANGE_MAX_DISTANCE = 26.0    # 变化判定阈值（越小越保守，源图细节保留得越多）
# ---- 只保留区域改写（结构性开关，治"整图发糊"的根本）----
# 实测结论（2026-10-03）：这个开关**在掩版质量达标之前应保持关闭**。
# 原因：区域外保留的是源图（写实照片），区域内是扩散出的动漫角色，
# 两者媒介不同，拼接边界必然露馅（实测出图有明显"抠图感"与生硬折线）。
# 掩版改造成沿头发/衣物的有机轮廓、且媒介风格锚定做好之后，再打开它。
REGION_ONLY_COMPOSITION = False
REGION_ONLY_EXTRA_DISTANCE = None   # 额外的变化阈值；None=严格只在区域内采用（边界最干净）
# ---- 人物：整块主体沿有机轮廓合成（取代"只改头发+衣服"的局部拼接）----
# 用户的核心不满之一是"分界处没有做好分界、画风与原图不一致"。根因是**媒介拼接**：
# 局部重绘只覆盖头发与衣服，脸、脖子、手臂仍来自源照片，于是成品里
# "写实脸 + 动漫衣服"硬拼在一起，边界越清楚越难看。
# 打开本开关后：主体掩版由"生成差异 ∩ 区域邻域"给出（贴着头发/衣服的真实轮廓），
# 整块主体取自生成结果，背景逐像素保留源图——主体内部画风一致，主体之外原图完全不变。
SUBJECT_SILHOUETTE_COMPOSITION = False
# **默认关闭**（本轮最终判定）。机制完整保留，置 True 可启用。
#
# 为什么关：本开关的目的是"整块主体都取自生成结果，消除写实脸/动漫衣服的媒介拼接"。
# 但它依赖一张**覆盖整个人物**的掩版，而现有掩版的来源都有实测问题：
#   * 视觉模型的头发框会扩到半张画幅（实测 0.378~1.0 × 0.171~1.0），SAM 按它细化出来的
#     掩版自然溢出到天空与海面；
#   * "生成差异 ∩ 区域邻域"会产生**筛网状**掩版，合成后出现"闪粉/星点"伪影；
#   * 只用区域并集虽然平滑，但形状取决于视觉模型的框，同样会溢出。
# 溢出之后，"主体外逐像素保留源图"这条硬约束就失效了：天空与海面被生成结果覆盖，
# 观感上是一层雾，靠外的部分还出现星点。
# → 在拿到**可靠的人物整体分割**之前，回到"按点名区域（头发 / 衣服）合成"：
#   这两个掩版来自 SAM 对该区域的像素级细化，形状与位置都可靠，
#   区域外由"只在确有变化处采用生成结果"兜底，出图稳定、无伪影。
SUBJECT_SILHOUETTE_NEIGHBOUR_PAD = 61
# "生成差异 ∩ 区域邻域"里邻域的半径（像素，奇数核）。实测必要性：点名区域（头发/衣服）
# 只覆盖画面约 34%，而人物约占 55%；把邻域放大到能覆盖脸、脖子、垂下的长发之后，
# 差异法给出的掩版就正好是"主体真正被改写的部分"，形状贴着头发与衣服的真实轮廓，
# 而不是视觉模型矩形框那条台阶状硬边。调大=主体覆盖更全（代价是可能带上一点背景）。
SUBJECT_SILHOUETTE_DIFF_DISTANCE = 22.0
# 判定"这个像素确实被改写了"的颜色距离阈值（与 LIMIT_CHANGE_MAX_DISTANCE 同量纲）。
SUBJECT_SILHOUETTE_DILATE = 13
# 合成前把主体轮廓外扩的像素数（奇数核）。实测必要性：SAM 对**飘散的发丝**给不出
# 掩版，边界落在头发主体上；外扩不足时发丝留在源图里，与生成结果叠在一起，
# 在头部周围形成一圈"半透明幽灵状"的浅色带。外扩量取 13：既能盖住贴边发丝，
# 又不会把大片背景（天空、海面）拉进生成范围（实测 31 时边界外出现星点状伪影）。
SUBJECT_SILHOUETTE_FEATHER = 11
# 主体轮廓合成时的羽化像素（**向外**过渡：合成前把掩版向外扩张，再用向外羽化的
# alpha 混合，边界落在背景上，因此不会出现"糊—锐"突变）。羽化不宜过大：
# 过渡带越宽，越容易在边界上显出两种画风的混叠（实测 21 时有星点感）。
REGION_SHARPEN = True               # 被改写区域合成前做反锐化，补回扩散造成的软化
REGION_SHARPEN_AMOUNT = 0.55        # 反锐化强度（0=关闭；过高会产生白边）
REGION_SHARPEN_RADIUS = 1.4         # 反锐化半径（像素）
EDGE_CLEAN = True                   # 把羽化过渡带收窄，减少"分界不清"
EDGE_CLEAN_LOW = 0.06               # 低于此比例的 alpha 直接压成 0
EDGE_CLEAN_HIGH = 0.94              # 高于此比例的 alpha 直接抬成 255
# 下面两个开关用于对照实验（环境变量覆盖，便于不改源码就做 A/B）：
#   ENDLESS_NO_PRESERVE=1       关闭源图保留
#   ENDLESS_NO_COLOR_HINT=1     关闭初值配色改写
import os as _os
if _os.environ.get("ENDLESS_NO_PRESERVE") == "1":
    PRESERVE_SOURCE_OUTSIDE_REGIONS = False
if _os.environ.get("ENDLESS_NO_COLOR_HINT") == "1":
    COLOR_HINT = False
if _os.environ.get("ENDLESS_COLOR_HINT") == "1":
    COLOR_HINT = True        # 显式打开配色锚定（做对照实验用）

# 通用数值覆盖：`ENDLESS_OVERRIDE="KEY=值,KEY=值"`（只允许覆盖**已存在**的常量）。
# 为什么需要：调参对照（例如"明度位移上限取多少最好"）以前要改源码 + 重编译，
# 一次对照要跑十几分钟，改一次源码就多一份"源码与结果不一致"的风险。
# 只接受本文件里已经定义过的名字，避免一个环境变量凭空造出配置项。
#
# **这段逻辑必须放在文件最末尾**（实测踩过，代价是一次无效对照）：
# 早期它写在文件中部，于是**只有排在它之前的常量能被覆盖**，
# 排在它之后的（例如 `USE_REFERENCE_ENHANCEMENT`、`REFERENCE_FACE_SCALE`）
# 覆盖请求会被当成"未知项"忽略——对照实验静默地跑了两遍同一个配置，
# 两次输出逐字节相同，而我差点据此得出"参考增强没用"的结论。
# `apply_override` 提到模块级并在末尾调用，就不会再有时序问题。
def _apply_override():
    spec = _os.environ.get("ENDLESS_OVERRIDE") or ""
    if not spec:
        return
    for item in spec.split(","):
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        key, raw = key.strip(), raw.strip()
        if key not in globals() or key.startswith("_"):
            print(f"[config] 忽略未知覆盖项: {key}")
            continue
        current = globals()[key]
        try:
            if isinstance(current, bool):
                value = raw.lower() in {"1", "true", "yes", "on"}
            elif isinstance(current, int) and not isinstance(current, bool):
                value = int(float(raw))
            elif isinstance(current, float):
                value = float(raw)
            elif isinstance(current, str):
                value = raw
            else:
                print(f"[config] 忽略不支持覆盖的类型: {key}")
                continue
        except ValueError:
            print(f"[config] 覆盖值无法解析: {key}={raw}")
            continue
        globals()[key] = value
        print(f"[config] 覆盖 {key} = {value}")


# LoRA 权重强度：身份条件与结构条件的取舍旋钮
# 实测（权重真正生效之后重标）：1.0 时角色特征偏弱（发色几乎没被改写），
# 1.5 时头发明显变浅、服装也换成角色的白蓝配色，但再往上画面会过度风格化。
LORA_SCALE = 0.90                   # 通用起点
HUMAN_LORA_SCALE = 1.45
ANIMAL_LORA_SCALE = 0.80
OBJECT_LORA_SCALE = 0.85
MULTI_ADAPTER_ALLOWED = True        # 允许联合加载同一底模的多个分支权重（须先验证兼容性）

# ---- 受控参考增强（方案 §8）：补 LoRA 对局部细节的遗漏，不接管角色身份学习 ----
# 方案明确要求先比较「仅LoRA / 仅参考条件 / LoRA+参考条件」，因此默认关闭。
USE_REFERENCE_ENHANCEMENT = False
REFERENCE_ADAPTER_SUBFOLDER = "sdxl_models"
REFERENCE_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
# 备用权重：plus 版在 diffusers 0.31 上会因切片配置缺失而报 TypeError，此时自动改用这一份
REFERENCE_ADAPTER_FALLBACK = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
REFERENCE_FACE_SCALE = 0.35         # 脸部参考槽强度（只作用于提示区域）
REFERENCE_COSTUME_SCALE = 0.40      # 服装参考槽强度
REFERENCE_MAX = 3                   # 最多使用的参考图数量（脸 + 服装 + 单独标志参考图各一）。
                                    # 单独标志参考槽是规格 §3 要求的"推理时保留真实参考图"，
                                    # 因此必须给它留位置：2 个槽位时它永远轮不到（实测）。
REFERENCE_RENDER = 512              # 参考槽输入分辨率

# =========================
# 模块开关（关闭即退化为明确记录的条件，不静默换实现）
# =========================
USE_VLM = True                      # 自动标注、主体理解、角色档案生成
USE_SAM = True                      # 主体分割（非人物分支必需）
USE_CONTROLNET = True               # 结构条件
VLM_DTYPE = "fp16"
VLM_LOAD_IN_4BIT = True             # 4B 模型在 8GB 显存上必须量化
VLM_OFFLOAD_CPU = False             # 与扩散段分时占用显存即可
VLM_MAX_NEW_TOKENS = 512
VLM_TEMPERATURE = 0.2

# =========================
# 数据登记（方案 §12）
# =========================
# 单一数据访问层（data_registry.py），两种后端共用同一份 SQL 与同一套调用；
# 训练启动时一次性批量读取清单，训练循环内不查询数据库。
# 连接信息优先级：环境变量 > 本地配置 config.local.json > 内置默认值。
def _local_settings() -> dict:
    import json as _json
    path = os.path.join(PROJECT_ROOT, "config.local.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as _stream:
            data = _json.load(_stream)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as _exc:
        print(f"[config] 警告: 本地配置读取失败，已忽略: {path} ({_exc})")
        return {}


_LOCAL = _local_settings()
_MYSQL_LOCAL = _LOCAL.get("mysql") or {}

REGISTRY_BACKEND = (os.environ.get("ENDLESS_REGISTRY_BACKEND")
                    or _LOCAL.get("registry_backend") or "mysql")
MYSQL = {
    "host": os.environ.get("ENDLESS_MYSQL_HOST") or _MYSQL_LOCAL.get("host", "127.0.0.1"),
    "port": int(os.environ.get("ENDLESS_MYSQL_PORT") or _MYSQL_LOCAL.get("port", 3306)),
    "user": os.environ.get("ENDLESS_MYSQL_USER") or _MYSQL_LOCAL.get("user", "endless"),
    "password": (os.environ.get("ENDLESS_MYSQL_PASSWORD")
                 or _MYSQL_LOCAL.get("password", "")),
    "database": (os.environ.get("ENDLESS_MYSQL_DATABASE")
                 or _MYSQL_LOCAL.get("database", "endless_xxx")),
    "charset": "utf8mb4",
}
REGISTRY_REQUIRED = True            # True=连接失败即报错；False=允许用清单文件恢复既有任务
LABEL_RULE_VERSION = "v1"           # 标注规则版本；规则或观察模型变化时递增
LABEL_TOKEN_BUDGET = 77             # 训练标注的 CLIP token 预算（`train.py` 会硬性检查，
                                    # 超了直接抛 ValueError）。这是**整条训练提示词**的预算：
                                    # 触发词 + 身份条件 + 质量词 + 该图标注 + 该图标志条件。
                                    # 实测（2026-10-05）曾有 21 条到 176/77——因为
                                    # `enrich_captions` 只检查标注本身，没算上其余部分。
PROFILE_FINGERPRINT_INCLUDES_CAPTIONS = False
                                    # 角色档案的缓存键是否包含 captions.txt。**默认否**。
                                    # 实测踩过：captions.txt 由 `enrich_captions` 每轮自动重写，
                                    # 而档案生成发生在它之前，于是
                                    # "档案写好 → captions 被重写 → 档案立刻过期"成了固定循环：
                                    # 每次训练/推理都重建一次档案（多约 8 分钟 VLM），
                                    # 且 `test.py` 会在启动时因"语义档案缺失"直接退出。
                                    # 档案内容来自图像观察、不依赖 captions，故不该由它决定过期。

# =========================
# 环境
# =========================
HF_ENDPOINT = "https://hf-mirror.com"
HF_OFFLINE = True                   # 本地缓存已具备所需配置；在线拉取会在国内网络下超时


def get_character_config(folder: str) -> dict:
    """返回该角色的训练配置（角色间无差异，预留按角色覆盖的入口）。"""
    return {
        "resolution": RESOLUTION,
        "max_train_steps": MAX_TRAIN_STEPS,
        "trigger": get_trigger(folder),
        "min_images": MIN_DATASET_IMAGES,
    }


# 覆盖必须在**文件末尾**执行：只有到这里，本文件定义的全部常量才都已就位。
# 早期它写在文件中部，排在它之后的常量（如 USE_REFERENCE_ENHANCEMENT）
# 覆盖请求会被当成"未知项"忽略，对照实验静默跑了两遍同一配置。
_apply_override()


if __name__ == "__main__":
    print("=" * 64)
    print("config.py 配置摘要")
    print("=" * 64)
    for name in ("PROJECT_ROOT", "DATASET_DIR", "MODEL_DIR", "INPUT_DIR", "OUTPUT_DIR",
                 "TRAIN_MODELS_DIR", "CHARACTERS_DIR", "SEMANTIC_DIR", "RESULT_DIR",
                 "LOGS_DIR", "MANIFEST_DIR"):
        print(f"[路径] {name:16} = {globals()[name]}")
    for name in ("BASE_MODEL_PATH", "BASE_VAE_PATH", "CONTROLNET_PATH", "SAM_PATH", "VLM_PATH"):
        value = globals()[name]
        exists = os.path.isfile(value) or os.path.isdir(value)
        print(f"[模型] {name:16} = {value}  {'存在' if exists else '缺失'}")
    print(f"[角色] CHARACTER_ID={CHARACTER_ID} TRIGGER_WORD={TRIGGER_WORD} "
          f"最低有效图片={MIN_DATASET_IMAGES}")
    print(f"[训练] res={RESOLUTION} batch={TRAIN_BATCH_SIZE} rank={LORA_RANK} "
          f"lr={LEARNING_RATE}/{LR_SCHEDULER} steps={MAX_TRAIN_STEPS} "
          f"detail_crop_ratio={DETAIL_CROP_RATIO} concept={CONCEPT_TRAIN}")
    print(f"[推理] steps={NUM_INFERENCE_STEPS} cfg={GUIDANCE_SCALE} "
          f"img2img={HUMAN_IMG2IMG_STRENGTH} cn={HUMAN_CONTROLNET_SCALE} "
          f"res={INFERENCE_RESOLUTION} seed={INFERENCE_SEED}")
    print(f"[权重] human={HUMAN_LORA_SCALE} animal={ANIMAL_LORA_SCALE} "
          f"object={OBJECT_LORA_SCALE} concept_non_human={CONCEPT_NON_HUMAN_SCALE}")
    print(f"[VLM ] {VLM_PATH} 4bit={VLM_LOAD_IN_4BIT} offload={VLM_OFFLOAD_CPU}")
    print(f"[数据] backend={REGISTRY_BACKEND} required={REGISTRY_REQUIRED} "
          f"rule_version={LABEL_RULE_VERSION}")
    print("=" * 64)
# ---- 脸部裁剪训练样本 ----
# 实测依据（2026-10-05）：22 张训练图里只有 **1 张** `face close-up`，
# `upper body` 构图下脸只占画面高度 15%~25%——512 训练分辨率下脸部约 80~130 像素，
# 身份权重因此几乎没学到"这个角色的脸长什么样"（整图 eval_loss 0.20 看着不错，
# 是因为脸部信号被全身像素摊薄）。补脸部裁剪样本就是给脸足够的像素预算。
# 脸部框由"主体外接框 + 构图标签"确定性推导（dataset.face_box_from_subject），
# 复用与标志裁剪同一条 DetailCrop 训练链路，不引入第二套裁剪机制。
FACE_SAMPLE_ENABLED = True# 脸部裁剪样本：**已启用**（2026-10-05 调整，附验证证据）。
#
# 目标：给脸足够的像素预算——22 张训练图里只有 1 张 `face close-up`，
# `upper body` 构图下脸只占画面高度 15%~25%，512 分辨率下约 80~130 像素，
# 身份权重因此几乎没学到"脸长什么样"。
#
# 取框走过三条路，前两条都被实测否掉（记录在此，避免重走）：
#   1. "主体掩版外接框 + 构图比例推导"：`002.png` 的 SAM 主体掩版落在**胸甲**上
#      （不是人物），裁出来是胸口与护甲——错对象上套比例没有任何纠错能力；
#   2. "直接问视觉模型脸在哪"：22 张定位到 13 张，其中约 6 张是误定位
#      （头发 / 手+天空 / 背景）；
#   3. "用 OpenCV Haar 复核"：动漫图**检出 0 张**（只对真人脸有效），
#      既不能确认也不能否证。
#
# **最终采用的做法**：让视觉模型**复核自己给出的框**——把候选裁剪再问一次
# "这里面有脸吗"。实测这条判据有效：误定位的候选（头发/手+天空/纯背景）
# 稳定返回空，真正的脸稳定返回一个框。复核通过后再**向下扩展补齐下巴**
# 并复核一次，不过就退回原框。
#
# 验证证据（`docs/_run_face_final.py`，contact sheet
# `output/_probe/faces/face_final_sheet.png`）：22 张里通过 6 张，
# **人工核对 6 格全部是脸**（019 整张脸最佳、018/008/009/016 清晰、003 含双眼），
# 未通过的 16 格全部是黑格（即没有可用脸）。
# 6 张高质量脸部样本相对原先的 1 张脸特写是实质提升，因此启用。
FACE_SAMPLE_MAX = 0                 # 最多加入多少条脸部样本（0=不限制，每张图一条）

# ---- 脸部框的几何合理性判据（与视觉模型复核**互相独立**的第二道判据）----
# 实测依据（2026-10-13 复核）：`FACE_SAMPLE_ENABLED=True` 之后仍有个别不合格的框
# 通过了"裁出来再问一次有没有脸"的复核——例如 `007.png` 拿到
# `[0.541, 0.027, 1.0, 0.916]`（占画面宽 46%、高 89%）。那样的"脸部样本"其实是一张
# 大半身像，与"给脸足够像素"的目的正好相反，会把身份权重重新摊薄。
# 阈值取得很宽，只挡明显不合理的框；实测记录里通过复核的 6 张脸都在阈内。
FACE_BOX_MIN_SIDE = 0.03            # 框的最小边长（归一化），低于此判为没框住
FACE_BOX_MAX_WIDTH = 0.60           # 宽度上限（占画面）：超过判为半身像
FACE_BOX_MAX_HEIGHT = 0.55          # 高度上限（占画面）：超过判为半身像
FACE_BOX_MIN_ASPECT = 0.45          # 宽/高 下限
FACE_BOX_MAX_ASPECT = 2.40          # 宽/高 上限
FACE_BOX_MAX_TOP = 0.80             # 上沿位置的允许下限：脸不会出现在画面最下方
FACE_VERIFY_UPSCALE = 2.5           # 复核脸部候选时先放大几倍（只影响"复核"这一问）
FACE_VERIFY_MAX_SIDE = 1024         # 放大后的长边上限（避免复核开销失控）
# 为什么复核要放大（实测，2026-10-13）：候选裁剪往往只有一两百像素，
# 直接送进视觉模型时小脸容易"看不见"，于是**本应通过的框被大量误拒**——
# 实测 22 张里复核通过 0 张，日志几乎全是"复核未检出人脸"，
# 而定位阶段明明给出了看起来合理的框（例如 275x177）。
# 放大只用于复核（读取已有像素、不产生新信息），真正裁给训练的样本仍按原分辨率走。

# 覆盖必须在**文件末尾**执行：只有到这里，本文件定义的全部常量才都已就位。
_apply_override()
# ---- 角色"风格特征"迁移（非人物承载的第一步）----
# 用户反馈（2026-10-06）："看起来就只是简单的单独标志图的粘贴，
# 根本没有学习到这种角色的特征并运用上"。这个判断准确——上一版贴的是
# **一张图的像素**，与器物自身的受光/材质/透视无关，也没有用到"这个角色的颜色关系"。
# 现在改为：先从角色自己的图像证据里 k-means 出主色（这是"角色的颜色关系"），
# 再把目标表面按它**重新着色**——只迁移色度，**明度通道完全保留**，
# 因此器物的圆润、高光、阴影、材质纹理全部还在，变的是"它属于哪个角色"。
MARK_STYLE_ENABLED = True
MARK_STYLE_PALETTE_K = 5            # 提取几个主色（按占比排序）
MARK_STYLE_STRENGTH = 0.88          # 表面重新着色的强度（0=不动；1=完全移到调色板色度）
MARK_STYLE_ASSET_STRENGTH = 0.70    # 标志素材按同一调色板重着色的强度
MARK_STYLE_FEATHER = 14.0            # 区域边缘羽化像素（避免硬边）

_apply_override()
MARK_STYLE_SAT_MIN = 40.0           # 调色板只统计"有色"像素的饱和度下限（0~255）。
                                    # 实测教训：不排除近白像素时，浅蓝近白的头发 + 白瓷皮肤
                                    # 让"角色配色"变成"近白+灰紫"，迁移到白瓷上完全看不出——
                                    # 而真正该学的点缀色（金/粉/深蓝）被淹没了。
MARK_STYLE_VAL_MIN = 35.0           # 明度下限（排除死黑阴影）

_apply_override()
MARK_STYLE_HIGHLIGHT_PROTECT = 0.30 # 高光保护：区域内最亮的一小撮少迁移，保住器物光泽。
                                    # 不做保护的话整只杯子会变成"哑光漆"，失去瓷器的反光。

_apply_override()
MARK_STYLE_DOMINANT_PULL = 0.60     # 中间调额外朝"角色主色"靠拢的比例。
                                    # 实测：纯明度插值会把中灰映射成浅藕荷，整只杯子变"淡紫加粉"，
                                    # 看不清是哪个角色；角色之所以是那个角色靠的是**主色**。

_apply_override()
# ---- 配色迁移用的"整个主体"判据（与"可贴区域"分开）----
# 实测教训：可贴区域的判据是"亮+低饱和+极平滑"，它要找"能看清图案的平坦区"，
# 在陶杯上只覆盖杯身左半边一小块（亮≥207 低饱和≤24 平滑≤2 都是画面分位数，很严）。
# 用它做配色迁移会得到"一条竖色带"，而不是"整只杯子变成角色的"。
# 因此配色迁移另用一组宽松判据定位**主体本身**。
MARK_STYLE_FULL_SUBJECT = True      # 配色迁移覆盖整个主体（而不是只覆盖平坦可贴区）
MARK_STYLE_VALUE_PCT = 88.0
MARK_STYLE_SAT_PCT = 72.0           # 低饱和分位（陶瓷/塑料表面比彩色背景更灰）

_apply_override()
MARK_STYLE_SMOOTH_PCT = 88.0        # 配色迁移掩版：局部方差分位上限（主体比背景平滑）。
                                    # 实测陶杯：主体局部方差中位 2.24、背景 3.85，区分度明确。
# 注：亮度的"主体/背景"分界改用 **Otsu 自动求**，不再用固定分位。
# 实测教训：62 分位在本样例只有 157，而主体 V 中位 200、背景 142——
# 整幅画面（含桌面）被选进来，掩版外接框 x 0~767 y 0~767，配色迁移把背景也染了。

_apply_override()
MARK_STYLE_GROW = 0.25              # 配色迁移掩版：种子外扩比例（把主体偏暗部分补进来）
MARK_STYLE_SUPPORT_SMOOTH_PCT = 94.0 # 外扩填充时的平滑度上限（比可贴区域的 62 宽松，
                                     # 因为这里要的是"整个主体"而不是"平坦可贴区"）
MARK_STYLE_SUPPORT_VALUE_RATIO = 0.80 # 外扩填充时的亮度下限（种子阈值的比例）

_apply_override()
MARK_STYLE_DILATE_PX = 26           # 外扩填充的邻域半径（像素）。
                                    # 实测：不限制邻域时，画面两侧被照亮的桌面也满足亮度条件，
                                    # 掩版从 x 258~581 扩到 x 178~661，左右各多约 80 像素被染色。

_apply_override()
MARK_STYLE_GLOBAL_FLOOR = 1.0       # 种子亮度的全局下限比例（乘整幅分位阈值）。
                                    # 实测两种取法各有一个坑：整幅取分位 -> 只能盖最亮部分，
                                    # 向外填充会把杯外亮桌面拉进来；框内取分位 -> 分位被抬高到
                                    # 几乎不过滤，掩版覆盖整幅。因此"框内阈值 + 全局下限"一起用。

_apply_override()
# ---- 标志"空间用法"判定（来自角色自己的图像证据）----
# 用户指出的方向（2026-10-06）：标志在角色身上**怎么分布**和它长什么样同样重要——
# 樱花是"零星覆盖在裙摆上"，金色图案"虽大但不占视觉主体、常与祥云线条组合"。
# 这些都能从逐图证据的框量出来，不需要写死角色：
#   coverage = 框面积中位数（视觉权重）；spread = 框中心的归一化标准差（散布程度）。
MARK_LAYOUT_FOCAL_COVERAGE = 0.030  # 覆盖率达到此值视为"视觉主体"
MARK_LAYOUT_SCATTER_SPREAD = 0.16   # 散布度达到此值视为"零星散布"（应重复铺开而非单张大图）

_apply_override()
# ---- 平铺单元按"角色上的实际占比"缩放 ----
# 用户指出（2026-10-06）："金色图案在角色服饰中虽然较大但不会说占较大视觉形成视觉主体"。
# 实测该标志在角色画面上只占 0.31% 面积，却因固定单元尺寸被放到近 200 像素见方，
# 成了整只杯子的视觉主体。因此用 coverage 反推单元缩放（面积 -> 边长取开方）。
MARK_CARRY_UNIT_SCALE_RANGE = (0.35, 1.15)  # 单元缩放系数的夹取范围
MARK_CARRY_UNIT_SCALE_REF = 0.06            # 参考占比：coverage 等于它时缩放为 1.0
MARK_CARRY_PATTERN_MIN_COLUMNS = 2.0        # 放不下时至少保证的列数（用于缩小重试）

# ---- 加法式着色（非人物）：保留主体形态与原画面风格 ----
# 用户要求"保留原图像主体的形态、比例"与"原图像的画面风格"。
# 实测：把明度整体映射到角色调色板会把器物洗成淡彩块
# （区域内平均 RGB [184,186,175] -> [119,130,153]，明度掉约 35%），属于替换而非加法；
# 而且 RGB->LAB->RGB 往返不是无损的，区域外会出现 4~5 的量化改动。
# 因此改为：明度只做很轻的跟随，彩度朝角色主导色偏移，并在合成阶段
# 让区域外逐像素取回原值（硬保证"只做加法"）。
MARK_STYLE_LUMA_FOLLOW = 0.22       # 明度跟随比例（0=完全保留原明度层次）
MARK_STYLE_ERODE_PX = 10            # 羽化前把掩版内缩的像素数（羽化带留在器物内侧）

# 内容物（杯中咖啡/水）不是器物表面
# 实测：杯口内壁与杯中液体在亮度/饱和度/平滑度上与器物本体几乎一致，
# SAM 会把它们与器物圈成同一个连通块，配色迁移**把咖啡染成了紫蓝色**。
# 打开后按器物口沿（横向强边）排除口沿之上的内容物。
MARK_STYLE_EXCLUDE_CONTENTS = True

# ---- 承载素材纯度闸门 ----
# 本轮实测的**最严重异常**：mark_refs.txt 指向的其实是**整张插画**
# （用户说明：这些是 images 内保存的单独象征标志，即该标志的**样例图**），
# 直接当承载素材会把整幅角色立绘平铺到器物上（实测出图出现九宫格完整立绘）。
# 旧的"背景透传"形状分离对插画无效（ref_006.png 抠完仍剩 68% 前景）。
# 因此设一道只依赖图像自身统计量的闸门：前景占比过高的素材判为插画，不使用。
MARK_ASSET_MAX_FOREGROUND = 0.55

# ---- 承载素材形状闸门 ----
# 实测：逐图裁剪有时会圈到"一缕头发 / 一条衣边"，铺上去就是几条弯线，
# 完全看不出是标志（出图上出现过一排弯线）。真正的纹样/徽记接近成块，
# 长宽比不极端、前景在外接框内填充率高。
MARK_ASSET_MAX_ASPECT = 2.2         # 前景外接框的长宽比上限
MARK_ASSET_MIN_FILL = 0.30          # 前景像素在其外接框内的最低填充率

# ---- 参考图内容裁切 ----
# 本轮实测的根因：reference_crop 原先只按**透明**范围裁，而用户的参考图是
# **白底不透明**的（如 images/021.png = 白底 + 黑色樱花 + 金色花心）。
# 对不透明图，"非透明外接框"就是整张图，于是 ref_*.png 原样保留整幅画面；
# 若那张参考图是整张插画，承载阶段就会把整幅角色立绘平铺到器物上
# （实测出图出现九宫格完整立绘）。改为按**主导背景色**去底后裁到内容外接框。
MARK_REF_BG_TOLERANCE = 26.0        # 与四边中位背景色的最大通道差（小于它算背景）
MARK_REF_ARTWORK_FILL = 0.70        # 内容占整幅达到此比例即判为"插画/整图"，不作贴图素材

# ---- 参考图结构判据 ----
# 判"这张参考图是**干净标志**还是**整张插画**"靠"内容占画布比例"是区分不了的
# （实测：021.png 的白底黑樱花占 85% 却完全可用；006.jpg 的角色立绘占 100% 不可用）。
# 真正的差别在结构：干净标志是"**一块**图形 + 简单背景"，
# 插画是"人物 + 场景 + 各处零散背景"。因此用**最大内容连通块占内容像素的比例**判定。
MARK_REF_MIN_LARGEST = 0.55         # 最大内容连通块占比下限（低于它判为插画/整图）

# ---- 参考图平面度判据 ----
# 判"参考图是干净标志还是渲染插画"，实测排除了两个判据：
#   * 内容占画布比例：021.png（白底黑樱花）占 85% 可用，006.jpg（立绘）占 100% 不可用，
#     比例几乎一样；
#   * 最大内容连通块占比：022.png 是**细线条金色徽记**，线条彼此分离，
#     最大连通块只占内容 18%；插画占 99% —— 刚好反着来，会误判好素材。
# 稳定判据是**去底后的显著颜色数**：平面标志是白底 + 一两种色（< 40），
# 渲染插画有渐变/阴影/光照（成百上千）。
MARK_REF_MAX_COLORS = 150.0          # 去底后显著颜色数上限（超过判为渲染插画）

# ---- 零星放置承载（替代规则网格平铺）----
# 用户评价"效果很差"的直接原因之一：两个标志各铺 6 次、器物表面 94% 像素被改写，
# 观感像"包装纸"或"重新刷过的杯子"，而不是"这件器物属于这个角色"。
# 需求是"加法"且"看得出是这个角色的标志"，因此改为**少量、看得清**的散布放置。
MARK_CARRY_SCATTERED_COUNT = 3      # `scattered`（零星散布）类标志放置几处
MARK_CARRY_MARK_MAX_RATIO = 0.34    # 单个标志尺寸上限（占区域短边比例）
MARK_ASSET_MIN_SIDE_PX = 96         # 素材短边的绝对像素下限。
                                    # 承载单元在成品上约 60~200 像素，素材短边低于
                                    # 这个量级必然放大发糊（实测白色樱花素材只有 50x67，
                                    # 出图上看不出是樱花）。用绝对像素而不是相对区域比例：
                                    # `_asset_for` 拿不到区域掩版，用比例会在运行时 NameError。

# ---- 曲面承印（模型出平面纹样 + 几何承印）----
# 用户批评（2026-10-08）："你把标志图片规整得虽小复制排列，这并不能体现模型
# 学习了角色的视觉特征并自然得运用这些"。
# 实测结论：让模型直接把纹样画到摄影器物的曲面上 -> 画不出来；
# 让模型在平面画布上画角色纹样 -> 画得出来，但配色偏"泛用日式"。
# 因此分工：**模型负责画什么，几何负责怎么落到器物上**。
SURFACE_CYLINDER = True             # 按柱面重投影（图案随器物圆润而横向收窄）
SURFACE_CYLINDER_STRENGTH = 1.0     # 重投影强度（0=不映射，1=完全按柱面采样）
SURFACE_RECOLOUR_STRENGTH = 0.6     # 把生成的纹样朝角色调色板着色的强度
SURFACE_SHADING = True              # 让纹样跟随器物明暗（高光处亮、背光处暗）
SURFACE_LUMINANCE_TRANSFER = 0.45   # 用器物明暗低频替换纹样低频的比例
SURFACE_SHADING_STRENGTH = 0.55     # 按器物明度调制纹样亮度的强度
SURFACE_EDGE_FEATHER = 6            # 纹样与器物交界的羽化像素

SURFACE_EDGE_FADE = 6.0               # 沿器物轮廓羽化的像素数（羽化带内缩，不溢出到背景）

# ---- 曲面承印之后要**还原标志落笔区** ----
# 实测问题（2026-10-13 定位）：确定性承载先把标志按角色用法放好（日志"铺放 3 次"），
# 紧接着曲面承印把**整个可承载区域**整块覆盖成生成出来的底纹，
# 于是成品上**一枚标志都看不见**，只剩一片喷绘图案——用户看到的就是
# "规整复制排列的图案"，而不是"角色的标志被自然地运用"。
# 职责划分应当是：**标志是视觉主体，底纹是衬底**。
# 还原用的是承载阶段留下的快照（标志已融合好、底纹未印），
# 因此标志的受光、材质与边缘都保持融合后的样子，而不是"再贴一次"。
SURFACE_PRINT_RESTORE_MARKS = True   # 承印后还原标志落笔区（False=旧行为，整块被底纹覆盖）
SURFACE_PRINT_RESERVE_PAD = 6        # 还原区外扩像素（让标志边缘不被底纹啃掉）

# ---- 某条标志自己的素材全部不可用时，允许在**同一角色库内**另找一份可印素材 ----
# 实测依据：`白色樱花图案` 的四张候选全部不可用——三张是插画、一张（`crop_009`）
# 是"平坦蓝底 + 金属斜边"，印上去就是方形贴片；而同一角色的 `ref_021.png`
# 恰好是一枚**干净的樱花徽记**（白底黑花、形状分离正确），只是被 `mark_refs.txt`
# 归到了另一条标志名下。用户的要求是"非人物上要一眼看得出是这个角色的标志"，
# 因此"因为归档归属而让杯子上一个樱花都没有"是更差的结果。
# 替换**仍然只使用有真实像素证据的库内素材**（不生成、不用通用花纹），
# 并如实写进审计。置 False 可关闭。
MARK_SUBSTITUTE_ASSET = True         # 允许库内代用素材
MARK_SUBSTITUTE_MIN_SIDE = 160       # 代用素材的短边下限（保证印上去清晰）

# ---- 曲面承印：生成纹样时的权重强度 ----
# 实测现象：在主链路里生成"角色纹样"，得到的是一张**柔和人物特写**
# （紫粉色调 + 丝带结），完全不是纺织品图案；而单独只挂概念权重、scale=1.0 时，
# 同一个提示词能生成干净的蓝底金花纺织图案。
# 原因是主链路里 **身份权重 scale=0.9 在主导**，概念只有 0.55——
# 身份分支学的是"这个角色长什么样"，会把画面拉回人物。
# 因此生成纹样这一步临时把身份压到 0、概念提到 1，生成完恢复。
SURFACE_DESIGN_IDENTITY_SCALE = 0.0
SURFACE_DESIGN_CONCEPT_SCALE = 1.0
SURFACE_DESIGN_SEED = 20261008
SURFACE_DESIGN_NEGATIVE = ("photo, photograph, 3d, person, girl, human, face, "
                           "portrait, text, watermark, blurry, low quality, "
                           "worst quality")

# ---- 表面承印融合（把标志**印**到器物上，而不是贴上画面）----
# 用户批评（2026-10-08）："你把标志图片规整得虽小复制排列，这并不能体现模型
# 学习了角色的视觉特征并自然得运用这些"。此前的承载是 alpha 合成（`mark_carry.paste_rgba`），
# **只能改颜色、改不了光照关系**，因此无论素材多干净、边缘多羽化，观感仍是"贴上去的一块"。
# 实测四个指标（`vision/metrics.py`，已在 `_check_metrics.py` 里验证过有区分度）：
#   * `follow_cos` 平贴 -0.20 ~ -0.33（纹样不跟随器物受光）；
#   * `boundary_edge_excess` 平贴 2.8 ~ 12.0（边界一圈硬"框"）；
#   * `texture_retention` 平贴 5.8 ~ 18.5（器物自身纹理被整块盖掉）；
# 融合的做法是"**图案提供颜色与花纹，器物提供明暗与材质**"：
#   去光（拿掉素材自身受光）-> 柱面预畸变 -> 重光（乘回器物受光）-> 合材（混入器物纹理）。
# 全部是确定性运算，可复现、可逐项核对，不引入扩散"随机画错"的风险。
MARK_FUSION_ENABLED = True          # 承载时走 `vision/surface_fusion`（False=退回旧的 alpha 平贴）
MARK_FUSION_DETAIL = 0.75           # 素材自身花纹层次的保留比例（0=只剩平色，1=原样）
MARK_FUSION_TEXTURE = 0.60          # 器物自身高频颗粒混入的比例（让图案"长在同一材质上"）
MARK_FUSION_CONFORMITY = 0.45       # 图案整体明度向器物看齐的比例（0=保留素材本色，1=完全看齐）
MARK_FUSION_FEATHER = 1.8           # 轮廓羽化像素（硬切边本身就是"贴纸感"的来源）
MARK_FUSION_ALPHA_BIAS = 0.35       # 形状 alpha 的**收紧阈值**（低于它直接压成 0）
MARK_FUSION_ALPHA_GAMMA = 1.6       # 收紧后把中间地带拉向两端的指数
# 为什么要收紧（实测缺陷，2026-10-13）：形状分离输出的 alpha 在背景处往往不是 0，
# 而是 0.2~0.3 的淡影（判据是颜色距离阈值，背景色附近总会残留）。
# 直接当权重用，背景会以 20%~30% 的比例混进成品——出图上是
# **黑色樱花周围一块淡蓝色方片**，因为底纹更亮，方片反而更显眼。
# 收紧必须发生在**羽化之前**：羽化是给轮廓用的，不能用来掩盖残留背景。
# ---- 素材自带背景时的抠除（实测缺陷的直接对策）----
# 实测（2026-10-13）：`crop_009`（所谓的白色樱花裁剪）实际内容是"一片平坦的蓝色 +
# 一块粉紫金属斜边"，而形状分离只透掉 73% 的像素，剩下的**平坦背景被当成标志**
# 一起贴上去——出图上是三块带直角边的方形贴片。素材自带背景并不罕见，
# 不能假设上游的形状分离一定干净，因此融合侧再设一道：
#   * 形状在外接框内的**实心度**超过 `MARK_FUSION_MAX_BLOCKINESS` 时判为"矩形块"，
#     改用按边框背景色重新抠除；抠不出来就**拒绝执行**，而不是照贴方片；
#   * 抠除用**测地距离**（必须能从边缘连通到背景）而不是欧氏距离，
#     否则徽记内部与背景相近的颜色会被整块剔掉、只剩一圈描边。
MARK_FUSION_MAX_BLOCKINESS = 0.62   # 形状在其外接框内的实心度上限
MARK_FUSION_MAX_FILL = 0.90         # 形状覆盖率上限；**与上一条同时成立**才判为矩形块
# 为什么必须两条同时成立（实测教训）：只看实心度会误伤干净的大素材——
# `ref_021.png`（1349x1248 的黑色樱花徽记）实心度 0.64，但它的形状分离是正确的
# （覆盖率 0.35、背景已透掉），而且它正是"白色樱花图案"最需要的干净素材。
MARK_FUSION_MATTE_TOLERANCE = 22.0  # 背景色距离容差（0~441）
MARK_FUSION_MATTE_RAMP = 10.0       # 容差到"完全前景"的过渡宽度
MARK_FUSION_BG_SPREAD = 26.0        # 边框颜色的离散度上限（超过说明没有平坦背景，不硬抠）
MARK_FUSION_MIN_ASSET_SIDE = 48     # 素材短边的绝对下限（低于此放大后必然糊）
MARK_FUSION_MIN_CONTOUR_DETAIL = 4.0  # 形状边界像素的明度标准差下限（低于此判为一整块纯色）
MARK_CARRY_CYLINDER = 0.55          # 柱面预畸变强度（0=平面，1=强烈收窄 + 上下边明显起弧）

_apply_override()
