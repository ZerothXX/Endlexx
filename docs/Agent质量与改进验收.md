# Agent 质量与改进验收

日期：2026-10-25。环境：Windows、RTX 4060 Laptop 8GB。质量评测调用实际本地模型；功能验收使用独立 SQLite 数据库，不向原角色知识库写入虚构设定。

## 本轮结论

已实测现有 Qwen3-VL-4B、Qwen3 4B Instruct 和 Qwen3 8B。模型类别本身不能保证 Agent 能力；本机 8B 在完整多步流程上表现较好，仍未全部通过。项目以多步图像任务为主，因此选择已部署的 `qwen3:8b` 作为新对话默认主控，图片交给原有 Qwen3-VL，并保留原方案供快速交流。已有对话保留原来的模型选择，可在输入框下方切换；已有任务仍使用提交时的快照。

自由生成摘要确有事实遗漏、擅自添加待办和改写文件名的问题。本轮改用「模型选原文编号，程序复制原文」，并优先完整保留能放进预算的用户原文。条件规划同时增加执行条件、来源图片核对、跳过原因及重放凭据，避免把不确定检查当作修改许可。

## 评测方法与适用范围

用例在 [agent_quality.json](../evals/agent_quality.json)，实际执行器为 [evaluate_agent.py](../scripts/evaluate_agent.py)。共 30 个自建中文开发回归案例：工具选择 16、复杂依赖/条件规划 8、摘要事实 6。每个案例建立独立对话，直接调用生产规划与上下文代码；不是只测试一个另外编写的提示词。

工具选择按完整工具顺序比对；条件步骤要求对应的 `if_mismatch`，不能只把条件写进说明。摘要用明确事实的替代词匹配，同时保存原文供人工复核。词匹配不是语义正确性证明，程序护栏通过也不等于模型自身能够可靠记忆。这组案例用于开发迭代，已经参与修正，不是独立保留的通用 Agent 基准；单轮、小样本和加载缓存差异均限制结论。

原流程对比复用了冻结的 `agent/context/providers` 源码；新运行保存代码快照、实现与用例 SHA256、逐案结果、原始模型输出和耗时。摘要用例把关键事实放在早期，并加入无新要求的长对话，强制经过压缩。

### 原流程

| 主控 | 工具选择 | 复杂规划 | 摘要事实 | 工具选择 P50 |
|---|---:|---:|---:|---:|
| Qwen3-VL-4B，NF4 | 13/16 | 6/8 | 2/6 | 2.03 秒 |
| Qwen3 4B Instruct，Q4_K_M | 13/16 | 6/8 | 3/6 | 5.65 秒 |

原始报告：`data/verification/agent-quality/baseline-vl4b/report.json`、`baseline-text4b/report.json`。例如原摘要把「检查失败先汇报，不自动修改」遗漏为普通待办；另一次将 `draft_blue_2026.png` 改写成不同文件名。这些是引入原文摘录与保真护栏的依据。

### 修正后的流程

| 主控 | 工具选择 | 复杂规划 | 摘要事实 | 工具选择 P50 | 复杂规划 P50 | 摘要 P50 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-4B，NF4 | 16/16 | 5/8 | 6/6 | 2.45 秒 | 3.53 秒 | 3.13 秒 |
| Qwen3 8B，Q4_K_M | 15/16 | 7/8 | 6/6 | 6.33 秒 | 6.62 秒 | 12.21 秒 |

这两轮的实现 SHA256 均为 `17b1ea459dc88e37790972e33d8cffedb10eca91e13a0c86ebccc5007fc38172`，用例也相同。完整报告与源代码快照位于 `data/verification/agent-quality/release-native/` 和 `release-text8b/`。前面的中间迭代报告不与最终结果拼接成成绩。

VLM 在「修复→迁移→检查」「修复→查询→检查」中把检查标成条件修改，重试后仍不合法；在「不确定就停止」的条件链中漏掉停止标志。8B 通过上述两条无条件多步链，但仍漏掉后一条的停止标志，重试后请求澄清。8B 的另一个失败是单纯迁移额外查询了一次角色知识库。失败均计入分母，没有因为执行器挡住错误就记成规划正确。

因此默认选择 8B 是本项目对多步完成率与时延的取舍，不是「LLM 一定胜过 VLM」的结论。8B 仍需护栏，不能称为复杂任务完全可靠。其每次调用后卸载以与图像引擎顺序复用 8GB 显存，加载时间也包含在耗时中；没有将原模型常驻时延与 8B 纯解码时延混比。最终摘要两者均为 6/6 是摘录与保留规则后的系统结果，不能归因于模型自身记忆能力。

## 已修复与补齐

| 问题 | 改动 | 验证依据 |
|---|---|---|
| 普通说明、否定要求与历史操作容易触发错误工具 | 明确工具边界，历史作为数据传递，格式重试包含具体校验原因；无工具操作规范化为普通交流 | 真实逐案评测与结构校验测试 |
| 「先检查，只有不符才修改」被遗漏或直接无条件执行 | 条件与工具逐项关联；仅明确 mismatch 才执行；要求不确定即停止时，unknown/not_visible 也阻止修改 | mismatch、unknown、不可见、过期图片和凭据重放测试 |
| 明确的条件修改步骤可能只写在摘要中 | 对可明确识别的中文条件命令检查计划完整性；不完整先重试，仍不合法则请求澄清，不执行半套计划 | 条件遗漏与引用/否定反例测试 |
| 自由摘要漏约束、编待办、改文件名 | 摘录完整原文，检验索引与字节预算，保留来源角色和时间顺序；能放下的用户原文全部保留 | 摘录/否定/文件名测试及真实长对话往返 |
| 旧摘要缓存继续传播错误 | 识别摘要格式版本，从原消息重建旧自由摘要；失败不推进压缩边界 | 缓存隔离与失败保留测试 |
| 纯文字主控不能接入或被当成视觉模型使用 | 部署时验证 completion 和真实生成，图片独立路由到已有本地 VLM；列表说明视觉分工 | 实际 8B 部署、任务快照与图片问答 |
| API 新建会话固定使用旧默认模型 | 未显式选择时继承当前启用的模型连接；显式 local 与已有会话保留原选择 | 切换默认模型回归测试及真实 API 创建验收 |
| thinking 占满短输出预算，空答复被误判为上下文过长 | Ollama 显式关闭 thinking；空最终答复按生成失败处理 | 供应器协议测试与真实模型调用 |
| 向量/重排数量异常导致文档被静默丢弃或排名失真 | 验证向量形状、数量与有限值；失败保留完整 BM25 文档/融合排序并给出提示 | 部分向量、NaN、Inf、空分值测试 |
| 离线 SDXL 适配器重复传入 config，迁移启动失败 | 保留 Diffusers 内部已传入的参数，只补充缺省值 | 复现嵌套 VAE 调用的回归测试与正式入口重跑 |
| 迁移重试可能拿到旧文件，或使用空权重清单 | 拒绝空权重清单；要求正式入口更新本次输出文件；保留原文件 | 旧产物不可冒充新输出的测试 |
| 视觉复核重复上一轮的待定判断 | 复核只提供要求、承载部位和索引，让模型重新看原图；原观察留在审计中 | 复核不注入旧观察/旧判定的测试；视觉效果另列 |
| 后台停止命令在权限不足时假装成功 | 检查失败/结束失败保留 PID，并返回失败状态；检查进程归属后再结束 | 权限失败和 PID 复用测试 |
| 重复界面入口与验收工具不可靠 | 删除未使用的旧 API 配置组件及样式；验收脚本使用独立数据库，修正幂等角色导入和 MCP 环境传递 | 浏览器、脚本与真实 MCP 调用 |

同一角色下的新建/删除对话、删除时消息/摘要/任务/检查点清理、保留角色资料与图片、忙碌对话保护，以及会话之间的摘要隔离，均纳入回归验收。

## 功能验收

本节的 `data/verification/` 原始功能记录留在开发机器，不随源码发布；公开包仅附虚构开发用例的两份原始主控评测报告，见 `evals/reports/`。图片验收需要另行准备原样例；`smoke_agent.py --image PATH` 可显式指定路径，视觉断言仍针对固定样例，并非任意图片评分器。

自动测试：Python **78 项通过**（12.12 秒），Ruff 通过；前端 TypeScript/Vite 构建通过，浏览器 **9 项通过**。包括同角色多会话创建/删除、忙碌保护、级联清理、上下文隔离、模型管理与图片路由。测试通过不代表下面所有真实模型效果达标。

真实功能结果如下。原始 JSON 均位于 `data/verification/`，模型推理按顺序运行，测试资料保存在独立数据库 `full-acceptance.db`，检索消融另用 `rag-evaluation.db`。

| 功能 | 结果 | 原始证据 |
|---|---|---|
| 默认主控 | 新会话继承 Qwen3 8B；显式 local 保留；原会话不被默认切换改写 | `default-model.json`，模型管理回归测试 |
| 知识问答 | 原 VLM、8B 均通过；真实 BGE 向量与重排，无降级警告，正确引用右肩珐琅设定 | `acceptance-knowledge-local.json`、`acceptance-knowledge-m_5c0989dd25b8.json` |
| 图片问答 | 8B 规划、原生 VLM 看图，正确描述白色挂脖裙 | `acceptance-image_chat-m_5c0989dd25b8.json` |
| 局部修复 | 真正调用 LaMa，生成新图；保护区变化像素为 0，人工查看补全效果通过 | `acceptance-repair-local.json`；输出资产 `e548bc24968f4250b37d13582228ff7e` |
| 长对话 | 两种主控均通过：压缩 10 条旧消息，保留全部原始消息；回答仍包含黑色手套、米色裙、准确文件名及禁止自动修改约束 | `acceptance-context-local.json`、`acceptance-context-m_5c0989dd25b8.json` |
| 一致性检查 | 报告与独立复核均完成；**质量未通过**：要求的眼镜缺失仍判为 unknown，鞋子不可见判断正确；更换文字主控后仍存在 | `acceptance-check-local.json`、`acceptance-check-m_5c0989dd25b8.json` |
| 角色迁移 | 正式研究入口执行通过、产出新图；**画质未通过**，存在明显色带与伪影，详见下文 | `acceptance-transfer-local.json`、`transfer-generation-audit.json`；输出资产 `812515093f2d44b18eed71ddc7ec606b` |
| MCP | 真实 stdio 握手、两个工具、检索调用通过；5 条引用、无警告；拒绝未知角色 | `mcp.json` |
| 检索消融 | 12 文档、12 查询；四种策略 Recall@3 均为 1.0，BM25/混合 MRR@3 为 0.9583，向量/混合重排为 1.0 | `rag-ablation.json` |

这组检索资料是明确标记的虚构验收设定，小样本成绩不能推导真实资料库检索质量。混合重排 P50 为 2.82 秒，BM25 为 0.0043 秒，向量为 1.37 秒，混合为 1.26 秒。外部 API 没有可用密钥，只做协议与异常处理测试。

一致性检查保留严格断言，两个真实检查脚本因此以失败状态退出；迁移执行脚本只验证入口、产物和权重，人工画质验收另记失败。没有用执行成功替代模型效果通过。

验收结束后已停止项目 API、两个隔离验收 API、原生模型运行时与托管 Ollama；8765、8766、8875、8877、11435 均验证不再监听，启动器 PID 文件已清除。证据见 `data/verification/shutdown.json`。默认模型配置保留，后台保持关闭。

### 迁移效果的进一步排查

本轮正式入口耗时 755.7 秒，主生成耗时 222.9 秒，头部强化耗时 226.4 秒；全过程约 13 分钟，实际保存了新的 768×928 结果及权重摘要。输出仍有明显色带、边缘断裂和局部伪影，人工视觉验收未通过。

查看锚定输入图后确认：生成前的发色锚定本身已有蓝色块状区域，最终结果则有更广泛的色带。生成审计还记录了按色差筛选的 `source_preserve.changed_ratio=0.8561`、`region_only=false`，以及局部明度调整。它们是下一步消融排查的具体线索，但目前缺少主生成、强化后、合成前后的独立中间图，不能据此断言唯一根因，也没有把更换文字主控说成画质修复。研究源码与生成参数未为通过验收而改写。

原始生成审计另存为 `data/verification/transfer-generation-audit.json`，避免研究项目后续同角色运行覆盖本次证据；正式任务及前后资产见 `acceptance-transfer-local.json`。需要在独立样例上对生成、配色锚定和阈值合成分别做消融，再决定图像侧改动。

## 保留的限制

- 8GB 本地小模型的完整规划仍不稳定；结构校验和条件护栏能限制错误执行，不能保证理解所有自然语言。条件完整性检查目前只覆盖可明确识别的中文表达。
- 原文摘录保证复制内容不被改写，但当用户原文本身超过预算时，模型仍可能漏选重要信息；多次压缩也可能遗忘早期事实。原始消息始终保留。后续应在独立长对话集上检验更正、冲突和跨轮引用。
- 自动修复仅覆盖可可靠定位的局部浅色/深色色块缺损；保护区零变化不能代替补全区域的视觉质量评价。PPO 旧权重缺少对应编码器，仍未接入。
- 角色迁移与一致性检测的执行成功不代表视觉质量达标；失败的视觉断言不删除、不改成宽松通过。实际样例结果另列。
- 无第三方 API 密钥，外部供应商仅完成协议和错误处理测试；没有声称实际供应商质量已验证。全新空白 Windows 机器的完整大模型下载、PostgreSQL/WSL 路径也未在本轮重新验收。

## 复现

常规检查：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check endlexx scripts tests
npm.cmd --prefix frontend run build
npm.cmd --prefix frontend run test:e2e
```

真实功能验收先启动隔离 API，使用独立数据库及 8875 端口；关闭原本机 API，避免两个本地 Worker 竞争同一进程锁。以下为 PowerShell 新会话中的命令，不改变 `.env`：

```powershell
$evaluationRoot=(Get-Location).Path.Replace('\','/')
$env:ENDLEXX_DATABASE_URL="sqlite:///$evaluationRoot/data/verification/full-acceptance.db"
$env:ENDLEXX_DATA_DIR="$evaluationRoot/data"
$env:ENDLEXX_TASK_BACKEND='local'
.\.venv\Scripts\python.exe -m uvicorn endlexx.api:app --host 127.0.0.1 --port 8875
```

测试终端需保持本地模型运行时可用；以下推理测试依次运行，避免显存与耗时互相影响。迁移需要已发布的 `ayaka_f` 角色包，可用 `--package` 显式指定其他测试包；当前图片断言与本项目固定样例对应。

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_agent.py --model native --label native-new-run
.\.venv\Scripts\python.exe scripts/evaluate_agent.py --model ollama:qwen3:8b --label text8b-new-run
.\.venv\Scripts\python.exe scripts/smoke_agent.py knowledge
.\.venv\Scripts\python.exe scripts/smoke_agent.py repair
.\.venv\Scripts\python.exe scripts/smoke_agent.py transfer
.\.venv\Scripts\python.exe scripts/smoke_agent.py check
.\.venv\Scripts\python.exe scripts/smoke_agent.py image_chat
.\.venv\Scripts\python.exe scripts/smoke_context.py
.\.venv\Scripts\python.exe scripts/evaluate_rag.py
$env:ENDLEXX_DATABASE_URL="sqlite:///$evaluationRoot/data/verification/rag-evaluation.db"
.\.venv\Scripts\python.exe scripts/smoke_mcp.py
```

新评测标签下保存报告与源代码快照；同一标签重跑会覆盖该标签报告。所有临时角色和任务保留在验收数据库中，便于复查；原有用户对话不被当作测试样本修改。
