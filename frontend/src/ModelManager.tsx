import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Check, ChevronRight, Cloud, Cpu, Database, Layers3, LoaderCircle, Plus, X } from "lucide-react";
import { api, post } from "./api";
import type { Deployment, ModelConnection, ModelRegistry } from "./api";
import "./models.css";
import EnvironmentPanel from "./EnvironmentPanel";

export const kindLabel = { api: "多模态 API", local: "本地部署", hybrid: "多模型组合" };
const stageLabel: Record<string, string> = { queued: "等待中", running: "进行中", completed: "已完成", failed: "失败", cancelled: "已取消", interrupted: "已中断" };
const bytes = (size: number) => size >= 1024 ** 3 ? (size / 1024 ** 3).toFixed(2) + " GB" : (size / 1024 ** 2).toFixed(1) + " MB";
const installing = (status: string) => ["queued", "running"].includes(status);

export default function ModelManager({ registry, onRefresh, onSelect, onClose, onError }: {
  registry: ModelRegistry;
  onRefresh: () => Promise<unknown>;
  onSelect: (id: string) => Promise<void>;
  onClose: () => void;
  onError: (error: unknown) => void;
}) {
  const [page, setPage] = useState<"list" | "add" | "api" | "local" | "hybrid" | "environment">("list");
  const [catalog, setCatalog] = useState<{local: {model: string; name: string; size: string}[]; bundle: string[]}>({local: [], bundle: []});
  const [jobs, setJobs] = useState<Record<string, Deployment>>({});
  const [edit, setEdit] = useState<ModelConnection | null>(null);
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [model, setModel] = useState("");
  const [visionModel, setVisionModel] = useState("");
  const [key, setKey] = useState("");
  const [localModel, setLocalModel] = useState("qwen3-vl:4b-instruct");
  const [custom, setCustom] = useState(false);
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const dialog = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.focus();
    api<typeof catalog>("/model-catalog").then(setCatalog).catch(onError);
    return () => previous?.focus();
  }, []);
  useEffect(() => {
    let live = true;
    let refreshing = false;
    const poll = async () => {
      if (refreshing) return;
      refreshing = true;
      try {
        const results = await Promise.all(registry.entries.filter(row => row.kind !== "api").map(async row => {
          const job = await api<Deployment | null>(`/model-connections/${row.id}/deployment`);
          return job ? [row.id, job] as const : null;
        }));
        if (!live) return;
        setJobs(Object.fromEntries(results.filter(row => row !== null)));
        if (results.some(row => row && !installing(row[1].status) && registry.entries.find(entry => entry.id === row[0])?.status === "installing")) await onRefresh();
      } catch (error) { if (live) onError(error); }
      finally { refreshing = false; }
    };
    void poll();
    const timer = setInterval(poll, 1600);
    return () => { live = false; clearInterval(timer); };
  }, [registry.entries.map(row => row.id + row.status).join(",")]);

  const run = async (action: () => Promise<unknown>) => {
    setBusy(true);
    try { await action(); await onRefresh(); } catch (error) { onError(error); }
    finally { setBusy(false); }
  };
  function openApi(entry: ModelConnection | null = null) {
    setEdit(entry); setName(entry?.name || ""); setUrl(entry?.base_url || "");
    setModel(entry?.model || ""); setVisionModel(entry?.vision_model || ""); setKey(""); setPage("api");
  }
  return <div className="model-overlay" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}>
    <div ref={dialog} className="model-dialog" role="dialog" aria-modal="true" aria-labelledby="model-dialog-title" tabIndex={-1}
      onKeyDown={event => {
        if (event.key === "Escape") onClose();
        if (event.key === "Tab") {
          const controls = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), input, select, [tabindex="0"]'));
          const first = controls[0], last = controls[controls.length - 1];
          if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog.current)) { event.preventDefault(); last?.focus(); }
          else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
        }
      }}>
      <aside className="model-nav"><h2>设置</h2><button className={page !== "environment" ? "selected" : ""} onClick={() => setPage("list")}><Database size={17}/>模型</button><button className={page === "environment" ? "selected" : ""} onClick={() => setPage("environment")}><Cpu size={17}/>环境检查</button><p>为你的角色工作空间<br/>连接合适的模型。</p></aside>
      <section className="model-content">
        <header><div>{page !== "list" && <button className="icon-button" aria-label="返回模型列表" onClick={() => setPage("list")}><ArrowLeft size={17}/></button>}<h2 id="model-dialog-title">{page === "environment" ? "环境检查" : page === "list" ? "模型" : page === "add" ? "添加模型" : page === "api" ? (edit ? "编辑 API" : "API 添加") : kindLabel[page]}</h2></div><button className="icon-button" aria-label="关闭模型设置" onClick={onClose}><X size={19}/></button></header>
        {page === "environment" ? <EnvironmentPanel onError={onError}/> : page === "list" ? <>
          <p className="model-description">管理模型连接，在角色聊天框下方随时切换。</p>
          {!registry.entries.length && <div className="models-empty"><Cpu size={30}/><h3>连接第一个模型</h3><p>使用 API 即可开始，或让 Endlexx 下载并部署本地模型。</p></div>}
          <div className="connection-list">{registry.entries.map(entry => {
            const job = jobs[entry.id];
            const inProgress = job ? installing(job.status) : entry.status === "installing";
            return <div className="connection-card" key={entry.id}>
              <div className="connection-row"><span className={"model-status-dot " + (entry.status === "ready" ? "ready" : inProgress ? "working" : "")}/><div className="connection-info"><strong>{entry.name}</strong><small>{kindLabel[entry.kind]} · {entry.model}{entry.vision_fallback === "native" ? " · 视觉：本地 Qwen3-VL" : ""}</small></div><div className="connection-actions">
                {entry.status === "ready" && <button disabled={busy} className={registry.active_id === entry.id ? "selected-model" : ""} onClick={() => run(() => onSelect(entry.id))}>{registry.active_id === entry.id ? <><Check size={13}/> 已选用</> : "选用"}</button>}
                {entry.kind === "api" && <button onClick={() => openApi(entry)}>编辑</button>}
                {!inProgress && <button className="danger-text" onClick={() => setDeleting(deleting === entry.id ? null : entry.id)}>删除</button>}
              </div></div>
              {deleting === entry.id && <div className="model-delete"><span>删除此连接与密钥？已下载的共享权重会保留。</span><button disabled={busy} onClick={() => run(async () => { await api(`/model-connections/${entry.id}`, {method: "DELETE"}); setDeleting(null); })}>确认删除</button><button onClick={() => setDeleting(null)}>取消</button></div>}
              {job && job.status !== "completed" && <div className="deployment-progress" aria-live="polite"><div className="deployment-title"><strong>{inProgress && <LoaderCircle size={13} className="spin"/>}{job.phase}</strong><span>{job.items.filter(item => item.status === "completed").length} / {job.items.length}</span></div>
                <ol>{job.items.map((item, index) => <li key={index} className={item.status}><div><span>{item.status === "completed" ? <Check size={13}/> : index + 1}</span><strong>{item.name}</strong><small>{stageLabel[item.status] || item.status}</small></div>{item.status === "running" && <><progress max={item.total || undefined} value={item.total ? item.completed : undefined}/><p>{item.total ? `${bytes(item.completed)} / ${bytes(item.total)} · ${Math.min(100, item.completed / item.total * 100).toFixed(1)}%` : "正在准备，请稍候"}{item.detail && <span> · {item.detail}</span>}</p></>}</li>)}</ol>
                {inProgress ? <button disabled={busy} onClick={() => run(() => post(`/model-connections/${entry.id}/deployment/cancel`))}>取消部署</button> : <button disabled={busy} onClick={() => run(() => post(`/model-connections/${entry.id}/deployment/retry`))}>重试部署</button>}
              </div>}
            </div>;
          })}</div>
          <button className="add-provider" onClick={() => setPage("add")}><Plus size={17}/>添加模型</button>
          <p className="model-footnote">切换只影响后续任务。API 会向所配置的服务发送必要的文字与图片；本地模型在本机运行。</p>
        </> : page === "add" ? <><p className="model-description">选择一种接入方式。</p><div className="model-methods">
          <button onClick={() => openApi()}><Cloud size={22}/><span><strong>API 添加</strong><small>连接多模态模型服务，填写 URL、模型名与 Key</small></span><ChevronRight size={17}/></button>
          <button onClick={() => setPage("local")}><Cpu size={22}/><span><strong>本地部署</strong><small>选择通用指令或视觉模型，自动下载运行时与模型</small></span><ChevronRight size={17}/></button>
          <button onClick={() => setPage("hybrid")}><Layers3 size={22}/><span><strong>多模型组合</strong><small>部署 Endless、角色检索与 LaMa 的固定模型组合</small></span><ChevronRight size={17}/></button>
        </div></> : page === "api" ? <form className="model-form" onSubmit={event => { event.preventDefault(); void run(async () => {
          const payload = {name, base_url: url, model, vision_model: visionModel || model, api_key: key || null};
          if (edit) await api(`/model-connections/${edit.id}`, {method: "PATCH", body: JSON.stringify(payload)});
          else await post("/model-connections/api", payload);
          setKey(""); setPage("list");
        }); }}>
          <p className="model-description">支持 OpenAI Chat Completions 兼容接口，请选择支持图片输入的模型。</p>
          <label>显示名称<input required maxLength={80} value={name} onChange={event => setName(event.target.value)} placeholder="例如：我的 Qwen API"/></label>
          <label>API Base URL<input required type="url" value={url} onChange={event => setUrl(event.target.value)} placeholder="https://服务地址/v1"/></label>
          <label>模型名称<input required value={model} onChange={event => setModel(event.target.value)} placeholder="服务提供商的模型 ID"/></label>
          <label>视觉模型名称 <small>可选，默认与上面相同</small><input value={visionModel} onChange={event => setVisionModel(event.target.value)} placeholder="需要单独指定时填写"/></label>
          <label>API Key<input required={!edit} type="password" autoComplete="new-password" value={key} onChange={event => setKey(event.target.value)} placeholder={edit ? "留空保留现有 Key" : "输入 API Key"}/></label>
          <p className="model-footnote">密钥保存在本机，不会显示在会话或任务记录中。图像迁移与局部修复还需要本地组合中的图像工具。</p>
          <button className="primary" disabled={busy}>{busy ? "保存中…" : "保存配置"}</button>
        </form> : <div className="model-form">
          {page === "local" ? <><p className="model-description">自动安装独立的 Ollama 运行时，再下载所选模型。通用语言模型负责规划、摘要与文字回答，图片任务使用已有本地 Qwen3-VL。</p><label>选择模型<select value={custom ? "custom" : localModel} onChange={event => { setCustom(event.target.value === "custom"); if (event.target.value !== "custom") setLocalModel(event.target.value); }}>{catalog.local.map(row => <option key={row.model} value={row.model}>{row.name} · {row.size}</option>)}<option value="custom">其他 Ollama 本地模型</option></select></label>{custom && <label>模型标签<input value={localModel} onChange={event => setLocalModel(event.target.value)} placeholder="例如 qwen3:4b-instruct"/></label>}<p className="model-footnote">支持本地通用指令与视觉模型，部署时会验证能力。纯文字模型需要先部署多模型组合，以提供图片理解。运行时和模型需要额外磁盘空间，速度取决于硬件；迁移、修复仍使用本地图像工具。</p></> : <><p className="model-description">以下模型构成固定的研究组合，将按顺序下载并部署；已有权重会自动复用。</p><ol className="bundle-manifest">{catalog.bundle.map((name, index) => <li key={name}><span>{index + 1}</span>{name}<small>必需</small></li>)}</ol><p className="model-footnote">自动准备独立的 PyTorch CUDA 环境。需要 NVIDIA 显卡及数十 GB 磁盘空间。角色专属权重随角色包导入；当前修复使用 LaMa，PPO 策略尚未接入。</p></>}
          <button className="primary" disabled={busy || (page === "local" && !localModel.trim())} onClick={() => run(async () => { await post("/model-connections/deploy", {kind: page, model: localModel}); setPage("list"); })}>{busy ? "创建部署任务…" : "下载并部署"}</button><p className="model-footnote">关闭窗口后部署继续运行，可随时回到这里查看进度、取消或重试。</p>
        </div>}
      </section>
    </div>
  </div>;
}
