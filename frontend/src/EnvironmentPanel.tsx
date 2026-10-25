import { useEffect, useState } from "react";
import { Check, LoaderCircle, RefreshCw } from "lucide-react";
import { api, post } from "./api";
import type { Deployment } from "./api";

type CheckItem = { id: string; name: string; status: string; detail: string; fixable: boolean };
type Report = { checks: CheckItem[]; combination: CheckItem[]; repair: Deployment | null; checked_at: number };
const active = (job: Deployment | null) => !!job && ["queued", "running"].includes(job.status);
const label: Record<string, string> = {ready: "可用", warning: "提示", blocked: "需处理", repairable: "可自动配置", queued: "等待", running: "进行中", completed: "完成", failed: "失败", cancelled: "已取消", interrupted: "已中断"};

export default function EnvironmentPanel({ onError }: {onError: (error: unknown) => void}) {
  const [report, setReport] = useState<Report | null>(null);
  const [job, setJob] = useState<Deployment | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const refresh = async () => {
    setLoading(true);
    try {
      const found = await api<Report>("/environment");
      setReport(found); setJob(found.repair);
    } catch (error) { onError(error); }
    finally { setLoading(false); }
  };
  useEffect(() => { void refresh(); }, []);
  useEffect(() => {
    if (!active(job)) return;
    let live = true, pending = false;
    const timer = setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const current = await api<Deployment | null>("/environment/repair");
        if (live) {
          setJob(current);
          if (!active(current)) void refresh();
        }
      } catch (error) { if (live) onError(error); }
      finally { pending = false; }
    }, 1800);
    return () => { live = false; clearInterval(timer); };
  }, [job?.status]);
  const repair = async (target: "local" | "hybrid") => {
    setBusy(true);
    try { setJob(await post<Deployment>("/environment/repair", {target})); }
    catch (error) { onError(error); }
    finally { setBusy(false); }
  };
  const rows = (items: CheckItem[]) => <div className="environment-checks">{items.map(item => <div key={item.id} className={"environment-check " + item.status}>
    <div><strong>{item.name}</strong><small>{label[item.status] || item.status}</small></div><p>{item.detail}</p>
  </div>)}</div>;
  return <div className="environment-panel">
    <p className="model-description">启动时检查应用环境，模型部署前检查硬件并自动补齐对应依赖。环境保存在本项目中。</p>
    <button className="environment-refresh" disabled={loading || busy} onClick={() => void refresh()}><RefreshCw size={14} className={loading ? "spin" : ""}/>{loading ? "正在检查环境…" : "重新检查"}</button>
    {report && <>
      {rows(report.checks)}
      <h3>自动配置</h3>
      <p className="model-footnote">配置运行环境，不下载模型权重。选择模型后的部署流程也会自动执行这些步骤。</p>
      <div className="environment-actions"><button disabled={loading || busy || active(job) || report.checks.some(item => item.status === "blocked")} onClick={() => void repair("local")}>配置本地模型环境</button>
        <button disabled={loading || busy || active(job) || report.combination.some(item => item.status === "blocked")} onClick={() => void repair("hybrid")}>配置组合推理环境</button></div>
      <h3>研究组合硬件要求</h3>{rows(report.combination.filter(item => ["gpu", "vram"].includes(item.id)))}
      <p className="model-footnote">无需手动安装 CUDA Toolkit。显卡、驱动权限、磁盘空间或系统重启无法通过安装 Python 包解决，检查未通过时会明确说明原因。API 和本地 CPU 主控仍可单独使用。</p>
    </>}
    {job && <div className="deployment-progress" aria-live="polite"><div className="deployment-title"><strong>{active(job) && <LoaderCircle size={13} className="spin"/>}{job.phase}</strong><span>{label[job.status] || job.status}</span></div>
      <ol>{job.items.map((item, index) => <li key={index} className={item.status}><div><span>{item.status === "completed" ? <Check size={13}/> : index + 1}</span><strong>{item.name}</strong><small>{label[item.status] || item.status}</small></div>
        {item.status === "running" && <><progress max={item.total || undefined} value={item.total ? item.completed : undefined}/><p>{item.total ? `${(item.completed / 1024 ** 2).toFixed(1)} / ${(item.total / 1024 ** 2).toFixed(1)} MB` : "正在配置，请稍候"}{item.detail && ` · ${item.detail}`}</p></>}
      </li>)}</ol>
      {active(job) && <button onClick={() => void post("/environment/repair/cancel").catch(onError)}>取消配置</button>}
    </div>}
  </div>;
}
