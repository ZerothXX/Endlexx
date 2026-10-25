import { useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  BookOpen,
  Check,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Download,
  FileText,
  ImagePlus,
  Layers3,
  LoaderCircle,
  Menu,
  MessageSquare,
  PanelRight,
  Plus,
  ScanLine,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  Square,
  Trash2,
  WandSparkles,
  X,
} from "lucide-react";
import { activeTask, api, imageUrl, post } from "./api";
import ModelManager from "./ModelManager";
import type {
  Asset,
  Character,
  Chat,
  ChatContextStatus,
  Citation,
  Doc,
  Message,
  ModelRegistry,
  Profile,
  Task,
  ToolResult,
} from "./api";

type Drawer = "knowledge" | "profile" | null;
const statusLabel: Record<string, string> = {
  queued: "排队中",
  running: "处理中",
  completed: "已完成",
  failed: "未完成",
  cancelled: "已取消",
  needs_input: "需要补充信息",
};
const checkLabel: Record<string, string> = {
  match: "符合",
  mismatch: "存在差异",
  not_visible: "不可见",
  unknown: "无法判断",
};

function Brand({ small = false }: { small?: boolean }) {
  return (
    <span className={"brand-mark " + (small ? "small" : "")}>
      <span>E</span>
      <i />
    </span>
  );
}

function Compare({
  result,
  onUse,
}: {
  result: ToolResult;
  onUse: (id: string) => void;
}) {
  const [position, setPosition] = useState(50);
  const [compare, setCompare] = useState(true);
  if (!result.asset) return null;
  return (
    <div className="result-card">
      <div className="result-head">
        <span>
          <CheckCheck size={15} />
          {result.tool === "repair" ? "局部修复" : "角色迁移"}
        </span>
        <button onClick={() => setCompare(!compare)}>
          {compare ? "查看结果" : "前后对比"}
        </button>
      </div>
      <div className="compare-frame">
        <img src={result.asset.url} alt="处理后的图片" />
        {compare && result.before && (
          <>
            <img
              className="before-image"
              src={imageUrl(result.before)}
              alt="原图"
              style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
            />
            <div className="compare-line" style={{ left: `${position}%` }}>
              <span>‹ ›</span>
            </div>
            <span className="image-label before">原图</span>
            <span className="image-label after">结果</span>
            <input
              type="range"
              min="0"
              max="100"
              value={position}
              onChange={(e) => setPosition(+e.target.value)}
              aria-label="拖动对比原图与结果"
            />
          </>
        )}
      </div>
      <div className="result-foot">
        <span>
          {result.metrics ? "保护区保持原样 · LaMa" : "Endless 角色流水线"} ·{" "}
          {result.asset.width} × {result.asset.height}
        </span>
        <a
          href={result.asset.url}
          download={result.asset.filename}
          title="下载结果"
        >
          <Download size={16} />
        </a>
      </div>
      <div className="result-actions">
        {result.before && (
          <button onClick={() => onUse(result.before!)}>使用原图继续</button>
        )}
        <button onClick={() => onUse(result.asset!.id)}>使用结果继续</button>
      </div>
    </div>
  );
}

function Citations({ items }: { items: Citation[] }) {
  return (
    <div className="citations">
      <div className="section-caption">
        <BookOpen size={13} />
        参考资料 · {items.length}
      </div>
      {items.map((item) => (
        <details key={item.number}>
          <summary>
            <span className="citation-number">{item.number}</span>
            <span>{item.title}</span>
            <small>{item.version}</small>
            <ChevronDown size={13} />
          </summary>
          <div className="citation-content">
            <p>{item.text}</p>
            <span>
              来源：{item.source} · 段落 {item.ordinal + 1}
            </span>
          </div>
        </details>
      ))}
    </div>
  );
}

function TaskCard({
  task,
  onCancel,
  onRetry,
  onDetails,
}: {
  task: Task;
  onCancel: () => void;
  onRetry: () => void;
  onDetails: () => void;
}) {
  return (
    <div className={"task-card " + task.status}>
      <div className="task-top">
        <span>
          {activeTask(task) ? (
            <LoaderCircle size={16} className="spin" />
          ) : task.status === "completed" ? (
            <Check size={16} />
          ) : (
            <CircleHelp size={16} />
          )}
          <strong>{statusLabel[task.status] || task.status}</strong>
        </span>
        <button onClick={onDetails}>
          任务详情 <ChevronRight size={12} />
        </button>
      </div>
      <p>{task.phase}</p>
      {activeTask(task) && (
        <div className="progress-track">
          <div style={{ width: Math.max(4, task.progress) + "%" }} />
        </div>
      )}
      <div className="task-bottom">
        <span>{task.snapshot.controller}</span>
        {activeTask(task) ? (
          <button onClick={onCancel}>
            <Square size={11} />
            取消
          </button>
        ) : (
          ["failed", "cancelled"].includes(task.status) && (
            <button onClick={onRetry}>重试</button>
          )
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [character, setCharacter] = useState<Character | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [chat, setChat] = useState<Chat | null>(null);
  const [contextStatus, setContextStatus] = useState<ChatContextStatus | null>(null);
  const [creatingChat, setCreatingChat] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Chat | null>(null);
  const [deletingChat, setDeletingChat] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [models, setModels] = useState<ModelRegistry>({ active_id: null, entries: [] });
  const [modelManager, setModelManager] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [input, setInput] = useState("");
  const [attachment, setAttachment] = useState<Asset | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [drawer, setDrawer] = useState<Drawer>(null);
  const [toast, setToast] = useState("");
  const [newRole, setNewRole] = useState(false);
  const [roleName, setRoleName] = useState("");
  const [taskDetail, setTaskDetail] = useState<Task | null>(null);
  const [sidebar, setSidebar] = useState(false);
  const [login, setLogin] = useState(false);
  const [token, setToken] = useState("");
  const [lightbox, setLightbox] = useState<string | null>(null);
  const roleSelection = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const chatIdRef = useRef<string | null>(null);
  const requestKey = useRef<string | null>(null);
  const creatingChatRef = useRef(false);

  const notify = (message: string) => setToast(message);
  const fail = (error: unknown) =>
    notify(error instanceof Error ? error.message : String(error));
  const running = tasks.find(activeTask);
  const reloadCharacters = () =>
    api<Character[]>("/characters").then(setCharacters);
  const reloadModels = async () => {
    const value = await api<ModelRegistry>("/model-connections");
    setModels(value);
    setSelectedModelId(current => value.entries.some(row => row.id === current && row.status === "ready") ? current : value.active_id || "");
  };
  const selectedModel = models.entries.find(row => row.id === selectedModelId && row.status === "ready");
  useEffect(() => {
    Promise.all([reloadCharacters(), reloadModels()]).catch((e) => {
      setLogin(true);
      fail(e);
    });
  }, []);
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(""), 5000);
      return () => clearTimeout(timer);
    }
  }, [toast]);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, tasks.map((t) => t.phase).join("")]);

  async function selectRole(role: Character | null) {
    const selection = ++roleSelection.current;
    setCharacter(role);
    setChat(null);
    setContextStatus(null);
    setMessages([]);
    setTasks([]);
    setAttachment(null);
    setSidebar(false);
    setDrawer(null);
    setInput("");
    setChats([]);
    chatIdRef.current = null;
    requestKey.current = null;
    const id = role?.id || null;
    if (!id) return;
    try {
      const found = await api<Chat[]>(
        "/chats" + (id ? "?character_id=" + id : ""),
      );
      if (selection !== roleSelection.current) return;
      setChats(found);
      if (found.length) await selectChat(found[0]);
    } catch (e) {
      fail(e);
    }
  }
  async function selectChat(item: Chat) {
    chatIdRef.current = item.id;
    requestKey.current = null;
    setChat(item);
    setMessages([]);
    setTasks([]);
    setContextStatus(null);
    setSelectedModelId(models.entries.some(row => row.id === item.model_profile && row.status === "ready") ? item.model_profile : models.active_id || "");
    setAttachment(null);
    setInput("");
    setSidebar(false);
    try {
      const [items, taskList, context] = await Promise.all([
        api<Message[]>(`/chats/${item.id}/messages`),
        api<Task[]>(`/chats/${item.id}/tasks`),
        api<ChatContextStatus>(`/chats/${item.id}/context`),
      ]);
      if (chatIdRef.current === item.id) {
        setMessages(items);
        setTasks(taskList);
        setContextStatus(context);
      }
    } catch (e) {
      fail(e);
    }
  }
  async function newChat() {
    if (!character) { setNewRole(true); return; }
    if (creatingChatRef.current) return;
    creatingChatRef.current = true;
    setCreatingChat(true);
    const selection = roleSelection.current;
    try {
      const created = await post<Chat>("/chats", {
        character_id: character?.id || null,
        model_profile: selectedModel?.id || "local",
      });
      if (selection !== roleSelection.current) return;
      setChats((list) => [created, ...list]);
      await selectChat(created);
    } catch (e) {
      fail(e);
    } finally {
      creatingChatRef.current = false;
      setCreatingChat(false);
    }
  }
  async function deleteChat(item: Chat) {
    if (deletingChat) return;
    setDeletingChat(true);
    const selection = roleSelection.current;
    try {
      await api(`/chats/${item.id}`, { method: "DELETE" });
      setDeleteTarget(null);
      if (selection !== roleSelection.current) return;
      const remaining = chats.filter(row => row.id !== item.id);
      setChats(remaining);
      if (chatIdRef.current === item.id) {
        if (remaining.length) await selectChat(remaining[0]);
        else {
          chatIdRef.current = null;
          requestKey.current = null;
          setChat(null);
          setMessages([]);
          setTasks([]);
          setContextStatus(null);
          setInput("");
          setAttachment(null);
        }
      }
      notify("对话已删除");
    } catch (e) {
      fail(e);
    } finally {
      setDeletingChat(false);
    }
  }
  async function refreshCurrent(id: string) {
    const [items, taskList, context] = await Promise.all([
      api<Message[]>(`/chats/${id}/messages`),
      api<Task[]>(`/chats/${id}/tasks`),
      api<ChatContextStatus>(`/chats/${id}/context`),
    ]);
    if (chatIdRef.current === id) {
      setMessages(items);
      setTasks(taskList);
      setContextStatus(context);
    }
  }
  useEffect(() => {
    if (!running || !chat) return;
    const id = chat.id,
      taskId = running.id;
    const source = new EventSource(`/api/tasks/${taskId}/events`);
    source.addEventListener("task", (event) => {
      const value = JSON.parse((event as MessageEvent).data) as Task;
      if (chatIdRef.current !== id) return;
      setTasks((list) => list.map((t) => (t.id === value.id ? value : t)));
      if (!activeTask(value)) {
        source.close();
        refreshCurrent(id).catch(fail);
        reloadModels().catch(() => {});
      }
    });
    // Polling also repairs missed terminal events after a browser/network reconnect.
    const timer = setInterval(() => refreshCurrent(id).catch(() => {}), 6000);
    return () => {
      source.close();
      clearInterval(timer);
    };
  }, [running?.id, chat?.id]);

  async function send(text = input) {
    if (!text.trim() || busy || running || uploading || creatingChat) return;
    if (!character) return;
    if (!selectedModel) { setModelManager(true); return; }
    const selection = roleSelection.current;
    const initialChatId = chatIdRef.current;
    setBusy(true);
    try {
      let target = chat;
      if (!target) {
        target = await post<Chat>("/chats", {
          character_id: character?.id || null,
          model_profile: selectedModelId,
        });
        if (selection !== roleSelection.current || chatIdRef.current !== initialChatId) return;
        setChat(target);
        chatIdRef.current = target.id;
        setChats((list) => [target!, ...list]);
      }
      if (target.model_profile !== selectedModelId) {
        target = await api<Chat>(`/chats/${target.id}`, {method: "PATCH", body: JSON.stringify({model_profile: selectedModelId})});
        if (selection !== roleSelection.current || chatIdRef.current !== target.id) return;
        setChat(target);
      }
      const key = requestKey.current || crypto.randomUUID();
      requestKey.current = key;
      const task = await post<Task>(`/chats/${target.id}/send`, {
        content: text.trim(),
        assets: attachment ? [attachment.id] : [],
        idempotency_key: key,
      });
      requestKey.current = null;
      if (selection !== roleSelection.current || chatIdRef.current !== target.id) return;
      setInput("");
      setAttachment(null);
      setTasks((list) => [task, ...list.filter((t) => t.id !== task.id)]);
      await refreshCurrent(target.id);
      const found = await api<Chat[]>(
        "/chats" + (character ? "?character_id=" + character.id : ""),
      );
      if (selection !== roleSelection.current || chatIdRef.current !== target.id) return;
      setChats(found);
      setChat(found.find((t) => t.id === target!.id) || target);
    } catch (e) {
      requestKey.current = null;
      fail(e);
    } finally {
      setBusy(false);
    }
  }
  async function upload(file?: File) {
    if (!file || !character) return;
    setUploading(true);
    try {
      const data = new FormData();
      data.append("file", file);
      setAttachment(
        await api<Asset>("/assets", { method: "POST", body: data }),
      );
    } catch (e) {
      fail(e);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }
  async function changeMode(mode: string) {
    await post(`/model-connections/${mode}/activate`);
    if (chat) {
        const changed = await api<Chat>(`/chats/${chat.id}`, {
          method: "PATCH",
          body: JSON.stringify({ model_profile: mode }),
        });
        setChat(changed);
    }
    setSelectedModelId(mode);
    await reloadModels();
  }
  const shortcuts = [
    {
      icon: WandSparkles,
      title: "角色迁移",
      sub: "让视觉特征融入新画面",
      text: "将这张图片的主体迁移为当前角色的视觉特征。",
    },
    {
      icon: BookOpen,
      title: "角色知识",
      sub: "从设定与文案中寻找依据",
      text: "请根据知识库介绍这个角色的外观与服装设定。",
    },
    {
      icon: ScanLine,
      title: "自动修复",
      sub: "描述需求，自动定位并补全",
      text: "修复这张图片中的局部缺损。",
    },
  ];
  return (
    <div className="app-shell">
      <aside className={"sidebar " + (sidebar ? "open" : "")}>
        <button className="brand" onClick={() => selectRole(null)}>
          <Brand />
          <span>
            Endlexx<small>角色创作工作台</small>
          </span>
          <span className="brand-beta">BETA</span>
        </button>
        <button className="new-chat" onClick={newChat} disabled={creatingChat || busy}>
          <Plus size={17} />
          新建会话
        </button>
        <div className="nav-caption">工作空间</div>
        <button
          className={"workspace-link " + (!character ? "selected" : "")}
          onClick={() => selectRole(null)}
        >
          <ScanLine size={17} />
          <span>工作台</span>
          <ChevronRight size={13} />
        </button>
        <div className="nav-caption role-caption">
          <span>
            我的角色 <i>{characters.length}</i>
          </span>
          <button title="添加角色" onClick={() => setNewRole(true)}>
            <Plus size={16} />
          </button>
        </div>
        <div className="roles">
          {characters.map((role) => (
            <div key={role.id} className="role-group">
              <button
                className={
                  "role-link " + (character?.id === role.id ? "selected" : "")
                }
                onClick={() => selectRole(role)}
              >
                {role.avatar_id ? (
                  <img src={imageUrl(role.avatar_id)} alt="" />
                ) : (
                  <span className="avatar-fallback">
                    {role.name.slice(0, 1).toUpperCase()}
                  </span>
                )}
                <span>
                  {role.name}
                  <small>
                    {role.package ? "已关联视觉角色包" : "知识与创作角色"}
                  </small>
                </span>
                {character?.id === role.id && <span className="selected-dot" />}
              </button>
              {character?.id === role.id && (
                <div className="chat-links">
                  <button className="role-new-chat" onClick={newChat} disabled={creatingChat || busy}>
                    {creatingChat ? <LoaderCircle size={13} className="spin" /> : <Plus size={13} />}
                    <span>新建对话</span>
                  </button>
                  {chats.map((item) => (
                    <div key={item.id} className={"chat-row " + (chat?.id === item.id ? "current" : "")}>
                      <button className="chat-select" onClick={() => selectChat(item)} title={item.title}>
                        <MessageSquare size={13} />
                        <span>{item.title}</span>
                      </button>
                      <button className="chat-delete" title="删除对话" aria-label={`删除对话：${item.title}`}
                        disabled={busy || (chat?.id === item.id && !!running)}
                        onClick={() => setDeleteTarget(item)}>
                        <Trash2 size={13} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          {!characters.length && (
            <div className="sidebar-empty">
              添加一个角色，或导入
              <br />
              已有的 Endless 角色包。
            </div>
          )}
        </div>
        <div className="sidebar-bottom">
          <button
            onClick={async () => {
              try {
                const values = await post<Character[]>("/endless/import");
                await reloadCharacters();
                notify(
                  values.length
                    ? `已导入 ${values.length} 个角色，档案请核实后使用`
                    : "已有角色均已导入，或尚无已发布的角色包",
                );
              } catch (e) {
                fail(e);
              }
            }}
          >
            <Layers3 size={16} />
            导入 Endless 角色
          </button>
          <button
            onClick={() => { setModelManager(true); setSidebar(false); }}
          >
            <Settings2 size={16} />
            模型
            <span
              className={
                "connection-dot " + (selectedModel ? "online" : "")
              }
            />
          </button>
          <div className="local-note">
            <ShieldCheck size={12} />
            {selectedModel?.kind === "api"
              ? "含外部 API · 文字或图片按分工发送"
              : "本地工作空间 · 素材不外发"}
          </div>
        </div>
      </aside>
      {sidebar && (
        <button
          className="sidebar-backdrop"
          aria-label="关闭导航"
          onClick={() => setSidebar(false)}
        />
      )}
      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-title">
            <button
              className="mobile-menu icon-button"
              aria-label="展开导航"
              onClick={() => setSidebar(!sidebar)}
            >
              <Menu size={20} />
            </button>
            <span className="workspace-dot" />
            {character?.name || "工作台"}
            <span className="title-slash">/</span>
            <span className="chat-title">{chat?.title || "新的创作"}</span>
          </div>
          <div className="topbar-actions">
            <button
              className={
                "icon-button " + (drawer === "knowledge" ? "active" : "")
              }
              title="角色知识库"
              aria-label="角色知识库"
              disabled={!character}
              onClick={() =>
                setDrawer(drawer === "knowledge" ? null : "knowledge")
              }
            >
              <BookOpen size={18} />
            </button>
            <button
              className={
                "icon-button " + (drawer === "profile" ? "active" : "")
              }
              title="角色档案"
              aria-label="角色档案"
              disabled={!character}
              onClick={() => setDrawer(drawer === "profile" ? null : "profile")}
            >
              <PanelRight size={18} />
            </button>
          </div>
        </header>
        <section
          className="conversation"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            upload(e.dataTransfer.files[0]);
          }}
        >
          {!messages.length ? (
            <div className="welcome">
              <div className="welcome-emblem">
                <Brand />
              </div>
              <div className="eyebrow">YOUR CHARACTER, EVERY DETAIL.</div>
              <h1>
                {character ? (
                  <>
                    让 {character.name} 的设定
                    <br />
                    落在每一张图里。
                  </>
                ) : (
                  <>
                    从一张图，
                    <br />
                    开始下一次创作。
                  </>
                )}
              </h1>
              <p>
                {character
                  ? "连接角色知识、视觉迁移与图像修复。用自然语言，让想法成为作品。"
                  : "选择或创建一个角色，在角色会话中连接知识、迁移视觉特征、修复图片。"}
              </p>
              <div className="shortcut-grid">
                {shortcuts.map(({ icon: Icon, title, sub }, index) => (
                  <div className="feature-card"
                    key={title}
                  >
                    <span className={"shortcut-icon tone-" + index}>
                      <Icon size={20} />
                    </span>
                    <strong>
                      {title}
                    </strong>
                    <small>{sub}</small>
                  </div>
                ))}
              </div>
              <div className="welcome-foot">
                <span />
                <span>为角色创作而设计</span>
                <span />
              </div>
            </div>
          ) : (
            <div className="message-list">
              {!!contextStatus?.compressed_messages && (
                <div className="context-notice" role="status">
                  <Layers3 size={13} />
                  较早的 {contextStatus.compressed_messages} 条消息已压缩为摘要，完整记录仍保留。
                </div>
              )}
              {messages.map((message) => (
                <article key={message.id} className={"message " + message.role}>
                  {message.role === "assistant" && (
                    <div className="assistant-avatar">
                      <Brand small />
                    </div>
                  )}
                  <div className="message-body">
                    {message.role === "assistant" && (
                      <div className="assistant-name">
                        Endlexx <span>创作助手</span>
                      </div>
                    )}
                    {!!message.assets.length &&
                      !message.details.results?.length && (
                        <div className="message-images">
                          {message.assets.map((id) => (
                            <button
                              key={id}
                              onClick={() => setLightbox(imageUrl(id))}
                            >
                              <img src={imageUrl(id)} alt="会话图片" />
                            </button>
                          ))}
                        </div>
                      )}
                    <div className="message-text">{message.content}</div>
                    {message.details.results?.map((result, i) =>
                      result.tool === "check" ? (
                        <div className="check-report" key={i}>
                          <h4>
                            <ShieldCheck size={16} />
                            角色一致性检查
                          </h4>
                          <p>{result.report?.summary}</p>
                          {result.report?.items.map((item, j) => (
                            <div className="check-item" key={j}>
                              <div>
                                <strong>{item.feature}</strong>
                                <span className={"check-status " + item.status}>
                                  {checkLabel[item.status] || "无法判断"}
                                </span>
                              </div>
                              <p>{item.observation}</p>
                              <small>设定依据：{item.expected}</small>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <Compare
                          key={i}
                          result={result}
                          onUse={(id) => {
                            api<Asset>(`/assets/${id}`)
                              .then((asset) => {
                                setAttachment(asset);
                                inputRef.current?.focus();
                              })
                              .catch(fail);
                          }}
                        />
                      ),
                    )}
                    {!!message.details.citations?.length && (
                      <Citations items={message.details.citations} />
                    )}{" "}
                    {!!message.details.retrieval?.warnings?.length && (
                      <div className="retrieval-note">
                        {message.details.retrieval.warnings.join("；")}
                      </div>
                    )}
                  </div>
                </article>
              ))}
              {tasks
                .filter(
                  (task) =>
                    activeTask(task) ||
                    ["failed", "cancelled"].includes(task.status),
                )
                .slice(0, 3)
                .map((task) => (
                  <div className="task-wrapper" key={task.id}>
                    <Brand small />
                    <TaskCard
                      task={task}
                      onCancel={() =>
                        post(`/tasks/${task.id}/cancel`)
                          .then(() => chat && refreshCurrent(chat.id))
                          .catch(fail)
                      }
                      onRetry={() =>
                        post(`/tasks/${task.id}/retry`)
                          .then(() => chat && refreshCurrent(chat.id))
                          .catch(fail)
                      }
                      onDetails={() =>
                        api<Task>(`/tasks/${task.id}`)
                          .then(setTaskDetail)
                          .catch(fail)
                      }
                    />
                  </div>
                ))}
              <div ref={bottomRef} />
            </div>
          )}
        </section>
        {character ? <div className="composer-area">
          <div
            className="composer"
            onPaste={(e) => {
              const file = Array.from(e.clipboardData.files).find((f) =>
                f.type.startsWith("image/"),
              );
              if (file) {
                e.preventDefault();
                upload(file);
              }
            }}
          >
            {attachment && (
              <div className="attachment-preview">
                <img src={attachment.url} alt="待发送图片" />
                <div>
                  <strong>{attachment.filename}</strong>
                  <small>
                    {attachment.width} × {attachment.height}
                  </small>
                </div>
                <button
                  className="icon-button"
                  onClick={() => setAttachment(null)}
                  aria-label="移除图片"
                >
                  <X size={14} />
                </button>
              </div>
            )}
            <textarea
              ref={inputRef}
              aria-label="输入创作需求"
              placeholder={
                character
                  ? "描述你的创作需求，或问问这个角色的设定…"
                  : "上传一张图片，告诉我你想修复什么…"
              }
              rows={2}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            <div className="composer-tools">
              <button
                className="icon-button upload-button"
                title="添加图片"
                aria-label="添加图片"
                disabled={uploading}
                onClick={() => fileRef.current?.click()}
              >
                {uploading ? (
                  <LoaderCircle className="spin" size={19} />
                ) : (
                  <Plus size={21} />
                )}
              </button>
              <span className="composer-divider" />
              <div className="composer-model"><select aria-label="选择聊天模型" value={selectedModelId} disabled={busy || !!running} onChange={event => changeMode(event.target.value).catch(fail)}>
                {!selectedModel && <option value="">请添加或选择模型</option>}
                {models.entries.map(entry => <option key={entry.id} value={entry.id} disabled={entry.status !== "ready"}>{entry.name}{entry.status !== "ready" ? " · 尚未就绪" : ""}</option>)}
              </select><button title="管理模型" aria-label="管理模型" onClick={() => setModelManager(true)}><Settings2 size={14}/></button></div>
              <span className="composer-spacer" />
              <button
                className="send-button"
                title="发送"
                aria-label="发送"
                disabled={!input.trim() || busy || !!running || uploading || !selectedModel}
                onClick={() => send()}
              >
                {busy ? (
                  <LoaderCircle className="spin" size={18} />
                ) : (
                  <ArrowUp size={19} />
                )}
              </button>
            </div>
          </div>
          <div className="composer-note">
            {running
              ? "任务正在处理，可在任务卡中取消。"
              : "Enter 发送 · Shift + Enter 换行"}
            <span>结果保留原图，可随时对比</span>
          </div>
        </div> : <div className="home-invitation">从左侧选择角色开始创作 · 模型可在侧栏底部添加</div>}
      </main>
      {modelManager && <ModelManager registry={models} onRefresh={reloadModels} onSelect={changeMode} onClose={() => setModelManager(false)} onError={fail}/>}
      {drawer && (
        <aside className="detail-drawer">
          <div className="drawer-title">
            <div>
              <small>WORKSPACE</small>
              <h2>
                {drawer === "knowledge"
                  ? "角色知识库"
                  : drawer === "profile"
                    ? "角色档案"
                    : "模型与运行状态"}
              </h2>
            </div>
            <button
              className="icon-button"
              aria-label="关闭侧栏"
              onClick={() => setDrawer(null)}
            >
              <X size={18} />
            </button>
          </div>
          {!character ? (
            <div className="drawer-empty">
              <BookOpen size={30} />
              <h3>先选择一个角色</h3>
              <p>角色档案与知识在同一角色的多个会话间共享。</p>
              <button className="primary" onClick={() => setNewRole(true)}>
                创建角色
              </button>
            </div>
          ) : drawer === "knowledge" ? (
            <KnowledgePanel
              key={character.id}
              character={character}
              fail={fail}
              notify={notify}
            />
          ) : (
            <ProfilePanel
              key={character.id}
              character={character}
              fail={fail}
              onSaved={async () => {
                const values = await api<Character[]>("/characters");
                setCharacters(values);
                setCharacter(
                  values.find((c) => c.id === character.id) || character,
                );
                notify("角色设定版本已保存");
              }}
            />
          )}
        </aside>
      )}
      <input
        type="file"
        ref={fileRef}
        hidden
        accept="image/png,image/jpeg,image/webp"
        onChange={(e) => upload(e.target.files?.[0])}
      />
      {deleteTarget && (
        <div className="modal-backdrop" onClick={() => !deletingChat && setDeleteTarget(null)}>
          <div className="modal" role="dialog" aria-modal="true" aria-labelledby="delete-chat-title"
            onClick={e => e.stopPropagation()} onKeyDown={e => {
              if (e.key === "Escape" && !deletingChat) setDeleteTarget(null);
            }}>
            <div className="modal-heading"><h2 id="delete-chat-title">删除对话</h2></div>
            <p>确定删除“{deleteTarget.title}”吗？该对话的消息、任务记录与上下文摘要将被删除，无法恢复。角色资料、知识库和图片仍保留。</p>
            <div className="modal-actions">
              <button className="secondary" autoFocus disabled={deletingChat} onClick={() => setDeleteTarget(null)}>取消</button>
              <button className="primary danger" disabled={deletingChat} onClick={() => deleteChat(deleteTarget)}>
                {deletingChat ? "正在删除…" : "确认删除"}
              </button>
            </div>
          </div>
        </div>
      )}
      {newRole && (
        <div className="modal-backdrop" onClick={() => setNewRole(false)}>
          <form
            className="modal"
            onClick={(e) => e.stopPropagation()}
            onSubmit={async (e) => {
              e.preventDefault();
              try {
                const value = await post<Character>("/characters", {
                  name: roleName,
                });
                await reloadCharacters();
                setNewRole(false);
                setRoleName("");
                await selectRole(value);
                setDrawer("profile");
              } catch (error) {
                fail(error);
              }
            }}
          >
            <div className="modal-heading">
              <h2>创建角色</h2>
              <button
                type="button"
                className="icon-button"
                aria-label="关闭"
                onClick={() => setNewRole(false)}
              >
                <X size={18} />
              </button>
            </div>
            <p>为角色的知识、参考图与创作会话建立一个空间。</p>
            <label>
              角色名称
              <input
                autoFocus
                maxLength={80}
                required
                value={roleName}
                onChange={(e) => setRoleName(e.target.value)}
                placeholder="例如：原创角色 · 星遥"
              />
            </label>
            <div className="modal-actions">
              <button
                type="button"
                className="secondary"
                onClick={() => setNewRole(false)}
              >
                取消
              </button>
              <button className="primary" type="submit">
                创建角色
              </button>
            </div>
          </form>
        </div>
      )}
      {taskDetail && (
        <div className="modal-backdrop" onClick={() => setTaskDetail(null)}>
          <div
            className="modal task-detail"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-heading">
              <h2>任务详情</h2>
              <button
                className="icon-button"
                onClick={() => setTaskDetail(null)}
                aria-label="关闭"
              >
                <X size={18} />
              </button>
            </div>
            <p>{taskDetail.phase}</p>
            <div className="trace-id">{taskDetail.id}</div>
            {taskDetail.traces?.map((trace, i) => (
              <div className="trace-row" key={i}>
                <span>{i + 1}</span>
                <strong>{trace.node}</strong>
                <small>
                  {typeof trace.data.duration_ms === "number"
                    ? `${trace.data.duration_ms} ms`
                    : String(trace.data.phase || "已记录")}
                </small>
              </div>
            ))}
          </div>
        </div>
      )}
      {login && (
        <div className="modal-backdrop">
          <form
            className="modal"
            onSubmit={async (e) => {
              e.preventDefault();
              try {
                await post("/auth", { token });
                await Promise.all([reloadCharacters(), reloadModels()]);
                setLogin(false);
              } catch (error) {
                fail(error);
              }
            }}
          >
            <h2>连接 Endlexx</h2>
            <p>
              如果启用了访问保护，请输入工作台访问令牌。若服务未启动，请先运行启动脚本。
            </p>
            <input
              type="password"
              aria-label="访问令牌"
              value={token}
              onChange={(e) => setToken(e.target.value)}
            />
            <button className="primary full">连接工作台</button>
          </form>
        </div>
      )}
      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)}>
          <button className="icon-button" aria-label="关闭预览">
            <X size={24} />
          </button>
          <img src={lightbox} alt="图片预览" />
        </div>
      )}
      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}

function KnowledgePanel({
  character,
  fail,
  notify,
}: {
  character: Character;
  fail: (e: unknown) => void;
  notify: (text: string) => void;
}) {
  const [docs, setDocs] = useState<Doc[]>([]),
    [editing, setEditing] = useState(false),
    [saving, setSaving] = useState(false);
  const [title, setTitle] = useState(""),
    [content, setContent] = useState(""),
    [source, setSource] = useState("用户上传"),
    [category, setCategory] = useState("user"),
    [version, setVersion] = useState("默认"),
    [query, setQuery] = useState(""),
    [results, setResults] = useState<Citation[] | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const reload = () =>
    api<Doc[]>(`/characters/${character.id}/documents`).then(setDocs);
  useEffect(() => {
    reload().catch(fail);
  }, [character.id]);
  async function save() {
    setSaving(true);
    try {
      const result = await post<{ warning?: string }>(
        `/characters/${character.id}/documents`,
        { title, content, source, category, version },
      );
      notify(result.warning || "资料已入库并建立检索索引");
      setEditing(false);
      setTitle("");
      setContent("");
      await reload();
    } catch (e) {
      fail(e);
    } finally {
      setSaving(false);
    }
  }
  return (
    <div className="drawer-content">
      <p className="drawer-description">
        资料只在「{character.name}
        」范围检索。回答附引用，聊天内容不会自动成为角色事实。
      </p>
      {editing ? (
        <div className="editor-form">
          <label>
            资料标题
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              maxLength={160}
            />
          </label>
          <div className="field-row">
            <label>
              资料类型
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
              >
                <option value="user">用户补充</option>
                <option value="official">原始设定</option>
                <option value="dialogue">台词 / 对话</option>
                <option value="creative">创作文案</option>
              </select>
            </label>
            <label>
              版本
              <input
                value={version}
                onChange={(e) => setVersion(e.target.value)}
              />
            </label>
          </div>
          <label>
            来源
            <input
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder="作品名、章节或来源链接"
            />
          </label>
          <label>
            正文
            <textarea
              rows={11}
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="粘贴角色设定、台词或文案…"
            />
          </label>
          <div className="field-row">
            <button className="secondary" onClick={() => setEditing(false)}>
              取消
            </button>
            <button
              className="primary"
              disabled={!title || !content || saving}
              onClick={save}
            >
              {saving ? (
                <LoaderCircle size={15} className="spin" />
              ) : (
                <Check size={15} />
              )}
              保存并索引
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="field-row">
            <button className="primary" onClick={() => setEditing(true)}>
              <Plus size={15} />
              添加资料
            </button>
            <button
              className="secondary"
              onClick={() => fileRef.current?.click()}
            >
              <FileText size={15} />
              导入文件
            </button>
          </div>
          <div className="search-field">
            <Search size={15} />
            <input
              placeholder="检索角色资料…"
              aria-label="检索角色资料"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && query)
                  post<{ citations: Citation[] }>(
                    `/characters/${character.id}/search`,
                    { query },
                  )
                    .then((data) => setResults(data.citations))
                    .catch(fail);
              }}
            />
            <button
              onClick={() =>
                query &&
                post<{ citations: Citation[] }>(
                  `/characters/${character.id}/search`,
                  { query },
                )
                  .then((data) => setResults(data.citations))
                  .catch(fail)
              }
            >
              <ArrowUp size={15} />
            </button>
          </div>
          {results && (
            <div className="search-results">
              <div className="section-caption">
                检索结果
                <button onClick={() => setResults(null)}>
                  <X size={13} />
                </button>
              </div>
              {results.length ? (
                <Citations items={results} />
              ) : (
                <p className="muted">没有找到足够相关的资料。</p>
              )}
            </div>
          )}
          <div className="section-caption">
            全部资料 <span>{docs.length}</span>
          </div>
          {!docs.length && (
            <div className="empty-inline">
              <BookOpen size={25} />
              <p>还没有角色资料</p>
              <small>
                添加设定、台词或文案，
                <br />
                让每次回答都有据可查。
              </small>
            </div>
          )}
          {docs.map((doc) => (
            <details className="document-card" key={doc.id}>
              <summary>
                <FileText size={18} />
                <span>
                  <strong>{doc.title}</strong>
                  <small>
                    {doc.version} ·{" "}
                    {doc.vector_indexed ? "混合检索已就绪" : "关键词检索"}
                  </small>
                </span>
                <ChevronDown size={14} />
              </summary>
              <div className="document-content">
                <p>{doc.content}</p>
                <small>来源：{doc.source}</small>
                <div className="document-actions">
                  <button
                    onClick={() =>
                      post(`/documents/${doc.id}/reindex`)
                        .then(() => {
                          notify("已重新建立索引");
                          reload();
                        })
                        .catch(fail)
                    }
                  >
                    重新索引
                  </button>
                  <button
                    title="删除资料"
                    onClick={() => {
                      if (window.confirm(`删除「${doc.title}」及其检索索引？`))
                        api(`/documents/${doc.id}`, { method: "DELETE" })
                          .then(reload)
                          .catch(fail);
                    }}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            </details>
          ))}
        </>
      )}
      <input
        ref={fileRef}
        type="file"
        hidden
        accept=".txt,.md,.json"
        onChange={async (e) => {
          const file = e.target.files?.[0];
          if (file) {
            if (file.size > 1024 * 1024) {
              fail(new Error("资料最大支持 1MB"));
              return;
            }
            setTitle(file.name);
            setContent(await file.text());
            setEditing(true);
          }
        }}
      />
    </div>
  );
}

function ProfilePanel({
  character,
  fail,
  onSaved,
}: {
  character: Character;
  fail: (e: unknown) => void;
  onSaved: () => Promise<void>;
}) {
  const [profiles, setProfiles] = useState<Profile[]>([]),
    [label, setLabel] = useState("默认设定"),
    [description, setDescription] = useState(""),
    [source, setSource] = useState("用户设定"),
    [references, setReferences] = useState<string[]>([]),
    [saving, setSaving] = useState(false),
    [confirmed, setConfirmed] = useState(true),
    [editing, setEditing] = useState(false);
  const referenceInput = useRef<HTMLInputElement>(null);
  const [packages, setPackages] = useState<{ name: string; ready: boolean }[]>(
    [],
  );
  const load = () =>
    api<Profile[]>(`/characters/${character.id}/profiles`).then(setProfiles);
  useEffect(() => {
    load().catch(fail);
    api<{ name: string; ready: boolean }[]>("/endless/packages")
      .then(setPackages)
      .catch(fail);
  }, [character.id]);
  const current = profiles.find((p) => p.id === character.active_version);
  const beginEdit = () => {
    setLabel(current?.label || "默认设定");
    setDescription(
      current
        ? String(
            current.constraints["设定说明"] ||
              JSON.stringify(current.constraints, null, 2),
          )
        : "",
    );
    setReferences(current?.references || []);
    setSource(current?.provenance || "用户设定");
    setConfirmed(current?.confirmed ?? true);
    setEditing(true);
  };
  async function save() {
    setSaving(true);
    try {
      await post(`/characters/${character.id}/profiles`, {
        label,
        constraints: { 设定说明: description },
        references,
        provenance: source,
        confirmed,
      });
      await load();
      await onSaved();
      setEditing(false);
    } catch (e) {
      fail(e);
    } finally {
      setSaving(false);
    }
  }
  return (
    <div className="drawer-content">
      <div className="profile-heading">
        {character.avatar_id ? (
          <img src={imageUrl(character.avatar_id)} alt="角色参考" />
        ) : (
          <div className="profile-placeholder">{character.name[0]}</div>
        )}
        <h3>{character.name}</h3>
        <p>
          {character.package
            ? "视觉角色包：" + character.package
            : "尚未关联视觉角色包"}
        </p>
      </div>
      <div className="editor-form package-selector">
        <label>
          关联视觉角色包
          <select
            value={character.package || ""}
            onChange={async (e) => {
              try {
                await api(`/characters/${character.id}`, {
                  method: "PATCH",
                  body: JSON.stringify({
                    name: character.name,
                    description: character.description,
                    package: e.target.value || null,
                  }),
                });
                await onSaved();
              } catch (error) {
                fail(error);
              }
            }}
          >
            <option value="">仅知识与参考图，暂不关联</option>
            {packages.map((p) => (
              <option key={p.name} value={p.name} disabled={!p.ready}>
                {p.name}
                {p.ready ? "" : "（权重不完整）"}
              </option>
            ))}
          </select>
        </label>
      </div>
      {editing ? (
        <div className="editor-form">
          <label>
            设定版本名称
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              maxLength={80}
            />
          </label>
          <label>
            外观与视觉约束
            <textarea
              rows={12}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="记录发色、瞳色、服装、配色和标志配饰，可注明遮挡时无需补画…"
            />
          </label>
          <label>
            设定依据
            <input value={source} onChange={(e) => setSource(e.target.value)} />
          </label>
          <div className="section-caption">
            参考图片{" "}
            <button
              onClick={() => referenceInput.current?.click()}
              disabled={references.length >= 6}
            >
              <ImagePlus size={15} />
            </button>
          </div>
          <div className="reference-grid">
            {references.map((id) => (
              <div key={id}>
                <img src={imageUrl(id)} alt="角色参考图" />
                <button
                  onClick={() =>
                    setReferences((items) =>
                      items.filter((item) => item !== id),
                    )
                  }
                  aria-label="移除参考图"
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={confirmed}
              onChange={(e) => setConfirmed(e.target.checked)}
            />
            我已核实此版本，允许作为一致性检查依据
          </label>
          <div className="field-row">
            <button className="secondary" onClick={() => setEditing(false)}>
              取消
            </button>
            <button
              className="primary"
              disabled={!label || !description || saving}
              onClick={save}
            >
              {saving ? (
                <LoaderCircle className="spin" size={15} />
              ) : (
                <Check size={15} />
              )}
              保存为新版本
            </button>
          </div>
        </div>
      ) : (
        <>
          {current ? (
            <>
              <div className="section-caption">
                当前设定{" "}
                <span
                  className={
                    "availability " + (current.confirmed ? "ready" : "")
                  }
                >
                  {current.confirmed ? "已核实" : "待核实"}
                </span>
              </div>
              <div className="profile-version">
                <h4>{current.label}</h4>
                <p className="profile-description">
                  {String(
                    current.constraints["设定说明"] ||
                      JSON.stringify(current.constraints, null, 2),
                  )}
                </p>
                <small>依据：{current.provenance}</small>
              </div>
              <div className="reference-grid">
                {current.references.map((id) => (
                  <img src={imageUrl(id)} alt="角色参考图" key={id} />
                ))}
              </div>
            </>
          ) : (
            <div className="empty-inline">
              <ShieldCheck size={26} />
              <p>为角色建立一份视觉档案</p>
              <small>
                明确的设定与参考图，
                <br />
                是一致性检查的依据。
              </small>
            </div>
          )}
          <button className="primary full" onClick={beginEdit}>
            <Plus size={15} />
            {current ? "编辑并保存新版本" : "建立角色档案"}
          </button>
          {profiles.length > 1 && (
            <>
              <div className="section-caption version-caption">
                历史设定版本
              </div>
              {profiles.map((p) => (
                <button
                  className="version-row"
                  key={p.id}
                  onClick={() =>
                    post(
                      `/characters/${character.id}/profiles/${p.id}/activate`,
                    )
                      .then(onSaved)
                      .catch(fail)
                  }
                >
                  <span>
                    {p.label}
                    <small>{p.confirmed ? "已核实" : "待核实"}</small>
                  </span>
                  {p.id === character.active_version ? (
                    <Check size={15} />
                  ) : (
                    <ChevronRight size={15} />
                  )}
                </button>
              ))}
            </>
          )}
        </>
      )}
      <input
        ref={referenceInput}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        hidden
        onChange={async (e) => {
          const file = e.target.files?.[0];
          if (file) {
            try {
              const body = new FormData();
              body.append("file", file);
              const asset = await api<Asset>("/assets", {
                method: "POST",
                body,
              });
              setReferences((items) => [...items, asset.id]);
            } catch (error) {
              fail(error);
            }
          }
        }}
      />
    </div>
  );
}
