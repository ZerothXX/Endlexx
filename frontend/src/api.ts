export type Character = {
  id: string;
  name: string;
  description: string;
  package: string | null;
  avatar_id: string | null;
  active_version: string | null;
};
export type Chat = {
  id: string;
  character_id: string | null;
  title: string;
  model_profile: string;
};
export type ChatContextStatus = { compressed_messages: number; updated_at: string | null };
export type Asset = {
  id: string;
  url: string;
  filename: string;
  parent_id: string | null;
  width: number;
  height: number;
};
export type Citation = {
  number: number;
  title: string;
  source: string;
  text: string;
  version: string;
  document_id: string;
  ordinal: number;
};
export type CheckItem = {
  feature: string;
  status: string;
  observation: string;
  expected: string;
};
export type ToolResult = {
  tool: string;
  asset?: Asset;
  before?: string;
  report?: { items: CheckItem[]; summary: string };
  metrics?: { protected_pixels_changed: number; mask_fraction: number };
};
export type Details = {
  citations?: Citation[];
  results?: ToolResult[];
  retrieval?: { method: string; warnings: string[] };
  plan?: string[];
};
export type Message = {
  id: string;
  role: string;
  content: string;
  assets: string[];
  details: Details;
  task_id: string | null;
};
export type Task = {
  id: string;
  status: string;
  phase: string;
  progress: number;
  error: string | null;
  result: Details;
  snapshot: { controller: string; mode: string };
  traces?: { node: string; data: Record<string, unknown> }[];
};
export type Profile = {
  id: string;
  label: string;
  constraints: Record<string, unknown>;
  references: string[];
  provenance: string;
  confirmed: boolean;
};
export type Doc = {
  id: string;
  title: string;
  source: string;
  content: string;
  version: string;
  category: string;
  vector_indexed: boolean;
};
export type Models = {
  profiles: {
    id: string;
    name: string;
    available: boolean;
    description: string;
    external: boolean;
  }[];
  runtime: {
    ready: boolean;
    loaded?: boolean;
    embedding_ready?: boolean;
    reranker_ready?: boolean;
  };
  repair: { weights_present: boolean; ppo_available: boolean; reason: string };
  storage: string;
  queue: string;
};
export type ModelConnection = {
  id: string;
  name: string;
  kind: "api" | "local" | "hybrid";
  engine: string;
  model: string;
  vision_model?: string;
  vision_fallback?: "native" | null;
  base_url?: string;
  status: string;
};
export type ModelRegistry = { active_id: string | null; entries: ModelConnection[] };
export type Deployment = {
  model_id: string;
  status: string;
  phase: string;
  current: number;
  error: string | null;
  items: { name: string; status: string; completed: number; total: number | null; detail?: string }[];
};
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch("/api" + path, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const value = await response.json().catch(() => ({}));
    throw new Error(
      typeof value.detail === "string"
        ? value.detail
        : value.detail?.message || `请求失败 (${response.status})`,
    );
  }
  return response.status === 204 ? (undefined as T) : response.json();
}
export const post = <T>(path: string, value: unknown = {}) =>
  api<T>(path, { method: "POST", body: JSON.stringify(value) });
export const imageUrl = (id: string) => `/api/assets/${id}/file`;
export const activeTask = (task: Task) =>
  ["queued", "running"].includes(task.status);
