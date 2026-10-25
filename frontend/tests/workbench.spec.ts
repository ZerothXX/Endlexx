import { expect, test } from "@playwright/test";

test("role conversations can be created and deleted independently without a configured model", async ({page, request}) => {
  await page.route("**/api/model-connections", route => route.fulfill({json: {active_id: null, entries: []}}));
  const role = await (await request.post("/api/characters", {data: {name: "对话管理验收-" + Date.now()}})).json();
  for (let i = 1; i <= 3; i++) {
    await request.post("/api/chats", {data: {character_id: role.id, title: `已有对话${i}`}});
  }
  await page.goto("/");
  await page.locator(".role-link").filter({hasText: role.name}).click();
  const group = page.locator(".role-group").filter({hasText: role.name});
  await expect(group.locator(".chat-row")).toHaveCount(3);
  await group.getByRole("button", {name: "新建对话", exact: true}).click();
  await expect(group.locator(".chat-row")).toHaveCount(4);
  await expect(group.locator(".chat-row.current .chat-select")).toHaveText("新会话");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.screenshot({path: "../data/verification/conversation-actions.png", fullPage: true});

  await group.getByRole("button", {name: "删除对话：已有对话2", exact: true}).click();
  await page.getByRole("dialog").getByRole("button", {name: "取消", exact: true}).click();
  await expect(group.locator(".chat-row")).toHaveCount(4);
  await group.getByRole("button", {name: "删除对话：已有对话2", exact: true}).click();
  await page.getByRole("dialog").getByRole("button", {name: "确认删除", exact: true}).click();
  await expect(group.locator(".chat-row")).toHaveCount(3);
  await expect(group.locator(".chat-row.current .chat-select")).toHaveText("新会话");
  await group.getByRole("button", {name: "删除对话：新会话", exact: true}).click();
  await page.getByRole("dialog").getByRole("button", {name: "确认删除", exact: true}).click();
  await expect(group.locator(".chat-row")).toHaveCount(2);
  await expect(group.locator(".chat-row.current")).toHaveCount(1);
  for (let remaining = 2; remaining > 0; remaining--) {
    await group.locator(".chat-row.current .chat-delete").click();
    await page.getByRole("dialog").getByRole("button", {name: "确认删除", exact: true}).click();
    await expect(group.locator(".chat-row")).toHaveCount(remaining - 1);
  }
  await expect(page.getByLabel("输入创作需求")).toBeVisible();
  await group.getByRole("button", {name: "新建对话", exact: true}).click();
  await expect(group.locator(".chat-row")).toHaveCount(1);
  await page.reload();
  await page.locator(".role-link").filter({hasText: role.name}).click();
  await expect(group.locator(".chat-row")).toHaveCount(1);
});

test("role conversation shows compression status and blocks deleting a running task", async ({page, request}) => {
  const role = await (await request.post("/api/characters", {data: {name: "摘要提示验收-" + Date.now()}})).json();
  const chat = await (await request.post("/api/chats", {data: {character_id: role.id, title: "长对话"}})).json();
  await page.route(`**/api/chats/${chat.id}/context`, route => route.fulfill({json: {compressed_messages: 28, updated_at: null}}));
  await page.route(`**/api/chats/${chat.id}/messages`, route => route.fulfill({json: [
    {id: "old", role: "user", content: "较早消息仍可查看", assets: [], details: {}, task_id: null},
  ]}));
  await page.route(`**/api/chats/${chat.id}/tasks`, route => route.fulfill({json: [
    {id: "running-fixture", status: "running", phase: "正在压缩上下文", progress: 10, error: null, result: {}, snapshot: {controller: "test"}},
  ]}));
  await page.route("**/api/tasks/running-fixture/events", route => route.fulfill({status: 200, contentType: "text/event-stream", body: ": heartbeat\n\n"}));
  await page.goto("/");
  await page.locator(".role-link").filter({hasText: role.name}).click();
  await expect(page.getByRole("status")).toContainText("28 条消息已压缩为摘要");
  await expect(page.getByText("较早消息仍可查看", {exact: true})).toBeVisible();
  await expect(page.getByRole("button", {name: "删除对话：长对话", exact: true})).toBeDisabled();
  await page.screenshot({path: "../data/verification/conversation-compression.png", fullPage: true});
});

test("environment checks explain hardware limits and preserve repair progress", async ({page}) => {
  let job: Record<string, unknown> | null = null;
  await page.route("**/api/environment", route => route.fulfill({json: {
    checked_at: Date.now() / 1000,
    checks: [{id: "app", name: "应用 Python 与依赖", status: "ready", detail: "已验证", fixable: false}, {id: "ollama", name: "本地模型运行时", status: "repairable", detail: "自动配置独立运行时", fixable: true}],
    combination: [{id: "gpu", name: "研究组合 NVIDIA 驱动", status: "blocked", detail: "未检测到可用 NVIDIA 驱动", fixable: false}], repair: job
  }}));
  await page.route("**/api/environment/repair", route => {
    if (route.request().method() === "POST") job = {status: "running", current: 1, phase: "准备 Ollama 运行时", items: [{name: "硬件检查", status: "completed", completed: 0, total: null}, {name: "Ollama 运行时", status: "running", completed: 1024, total: 2048}], error: null};
    return route.fulfill({json: job});
  });
  await page.goto("/");
  await page.getByRole("button", {name: "模型", exact: true}).click();
  await page.getByRole("button", {name: "环境检查", exact: true}).click();
  await expect(page.getByText("应用 Python 与依赖", {exact: true})).toBeVisible();
  await expect(page.getByRole("button", {name: "配置组合推理环境"})).toBeDisabled();
  await page.getByRole("button", {name: "配置本地模型环境"}).click();
  await expect(page.getByText("准备 Ollama 运行时", {exact: true})).toBeVisible();
  await page.getByRole("button", {name: "关闭模型设置"}).click();
  await page.getByRole("button", {name: "模型", exact: true}).click();
  await page.getByRole("button", {name: "环境检查", exact: true}).click();
  await expect(page.getByText("准备 Ollama 运行时", {exact: true})).toBeVisible();
  await expect(page.getByRole("button", {name: "取消配置", exact: true})).toBeVisible();
});

test.beforeEach(async ({ request }) => {
  const roles = await (await request.get("/api/characters")).json();
  if (!roles.some((row: { name: string }) => row.name === "界面验收角色")) {
    await request.post("/api/characters", {data: {name: "界面验收角色"}});
  }
});

test("home is display-only and composer belongs to a role", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("heading", {name: "从一张图， 开始下一次创作。"})).toBeVisible();
  await expect(page.locator(".feature-card")).toHaveCount(3);
  await expect(page.locator(".shortcut-grid button")).toHaveCount(0);
  await expect(page.getByLabel("输入创作需求")).toHaveCount(0);
  await expect(page.getByRole("button", {name: "角色知识库", exact: true})).toBeDisabled();
  await page.screenshot({path: "../data/verification/model-home.png", fullPage: true});
  await page.locator(".role-link").filter({hasText: "界面验收角色"}).click();
  await expect(page.getByLabel("输入创作需求")).toBeVisible();
  await expect(page.locator(".composer-tools").getByLabel("选择聊天模型")).toBeVisible();
  await expect(page.locator(".topbar select")).toHaveCount(0);
  await page.getByRole("button", {name: "角色知识库", exact: true}).click();
  await expect(page.getByRole("heading", {name: "角色知识库"})).toBeVisible();
  await page.getByRole("button", {name: "关闭侧栏"}).click();
  await page.screenshot({path: "../data/verification/model-composer.png", fullPage: true});
  expect(errors).toEqual([]);
});

test("API connection add edit selection persistence and deletion", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", {name: "模型", exact: true}).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByRole("heading", {name: "模型", exact: true})).toBeVisible();
  await page.screenshot({path: "../data/verification/model-settings.png", fullPage: true});
  await dialog.getByRole("button", {name: "添加模型", exact: true}).click();
  await dialog.getByRole("button", {name: "API 添加", exact: false}).click();
  const name = "浏览器验收API-" + Date.now();
  await dialog.getByLabel("显示名称").fill(name);
  await dialog.getByLabel("API Base URL").fill("https://example.invalid/v1");
  await dialog.getByLabel("模型名称", {exact: true}).fill("vision-fixture");
  await dialog.getByLabel("API Key").fill("fixture-secret-not-a-real-key");
  await dialog.getByRole("button", {name: "保存配置", exact: true}).click();
  const row = dialog.locator(".connection-card").filter({hasText: name});
  await expect(row).toBeVisible();
  await expect(dialog).not.toContainText("fixture-secret-not-a-real-key");
  await row.getByRole("button", {name: "选用", exact: true}).click();
  await expect(row.getByText("已选用", {exact: true})).toBeVisible();
  await row.getByRole("button", {name: "编辑", exact: true}).click();
  await expect(dialog.getByLabel("API Key")).toHaveValue("");
  await dialog.getByLabel("模型名称", {exact: true}).fill("vision-updated");
  await dialog.getByRole("button", {name: "保存配置", exact: true}).click();
  await expect(row).toContainText("vision-updated");
  await page.getByRole("button", {name: "关闭模型设置"}).click();
  await page.reload();
  await page.locator(".role-link").filter({hasText: "界面验收角色"}).click();
  await expect(page.getByLabel("选择聊天模型")).toContainText(name);
  await page.getByRole("button", {name: "管理模型"}).click();
  const restored = page.getByRole("dialog").locator(".connection-card").filter({hasText: name});
  await restored.getByRole("button", {name: "删除", exact: true}).click();
  await restored.getByRole("button", {name: "确认删除", exact: true}).click();
  await expect(restored).toHaveCount(0);
});

test("local and fixed bundle deployment forms", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", {name: "模型", exact: true}).click();
  const dialog = page.getByRole("dialog");
  await dialog.getByRole("button", {name: "添加模型", exact: true}).click();
  await dialog.getByRole("button", {name: "本地部署", exact: false}).click();
  await expect(dialog.getByLabel("选择模型")).toContainText("Qwen3-VL 4B");
  await dialog.getByLabel("选择模型").selectOption("custom");
  await expect(dialog.getByLabel("模型标签")).toBeVisible();
  await dialog.getByRole("button", {name: "返回模型列表"}).click();
  await dialog.getByRole("button", {name: "添加模型", exact: true}).click();
  await dialog.getByRole("button", {name: "多模型组合", exact: false}).click();
  await expect(dialog.locator(".bundle-manifest li")).toHaveCount(9);
  await expect(dialog.locator(".bundle-manifest li").filter({hasText: "Anime LaMa"})).toBeVisible();
  await expect(dialog.getByRole("checkbox")).toHaveCount(0);
  await expect(dialog.getByRole("combobox")).toHaveCount(0);
  await page.screenshot({path: "../data/verification/model-bundle.png", fullPage: true});
});

test("deployment progress stays visible after reopening the dialog", async ({page}) => {
  const entry = {id: "m_progress", name: "组合下载验收", kind: "hybrid", engine: "native", model: "Qwen3-VL", status: "installing"};
  await page.route("**/api/model-connections", route => route.fulfill({json: {active_id: null, entries: [entry]}}));
  await page.route("**/api/model-connections/m_progress/deployment", route => route.fulfill({json: {
    model_id: entry.id, status: "running", phase: "下载 Illustrious XL", current: 2, error: null,
    items: [{name: "隔离环境", status: "completed", completed: 0, total: null},
      {name: "Qwen3-VL", status: "completed", completed: 2048, total: 2048},
      {name: "Illustrious XL", status: "running", completed: 52428800, total: 104857600},
      {name: "LaMa", status: "queued", completed: 0, total: null}]
  }}));
  await page.goto("/");
  await page.getByRole("button", {name: "模型", exact: true}).click();
  await expect(page.getByText("50.0 MB / 100.0 MB · 50.0%", {exact: true})).toBeVisible();
  await expect(page.locator(".deployment-progress li").last()).toContainText("等待中");
  await page.getByRole("button", {name: "关闭模型设置"}).click();
  await page.getByRole("button", {name: "模型", exact: true}).click();
  await expect(page.getByRole("progressbar")).toHaveAttribute("value", "52428800");
  await page.screenshot({path: "../data/verification/model-download-progress.png", fullPage: true});
});

test("mobile model dialog and role composer fit the viewport", async ({ page }) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.goto("/");
  await page.getByRole("button", {name: "展开导航"}).click();
  await page.getByRole("button", {name: "模型", exact: true}).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.screenshot({path: "../data/verification/model-mobile.png", fullPage: true});
  await page.keyboard.press("Escape");
  await page.getByRole("button", {name: "展开导航"}).click();
  await page.locator(".role-link").filter({hasText: "界面验收角色"}).click();
  await expect(page.getByLabel("输入创作需求")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("text controller shows its independent visual model", async ({page}) => {
  const entry = {id: "m_text", name: "通用主控验收", kind: "local", engine: "ollama", model: "qwen3:8b", status: "ready", vision_fallback: "native"};
  await page.route("**/api/model-connections", route => route.fulfill({json: {active_id: entry.id, entries: [entry]}}));
  await page.route("**/api/model-connections/m_text/deployment", route => route.fulfill({json: null}));
  await page.goto("/");
  await page.getByRole("button", {name: "模型", exact: true}).click();
  await expect(page.locator(".connection-info")).toContainText("视觉：本地 Qwen3-VL");
  await page.getByRole("button", {name: "添加模型", exact: true}).click();
  await page.getByRole("button", {name: "本地部署", exact: false}).click();
  await expect(page.getByText("通用语言模型负责规划、摘要与文字回答", {exact: false})).toBeVisible();
  await expect(page.getByRole("combobox")).toContainText("通用主控");
});
