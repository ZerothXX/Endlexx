# -*- coding: utf-8 -*-
"""data_registry.py —— 训练数据登记与固定清单（单一数据访问层）

对应方案 §12：图片留在 dataset/<角色>/images/，数据库只存路径、标签与训练前元数据；
训练开始时一次性批量读取，形成不可变清单；**训练循环内不查询数据库**。

两种后端共用同一份 SQL 与同一套调用（唯一实现）：
    sqlite  （默认）        output/registry/characters.db，本机即开即用
    mysql   （可切换）      ENDLESS_REGISTRY_BACKEND=mysql + ENDLESS_MYSQL_* 环境变量

表结构（方案 §12 最小集）：
    characters     角色：稳定 ID、名称、数据目录、触发词、当前已发布数据版本
    image_samples  样本：图片 ID、角色 ID、数据版本、相对路径、原始文件名、哈希、
                   尺寸、标签、标注模型及规则版本、有效状态、样本类型与来源图片
    training_runs  训练任务：任务 ID、角色 ID、数据版本、清单路径与哈希、底模与配置、
                   状态、实际完成步数、输出权重、错误信息
"""
import hashlib
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime

import config

SQLITE_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS characters (
        character_id   TEXT PRIMARY KEY,
        name           TEXT NOT NULL,
        data_dir       TEXT NOT NULL,
        trigger_word   TEXT NOT NULL,
        data_version   TEXT,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS image_samples (
        sample_id      TEXT PRIMARY KEY,
        character_id   TEXT NOT NULL,
        data_version   TEXT NOT NULL,
        rel_path       TEXT NOT NULL,
        original_name  TEXT,
        file_hash      TEXT NOT NULL,
        width          INTEGER,
        height         INTEGER,
        tags           TEXT,
        label_model    TEXT,
        rule_version   TEXT,
        valid          INTEGER NOT NULL DEFAULT 1,
        sample_type    TEXT NOT NULL DEFAULT 'original',
        source_sample  TEXT,
        region_json    TEXT,
        created_at     TEXT NOT NULL,
        UNIQUE (character_id, data_version, rel_path)
    )""",
    """CREATE TABLE IF NOT EXISTS training_runs (
        run_id         TEXT PRIMARY KEY,
        character_id   TEXT NOT NULL,
        data_version   TEXT NOT NULL,
        manifest_path  TEXT,
        manifest_hash  TEXT,
        base_model     TEXT,
        settings_json  TEXT,
        status         TEXT NOT NULL,
        completed_steps INTEGER DEFAULT 0,
        weights_json   TEXT,
        error          TEXT,
        started_at     TEXT,
        finished_at    TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_samples_character ON image_samples (character_id, data_version)",
    "CREATE INDEX IF NOT EXISTS idx_runs_character ON training_runs (character_id, status)",
]

MYSQL_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS characters (
        character_id   VARCHAR(64) PRIMARY KEY,
        name           VARCHAR(191) NOT NULL,
        data_dir       VARCHAR(512) NOT NULL,
        trigger_word   VARCHAR(191) NOT NULL,
        data_version   VARCHAR(64),
        created_at     VARCHAR(32) NOT NULL,
        updated_at     VARCHAR(32) NOT NULL
    ) DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS image_samples (
        sample_id      VARCHAR(64) PRIMARY KEY,
        character_id   VARCHAR(64) NOT NULL,
        data_version   VARCHAR(64) NOT NULL,
        rel_path       VARCHAR(512) NOT NULL,
        original_name  VARCHAR(512),
        file_hash      VARCHAR(64) NOT NULL,
        width          INT,
        height         INT,
        tags           TEXT,
        label_model    VARCHAR(64),
        rule_version   VARCHAR(32),
        valid          TINYINT NOT NULL DEFAULT 1,
        sample_type    VARCHAR(32) NOT NULL DEFAULT 'original',
        source_sample  VARCHAR(512),
        region_json    TEXT,
        created_at     VARCHAR(32) NOT NULL,
        UNIQUE KEY uq_sample (character_id, data_version, rel_path(180))
    ) DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS training_runs (
        run_id          VARCHAR(64) PRIMARY KEY,
        character_id    VARCHAR(64) NOT NULL,
        data_version    VARCHAR(64) NOT NULL,
        manifest_path   VARCHAR(512),
        manifest_hash   VARCHAR(64),
        base_model      VARCHAR(512),
        settings_json   TEXT,
        status          VARCHAR(32) NOT NULL,
        completed_steps INT DEFAULT 0,
        weights_json    TEXT,
        error           TEXT,
        started_at      VARCHAR(32),
        finished_at     VARCHAR(32)
    ) DEFAULT CHARSET=utf8mb4""",
    "CREATE INDEX idx_samples_character ON image_samples (character_id, data_version)",
    "CREATE INDEX idx_runs_character ON training_runs (character_id, status)",
]


class RegistryError(RuntimeError):
    """数据库不可用或数据不一致；调用方据此决定报错还是走清单文件恢复。"""


def _placeholder(backend):
    return "?" if backend == "sqlite" else "%s"


class Registry:
    """数据登记统一入口。close() 之后连接不再复用（训练循环也不会用到它）。"""

    def __init__(self, backend=None, auto_create=True):
        self.backend = (backend or config.REGISTRY_BACKEND).lower()
        if self.backend not in {"sqlite", "mysql"}:
            raise RegistryError(f"未知的数据登记后端: {self.backend}")
        self.conn = None
        self._connect(auto_create=auto_create)

    # ---------- 连接与建表 ----------
    def _connect(self, auto_create=True):
        if self.backend == "sqlite":
            path = config.REGISTRY_DB_PATH
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.conn = sqlite3.connect(path)
            self.conn.row_factory = sqlite3.Row
            for statement in SQLITE_SCHEMA:
                self.conn.execute(statement)
            self.conn.commit()
            print(f"[registry] 使用 SQLite: {path}")
            return
        mysql = config.MYSQL
        if not mysql.get("password"):
            raise RegistryError(
                "MySQL 后端已选择但未提供密码。请设置 ENDLESS_MYSQL_PASSWORD（以及可选的 "
                "ENDLESS_MYSQL_HOST/PORT/USER/DATABASE），或改回 SQLite 后端。")
        try:
            import pymysql
        except ImportError as exc:
            raise RegistryError(f"缺少 pymysql，无法连接 MySQL: {exc}") from exc
        try:
            self.conn = pymysql.connect(host=mysql["host"], port=int(mysql["port"]),
                                        user=mysql["user"], password=mysql["password"],
                                        database=mysql["database"], charset=mysql["charset"],
                                        autocommit=False)
        except Exception as exc:
            raise RegistryError(
                f"MySQL 连接失败（{mysql['user']}@{mysql['host']}:{mysql['port']}/"
                f"{mysql['database']}）: {type(exc).__name__}: {exc}") from exc
        if auto_create:
            with self.conn.cursor() as cursor:
                for statement in MYSQL_SCHEMA:
                    try:
                        cursor.execute(statement)
                    except Exception as exc:
                        # 1050=表已存在 1061=索引名重复：都是幂等初始化的正常情况
                        code = getattr(exc, "args", [None])[0]
                        if code in (1050, 1061) or "exist" in str(exc).lower() \
                                or "duplicate key name" in str(exc).lower():
                            continue
                        raise
            self.conn.commit()
        print(f"[registry] 使用 MySQL: {mysql['user']}@{mysql['host']}/"
              f"{mysql['database']}")

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None

    def _rows(self, sql, params=()):
        sql = sql.replace("?", _placeholder(self.backend))
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            if self.backend == "sqlite":
                return [dict(row) for row in rows]
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cursor.close()

    def _execute(self, sql, params=(), commit=True):
        sql = sql.replace("?", _placeholder(self.backend))
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql, params)
            if commit:
                self.conn.commit()
        finally:
            cursor.close()

    # ---------- 角色 ----------
    def upsert_character(self, folder, data_version=None, trigger=None):
        now = datetime.now().isoformat(timespec="seconds")
        data_dir = f"{folder}/images"
        existing = self._rows("SELECT character_id, data_version FROM characters WHERE character_id=?",
                              (folder,))
        if existing:
            self._execute("UPDATE characters SET data_version=?, updated_at=? WHERE character_id=?",
                          (data_version or existing[0]["data_version"], now, folder))
        else:
            self._execute(
                "INSERT INTO characters (character_id, name, data_dir, trigger_word, "
                "data_version, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (folder, folder, data_dir, trigger or config.get_trigger(folder),
                 data_version, now, now))
        return folder

    def current_version(self, folder):
        rows = self._rows("SELECT data_version FROM characters WHERE character_id=?", (folder,))
        return rows[0]["data_version"] if rows else None

    # ---------- 样本 ----------
    def publish_samples(self, folder, samples, data_version, label_model="vlm",
                        rule_version=None):
        """登记一个数据版本的样本集合（幂等：同一 (角色,版本,路径) 覆盖更新）。"""
        rule_version = rule_version or config.LABEL_RULE_VERSION
        now = datetime.now().isoformat(timespec="seconds")
        self.upsert_character(folder, data_version=data_version)
        for sample in samples:
            tags = sample.get("tags")
            tags_text = json.dumps(tags, ensure_ascii=False) if isinstance(tags, list) else (tags or "")
            self._execute(
                "DELETE FROM image_samples WHERE character_id=? AND data_version=? AND rel_path=?",
                (folder, data_version, sample["rel_path"]))
            self._execute(
                "INSERT INTO image_samples (sample_id, character_id, data_version, rel_path, "
                "original_name, file_hash, width, height, tags, label_model, rule_version, "
                "valid, sample_type, source_sample, region_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sample.get("sample_id") or uuid.uuid4().hex, folder, data_version,
                 sample["rel_path"], sample.get("original_name"), sample["file_hash"],
                 sample.get("width"), sample.get("height"), tags_text, label_model,
                 rule_version, 1 if sample.get("valid", True) else 0,
                 sample.get("sample_type", "original"), sample.get("source_sample"),
                 json.dumps(sample.get("regions") or {}, ensure_ascii=False), now))
        self._execute("UPDATE characters SET data_version=?, updated_at=? WHERE character_id=?",
                      (data_version, now, folder))
        print(f"[registry] 已登记 {len(samples)} 个样本（角色={folder} 版本={data_version}）")

    def fetch_samples(self, folder, data_version=None):
        """训练前一次性批量读取该角色指定版本的样本（训练循环不再查询数据库）。"""
        version = data_version or self.current_version(folder)
        if not version:
            return []
        return self._rows(
            "SELECT sample_id, rel_path, original_name, file_hash, tags, sample_type, "
            "source_sample, valid FROM image_samples "
            "WHERE character_id=? AND data_version=? ORDER BY rel_path",
            (folder, version))

    def verify_samples(self, folder, samples):
        """校验磁盘上的图片与登记一致；返回不一致项列表（不静默换数据）。"""
        problems = []
        root = os.path.realpath(os.path.join(config.DATASET_DIR, folder, "images"))
        for sample in samples:
            path = os.path.realpath(os.path.join(root, os.path.basename(sample["rel_path"])))
            if not path.startswith(root):
                problems.append({"rel_path": sample["rel_path"], "reason": "路径越出角色图片目录"})
                continue
            if not os.path.isfile(path):
                problems.append({"rel_path": sample["rel_path"], "reason": "文件不存在"})
                continue
            actual = file_hash(path)
            if actual != sample["file_hash"]:
                problems.append({"rel_path": sample["rel_path"],
                                 "reason": f"内容哈希变化（登记 {sample['file_hash'][:12]}… "
                                           f"实际 {actual[:12]}…）"})
        return problems

    # ---------- 训练任务 ----------
    def start_run(self, folder, data_version, manifest_path, manifest_hash,
                  base_model, settings):
        run_id = uuid.uuid4().hex[:16]
        self._execute(
            "INSERT INTO training_runs (run_id, character_id, data_version, manifest_path, "
            "manifest_hash, base_model, settings_json, status, started_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, folder, data_version, manifest_path, manifest_hash, base_model,
             json.dumps(settings, ensure_ascii=False), "running",
             datetime.now().isoformat(timespec="seconds")))
        print(f"[registry] 训练任务已登记: run_id={run_id}")
        return run_id

    def finish_run(self, run_id, status, completed_steps=0, weights=None, error=None):
        self._execute(
            "UPDATE training_runs SET status=?, completed_steps=?, weights_json=?, error=?, "
            "finished_at=? WHERE run_id=?",
            (status, int(completed_steps),
             json.dumps(weights or [], ensure_ascii=False), error,
             datetime.now().isoformat(timespec="seconds"), run_id))

    def pending_runs(self):
        """状态更新失败时留下的本地任务记录，重连后按 run_id 幂等补写的依据。"""
        return self._rows("SELECT run_id, character_id, status FROM training_runs "
                          "WHERE status='running' ORDER BY started_at")


# ---------------------------------------------------------------------------
# 清单与指纹
# ---------------------------------------------------------------------------
def file_hash(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def new_data_version():
    return datetime.now().strftime("v%Y%m%d%H%M%S")


def write_manifest(folder, data_version, entries, meta=None) -> dict:
    """写出不可变训练清单：样本 ID、路径、标签、哈希、来源与版本（方案 §12）。"""
    os.makedirs(config.MANIFEST_DIR, exist_ok=True)
    payload = {
        "schema": 1,
        "character": folder,
        "data_version": data_version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": os.path.relpath(config.BASE_MODEL_PATH, config.PROJECT_ROOT),
        "samples": entries,
        "meta": meta or {},
    }
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    payload["manifest_hash"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    path = os.path.join(config.MANIFEST_DIR, f"{folder}_{data_version}.json")
    if os.path.isfile(path):
        raise FileExistsError(f"清单已存在，不覆盖不可变清单: {path}")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print(f"[registry] 训练清单已保存: {path}（{len(entries)} 条，"
          f"hash={payload['manifest_hash'][:12]}…）")
    return payload


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def open_registry(required=None):
    """按配置打开登记库；不可用时按 config.REGISTRY_REQUIRED 决定报错还是返回 None。"""
    required = config.REGISTRY_REQUIRED if required is None else required
    try:
        return Registry()
    except RegistryError as exc:
        if required:
            raise
        print(f"[registry] 警告: {exc}；按配置继续（训练清单需从既有文件恢复）")
        return None


def wait_for_connection(retries=3, delay=2.0):
    """训练前连接重试：MySQL 首次启动可能尚未就绪。"""
    last = None
    for attempt in range(1, retries + 1):
        try:
            return Registry()
        except RegistryError as exc:
            last = exc
            print(f"[registry] 连接失败（第 {attempt}/{retries} 次）: {exc}")
            time.sleep(delay)
    raise last
