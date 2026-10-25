"""Configure local WSL services with generated application-only credentials."""

import json
import secrets
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wsl(command, stdin=None):
    result = subprocess.run(
        ["wsl", "-u", "root", "--", "sh", "-c", command],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    return result.stdout.strip()


def main():
    env_file = ROOT / ".env"
    values = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    credential_file = ROOT / "data/local-services.json"
    credential_file.parent.mkdir(exist_ok=True)
    creds = (
        json.loads(credential_file.read_text())
        if credential_file.exists()
        else {"postgres": secrets.token_hex(20), "redis": secrets.token_hex(20)}
    )
    credential_file.write_text(json.dumps(creds), encoding="utf-8")
    wsl("pg_ctlcluster 16 main start || true")
    address = "127.0.0.1"
    # Set listen addresses only for this newly installed PostgreSQL cluster.
    wsl(
        "runuser -u postgres -- psql -v ON_ERROR_STOP=1", "ALTER SYSTEM SET listen_addresses = 'localhost';\n"
    )
    wsl("pg_ctlcluster 16 main restart")
    sql = (
        "SELECT 'CREATE ROLE endlexx LOGIN' WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname='endlexx')\\gexec\n"
        f"ALTER ROLE endlexx PASSWORD '{creds['postgres']}';\n"
        "SELECT 'CREATE DATABASE endlexx OWNER endlexx' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname='endlexx')\\gexec\n"
        "\\connect endlexx\nCREATE EXTENSION IF NOT EXISTS vector;\n"
    )
    wsl("runuser -u postgres -- psql -v ON_ERROR_STOP=1", sql)
    redis_config = f"bind 127.0.0.1\nport 6381\nrequirepass {creds['redis']}\ndaemonize yes\ndir /var/lib/redis\npidfile /var/run/redis/endlexx.pid\nlogfile /var/log/redis/endlexx.log\nappendonly yes\n"
    wsl("cat > /etc/redis/endlexx.conf", redis_config)
    # Stop only this app's Redis process, if already configured.
    wsl("test ! -f /var/run/redis/endlexx.pid || kill $(cat /var/run/redis/endlexx.pid)")
    wsl("redis-server /etc/redis/endlexx.conf")
    values.update(
        {
            "ENDLEXX_DATABASE_URL": f"postgresql+psycopg://endlexx:{creds['postgres']}@{address}:5432/endlexx?connect_timeout=5",
            "ENDLEXX_REDIS_URL": f"redis://:{creds['redis']}@{address}:6381/0",
            "ENDLEXX_TASK_BACKEND": "celery",
            "ENDLEXX_MODEL_RUNTIME_TOKEN": values.get("ENDLEXX_MODEL_RUNTIME_TOKEN") or secrets.token_hex(24),
            "ENDLEXX_MODEL_PYTHON": values.get("ENDLEXX_MODEL_PYTHON")
            or str(ROOT / ".modelenv/Scripts/python.exe").replace("\\", "/"),
        }
    )
    env_file.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    print("PostgreSQL + pgvector and Redis configured. Credentials saved only to ignored local files.")


if __name__ == "__main__":
    main()
