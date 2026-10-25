# Security / 安全说明

Endlexx 0.1.0 is a single-user local research preview. Bind HTTP to
`127.0.0.1`; it is not a hosted multi-user service. No stable/LTS security support
or independent security audit is claimed.

- Do not expose the default local configuration to the public internet.
- Shared access requires an access token and HTTPS proxy; tenant/asset authorization
  is still needed for multi-user hosting.
- API prompts and images are sent to the configured provider when API mode is selected.
- Credentials are stored separately in local `data/`. Backups and logs may contain
  private material; this directory is excluded from release archives.
- Install models and character packages only from sources you trust. Source packaging
  checks high-confidence token patterns, paths and hashes; it is not a comprehensive
  secret detector or a guarantee that a file contains no confidential text.

Report vulnerabilities privately through GitHub's private vulnerability reporting
if the repository owner has enabled it. Otherwise contact the owner through a
private channel; never post live keys, personal data, or an exploit against a live
third-party service in a public issue. No dedicated contact address is configured.

本项目面向单人本机使用。公开问题单中请勿包含真实密钥、对话、图片、数据库或攻击第三方服务的细节。

