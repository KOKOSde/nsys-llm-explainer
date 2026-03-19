#!/usr/bin/env bash
set -euo pipefail

# Provider-agnostic installer for nsys-llm-explainer API.
# Works on most Debian/Ubuntu/Amazon Linux hosts (Vast, EC2, personal GPU servers).

REPO_URL="https://github.com/KOKOSde/nsys-llm-explainer.git"
REPO_REF="v0.3.2"
INSTALL_DIR="/opt/nsys-llm-explainer"
VENV_DIR="/opt/nsys-venv"
SERVICE_NAME="nsys-llm-api"
BIND_HOST="0.0.0.0"
PORT="7860"
API_KEY="${NSYS_API_KEY:-}"

usage() {
  cat <<'USAGE'
Usage:
  sudo bash deploy/any_gpu/install_service.sh [options]

Options:
  --repo-url <url>         Git repo URL (default: https://github.com/KOKOSde/nsys-llm-explainer.git)
  --repo-ref <ref>         Git tag/branch/sha (default: v0.3.2)
  --install-dir <path>     Clone path (default: /opt/nsys-llm-explainer)
  --venv-dir <path>        Python venv path (default: /opt/nsys-venv)
  --bind-host <host>       API bind host (default: 0.0.0.0)
  --port <port>            API port (default: 7860)
  --api-key <secret>       Optional API key. If set, /v1 endpoints require auth.
  --help                   Show this help

Notes:
  - Do not pass secrets on shared shells. Prefer:
      export NSYS_API_KEY='your-secret'
      sudo -E bash deploy/any_gpu/install_service.sh
USAGE
}

require_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "This installer must run as root (sudo)." >&2
    exit 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --repo-url) REPO_URL="$2"; shift 2 ;;
      --repo-ref) REPO_REF="$2"; shift 2 ;;
      --install-dir) INSTALL_DIR="$2"; shift 2 ;;
      --venv-dir) VENV_DIR="$2"; shift 2 ;;
      --bind-host) BIND_HOST="$2"; shift 2 ;;
      --port) PORT="$2"; shift 2 ;;
      --api-key) API_KEY="$2"; shift 2 ;;
      --help|-h) usage; exit 0 ;;
      *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
  done
}

install_system_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y git python3 python3-venv python3-pip
    return
  fi
  if command -v dnf >/dev/null 2>&1; then
    dnf install -y git python3 python3-pip python3-setuptools
    return
  fi
  if command -v yum >/dev/null 2>&1; then
    yum install -y git python3 python3-pip
    return
  fi
  echo "Unsupported package manager. Install git/python3/python3-venv/python3-pip manually." >&2
  exit 1
}

sync_repo() {
  if [[ -d "${INSTALL_DIR}/.git" ]]; then
    git -C "${INSTALL_DIR}" fetch --tags --force origin
    git -C "${INSTALL_DIR}" checkout --force "${REPO_REF}"
    return
  fi
  rm -rf "${INSTALL_DIR}"
  git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" "${INSTALL_DIR}"
}

install_python_deps() {
  python3 -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV_DIR}/bin/python" -m pip install "${INSTALL_DIR}[api]"
}

install_service_files() {
  cat >"/usr/local/bin/start_nsys_api.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd ${INSTALL_DIR}
exec ${VENV_DIR}/bin/python -m nsys_llm_explainer.api --host ${BIND_HOST} --port ${PORT}
EOF
  chmod +x /usr/local/bin/start_nsys_api.sh

  cat >"/etc/default/${SERVICE_NAME}" <<EOF
NSYS_API_KEY=${API_KEY}
EOF

  cat >"/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=nsys-llm-explainer API service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
EnvironmentFile=-/etc/default/${SERVICE_NAME}
Environment=PORT=${PORT}
ExecStart=/usr/local/bin/start_nsys_api.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
}

start_service() {
  systemctl daemon-reload
  systemctl enable --now "${SERVICE_NAME}"
  sleep 2
  systemctl --no-pager --full status "${SERVICE_NAME}" || true
}

main() {
  parse_args "$@"
  require_root
  install_system_packages
  sync_repo
  install_python_deps
  install_service_files
  start_service
  echo
  echo "Installed ${SERVICE_NAME}."
  echo "Health check:"
  echo "  curl -sS http://127.0.0.1:${PORT}/healthz"
}

main "$@"
