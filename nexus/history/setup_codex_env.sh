#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/quants-lab"
ENV_YML="$REPO_DIR/environment.yml"

# Where environments live (no Miniconda needed)
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

# Pick micromamba binary for this CPU
case "$(uname -m)" in
  x86_64|amd64) MM_FILE="micromamba-linux-64" ;;
  aarch64|arm64) MM_FILE="micromamba-linux-aarch64" ;;
  *) echo "Unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac

fetch() {
  local url="$1" out="$2"
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$url" -o "$out"
  else wget -qO "$out" "$url"; fi
}

install_micromamba() {
  command -v micromamba >/dev/null 2>&1 && return
  echo "==> Installing micromamba…"
  install -d /usr/local/bin
  fetch "https://github.com/mamba-org/micromamba-releases/releases/latest/download/${MM_FILE}" "/usr/local/bin/micromamba"
  chmod +x /usr/local/bin/micromamba
}

wire_up_ca() {
  # Prefer system CA so libmamba/requests match curl’s trust
  for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
    if [[ -f "$f" ]]; then
      export SSL_CERT_FILE="$f" REQUESTS_CA_BUNDLE="$f"
      echo "==> Using system CA bundle: $f"
      break
    fi
  done
}

env_name_from_yaml() {
  awk '/^[[:space:]]*name:[[:space:]]*/{print $2; exit}' "$ENV_YML" 2>/dev/null || true
}

env_exists() {
  local name="$1"
  [[ -d "$MAMBA_ROOT_PREFIX/envs/$name" ]]
}

create_env() {
  [[ -f "$ENV_YML" ]] || { echo "Missing $ENV_YML" >&2; exit 1; }
  install_micromamba
  wire_up_ca

  local NAME; NAME="$(env_name_from_yaml)"; NAME="${NAME:-quants-lab}"
  echo "==> Target env name: $NAME"

  if env_exists "$NAME"; then
    echo "==> Environment '$NAME' already exists. Skipping creation."
  else
    echo "==> Creating environment with micromamba…"
    eval "$(micromamba shell hook -s bash)"
    micromamba env create -y -f "$ENV_YML"
  fi
}

activate_env() {
  local NAME; NAME="$(env_name_from_yaml)"; NAME="${NAME:-quants-lab}"
  eval "$(micromamba shell hook -s bash)"
  micromamba activate "$NAME"
  echo "==> Activated '$NAME' ($(which python))"
  python -V || true
}

download_dataset() {
  # Keep this exactly as you had it, but ensure it runs inside the env
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_ROOT_USER_ACTION=ignore
  mise settings add idiomatic_version_file_enable_tools "[]" || true
  pip install -q requests
  python -m nexus.history.codex
}

main() {
  [[ -d "$REPO_DIR" ]] || { echo "Missing repo: $REPO_DIR" >&2; exit 1; }
  cd "$REPO_DIR"
  create_env
  activate_env
  download_dataset

  echo
  echo "✔ Setup complete."
  echo "  To reuse the environment in a new shell:"
  echo "    eval \"\$(micromamba shell hook -s bash)\" && micromamba activate \$(awk '/^[[:space:]]*name:/{print \$2; exit}' \"$ENV_YML\" 2>/dev/null || echo quants-lab)"
}

main "$@"
