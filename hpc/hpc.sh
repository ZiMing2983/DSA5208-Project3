#!/usr/bin/env bash
# Helper for the NUS HPC clusters (Atlas, Vanda), DSA5208 Project 3.
# Off-campus, connect the NUS VPN (Cisco Secure Client) first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
CFG="$HERE/ssh_config"
REMOTE_DIR="dsa5208-p3"   # project folder on the cluster, relative to your home (no spaces)

usage() {
  cat <<'EOF'
usage: hpc/hpc.sh <command> [args]

  setup  <NUSNET_ID>              one-time: store your NUSNET ID in hpc/ssh_config
  login  <atlas|vanda> [--force]  interactive shell. The first login asks for your
                                  password and opens a shared connection (4 h).
                                  It reads the cluster's pre-login notice first; if
                                  the notice announces maintenance, it stops before
                                  the password prompt and logs the notice to
                                  hpc/env/<host>_downtime.txt (--force skips this)
  status <atlas|vanda>            show whether a shared connection is open
  run    <atlas|vanda> <command>  run a command on the login node
  sync   <atlas|vanda>            copy this project to ~/dsa5208-p3 on the cluster
  fetch  <atlas|vanda>            copy results/ and hpc/env/ back from the cluster
  logout <atlas|vanda>            close the shared connection

run, sync and fetch reuse the shared connection and never ask for a password.
EOF
  exit 1
}

need_host() {
  case "${1:-}" in
    atlas|vanda) ;;
    *) echo "error: host must be 'atlas' or 'vanda'" >&2; exit 1 ;;
  esac
  if grep -q '^    User NUSNET_ID$' "$CFG"; then
    echo "error: run 'hpc/hpc.sh setup <NUSNET_ID>' first" >&2
    exit 1
  fi
}

ssh_() { ssh -F "$CFG" "$@"; }

# rsync splits its -e string on spaces. The project path contains spaces, so run
# rsync from inside hpc/ and refer to the config by its relative name.
rsync_() { (cd "$HERE" && rsync -az -e "ssh -F ssh_config -o BatchMode=yes" "$@"); }

# run/sync/fetch only reuse the shared connection that `login` opens.
need_conn() {
  ssh_ -O check "$1" 2>/dev/null && return 0
  echo "error: no open connection to $1. Run 'hpc/hpc.sh login $1' first" >&2
  exit 1
}

# The cluster's pre-login banner, fetched without offering any password or key.
banner_() {
  ssh_ -o BatchMode=yes -o ControlMaster=no -o ControlPath=none \
       -o PubkeyAuthentication=no -o GSSAPIAuthentication=no \
       -o KbdInteractiveAuthentication=no -o PasswordAuthentication=no \
       -o ConnectTimeout=15 "$1" true 2>&1 | grep -v -e '^\*\* ' -e 'Permission denied' || true
}

cmd="${1:-}"
shift || true
case "$cmd" in
  setup)
    id="${1:-}"
    [[ "$id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "usage: hpc/hpc.sh setup <NUSNET_ID>" >&2; exit 1; }
    sed -i '' -E "s/^(    User ).*/\1$id/" "$CFG"
    echo "NUSNET ID set to '$id' in hpc/ssh_config"
    ;;
  login)
    need_host "${1:-}"
    h="$1"
    # A closed cluster may refuse even a correct password, and repeated refusals may
    # lock the NUSNET account (which the VPN, email and Canvas also use). So read the
    # pre-login notice before asking for the password, unless a connection is open.
    if [[ "${2:-}" != "--force" ]] && ! ssh_ -O check "$h" 2>/dev/null; then
      notice="$(banner_ "$h")"
      if grep -qiE 'maintenance|closed' <<<"$notice"; then
        mkdir -p "$HERE/env"
        printf '\n=== %s  %s pre-login notice (login skipped)\n%s\n' \
          "$(date '+%Y-%m-%d %H:%M %z')" "$h" "$notice" >> "$HERE/env/${h}_downtime.txt"
        printf '%s\n\n' "$notice" >&2
        echo "$h is announcing maintenance, so no password was asked for." >&2
        echo "The notice was logged to hpc/env/${h}_downtime.txt (evidence for a deadline extension)." >&2
        echo "Try again once the notice is gone, or override with: hpc/hpc.sh login $h --force" >&2
        exit 2
      fi
    fi
    ssh_ "$h"
    ;;
  status)
    need_host "${1:-}"
    ssh_ -O check "$1"
    ;;
  run)
    need_host "${1:-}"; need_conn "$1"
    h="$1"; shift
    ssh_ -o BatchMode=yes "$h" "$@"
    ;;
  sync)
    need_host "${1:-}"; need_conn "$1"
    rsync_ --exclude '.git/' --exclude 'build/' --exclude 'results/' --exclude 'hpc/env/' \
           --exclude '*.o' --exclude '.DS_Store' \
           --exclude 'hpc/smoke/hello_hybrid' --exclude 'hpc/smoke/cuda_aware_check' \
           "$ROOT/" "$1:$REMOTE_DIR/"
    echo "synced project -> $1:~/$REMOTE_DIR"
    ;;
  fetch)
    need_host "${1:-}"; need_conn "$1"
    mkdir -p "$ROOT/results" "$HERE/env"
    rsync_ "$1:$REMOTE_DIR/hpc/env/" "$HERE/env/"
    rsync_ "$1:$REMOTE_DIR/results/" "$ROOT/results/" 2>/dev/null || true
    echo "fetched hpc/env/ (and results/ if present) from $1"
    ;;
  logout)
    need_host "${1:-}"
    ssh_ -O exit "$1"
    ;;
  *)
    usage
    ;;
esac
