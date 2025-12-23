#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
VIDEOS_DIR="${DATA_DIR}/videos"
BDD_DIR="${DATA_DIR}/bdd100k"
KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
KAGGLE_JSON="${KAGGLE_CONFIG_DIR}/kaggle.json"
export KAGGLE_CONFIG_DIR

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install with: pip install kaggle"
  exit 1
fi

if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
  if [[ -z "${KAGGLE_USERNAME:-}" ]]; then
    echo "Set KAGGLE_USERNAME to use KAGGLE_API_TOKEN."
    exit 1
  fi
  if [[ ! -f "${KAGGLE_JSON}" ]]; then
    mkdir -p "${KAGGLE_CONFIG_DIR}"
    cat > "${KAGGLE_JSON}" <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_API_TOKEN}"}
EOF
    chmod 600 "${KAGGLE_JSON}"
  fi
else
  if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
    if [[ ! -f "${KAGGLE_JSON}" ]]; then
      echo "Kaggle credentials not found."
      echo "Set KAGGLE_API_TOKEN (and KAGGLE_USERNAME) or KAGGLE_USERNAME/KAGGLE_KEY, or place kaggle.json at ${KAGGLE_JSON}"
      exit 1
    fi
  fi
fi

mkdir -p "${VIDEOS_DIR}"

echo "Downloading test videos (various weather)..."
kaggle datasets download \
  -d ashikadnan/driving-video-for-lane-detection-various-weather \
  -p "${VIDEOS_DIR}" \
  --unzip

if [[ "${DOWNLOAD_BDD100K:-0}" == "1" ]]; then
  mkdir -p "${BDD_DIR}"
  echo "Downloading BDD100K (large)..."
  kaggle datasets download \
    -d marquis03/bdd100k \
    -p "${BDD_DIR}" \
    --unzip
fi

echo "Done."
