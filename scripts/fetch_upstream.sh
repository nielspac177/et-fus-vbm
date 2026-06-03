#!/usr/bin/env bash
# Fetch the Calvinwhow/vbm upstream into external/ (gitignored) for the MNI 2mm mask,
# ROI atlases and normative control distributions used by the (exploratory) normative
# z-scoring stream. Not required for the primary cerebellar / association analyses.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/external/vbm"
if [ -d "$DEST/.git" ]; then
  echo "Updating $DEST"; git -C "$DEST" pull --ff-only || true
else
  echo "Cloning Calvinwhow/vbm into $DEST"
  git clone --depth 1 https://github.com/Calvinwhow/vbm.git "$DEST"
fi
echo "Upstream assets at: $DEST/assets and $DEST/rois"
