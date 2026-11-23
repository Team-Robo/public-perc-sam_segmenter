#!/usr/bin/env bash
set -euo pipefail
sam_model_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "${sam_model_root}/models/model_manifest.json" ]]; then
  sam_model_root="$(cd -- "${sam_model_root}/../share/sam_segmenter" && pwd)"
fi
sam_model_destination="${1:-${sam_model_root}/models}"
mkdir -p -- "${sam_model_destination}"
python3 - "${sam_model_root}/models/model_manifest.json" <<'PY' |
import json, sys
for model in json.load(open(sys.argv[1], encoding='utf-8'))['models']:
    print(model['filename'], model['sha256'], model['url'], sep='\t')
PY
while IFS=$'\t' read -r filename digest url; do
  target="${sam_model_destination}/${filename}"
  if [[ -f "${target}" ]] && printf '%s  %s\n' "${digest}" "${target}" | sha256sum --check --status; then
    printf 'Verified %s\n' "${filename}"
    continue
  fi
  curl --fail --location --retry 3 --continue-at - --output "${target}.part" "${url}"
  printf '%s  %s\n' "${digest}" "${target}.part" | sha256sum --check
  mv -- "${target}.part" "${target}"
done
