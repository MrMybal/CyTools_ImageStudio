#!/bin/sh
set -eu
app=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$#" -ne 1 ]; then
  printf '%s\n' 'Usage: sh app/install_linux.sh /path/to/cytools_core-0.12.0-<linux>.whl' >&2
  exit 1
fi
python3 -m venv "$app/../runtime/python"
"$app/../runtime/python/bin/python" -m pip install "$1" -r "$app/requirements.txt"
