#!/usr/bin/env bash
# Download grandmaster game archives from pgnmentor.com into data/raw_pgn/.
# Usage: bash scripts/download_data.sh
set -euo pipefail
OUT=data/raw_pgn
mkdir -p "$OUT/zips"
PLAYERS="Carlsen Kasparov Fischer Anand Nakamura Caruana Nepomniachtchi Ding Karpov Tal \
Capablanca Kramnik Aronian Giri So Firouzja Rapport Grischuk Topalov"
for p in $PLAYERS; do
  if [ ! -f "$OUT/$p.pgn" ]; then
    echo "downloading $p"
    curl -sfL --retry 3 --max-time 60 -o "$OUT/zips/$p.zip" "https://www.pgnmentor.com/players/$p.zip"
    unzip -o -q "$OUT/zips/$p.zip" -d "$OUT" || echo "  could not unzip $p.zip, skipping"
  fi
done
ls -lh "$OUT"/*.pgn
