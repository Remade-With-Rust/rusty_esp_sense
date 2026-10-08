#!/usr/bin/env bash
# Every output the benchmarks and the CLI produce, timing lines stripped.
# Usage: golden.sh <binary> <outdir>
B="$1"; O="$2"; mkdir -p "$O"
D="${CUENCA:?set CUENCA to the unzipped Cuenca dataset}"
strip() { grep -v -E '^[0-9]+\.[0-9]+ s$' ; }
"$B" bench-cuenca "$D" 2>/dev/null | strip > "$O/cuenca-default.txt"
"$B" bench-cuenca "$D" --features 0 2>/dev/null | strip > "$O/cuenca-linear.txt"
"$B" bench-cuenca "$D" --raw-window 2>/dev/null | strip > "$O/cuenca-raw.txt"
"$B" bench-cuenca "$D" --raw-window --no-centre 2>/dev/null | strip > "$O/cuenca-uncentred.txt"
"$B" bench-cuenca "$D" --raw-window --features 0 2>/dev/null | strip > "$O/cuenca-raw-linear.txt"
"$B" bench-fall "$D" 2>/dev/null > "$O/fall.txt"
"$B" bench-night "$D" 2>/dev/null > "$O/night.txt"
cd "$D"
M="$O/room.safetensors"
"$B" fit --out "$M" --layout c6 "empty=Escenario 1/linea_base_iter_1_20260513_122505.csv" "occupied=Escenario 2/movimiento_humano_iter_1_20260520_152220.csv" 2>/dev/null
for f in "Escenario 1/linea_base_iter_15_20260513_123905.csv" "Escenario 3/trafico_udp_10mbps_iter_15_20260520_173029.csv" "Escenario 4/mov_trafico_iter_10_20260617_160648.csv"; do
  "$B" run --model "$M" --layout c6 "$f"
done > "$O/run.jsonl" 2>/dev/null
"$B" night --layout c6 "Escenario 2/movimiento_humano_iter_20_20260520_154120.csv" > "$O/night-one.jsonl" 2>/dev/null
sha256sum "$O"/*.txt "$O"/*.jsonl "$M" | sed "s#$O/##"
