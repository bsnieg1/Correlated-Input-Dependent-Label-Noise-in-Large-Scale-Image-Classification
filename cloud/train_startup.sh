#!/bin/bash
# GCE startup script for a GPU training VM (runs on every boot; launched by cloud/run_train.sh).
# Pulls the code and data from GCS, runs src.run_grid, syncs results to GCS every 5 min, powers off when done.
# Runs that already have a summary.json (locally or in GCS) are skipped, so a reboot simply continues the grid.
set -uo pipefail
meta() { curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
BUCKET=$(meta bucket); PATTERN=$(meta grid-pattern); SEEDS=$(meta grid-seeds); RUN=$(meta run-name)
WORK=/opt/het
OUT="gs://$BUCKET/runs/$RUN"
mkdir -p $WORK && cd $WORK
log() { echo "[het] $(date +%H:%M:%S) $*" | tee -a $WORK/startup.log; }

log "waiting for the GPU driver"
for i in $(seq 1 60); do nvidia-smi > /dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a $WORK/startup.log

log "fetching code and data"
gcloud storage cp "gs://$BUCKET/code/het_code.tar.gz" . --quiet && tar xzf het_code.tar.gz
mkdir -p data/animal10n results
[ -f data/animal10n/animal10n.pt ] || gcloud storage cp "gs://$BUCKET/animal10n/*" data/animal10n/ --quiet
gcloud storage rsync -r "$OUT/results" results --quiet 2>/dev/null || true

PY=""
for p in /opt/conda/bin/python /opt/python/bin/python3 /usr/bin/python3; do
  [ -x "$p" ] && "$p" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && PY=$p && break
done
if [ -z "$PY" ]; then
  log "no python with CUDA torch found, creating a venv"
  python3 -m venv $WORK/venv && PY=$WORK/venv/bin/python
  $PY -m pip install -q torch torchvision
fi
$PY -m pip install -q pyyaml numpy scipy scikit-learn pandas matplotlib tabulate
log "python: $PY, torch $($PY -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())')"

sync() { gcloud storage rsync -r results "$OUT/results" --quiet; cp $WORK/*.log results/ 2>/dev/null; }
( while true; do sleep 300; sync; done ) &
SYNC_PID=$!

log "grid: pattern=$PATTERN seeds=$SEEDS"
$PY -u -W ignore -m src.run_grid --pattern "$PATTERN" --seeds $SEEDS >> $WORK/grid.log 2>&1
log "grid finished (exit $?)"
kill $SYNC_PID
sync
gcloud storage cp $WORK/grid.log $WORK/startup.log "$OUT/" --quiet
echo done | gcloud storage cp - "$OUT/_DONE"
log "powering off"
shutdown -h now
