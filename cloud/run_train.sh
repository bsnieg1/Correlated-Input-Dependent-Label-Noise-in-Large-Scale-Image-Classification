#!/bin/bash
# Launches a GPU VM (1x NVIDIA L4) that runs one experiment grid and powers itself off when done.
#
#   bash cloud/run_train.sh upload-data D:/inzynier_temp/animal10n_download  # once: animal10n.pt -> GCS
#   bash cloud/run_train.sh start a1 "animal_a1_*"                      # run name, config pattern [, seeds]
#   bash cloud/run_train.sh status a1                                   # last lines of the grid log
#   bash cloud/run_train.sh fetch a1                                    # results -> results/ (local)
#   bash cloud/run_train.sh delete a1                                   # delete the (stopped) VM
#
# A stopped VM costs only its disk (~$0.01/h); delete it after fetching the results.
set -euo pipefail
PROJECT=het-thesis-bsnieg
ZONE=${ZONE:-europe-west4-a}
BUCKET=het-thesis-bsnieg-data
IMAGE_FAMILY=pytorch-2-9-cu129-ubuntu-2204-nvidia-580
GPU=${GPU:-l4}  # l4 (g2-standard-4) or t4 (n1-standard-4); L4 is often out of stock: GPU=t4 ZONE=... bash ...
case $GPU in
  l4) MACHINE="--machine-type g2-standard-4 --accelerator type=nvidia-l4,count=1" ;;
  t4) MACHINE="--machine-type n1-standard-4 --accelerator type=nvidia-tesla-t4,count=1" ;;
  *) echo "GPU must be l4 or t4"; exit 1 ;;
esac
cd "$(dirname "$0")/.."  # repo root; relative paths below also work with gcloud on Windows (Git Bash)
cmd=$1; run=${2:-}
vm="het-$run"

case $cmd in
upload-data)
  gcloud storage cp "$2"/animal10n.pt "gs://$BUCKET/animal10n/" ;;  # tensor cache made by src.animal_data
start)
  pattern=$3; seeds=${4:-"0 1 2"}
  # Code snapshot (working tree, not git HEAD) -> GCS.
  tar czf het_code.tar.gz --exclude=__pycache__ src configs analysis requirements.txt
  gcloud storage cp het_code.tar.gz "gs://$BUCKET/code/het_code.tar.gz" && rm het_code.tar.gz
  gcloud compute instances create "$vm" --project $PROJECT --zone "$ZONE" \
    $MACHINE --maintenance-policy TERMINATE \
    --image-family $IMAGE_FAMILY --image-project deeplearning-platform-release \
    --boot-disk-size 100GB --scopes storage-rw \
    --metadata "bucket=$BUCKET,run-name=$run,grid-pattern=$pattern,grid-seeds=$seeds,install-nvidia-driver=True" \
    --metadata-from-file startup-script=cloud/train_startup.sh ;;
status)
  gcloud compute instances describe "$vm" --project $PROJECT --zone "$ZONE" --format="value(status)" || true
  gcloud compute ssh "$vm" --project $PROJECT --zone "$ZONE" --command \
    "tail -n 4 /opt/het/startup.log; grep -E '^=== |^epoch' /opt/het/grid.log | tail -n 6" 2>/dev/null \
    || gcloud storage cat "gs://$BUCKET/runs/$run/results/grid.log" 2>/dev/null | grep -E '^=== |^epoch' | tail -n 6 ;;
fetch)
  gcloud storage rsync -r "gs://$BUCKET/runs/$run/results" results ;;
delete)
  gcloud compute instances delete "$vm" --project $PROJECT --zone "$ZONE" --quiet ;;
*)
  echo "unknown command $cmd"; exit 1 ;;
esac
