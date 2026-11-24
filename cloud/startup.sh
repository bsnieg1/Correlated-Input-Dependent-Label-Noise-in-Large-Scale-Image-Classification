#!/bin/bash
# GCE startup script: download the Zenodo record, downscale the photos, upload to GCS, power off.
# Launched by cloud/run_prepare.sh; progress: gcloud compute instances get-serial-port-output.
set -euo pipefail
RECORD=20425725
BUCKET=$(curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket")
WORK=/mnt/work
mkdir -p $WORK/zips $WORK/out
cd $WORK

echo "[prep] installing packages"
apt-get update -qq && apt-get install -y -qq python3-pil jq > /dev/null
curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/prepare-py" > prepare_images.py

echo "[prep] downloading record $RECORD"
curl -s "https://zenodo.org/api/records/$RECORD" > record.json
jq -r '.files[] | "\(.key) \(.links.self) \(.checksum)"' record.json | while read -r key url checksum; do
  for attempt in 1 2 3 4 5; do
    curl -s -L --fail -o "zips/$key" "$url" && \
      [ "md5:$(md5sum "zips/$key" | cut -d' ' -f1)" = "$checksum" ] && break
    echo "[prep] retry $attempt for $key"; sleep 30
  done
  echo "[prep] downloaded $key"
done

echo "[prep] downscaling"
python3 prepare_images.py zips out | tee out/prepare_log.txt
cp record.json out/zenodo_record.json

echo "[prep] uploading to gs://$BUCKET/zenodo_1024/"
gcloud storage cp -r out/* "gs://$BUCKET/zenodo_1024/" > /dev/null
echo done | gcloud storage cp - "gs://$BUCKET/zenodo_1024/_DONE"
echo "[prep] finished, powering off"
shutdown -h now
