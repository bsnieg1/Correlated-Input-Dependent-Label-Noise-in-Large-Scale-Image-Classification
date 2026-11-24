#!/bin/bash
# One-off: CPU VM that downloads Zenodo 20425725, downscales the photos and uploads them to GCS.
# The VM powers itself off when done; delete it afterwards:
#   gcloud compute instances delete het-prep --zone $ZONE --project $PROJECT --quiet
set -euo pipefail
PROJECT=het-thesis-bsnieg
ZONE=europe-west4-a
BUCKET=het-thesis-bsnieg-data
cd "$(dirname "$0")"

gcloud storage buckets describe "gs://$BUCKET" --project $PROJECT > /dev/null 2>&1 || \
  gcloud storage buckets create "gs://$BUCKET" --project $PROJECT --location europe-west4 \
    --uniform-bucket-level-access
gcloud compute instances create het-prep --project $PROJECT --zone $ZONE \
  --machine-type e2-standard-8 --image-family debian-12 --image-project debian-cloud \
  --boot-disk-size 50GB --scopes storage-rw \
  --metadata bucket=$BUCKET \
  --metadata-from-file startup-script=startup.sh,prepare-py=prepare_images.py
