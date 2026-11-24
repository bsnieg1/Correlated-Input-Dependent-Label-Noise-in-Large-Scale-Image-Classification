"""Downscales the Multispecies Campus UJ camera-trap photos (Zenodo 20425725) for training.

    python3 prepare_images.py <zips_dir> <out_dir>

For every JPG in every ZIP: resize so the longer side is MAX_SIDE (aspect kept, no cropping;
the camera info bar is cut later in the data loader, where the crop can be tuned per camera),
save as JPEG quality 92, and record a manifest row with the original size and EXIF timestamp
(needed to split train/test by photo sequence instead of by single photo).
"""
import csv
import io
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

MAX_SIDE = 1024
EXIF_DATETIME_ORIGINAL, EXIF_DATETIME = 36867, 306


def process_zip(args):
    zip_path, out_dir = args
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".jpg", ".jpeg")):
                continue
            folder = Path(zip_path).stem
            name = Path(info.filename).name
            try:
                im = Image.open(io.BytesIO(z.read(info)))
                exif = im.getexif()
                taken = exif.get_ifd(0x8769).get(EXIF_DATETIME_ORIGINAL) or exif.get(EXIF_DATETIME, "")
                w, h = im.size
                im = im.convert("RGB")
                im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
                dst = Path(out_dir) / folder / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                im.save(dst, "JPEG", quality=92)
                rows.append([folder, name, w, h, im.size[0], im.size[1], taken, info.file_size, ""])
            except Exception as e:  # keep going, record the failure
                rows.append([folder, name, "", "", "", "", "", info.file_size, repr(e)])
    return rows


def main():
    zips_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    zips = sorted(zips_dir.glob("*.zip"))
    with Pool() as pool:
        results = pool.map(process_zip, [(str(z), str(out_dir)) for z in zips])
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "name", "orig_w", "orig_h", "w", "h", "exif_datetime", "orig_bytes", "error"])
        for rows in results:
            writer.writerows(rows)
    n = sum(len(r) for r in results)
    errors = sum(1 for r in results for row in r if row[-1])
    print(f"{len(zips)} zips, {n} images, {errors} errors")


if __name__ == "__main__":
    main()
