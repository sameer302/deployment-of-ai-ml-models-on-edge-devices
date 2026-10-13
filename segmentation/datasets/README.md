# Datasets

Dataset files are not committed to git (size, and licences that forbid redistribution). This file records where each dataset comes from and how to restore the exact copy used for the results.

## Cityscapes (semantic segmentation)

| Item | Value |
|---|---|
| Website | https://www.cityscapes-dataset.com/ (free account required, downloads at https://www.cityscapes-dataset.com/downloads/) |
| Packages | `gtFine_trainvaltest.zip` (fine annotations, ~241 MB) and `leftImg8bit_trainvaltest.zip` (left 8-bit images, ~11 GB) |
| Split used | **val only**: 500 images from frankfurt (267), lindau (59), munster (174) |
| Image size | 2048×1024 RGB PNG |
| Labels | `*_gtFine_labelIds.png` (34 label IDs), mapped to the 19 standard train IDs in code; all other IDs = 255 (ignored) |
| Licence | Non-commercial research use; redistribution not allowed. Cite: Cordts et al., "The Cityscapes Dataset for Semantic Urban Scene Understanding", CVPR 2016 |
| Used by | `semantic/01_fcn/fcn8_resnet18` (later: STDC, SegFormer) |

### Restore
1. Log in at the Cityscapes downloads page and download both packages (done on a PC on 2026-10-10).
2. Unzip both on the PC and copy only the val folders to the Pi, keeping this layout:
   ```
   datasets/cityscapes/
     leftImg8bit/val/{frankfurt,lindau,munster}/*_leftImg8bit.png
     gtFine/val/{frankfurt,lindau,munster}/*_gtFine_{color,instanceIds,labelIds}.png, *_gtFine_polygons.json
   ```
   ```bash
   # on the Pi
   mkdir -p ~/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/segmentation/datasets/cityscapes/{leftImg8bit,gtFine}
   # on the PC, from the folder with the unzipped data
   scp -r leftImg8bit/val sameer@<pi-address>:~/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/segmentation/datasets/cityscapes/leftImg8bit/
   scp -r gtFine/val      sameer@<pi-address>:~/Desktop/Edge_AI/Deployment-of-AI-ML-Models-on-Edge-Devices/segmentation/datasets/cityscapes/gtFine/
   ```
3. Verify it is the same copy used for the results (run from `datasets/cityscapes/`):
   ```bash
   for d in leftImg8bit/val gtFine/val; do
     echo "$d files=$(find $d -type f | wc -l) manifest_sha256=$(find $d -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
   done
   ```
   Expected:
   ```
   leftImg8bit/val files=500  manifest_sha256=cb4fb3d5a6bf82c9a5979e81b61657bfa7b1e358d067f10820085818564ecd6a
   gtFine/val      files=2000 manifest_sha256=39c1769638ba04ece59271d0a2decbaddc35a4ac72ba23caea12262ef1b831e7
   ```
   (The manifest hash covers every file's name and content, so it only matches if the folder is identical.)
