# Brain Tumour Detection Using CNNs (ICT-4442, Team 11)

Binary classification (tumor vs no tumor) of brain MRI slices. We compare an MLP baseline,
VGG16, ResNet50 and a hybrid VGG16-ResNet50 on one common split.

Team: Pihu Pandey (230953134), Mihika Agarwal (2309531104), Kumari Udita (230953094)
School of Computer Engineering, MIT Manipal, MAHE.

## Dataset
Brain Tumor MRI Dataset (Masoud Nickparvar), Kaggle. Not stored in this repo (about 165 MB).
Download it, unzip so that `dataset/Training` and `dataset/Testing` exist, then:

    python src/data.py --data-dir dataset --out-dir data

This verifies every image, merges glioma/meningioma/pituitary into `tumor` (1) vs `notumor` (0),
hashes files to find exact duplicates, and writes a seeded, stratified 80/20 train/val split
(`data/train.csv`, `val.csv`, `test.csv`, `data_report.json`). The official `Testing` folder is the
held-out test set and is not used for tuning. A copy of the report for our download is in `docs/data_report.json`.

## Run
    pip install -r requirements.txt
    python -m src.train --model vgg16
    python -m src.train --model resnet50
    python -m src.train --model mlp
Metrics (accuracy, precision, tumor recall, F1, confusion matrix) go to `results/*.json`.
Use `--final-test` only for the final report.

## Status (interim)
| Model | Owner | Status |
|---|---|---|
| MLP baseline | all | implemented |
| VGG16 | Mihika Agarwal | implemented |
| ResNet50 | Pihu Pandey | implemented |
| Hybrid VGG16-ResNet50 | Kumari Udita | in progress |

`notebooks/prototype_253_images.ipynb` is the earlier prototype on a 253-image dataset; its numbers
are not comparable with the main pipeline in `src/`.
