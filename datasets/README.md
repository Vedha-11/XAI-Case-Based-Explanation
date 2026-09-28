# Dataset Documentation

## CUB-200-2011 Dataset

The primary dataset used in Phase 1 replication is **CUB-200-2011** (Caltech-UCSD Birds-200-2011).

- **Official Source**: [Caltech-UCSD Birds-200-2011 Dataset Page](https://www.vision.caltech.edu/datasets/CUB_200_2011/)
- **Description**: Contains 11,788 images of 200 bird species, with detailed annotations including bounding boxes, part locations, and attribute labels.

## Local Dataset Path & Structure

The repository expects the CUB-200-2011 dataset to be located at:
```text
original_repo/Expt 1/CUB/data/CUB_200_2011/
├── images/
│   ├── 001.Black_footed_Albatross/
│   ├── ...
│   └── 200.Common_Yellowthroat/
├── images.txt
├── image_class_labels.txt
├── train_test_split.txt
├── bounding_boxes.txt
└── parts/
```

> **Note**: Dataset image files, bounding box annotations, and large raw data are strictly excluded from Git version control via `.gitignore`.
