## Environment Setup

### 1. Create and activate environment
```bash
conda create -n SIRI python=3.12 -y
conda activate SIRI

# conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install matplotlib scikit-learn scikit-image opencv-python tqdm
pip install numpy pyyaml requests scipy

```

pip install -r requirements.txt

Dataset Preparation

Please place datasets under the dataset/ directory with the following structure:

```bash
dataset/
└── Rain100L/
    └── input/
        ├── image1.png
        ├── image2.png
        └── ...
```

1. Generate LDGP and SDR
```bash
python ldgp.py --dataset Rain100L
python sdr_new_test.py --dataset Rain100L
```

2. Run SIRI
```bash
bash run_all.sh
```