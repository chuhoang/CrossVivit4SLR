# 🧠 Sign Language Recognition (SLR) Model

## 📁 Dataset Structure

Make sure your dataset is located in the following directory:

```
dataset/
└── SLR/
    ├── class_1/
    │   ├── sample1.mp4
    │   └── ...
    ├── class_2/
    └── ...
```

## 🚀 Getting Started

### 1. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Use the following command to start training:

```bash
python train.py --data-path 'dataset/SLR' --batch-size 4 --epoch 100
```

#### Arguments

- `--data-path`: Path to your dataset folder (default: `'dataset/SLR'`)
- `--batch-size`: Batch size used for training (default: `4`)
- `--epoch`: Total number of training epochs (default: `100`)

### 3. Output

Training progress and model checkpoints will be saved to a directory (e.g., `checkpoints/`). You can configure this in `train.py`.

## 🛠 Project Structure

```
.
├── dataset/
│   └── SLR/              # Dataset folder
├── train.py              # Main training script
├── model.py            # Model architecture
├── utils.py                # Utility functions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
