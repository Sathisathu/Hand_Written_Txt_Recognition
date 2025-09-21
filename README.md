# Handwritten Text Recognition (HTR) System

A PyTorch-based deep learning system for recognizing handwritten text from line images using Convolutional Neural Networks (CNN) and Bidirectional LSTM with CTC (Connectionist Temporal Classification) loss.

## 🎯 Overview

This project implements a complete pipeline for handwritten text recognition, trained on the IAM Handwriting Database. The model combines CNN feature extraction with BiLSTM sequence modeling to transcribe handwritten text images into digital text.

## 🏗️ Architecture

The HTR model consists of three main components:

1. **CNN Feature Extractor**: 
   - Two convolutional layers (64 and 128 filters)
   - Max pooling for dimensionality reduction
   - Extracts visual features from input images

2. **BiLSTM Sequence Processor**:
   - 2-layer bidirectional LSTM with 256 hidden units
   - Processes sequential features from CNN
   - Captures temporal dependencies in handwriting

3. **CTC Output Layer**:
   - Fully connected layer for character classification
   - CTC loss for sequence alignment without explicit segmentation

## 📁 Project Structure

```
Hand_Written_Txt_Recognition/
├── train.py                    # Training script
├── predict.py                  # Inference script
├── test_dataset.py            # Dataset testing utility
├── model/
│   └── htr_model.py           # HTR model architecture
├── data_loader/
│   └── dataset.py             # Custom PyTorch dataset for IAM
├── data/
│   ├── labels.txt             # Image-text pairs mapping
│   ├── prepare_iam_lines.py   # IAM dataset preparation script
│   ├── ascii/
│   │   └── lines.txt          # Original IAM annotations
│   ├── images/                # Flattened image directory
│   └── lines/                 # Original IAM line images
├── checkpoints/
│   └── htr_model_best.pth     # Best trained model weights
├── my_handwriting/            # Sample handwriting images
└── scripts/
    └── visualize_samples.py   # Data visualization utilities
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- torchvision
- PIL (Pillow)
- tqdm
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Hand_Written_Txt_Recognition
```

2. Install required packages:
```bash
pip install torch torchvision pillow tqdm
```

### Dataset Preparation

This project uses the IAM Handwriting Database. To prepare the dataset:

1. Download the IAM dataset from [https://fki.tic.heia-fr.ch/databases/iam-handwriting-database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

2. Extract and organize the IAM data, then run:
```python
from data.prepare_iam_lines import prepare_iam_lines
prepare_iam_lines("/path/to/iam/dataset", "data/images/", "data/labels.txt")
```

## 🎓 Training

To train the model from scratch:

```bash
python train.py
```

### Training Configuration

The training script includes the following hyperparameters:

- **Batch Size**: 16
- **Image Height**: 64 pixels (width is variable, max 256)
- **Epochs**: 25
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Loss Function**: CTC Loss

The model automatically saves the best checkpoint based on validation loss to `checkpoints/htr_model_best.pth`.

### Training Features

- **Mixed Precision Training**: Uses automatic mixed precision for faster training
- **GPU Support**: Automatically detects and uses CUDA if available
- **Progress Tracking**: Uses tqdm for training progress visualization
- **Best Model Saving**: Saves model with lowest loss during training

## 🔮 Inference

To recognize text from a handwritten image:

```python
python predict.py
```

### Custom Prediction

You can modify `predict.py` to predict on your own images:

```python
import torch
from predict import predict_text

# Predict text from an image
image_path = "my_handwriting/sample1.png"
predicted_text = predict_text(image_path)
print(f"Predicted text: {predicted_text}")
```

### Image Requirements

- **Format**: PNG, JPG, or other PIL-supported formats
- **Content**: Single line of handwritten text
- **Quality**: Clear, well-contrasted handwriting works best
- **Size**: Images are automatically resized to 64px height

## 📊 Model Performance

The model uses CTC (Connectionist Temporal Classification) which allows:

- **No explicit character segmentation** required
- **Variable-length sequence** input and output
- **End-to-end training** without character-level annotations

### Character Set

The model recognizes:
- **Uppercase and lowercase letters** (A-Z, a-z)
- **Numbers** (0-9)
- **Punctuation marks** and special characters
- **Spaces** and common symbols

Total vocabulary size: ~80 unique characters (varies with training data)

## 🛠️ Customization

### Model Architecture

You can modify the model architecture in `model/htr_model.py`:

- Change CNN layer configurations
- Adjust LSTM hidden dimensions
- Modify the number of LSTM layers

### Training Parameters

Edit hyperparameters in `train.py`:

```python
BATCH_SIZE = 16      # Batch size for training
IMG_HEIGHT = 64      # Fixed image height
MAX_IMG_WIDTH = 256  # Maximum image width
EPOCHS = 25          # Number of training epochs
LR = 1e-3           # Learning rate
```

### Dataset Configuration

Modify dataset parameters in `data_loader/dataset.py`:

- Image preprocessing transforms
- Data augmentation techniques
- Character vocabulary handling

## 🧪 Testing

Test the dataset loading functionality:

```bash
python test_dataset.py
```

This script verifies:
- Dataset loading and preprocessing
- Batch generation and collation
- Label encoding consistency

## 📝 Example Usage

```python
# Load and predict on a single image
from predict import predict_text

result = predict_text("path/to/handwriting.png")
print(f"Recognized text: {result}")

# Batch prediction (modify predict.py)
images = ["image1.png", "image2.png", "image3.png"]
for img in images:
    text = predict_text(img)
    print(f"{img}: {text}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **IAM Handwriting Database**: For providing the training dataset
- **PyTorch Team**: For the excellent deep learning framework
- **CTC Loss**: Original paper by Alex Graves et al.

## 📚 References

- Graves, A., et al. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks." ICML 2006.
- IAM Handwriting Database: [https://fki.tic.heia-fr.ch/databases/iam-handwriting-database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `train.py`
2. **Dataset Loading Errors**: Verify IAM dataset structure and paths
3. **Model Loading Issues**: Ensure checkpoint file exists and matches architecture
4. **Poor Recognition**: Try training longer or with more data

### Performance Tips

- Use GPU for training (significantly faster)
- Increase batch size if you have more GPU memory
- Consider data augmentation for better generalization
- Fine-tune on domain-specific data for better results

---

**Happy Text Recognition! 🚀📝**