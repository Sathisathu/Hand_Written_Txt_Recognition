
---

# Handwritten Text Recognition (HTR) with Web Application

This project provides an end-to-end deep learning system for recognizing handwritten text from images. It uses a powerful model built with PyTorch, combining Convolutional Neural Networks (CNN) for feature extraction and Bidirectional LSTMs (BiLSTM) with a Connectionist Temporal Classification (CTC) loss for sequence transcription.

The project also includes a user-friendly web application, allowing you to easily upload your own handwriting images and get instant predictions.

 <!-- Placeholder: Replace with a GIF of your webapp if you have one -->

## ✨ Features

- **Advanced Deep Learning Model**: Utilizes a CNN + BiLSTM architecture, a standard and effective approach for HTR.
- **End-to-End Training**: The model is trained with CTC loss, which eliminates the need for character-level segmentation of the training data.
- **Inference Scripts**: Predict handwritten text directly from the command line for single images.
- **Web Application**: An intuitive web interface (likely built with Flask or Streamlit) to upload images and visualize predictions in real-time.
- **Pre-trained Model**: Comes with a pre-trained model checkpoint, ready for immediate use.
- **Modular Code**: The project is well-structured, separating data loading, model architecture, training, and prediction logic into different modules.

## 🏗️ Model Architecture

The HTR model is composed of three key stages:

1.  **CNN Feature Extractor**:
    -   A series of convolutional and max-pooling layers that scan the input image and extract a sequence of high-level visual features.
2.  **BiLSTM Sequence Processor**:
    -   A 2-layer Bidirectional LSTM network that processes the feature sequences from the CNN. It captures contextual information from both past and future characters in the sequence, which is crucial for accurate handwriting recognition.
3.  **CTC Output Layer**:
    -   A final linear layer that outputs a probability distribution over all possible characters for each step in the sequence. The CTC loss function then decodes these probabilities into the final text output without requiring a direct alignment between the image frames and the characters.

## 📁 Project Structure

```
Hand_Written_Txt_Recognition/
├── checkpoints/
│   └── htr_model_best.pth      # Pre-trained best model weights
├── data/
│   └── ...                     # IAM dataset files and preparation scripts
├── data_loader/
│   └── dataset.py              # PyTorch custom dataset loader
├── model/
│   └── htr_model.py            # HTR model architecture definition
├── my_handwriting/
│   └── ...                     # Sample images for prediction
├── webapp/
│   ├── static/                 # CSS and JS files
│   ├── templates/              # HTML templates for the web app
│   └── app.py                  # Main web application script (e.g., Flask)
├── .gitignore
├── README.md                   # This file
├── predict.py                  # Script for command-line inference
├── requirements.txt            # Project dependencies
├── test_dataset.py             # Utility to test the data loader
└── train.py                    # Main model training script
```

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- Python 3.7+
- PyTorch
- A CUDA-compatible GPU is highly recommended for training.

### 2. Clone the Repository

Clone this repository to your local machine:
```bash
git clone https://github.com/Sathisathu/Hand_Written_Txt_Recognition.git
cd Hand_Written_Txt_Recognition
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file. It is recommended to do this in a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### 4. Dataset Preparation (for Training)

This project is trained on the **IAM Handwriting Database**. If you want to train the model yourself, you must download it first.

1.  **Download**: Get the `lines` dataset from the [IAM Handwriting Database website](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). You will need to register to get access.
2.  **Prepare**: After downloading and extracting the dataset, run the preparation script to organize the images and create the labels file. You will need to modify the paths in the script accordingly.

## 🔧 Usage

You can use the project in two ways: through the command line or via the web application.

### 1. Web Application (Recommended)

The web application provides the easiest way to test the model.

**To run the web app:**
```bash
# Navigate to the webapp directory
cd webapp

# Run the application
python app.py  # Or your main webapp script name
```
Now, open your web browser and go to `http://127.0.0.1:5000` (or the address shown in the terminal). You can upload an image of handwritten text and see the model's prediction.

### 2. Command-Line Prediction

To recognize text from a single image file using the command line, run the `predict.py` script. Make sure a trained model exists in the `checkpoints/` directory.

```bash
python predict.py --image_path /path/to/your/image.png
```

You can place your own sample images in the `my_handwriting/` folder for easy access.

## 🎓 Model Training

If you want to train the model from scratch on the IAM dataset:

1.  **Prepare the Dataset**: Make sure you have completed the "Dataset Preparation" step above.
2.  **Run the Training Script**:
    ```bash
    python train.py
    ```
- **Configuration**: You can modify hyperparameters such as `batch_size`, `epochs`, and `learning_rate` directly within the `train.py` script.
- **Checkpoints**: The script will automatically save the model with the best validation loss to `checkpoints/htr_model_best.pth`.
- **GPU Support**: The training script will automatically use a CUDA-enabled GPU if it is available, which significantly speeds up the training process.

## 🛠️ Customization

-   **Model Architecture**: To experiment with the model, you can modify the CNN or LSTM layers in `model/htr_model.py`.
-   **Dataset**: To train on a different dataset, you will need to create a new data loader by adapting the `data_loader/dataset.py` script to your dataset's format.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, please feel free to fork the repository, create a new feature branch, and open a pull request.

1.  **Fork** the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a **Pull Request**



## 🙏 Acknowledgments

-   This project relies on the excellent **IAM Handwriting Database**.
-   Built with the powerful **PyTorch** deep learning framework.