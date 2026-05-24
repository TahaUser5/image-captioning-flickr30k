# Image Captioning with InceptionV3 + LSTM on Flickr30k

A deep learning project that automatically generates captions for images using a CNN-LSTM architecture trained on the Flickr30k dataset.

## 🧠 Model Architecture
- **Encoder**: InceptionV3 (pretrained on ImageNet) — extracts 2048-dim image features
- **Decoder**: LSTM (512 units) with Embedding layer for sequential caption generation
- **Regularization**: Dropout (0.5) + Batch Normalization
- **Loss**: Categorical Crossentropy | **Optimizer**: Adam

## 📦 Dataset
- [Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) (~9GB, 31,000 images)
- 18,000 images used (60% of dataset) for training

## 🔧 Requirements
tensorflow
numpy
pandas
tqdm
nltk
matplotlib

## 🚀 How to Run
1. Open the notebook in Google Colab
2. Connect your Kaggle API key to download the dataset
3. Mount Google Drive for model checkpointing
4. Run cells sequentially

## 📊 Evaluation
- BLEU-1 Score used for caption quality evaluation
- Visual results displayed with predicted vs. ground truth captions

## 📁 Project Structure
├── image_captioning_flickr30k.ipynb   # Main notebook
├── README.md
└── LICENSE
