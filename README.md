# Arabic Image Caption Generator (CNN-LSTM)

This repository contains an end-to-end multimodal deep learning project for generating descriptive Arabic captions for images. The model bridges Computer Vision and Natural Language Processing by utilizing a Convolutional Neural Network (CNN) combined with a Long Short-Term Memory (LSTM) network, trained on the Arabic version of the Flickr8k dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Arabic Text Preprocessing](#arabic-text-preprocessing)
- [Requirements](#requirements)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)

## Overview
Image captioning requires an AI to understand both visual contexts and sequential text generation. This project extracts visual features from images using a pre-trained **VGG16** model and processes corresponding Arabic text sequences using an **LSTM** network. The two components are then merged into a decoder network that predicts the next word in a sequence to generate a complete, coherent Arabic caption.

## Dataset
- **Flickr8k (Arabic Version)**: The dataset consists of 8,091 images, each paired with 5 different Arabic captions. 
- The dataset is split into a training set (90%) and a testing set (10%).

## Architecture
The model is built using TensorFlow/Keras and follows an **Encoder-Decoder** architecture:
1. **Image Feature Extractor (Encoder)**: A pre-trained **VGG16** model (excluding the final classification layer) is used to extract 4096-dimensional visual feature vectors from each image. These features are passed through a Dense layer with Dropout for dimensionality reduction.
2. **Sequence Processor (Encoder)**: Arabic captions are tokenized, padded, and passed through a Word Embedding layer, followed by a Dropout layer and an **LSTM** layer to extract sequential contextual features.
3. **Decoder**: The outputs from both the image encoder and the sequence encoder are combined (added) and passed through a Dense layer. Finally, a Softmax layer predicts the probability of the next word across the entire vocabulary.

## Arabic Text Preprocessing
Handling Arabic text requires a specialized preprocessing pipeline to ensure quality training. We use [`camel-tools`](https://github.com/CAMeL-Lab/camel_tools) for:
- Removing Arabic diacritics (Dediacritization / Tashkeel removal).
- Normalizing characters (Alef, Alef Maksura, and Teh Marbuta).
- Filtering out non-Arabic characters, digits, and extra spaces.
- Appending `startseq` and `endseq` tags to mark the beginning and end boundaries of every caption.

## Requirements
Ensure you have the following libraries installed:
- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- NLTK
- `camel-tools` (for Arabic NLP)
- `rouge-score` (for evaluation)
- tqdm
- matplotlib
- Pillow (PIL)
#evaluation-metrics

The image captioning model — a VGG16 encoder paired with an LSTM decoder, trained on the Flickr8k dataset with Arabic captions — was evaluated at two training checkpoints: 20 epochs (Model 1) and 40 epochs (Model 2), using BLEU and ROUGE metrics against held-out test captions.

| Metric | Model 1 (20 epochs) | Model 2 (40 epochs) |
|---|---|---|
| BLEU-1 | 0.4447 | 0.4375 |
| BLEU-2 | 0.2535 | 0.2363 |
| ROUGE-1 F1 | 0.4973 | 0.4994 |
| ROUGE-2 F1 | 0.3294 | 0.3326 |
| ROUGE-1 Recall | 0.3312 | 0.3328 |
| ROUGE Precision (1/2/L) | ~1.00 | ~1.00 |
