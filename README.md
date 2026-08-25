# Speech Emotion-Aware Conversational Agent

## Speech Emotion-Based Response Generation

An emotion-aware conversational system that identifies the emotional state expressed in a user's speech and uses the detected emotion to condition conversational response generation.

## Objective

The primary objective is to develop a speech-driven conversational pipeline that combines speech emotion recognition with emotion-conditioned language generation.

## Architecture

Speech
↓
Audio Preprocessing
↓
Mel Spectrogram
↓
CNN-BiLSTM Emotion Classifier
↓
Detected Emotion
↓
Transcript + Emotion
↓
LLM Response Generation
↓
Emotion-Aware Response

## Dataset

### Primary Target Dataset

IEMOCAP (Interactive Emotional Dyadic Motion Capture)

https://sail.usc.edu/iemocap/

IEMOCAP contains approximately 12 hours of audiovisual recordings from 10 actors participating in dyadic interactions.

For the initial prototype, an accessible RAVDESS speech subset was used to validate the speech emotion recognition pipeline.

### Prototype Dataset

RAVDESS

The prototype uses four emotion classes:

- Angry
- Happy
- Sad
- Neutral

## Methodology

1. Audio preprocessing
2. Resampling to 16 kHz
3. Mel-spectrogram extraction
4. CNN-based acoustic feature extraction
5. BiLSTM temporal modeling
6. Emotion classification using Softmax
7. Emotion-conditioned prompt construction
8. LLM-based response generation

## Model

CNN-BiLSTM

### Training Parameters

- Sampling rate: 16 kHz
- Mel filters: 64
- Batch size: 32
- Optimizer: Adam
- Learning rate: 0.001
- Loss: Cross Entropy
- Dropout: 0.4
- Epochs: 20
- Gradient clipping: 1.0
- Learning-rate scheduling: ReduceLROnPlateau

## Evaluation

The emotion recognition model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The language generation component can additionally be evaluated using:

- Perplexity
- BLEU
- ROUGE-L
- Human evaluation of emotional appropriateness

## Limitations

- Initial prototype is not trained directly on IEMOCAP.
- Emotion labels can be subjective.
- Emotion classification errors can propagate into response generation.
- Perplexity does not directly measure empathy or emotional appropriateness.
- Acted emotional speech may differ from spontaneous real-world speech.

## Future Work

- Train directly on IEMOCAP
- Real-time microphone input
- Multimodal emotion recognition
- Continuous valence/arousal prediction
- Conversational memory
- Emotion-aware fine-tuning
- Multilingual support
- Emotional speech synthesis

## Project Structure

```text
speech-emotion-aware-agent/
│
├── README.md
├── requirements.txt
├── notebooks/
│   └── Speech_Emotion_Aware_Conversational_Agent.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── response_generator.py
│
├── models/
│   └── emotion_cnn_bilstm.pt
│
├── results/
│   ├── training_loss.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   └── metrics.csv
│
└── docs/
    └── architecture.png
