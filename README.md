<!-- ================= TL;DR ================= -->
<div style="border:1px solid #e5e7eb; border-radius:8px; padding:16px; background-color:#f9fafb; margin-bottom:32px;">
  <h2>📌 Recruiter TL;DR</h2>
  <ul>
    <li><strong>Problem:</strong> Most speech emotion models predict a single emotion, even though real speech often contains multiple emotions.</li>
    <li><strong>Solution:</strong> Built a multi-emotion speech recognition pipeline using <strong>Wav2Vec2 embeddings</strong> and soft-label prediction.</li>
    <li><strong>Data:</strong> IEMOCAP dataset with multi-label emotion distributions and severe class imbalance.</li>
    <li><strong>Approach:</strong> Fine-tuned Wav2Vec2 with MLP, single-head attention, and multi-head attention classification heads.</li>
    <li><strong>Key Result:</strong> Multi-head attention performed best on rare emotions and achieved the strongest class-balanced results.</li>
    <li><strong>Skills Demonstrated:</strong> Deep learning, audio embeddings, class imbalance handling, evaluation metrics, and applied ML research.</li>
  </ul>
</div>
<!-- ==================================================== -->

# 🎙️ Multi-Emotion Speech Recognition with Wav2Vec2

## Overview
This project explores whether **Wav2Vec2**, a self-supervised speech representation model, can be used to recognize **multiple emotions simultaneously** within a single speech sample. Unlike traditional emotion classifiers that predict only one dominant emotion, this project focuses on predicting **soft emotion distributions**, reflecting how people naturally express overlapping emotions.

The system is evaluated using the **IEMOCAP dataset**, which provides multi-label emotion annotations and presents realistic challenges such as class imbalance and annotator disagreement.

---

## Motivation
Most existing speech emotion recognition models:
- Predict a **single emotion**
- Struggle with **rare or subtle emotions**
- Do not account for **overlapping emotional states**

However, real-world applications such as customer support analysis, emotion-aware speech translation, and human–computer interaction require more nuanced emotion modeling. This project investigates whether **soft-label learning** and **attention-based pooling** can better capture emotional complexity.

---

## Dataset: IEMOCAP
- Approximately **12 hours** of speech data  
- **10 speakers** across scripted and improvised dialogues  
- Each utterance labeled by **6 human annotators**  
- Emotions include:  
  **Angry, Frustrated, Happy, Excited, Sad, Neutral, Fear, Disgust, Surprise**  
- Labels are provided as **emotion distributions**, not just single classes  

### Why IEMOCAP?
- Supports **multi-emotion annotations**
- Reflects real-world emotional ambiguity
- Introduces severe **class imbalance**, making it a strong testbed

---

## Key Challenges
- **Extreme class imbalance** (some emotions appear far less frequently)
- **Limited dataset size** for deep learning
- **Overlapping emotions** that are difficult to separate cleanly
- Human disagreement in emotion labeling

These challenges required careful modeling choices and evaluation strategies.

---

## Approach

### Feature Extraction
- Used **Wav2Vec2-Large (Meta)** as a pretrained speech encoder
- Applied **selective layer freezing** to preserve low-level acoustic features while adapting higher layers

### Label Strategy
- Converted labels into **soft emotion distributions**
- Used **KL-divergence loss** instead of cross-entropy to reflect emotional ambiguity

---

## Model Architectures

### 1. MLP (Baseline)
- Two-layer feedforward network
- Fast to train but limited in capturing temporal emotional cues

### 2. Single-Head Attention Pooling
- Learns to focus on emotionally relevant parts of the audio
- Improves detection of subtle emotional signals

### 3. Multi-Head Attention Pooling
- Uses multiple attention heads to capture different acoustic patterns
- Achieved the best performance on **rare emotions**
- Strongest **class-balanced results**

---

## Evaluation Metrics
- **Top-1 Accuracy** (dominant emotion)
- **Top-2 Accuracy** (captures secondary emotions)
- **Macro-F1 Score** (class-balanced evaluation)
- **Kendall’s Tau** (emotion ranking correlation)

This combination provides a more complete evaluation than accuracy alone.

---

## Results & Insights
- The MLP model performed well on common emotions but struggled with minority classes
- Attention-based models improved temporal focus and emotion ranking
- Multi-head attention provided:
  - Best Macro-F1 score
  - Improved rare emotion detection
  - More stable training behavior

A key insight is that the model often produced either very accurate or very incorrect emotion rankings, suggesting strong performance on clear emotional signals and difficulty with ambiguous cases.

---

## What I Learned
- Soft-label learning better reflects real-world emotion ambiguity
- Attention mechanisms improve emotion-focused representation learning
- Class imbalance must be explicitly addressed in emotional modeling
- Metrics beyond accuracy are critical for meaningful evaluation

---

## Technologies Used
- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Wav2Vec2**
- **scikit-learn**
- **Matplotlib / Seaborn**

---

## My Contributions
- Fine-tuned Wav2Vec2 with selective layer freezing  
- Implemented and evaluated MLP and attention-based classification heads  
- Designed preprocessing and strategies for handling rare emotion classes  
- Analyzed results using confusion matrices and rank-based metrics  

---

## Future Work
- Incorporate **visual features** for multimodal emotion recognition
- Evaluate impact on **speech-to-text and translation tasks**
- Extend
