# TIMBRE

Timbre-based Instrument Modeling for Balanced Recognition and Estimation.

[DATABASE DOWNLOAD](https://www.myqnapcloud.com/share/603e551inp2m2685r841a8ac_e30i116565pl2ooq0t1tu65z94cbhh47?session_id=2%7C1%3A0%7C10%3A1748440377%7C19%3Asession_portal_goto%7C48%3ANTQ3OTI0ZTUtODBkMS00ZmE2LTgyYTUtNWUxNGRmNjdmMjdh%7C33ba169b2ba8b91fb410ddde9566d7cb6e28bab6c04f5a7f53b6be8850175587#!/home)

## Pipeline

1) Import audio data. Use panda to prepare the dataframe.
2) Import associated annotations. Use df["label"].
3) Normalize the data as needed. ? librosa.load() returns floating-point values in the range [-1.0, 1.0], which is already normalized for most models.
4) Split the data into training, validation (optional), and test sets. 70%, 15%, 15% or 70%, 20%, 10%.
5) The feature we have seen in the lessons are:
    - Mel Spectrogram. A time–frequency representation of the audio signal where the frequency axis is transformed to the Mel scale, which aligns with human auditory perception. It represents how the spectral content of a sound evolves over time, emphasizing frequencies most relevant to human hearing.
        - Strengths: Captures both timbre and temporal patterns.
        - Use Case: Input for CNNs in classification tasks.
        - Visual: Looks like a heatmap with time (x-axis) and frequency (y-axis).
    - MFCC (Mel-Frequency Cepstral Coefficients). Derived from the Mel spectrogram, MFCCs capture the broad spectral shape (timbre) of a sound by applying a discrete cosine transform (DCT) to log-Mel energies.
        - Strengths: Effective at modeling timbre, compact representation.
        - Use Case: Traditional classifiers (SVM, RF), RNNs.
        - Tip: Often averaged over time or stacked as sequences.
    - Zero Crossing Rate (ZCR). Measures how often the audio waveform crosses zero amplitude. Higher ZCR indicates a more noisy or percussive signal.
        - Strengths: Simple and efficient.
        - Use Case: Detecting percussive or unvoiced components (e.g., hi-hat vs flute).
    - Spectral Flatness: Distinguishes noise-like vs tone-like signals. Quantifies how flat or “noisy” the spectrum is by comparing the geometric mean to the arithmetic mean of spectral magnitudes.
        - High flatness = white noise-like.
        - Low flatness = tonal or harmonic.
        - Use Case: Differentiating instruments with pure tones (e.g., flute) vs noise (e.g., snare).
    - Spectral Centroid. Indicates the “center of mass” of the spectrum — higher centroid = brighter sound.
        - Strengths: Strongly linked to perceived brightness or sharpness.
        - Use Case: Classifying between mellow (e.g., cello) and bright instruments (e.g., trumpet).
    - Pitch. Refers to the fundamental frequency of a sound, often estimated using:
        - YIN: Autocorrelation-based method.
        - CREPE: Deep learning-based pitch tracker.
        - Use Case: Detecting musical notes; feeding pitch into pitch-informed CNNs.
        - Tip: Combine with timbre features for better accuracy.
    - Chroma Features. Represent the intensity of each of the 12 pitch classes (C, C#, ..., B) regardless of octave.
        - Strengths: Useful for modeling harmony and tonal content.
        - Use Case: Works well with piano, strings, guitar.
    - Constant-Q Transform (CQT). Similar to a spectrogram but with logarithmic frequency bins, mimicking musical note spacing.
        - Strengths: Better pitch resolution at low frequencies.
        - Use Case: Instrument recognition where note identity matters.
        - Advantage over Mel: More aligned with musical scales.
    - Tempogram. Represents tempo changes over time by computing the autocorrelation of onset strength.
        - Strengths: Captures rhythmic patterns.
        - Use Case: Useful for instruments with repeating rhythmic structure (e.g., drums, bass).
    - Pitch-Activity Roll: Pitch + time + instrument info; used in pitch-informed models. Matrix that encodes which pitches are active over time, often from MIDI annotations or pitch tracking.
        - Shape: (pitches × time frames).
        - Use Case: Multi-instrument scenes, polyphonic pitch modeling.
        - Tip: Used in dual-branch CNNs alongside timbral features.
    - Embeddings: Learned features via CNNs or metric learning (e.g., triplet networks). High-level, learned representations from deep models (e.g., CNNs, triplet networks).
        - Types: OpenL3, PANNs, VGGish, or custom triplet-loss embeddings.
        - Use Case: Transfer learning, music similarity, or instrument retrieval.
        - Benefit: Capture abstract characteristics beyond hand-crafted features.
6) Set up and configure the modeling algorithm (e.g., classifier or regressor).

   - Classical Machine Learning Models
      1. Support Vector Machine (SVM)
           - **Description**: Finds the optimal hyperplane to separate feature vectors of different classes.
           - **Use Case**: Best for MFCCs + spectral features in binary or multi-class classification.
           - **Limitations**: Does not natively support multi-label tasks.

      2. Multi-Layer Perceptron (MLP)
           - **Description**: A fully connected feedforward neural network
           - **Use Case**: Works with MFCCs, ZCR, spectral features; suitable for low-dimensional feature vectors.
           - **Files**: Used in Labs 2 and 3.

      3. Random Forest (RF)
           - **Description**: An ensemble of decision trees; robust to noise and non-linear patterns.
           - **Use Case**: Effective with MFCC + statistical features.
           - **Files**: Introduced in Lab 2.

      4. KMeans Clustering
           - **Description**: Unsupervised clustering method for exploring structure in feature space.
           - **Use Case**: Grouping audio clips based on MFCCs, spectral features (no labels required).
           - **Files**: Used in Lab 1.

   - Deep Learning Models

      1. Convolutional Neural Network (CNN)
           - **Description**: Learns spatial features from 2D audio representations like Mel spectrograms or CQT.
           - **Use Case**: Best for instrument recognition using Mel spectrograms or CQT; can be extended to multi-label classification.
           - **Files**: Applied in Labs 3 and 4.

      2. CRNN (Convolutional Recurrent Neural Network) *(Mentioned)*
           - **Description**: Combines CNN for spatial features with RNN for temporal modeling.
           - **Use Case**: Instrument activity over time; sequential labeling.

   - Specialized Models

      1. CREPE (Convolutional Representation for Pitch Estimation)
           - **Description**: Deep neural model for accurate pitch tracking.
           - **Use Case**: Extract pitch to inform classification (e.g., pitch-activity roll).
           - **Files**: Introduced in MIR Part 5 (II).

To tackle the task of musical instrument recognition, we choose to use a Convolutional Neural Network (CNN) in combination with MFCC (Mel-Frequency Cepstral Coefficients) as input features. CNNs are particularly well-suited for analyzing 2D time–frequency representations like spectrograms, allowing them to learn local patterns that correspond to timbral and temporal characteristics of instruments. MFCCs, which compress the spectral shape of audio signals into a compact and perceptually meaningful form, are widely used in audio classification due to their ability to capture timbre-related information efficiently. Together, MFCCs provide a structured, low-dimensional input that complements the CNN’s strength in pattern recognition, enabling robust classification even in polyphonic settings where multiple instruments may be active simultaneously.
