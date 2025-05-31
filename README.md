# TIMBRE

Timbre-based Instrument Modeling for Balanced Recognition and Estimation.

[DATABASE DOWNLOAD](https://www.myqnapcloud.com/share/603e551inp2m2685r841a8ac_e30i116565pl2ooq0t1tu65z94cbhh47?session_id=2%7C1%3A0%7C10%3A1748440377%7C19%3Asession_portal_goto%7C48%3ANTQ3OTI0ZTUtODBkMS00ZmE2LTgyYTUtNWUxNGRmNjdmMjdh%7C33ba169b2ba8b91fb410ddde9566d7cb6e28bab6c04f5a7f53b6be8850175587#!/home)
## Some notes from the corse that could be helpfull
1) The most important features we have seen during the lessons are:
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
    - Spectral Spread. The Spectral Spread measures how dispersed the spectral energy is around the spectral centroid.
        - Low Spectral Spread → Most energy is concentrated around the centroid (e.g., tonal sounds like flute).
        - High Spectral Spread → The spectrum is widely distributed (e.g., noise, percussion).
    - Pitch. Refers to the fundamental frequency of a sound, often estimated using:
        - YIN: Autocorrelation-based method.
        - CREPE: Deep learning-based pitch tracker.
        - Use Case: Detecting musical notes; feeding pitch into pitch-informed CNNs.
        - Tip: Combine with timbre features for better accuracy.
    - Chroma Features. Represent the intensity of each of the 12 pitch classes (C, C#, ..., B) regardless of octave. It represents the spectral energy distribution across the 12 pitch classes of Western music (A, A#, B, C, C#, D, D#, E, F, F#, G, G#). Note that The octave information is removed, meaning that an A note at any octave is treated the same.
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
    - NON NEGATIVE MATRIX FACTORIZATION: is a matrix decomposition technique that factors a non-negative data matrix \( V \in \mathbb{R}^{F \times T}_{\geq 0} \) into two non-negative matrices: \[ V \approx WH \].
        - \( V \): Original matrix (e.g., magnitude spectrogram of an audio signal)  
        - \( W \): Basis matrix (\( F \times K \)) – contains **spectral patterns**  
        - \( H \): Activation matrix (\( K \times T \)) – contains **temporal activations**  
        - \( K \): Number of components (rank of the decomposition)  
    Can be used for source separation to Separate a mixture of sources (e.g., vocals and instruments) using learned spectral patterns and their activations.
2) The most important models seen in class are: 
    - Classical Machine Learning Models
        1. Support Vector Machine (SVM)
            - **Description**: Finds the optimal hyperplane to separate feature vectors of different classes.
            - **Use Case**: Best for MFCCs + spectral features in binary or multi-class classification.
            - **Limitations**: Does not natively support multi-label tasks.

        3. Random Forest (RF)
            - **Description**: An ensemble of decision trees; robust to noise and non-linear patterns.
            - **Use Case**: Effective with MFCC + statistical features.
            - **Files**: Introduced in Lab 2.

        4. KMeans Clustering
            - **Description**: Unsupervised clustering method for exploring structure in feature space.
            - **Use Case**: Grouping audio clips based on MFCCs, spectral features (no labels required).
            - **Files**: Used in Lab 1.

    - Deep Learning Models
        1. Multi-Layer Perceptron (MLP)
            - **Description**: An MLP consists of multiple layers of nodes (also called neurons) arranged sequentially: **Input Layers** take in the features of the input data. **Hidden Layer(s)**: One or more layers that process the input using weighted connections and non-linear activation functions, this enables the network to learn complex patterns. **Output Layer**: produces the final prediction (e.g., class probabilities or regression values). Number of neurons depends on the task (e.g., one per class for classification).
            - **Use Case**: Works with MFCCs, ZCR, spectral features; suitable for low-dimensional feature vectors.
            - **Files**: Used in Labs 2 and 3.

        2. Convolutional Neural Network (CNN)
            - **Description**: Learns spatial features from 2D audio representations like Mel spectrograms or CQT.
            - **Use Case**: Best for instrument recognition using Mel spectrograms or CQT; can be extended to multi-label classification.
            - **Files**: Applied in Labs 3 and 4.

        3. CRNN (Convolutional Recurrent Neural Network) *(Mentioned)*
            - **Description**: Combines CNN for spatial features with RNN for temporal modeling.
            - **Use Case**: Instrument activity over time; sequential labeling.

    - Specialized Models

        1. CREPE (Convolutional Representation for Pitch Estimation)
            - **Description**: Deep neural model for accurate pitch tracking.
            - **Use Case**: Extract pitch to inform classification (e.g., pitch-activity roll).
            - **Files**: Introduced in MIR Part 5 (II).
3) To tackle the task of musical instrument recognition, we choose to use a Convolutional Neural Network (CNN) in combination with MFCC (Mel-Frequency Cepstral Coefficients) as input features. CNNs are particularly well-suited for analyzing 2D time–frequency representations like spectrograms, allowing them to learn local patterns that correspond to timbral and temporal characteristics of instruments. MFCCs, which compress the spectral shape of audio signals into a compact and perceptually meaningful form, are widely used in audio classification due to their ability to capture timbre-related information efficiently. Together, MFCCs provide a structured, low-dimensional input that complements the CNN’s strength in pattern recognition, enabling robust classification even in polyphonic settings where multiple instruments may be active simultaneously.
4) Data preparation:
    - things about data label. We don't need.
    - possibility to apply a data augmentation tecnich. I don't think we need it ahhahah.
5) Evaluation of the model: is a fundamental part of system development in machine learning and signal processing. It involves assessing how well a system performs on new, unseen data.
    - **Cross Validation**: is a crucial technique for evaluating model performance and ensuring robustness, especially when data is limited. It enables the use of all available data for both training and validation. Allows efficient use of all data by rotating through different training and validation splits. The data is divided into k folds, creating k training/validation splits. Each fold serves once as a validation set, while the remaining folds are used for training. The overall performance is computed as the average of the metric scores across all folds.
    - **Evaluation Measures**: is the process of comparing a system's outputs with reference annotations on test data, using various performance metrics. The choice of metric depends on the task, class distribution, and desired performance perspective.
        - Intermediate Results per Class. Given a class \( c \), the outcomes can be:
            - *True Positive (TP)*: System and reference both say class \( c \) is present.
            - *True Negative (TN)*: System and reference both say class \( c \) is not present.
            - *False Positive (FP)*: System says class \( c \) is present; reference says it's not.
            - *False Negative (FN)*: System says class \( c \) is absent; reference says it's present.
        For a good visualization can be used a multiclass Confusion Matrix, is square matrix where rows are actual classes and columns are predicted classes. Each cell \( (i, j) \) indicates the number of examples with true label i predicted as class j.
        - Derived Metrics: from TP, TN, FP, and FN, we can calculate:
            - *Recall / Sensitivity / True Positive Rate (TPR)*:
            $$ \text{Recall} = \frac{TP}{TP + FN} $$
            - *Precision*:
            $$
            \text{Precision} = \frac{TP}{TP + FP}
            $$
            - *False Positive Rate (FPR)*:
            $$
            \text{FPR} = \frac{FP}{FP + TN}
            $$
            - *Specificity / True Negative Rate (TNR)*:
            $$
            \text{Specificity} = \frac{TN}{TN + FP}
            $$
            - *Accuracy (ACC)*: is the roportion of correct predictions over all predictions. Easy to understand and compute. Misleading for imbalanced data: High TN count can mask poor TP performance.
            $$
            \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
            $$
            - *F1 Score (F1)*: is the harmonic mean of precision and recall. Better for imbalanced datasets. Assumes equal importance of precision and recall.
            $$
            F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
            $$
        -  Averaging in Multiclass Scenarios
            - **Micro-Averaging**:
                - Aggregates TP, FP, FN **globally**.
                - Dominated by **frequent classes**.

            - **Macro-Averaging**:
                - Calculates metrics **per class**, then averages them.
                - Gives equal weight to each class.
                - Risk: **Undefined recall** if a class has \( TP + FN = 0 \).
            - **Hybrid Averaging**:
                - Uses **weighted averages** based on class importance or frequency.
        - Balanced accuracy: si designed for imbalanced datasets. Average of sensitivity and specificity:
            $$
            \text{Balanced Accuracy} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)
            $$
    - **Precision-Recall (P-R) Curve**. Many classifiers rely on a threshold applied to an internal decision variable (e.g., neural network output). Lowering the threshold increases recall but typically reduces precision. The P-R curve plots precision vs. recall across different thresholds. Useful for imbalanced classification tasks, where high recall and high precision are both important.
    - **ROC Curve (Receiver Operating Characteristic)**: plots the True Positive Rate (Recall) against the False Positive Rate (FPR)for various threshold values. Good for binary classification problems and not easily generalizable to multiclass tasks.

## Pipeline

1) Import audio data. Use panda to prepare the dataframe.
2) Import associated annotations. Use df["label"].
3) Normalize the data as needed:
    - Low pass filter
    - Apply a windowing    
4) Split the data into training (Fit model parameters. More training data → Better model performance.), validation (optional: Tune hyperparameters, prevent overfitting. More validation data → More accurate performance estimates.), and test sets (Final unbiased evaluation  ). 70%, 15%, 15% or 70%, 20%, 10%.
5) The best feature for thi stask is the *Mel Spectrogram*. A time–frequency representation of the audio signal where the frequency axis is transformed to the Mel scale, which aligns with human auditory perception. It represents how the spectral content of a sound evolves over time, emphasizing frequencies most relevant to human hearing.
        - Strengths: Captures both timbre and temporal patterns.
        - Use Case: Input for CNNs in classification tasks.
        - Visual: Looks like a heatmap with time (x-axis) and frequency (y-axis).

6) Set up and configure the modeling algorithm (e.g., classifier or regressor). For sure we need a classification model and not a regression model. CNN 2D is the best. Note that eith CNN the NMF is not needed. Indeed in traditional audio processing, **Non-negative Matrix Factorization (NMF)** is a popular method for **source separation**, where a spectrogram is decomposed into spectral templates and activation patterns. However, with the advent of **deep learning**, especially **Convolutional Neural Networks (CNNs)**, the need for hand-crafted techniques like NMF has significantly reduced. CNNs **learn directly from data** without requiring explicit factorization.  They can model **non-linear relationships** and **context** in time and frequency. With large datasets, CNNs outperform NMF in **source separation accuracy**. End-to-end models (e.g., waveform in → separated sources out) are now common. When Is NMF Still Useful?
    - In **low-data scenarios**
    - For **interpretable models**
    - When computational resources are limited
    - As a **baseline** or **preprocessing** method for hybrid models
