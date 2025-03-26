
## 3.5.1 Model Architecture Overview: StarClassifierFusion

The **StarClassifierFusion** model is a multimodal deep learning architecture designed to leverage both spectral and Gaia astrometric data for stellar classification. The model employs parallel processing branches for different data modalities, with optional cross-modal interaction capabilities, followed by late fusion for final classification.

### Key Components

1. **Dual Modality Encoders**: 
   * **Spectral Branch**: Processes high-dimensional spectral data (3,647 features) through a dedicated encoder.
   * **Gaia Branch**: Processes lower-dimensional astrometric data (18 features) through a separate encoder.

2. **Mamba2 Backbone**:
   * Both branches utilize Mamba2 layers, a state-of-the-art sequence modeling architecture that combines the efficiency of linear recurrent models with the expressivity of attention-based models.
   * Each branch consists of 12 stacked Mamba2 layers with 2048 hidden dimensions.

3. **Cross-Attention Mechanism**:
   * Facilitates information exchange between the two modality branches.
   * **Spectra-to-Gaia Attention**: The spectral branch attends to relevant features in the Gaia branch.
   * **Gaia-to-Spectra Attention**: The Gaia branch attends to relevant features in the spectral branch.
   * Each cross-attention block employs multi-head attention with 8 heads and residual connections with layer normalization.

4. **Late Fusion**:
   * Concatenation of the processed features from both branches results in a joint representation with 4096 dimensions.
   * Layer normalization is applied to the concatenated features for stable training.

5. **Classification Head**:
   * A linear layer maps the fused representation to 55 output logits corresponding to different stellar classes.

### Technical Specifications

* **Input Dimensions**:
  * Spectral data: B × 3647 (where B is batch size)
  * Gaia data: B × 18
  
* **Embedding Dimensions**:
  * Spectral branch: 2048
  * Gaia branch: 2048
  
* **Mamba2 Configuration**:
  * State dimension: 256
  * Convolution kernel size: 4
  * Expansion factor: 2

* **Model Size**:
  * Total parameters: 741,407,287
  * Model size: 2828.24 MB

### Design Rationale

This architecture was designed to effectively capture both the fine-grained spectral features and the complementary astrometric information from Gaia, while enabling cross-modal interaction through the attention mechanism. The Mamba2 backbone was selected for its efficiency in processing sequence data compared to Transformer models, while maintaining competitive performance.

The cross-attention fusion strategy allows the model to learn which features from one modality are most relevant to the other, enabling more effective multimodal learning than simple late fusion approaches. This is particularly important for stellar classification where certain spectral features may be more informative when considered in conjunction with specific astrometric properties.
