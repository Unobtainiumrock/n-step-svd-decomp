# Mathematical Flow Diagram - Residualized Tone Space

## Overview
This document maps the exact mathematical objects, dimensions, and transformations in the residualized tone space system, showing the linear flow of data from raw embeddings to tone scores.

**Key Insight**: This system is built on a **two-stage application of PCA, implemented via SVD** for signal separation.

## Mathematical Objects and Dimensions

### 1. Raw Embeddings
**Object**: $E \in \mathbb{R}^{N \times 1536}$
- **Dimensions**: $(80420, 1536)$
- **File**: `data/embeddings/ugc-posts_COMPLETE_embeddings.h5`
- **Content**: Raw OpenAI text-embedding-3-small vectors
- **Assumption**: Each row is a post embedding, columns are semantic dimensions

### 2. Mean-Centered Embeddings
**Object**: $E_{centered} \in \mathbb{R}^{N \times 1536}$
- **Dimensions**: $(80420, 1536)$
- **Content**: $E_{centered} = E - \bar{e}$ where $\bar{e} = \frac{1}{N} \sum_{i=1}^{N} E_i$
- **Assumption**: Mean-centering ensures PCA captures variance around data center

### 3. Topic Basis Vectors (Right Singular Vectors from SVD)
**Object**: $T \in \mathbb{R}^{160 \times 1536}$
- **Dimensions**: $(160, 1536)$
- **File**: `data/models/optimal_topic_vectors.npy`
- **Content**: **Right singular vectors (V^T)** from SVD of centered embeddings
- **Assumption**: 160 components capture 66.7% of total variance
- **Property**: Rows of $T$ are orthonormal ($T \cdot T^T = I_{160}$)
- **Note**: These are the **principal components** - directions of maximum variance in embedding space

### 4. Topic Projection Matrix
**Object**: $P_T \in \mathbb{R}^{1536 \times 1536}$
- **Dimensions**: $(1536, 1536)$
- **File**: `data/models/optimal_projection_matrix.npy`
- **Content**: $P_T = T^T T$ (direct construction from orthonormal basis)
- **Assumption**: Idempotent projection matrix ($P_T^2 = P_T$)
- **Property**: Since $T$ is orthonormal, $P_T = T^T T$ is the correct projection

### 5. Topic Scores
**Object**: $S_T \in \mathbb{R}^{N \times 160}$
- **Dimensions**: $(80420, 160)$
- **Content**: $S_T = E \cdot T^T$ (coordinates in topic subspace)
- **Assumption**: Projection of embeddings onto 160-D topic space

### 6. Topic Projections
**Object**: $E_T \in \mathbb{R}^{N \times 1536}$
- **Dimensions**: $(80420, 1536)$
- **File**: `data/models/optimal_topic_projections.npy`
- **Content**: $E_T = S_T \cdot T = (E \cdot T^T) \cdot T$
- **Assumption**: Topic components reconstructed in original space

### 7. Residual Vectors
**Object**: $R \in \mathbb{R}^{N \times 1536}$
- **Dimensions**: $(80420, 1536)$
- **File**: `data/models/optimal_residual_vectors.npy`
- **Content**: $R = E - E_T = E - (E \cdot T^T) \cdot T$
- **Assumption**: Tone information after topic removal

### 8. Tone Basis Vectors (Right Singular Vectors from SVD)
**Object**: $A \in \mathbb{R}^{130 \times 1536}$
- **Dimensions**: $(130, 1536)$
- **File**: `data/models/optimal_tone_axes.npy`
- **Content**: **Right singular vectors (V^T)** from SVD of residual vectors
- **Assumption**: 130 components capture 38.2% of residual variance
- **Property**: Rows of $A$ are orthonormal ($A \cdot A^T = I_{130}$)
- **Note**: These are the **principal components** of the residual space

### 9. Tone Scores
**Object**: $S_A \in \mathbb{R}^{N \times 130}$
- **Dimensions**: $(80420, 130)$
- **Content**: $S_A = R \cdot A^T$ (coordinates in tone subspace)
- **Assumption**: Projection of residuals onto 130-D tone space

### 10. User Preference Vector
**Object**: $w \in \mathbb{R}^{1536}$
- **Dimensions**: $(1536,)$
- **File**: `data/calibrations/demo_user_001_calibration.json`
- **Content**: Learned preference direction in full embedding space
- **Assumption**: Ridge-regularized solution from 5 user choices

### 11. Decomposed User Preferences
**Object**: $w_T, w_A \in \mathbb{R}^{1536}$
- **Content**: 
  - **Topic Component**: $w_T = (w \cdot T^T) \cdot T$
  - **Tone Component**: $w_A = (w \cdot A^T) \cdot A$
- **Assumption**: User preferences can be separated into topic and tone components

## Linear Transformations Flow

### Stage 1: Mean-Centering (PCA Requirement)
```
E (80420×1536) → Calculate Mean → E_centered = E - mean (80420×1536)
```
- **Transformation**: Subtract mean embedding from each row
- **Assumption**: PCA requires mean-centered data for covariance analysis
- **Output**: Centered embeddings for proper PCA

### Stage 2: Topic Subspace Construction (PCA via SVD)
```
E_centered (80420×1536) → SVD → T (160×1536)
```
- **Transformation**: SVD decomposition of centered embedding matrix
- **Assumption**: First 160 **right singular vectors (V^T)** are principal components
- **Output**: Orthonormal topic basis vectors (principal components)
- **Note**: $E_{centered} = U \Sigma V^T$, we extract $V^T$ as topic basis

### Stage 3: Topic Projection Matrix (Simplified)
```
T (160×1536) → P_T = T^T T (1536×1536)
```
- **Transformation**: Direct construction from orthonormal basis
- **Assumption**: Since $T$ is orthonormal, $P_T = T^T T$ is correct
- **Output**: Idempotent projection matrix

### Stage 4: Topic Scoring and Projection (Efficient)
```
E (80420×1536) × T^T (1536×160) → S_T (80420×160)
S_T (80420×160) × T (160×1536) → E_T (80420×1536)
```
- **Transformation**: Two-step projection via basis vectors
- **Assumption**: More efficient than single large matrix multiplication
- **Output**: Topic components in original space

### Stage 5: Residual Calculation
```
E (80420×1536) - E_T (80420×1536) → R (80420×1536)
```
- **Transformation**: Element-wise subtraction
- **Assumption**: Residuals contain tone information orthogonal to topics
- **Output**: Tone space vectors

### Stage 6: Tone Subspace Discovery (PCA via SVD)
```
R (80420×1536) → SVD → A (130×1536)
```
- **Transformation**: SVD decomposition of residual matrix
- **Assumption**: First 130 **right singular vectors (V^T)** are principal components
- **Output**: Orthonormal tone basis vectors (principal components)
- **Note**: $R = U \Sigma V^T$, we extract $V^T$ as tone basis

### Stage 7: Tone Score Calculation
```
R (80420×1536) × A^T (1536×130) → S_A (80420×130)
```
- **Transformation**: Matrix multiplication for tone scoring
- **Assumption**: Linear combination captures tone preferences
- **Output**: Tone scores for each post

### Stage 8: User Preference Learning
```
X (5×1536) → Ridge Regression → w (1536×1)
```
- **Transformation**: Ridge-regularized linear regression
- **Assumption**: 5 choices sufficient with regularization
- **Output**: Preference vector in full embedding space

### Stage 9: Preference Decomposition
```
w (1536×1) × T^T (1536×160) → w_T_coords (160×1)
w_T_coords (160×1) × T (160×1536) → w_T (1536×1)
w (1536×1) × A^T (1536×130) → w_A_coords (130×1)
w_A_coords (130×1) × A (130×1536) → w_A (1536×1)
```
- **Transformation**: Decompose user preferences into topic and tone components
- **Assumption**: User preferences can be separated into orthogonal components
- **Output**: Topic and tone components of user preferences

## File Mapping to Mathematical Objects

| Mathematical Object | File Location | Dimensions | Purpose |
|-------------------|---------------|------------|---------|
| $E$ | `data/embeddings/ugc-posts_COMPLETE_embeddings.h5` | $(80420, 1536)$ | Raw embeddings |
| $T$ | `data/models/optimal_topic_vectors.npy` | $(160, 1536)$ | Topic principal components (V^T) |
| $P_T$ | `data/models/optimal_projection_matrix.npy` | $(1536, 1536)$ | Topic projection matrix |
| $E_T$ | `data/models/optimal_topic_projections.npy` | $(80420, 1536)$ | Topic components |
| $R$ | `data/models/optimal_residual_vectors.npy` | $(80420, 1536)$ | Residual vectors |
| $A$ | `data/models/optimal_tone_axes.npy` | $(130, 1536)$ | Tone principal components (V^T) |
| $w$ | `data/calibrations/demo_user_001_calibration.json` | $(1536,)$ | User preferences |

## Production Files (Float32 Optimized)

| Mathematical Object | File Location | Dimensions | Purpose |
|-------------------|---------------|------------|---------|
| $P_T$ | `data/reports/optimal_160_projection_matrix.npy` | $(1536, 1536)$ | Production projection |
| $T$ | `data/reports/optimal_160_topic_vectors.npy` | $(160, 1536)$ | Production topic basis |
| $A$ | `data/reports/optimal_130_tone_axes.npy` | $(130, 1536)$ | Production tone basis |
| $E_T$ | `data/reports/optimal_160_topic_projections.npy` | $(80420, 1536)$ | Production topic projections |
| $R$ | `data/reports/optimal_residual_vectors.npy` | $(80420, 1536)$ | Production residuals |

## Key Assumptions at Each Stage

### Stage 1: Mean-Centering
- **Assumption**: PCA requires mean-centered data for proper variance analysis
- **Validation**: Ensures principal components capture variance around data center
- **Risk**: Without centering, first principal component may be dominated by mean

### Stage 2: Topic Discovery (PCA via SVD)
- **Assumption**: SVD captures semantic topic structure as principal components
- **Validation**: 66.7% variance explained by 160 components
- **Risk**: Topics may not align with human concepts
- **Note**: Using **right singular vectors (V^T)** as principal components

### Stage 3: Projection Matrix
- **Assumption**: Direct construction $P_T = T^T T$ is correct for orthonormal basis
- **Validation**: $P_T^2 = P_T$ (idempotent property)
- **Risk**: Numerical instability with large matrices

### Stage 4: Topic Projection
- **Assumption**: Two-step projection is more efficient than single large multiplication
- **Validation**: $E_T = (E \cdot T^T) \cdot T$ gives same result as $E \cdot P_T$
- **Risk**: Slightly more complex but computationally faster

### Stage 5: Residual Calculation
- **Assumption**: Residuals contain tone information orthogonal to topics
- **Validation**: $R \cdot T^T \approx 0$ (orthogonality)
- **Risk**: Topic-tone separation may be incomplete

### Stage 6: Tone Discovery (PCA via SVD)
- **Assumption**: SVD on residuals captures tone structure as principal components
- **Validation**: 38.2% of residual variance explained
- **Risk**: Tone axes may not align with human preferences
- **Note**: Using **right singular vectors (V^T)** as principal components

### Stage 7: Tone Scoring
- **Assumption**: Linear combination captures tone preferences
- **Validation**: Tone scores correlate with human judgments
- **Risk**: Non-linear tone relationships lost

### Stage 8: Preference Learning
- **Assumption**: Ridge regularization prevents overfitting
- **Validation**: 5 choices sufficient with $\alpha = 1.0$
- **Risk**: Underfitting with too few examples

### Stage 9: Preference Decomposition
- **Assumption**: User preferences can be separated into orthogonal topic and tone components
- **Validation**: $||w_T||$ and $||w_A||$ provide interpretable user preference analysis
- **Risk**: Non-orthogonal preferences may not separate cleanly

## Mathematical Properties

### Orthogonality
- **Topic-Tone**: $T \cdot A^T \approx 0$ (cross-correlation = 1.4e-16)
- **Topic Basis**: $T \cdot T^T = I_{160}$ (orthonormal)
- **Tone Basis**: $A \cdot A^T = I_{130}$ (orthonormal)

### Projection Properties
- **Idempotent**: $P_T^2 = P_T$
- **Symmetric**: $P_T = P_T^T$
- **Rank**: $\text{rank}(P_T) = 160$

### Variance Decomposition
- **Total Variance**: 100%
- **Topic Variance**: 66.7%
- **Tone Variance**: 12.7% (38.2% of residuals)
- **Unexplained**: 20.6%

### User Preference Analysis
- **Topic Preference Magnitude**: $||w_T||$
- **Tone Preference Magnitude**: $||w_A||$
- **Preference Balance**: $\frac{||w_T||}{||w_A||}$ (topic vs tone preference ratio)

## SVD Decomposition Details

### Topic Space SVD
- **Decomposition**: $E_{centered} = U_T \Sigma_T V_T^T$
- **Extracted Components**: $T = V_T^T$ (right singular vectors)
- **Interpretation**: $V_T^T$ contains directions of maximum variance in embedding space
- **Dimensions**: $T \in \mathbb{R}^{160 \times 1536}$

### Tone Space SVD
- **Decomposition**: $R = U_A \Sigma_A V_A^T$
- **Extracted Components**: $A = V_A^T$ (right singular vectors)
- **Interpretation**: $V_A^T$ contains directions of maximum variance in residual space
- **Dimensions**: $A \in \mathbb{R}^{130 \times 1536}$

### Why Right Singular Vectors?
- **Principal Components**: Right singular vectors (V^T) are the principal components
- **Feature Space**: They represent directions in the 1536-dimensional embedding space
- **Variance Maximization**: They capture directions of maximum variance across all posts
- **Orthonormal Basis**: They form an orthonormal basis for the subspace

## Computational Complexity

| Operation | Complexity | Memory | Time |
|-----------|------------|--------|------|
| Mean-Centering | $O(N \cdot 1536)$ | $O(1536)$ | ~1s |
| SVD (Topic) | $O(N \cdot 1536^2)$ | $O(1536^2)$ | ~20s |
| Projection Matrix | $O(1536^2 \cdot 160)$ | $O(1536^2)$ | ~1s |
| Topic Projection | $O(N \cdot 1536 \cdot 160)$ | $O(N \cdot 160)$ | ~3s |
| Residual Calculation | $O(N \cdot 1536)$ | $O(N \cdot 1536)$ | ~2s |
| SVD (Tone) | $O(N \cdot 1536^2)$ | $O(1536^2)$ | ~15s |
| Tone Scoring | $O(N \cdot 1536 \cdot 130)$ | $O(N \cdot 130)$ | ~3s |
| Preference Decomposition | $O(1536 \cdot 160 + 1536 \cdot 130)$ | $O(1536)$ | ~0.1s |

## Summary

The system performs a **two-stage PCA decomposition via SVD**:

1. **Topic Stage**: $E \rightarrow \text{mean-center} \rightarrow \text{SVD} \rightarrow V_T^T \rightarrow \text{project} \rightarrow E_T \rightarrow \text{subtract} \rightarrow R$
2. **Tone Stage**: $R \rightarrow \text{SVD} \rightarrow V_A^T \rightarrow \text{score} \rightarrow S_A$
3. **Preference Analysis**: $w \rightarrow \text{decompose} \rightarrow w_T, w_A$

**Key Improvements**:
- **Mean-centering** ensures proper PCA
- **Direct projection matrix construction** eliminates unnecessary QR decomposition
- **Efficient two-step projection** reduces computational complexity
- **Preference decomposition** enables interpretable user analysis
- **Correct SVD usage**: Using **right singular vectors (V^T)** as principal components

Each stage uses **linear transformations** (matrix multiplications and projections) to separate topic and tone information, with **Ridge regularization** handling the underdetermined preference learning problem.

**Critical Note**: The system correctly uses **right singular vectors (V^T)** from SVD as the principal components, not left singular vectors (U). This is the standard and correct approach for PCA via SVD.
