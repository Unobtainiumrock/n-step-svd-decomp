> Historical proposal: semantic labels, independence, sample efficiency, and drift claims below are unvalidated. See ../../README.md and ../HANDOFF.md for current status.

# Face ID for Tone

**Discovering content preferences through residualized embedding spaces**

A full-stack system that decomposes 1,536-dimensional OpenAI embeddings using SVD to separate topic from tone. The "residualized tone space" isolates stylistic preferences independent of subject matter. A 9-round convergent A/B testing interface calibrates user preferences across 80k+ posts, using Pinecone for vector search and Supabase for real-time state.

## Key Features

- **Novel SVD decomposition**: 160 topic + 130 tone dimensions from 1,536D embeddings
- **Residualized tone space** eliminates topic bias from preference signal
- **Interactive convergent search** with 9-round A/B calibration
- **80k+ post corpus** with real-time vector similarity search

## Tech Stack

Python | FastAPI | React | OpenAI Embeddings | SVD/PCA | Pinecone | Supabase

## Dataset

The system operates on a corpus of **80,000+ posts**, each embedded into 1,536-dimensional vectors via OpenAI's embedding API. These embeddings serve as the input to the SVD decomposition pipeline described below. The dataset is not included in this repository due to size constraints and API usage considerations, but the decomposition algorithm and preference scoring system are fully described here and implemented in the companion application.

## A Note on Machine Learning

This project deliberately avoids traditional ML classifiers (neural networks, gradient-boosted trees, etc.) for preference discovery. Instead, it relies entirely on **linear algebraic decomposition** (SVD) and **geometric operations** (projection, residualization) in embedding space.

The rationale:

1. **Interpretability**: Each SVD component corresponds to a concrete, orthogonal subspace with a semantic label (topic, tone, etc.). There are no opaque learned weights to interpret.
2. **Sample efficiency**: The preference scoring algorithm (Phase 2 below) works from as few as 5 user selections. ML models typically require orders of magnitude more labeled data to learn meaningful preference signals.
3. **Composability**: Because the decomposition produces orthogonal subspaces, preference scores for topic and tone are genuinely independent. An ML model trained end-to-end would entangle these signals, making it impossible to ask "does this user like the *tone* of this post, regardless of topic?"
4. **No training loop**: The SVD is computed once offline. User preference profiles are computed in real-time via matrix-vector products -- no model retraining, no drift, no feedback loops.

A future direction could involve training individual VAEs per subspace to learn nonlinear representations within each decomposed layer (see the end of this document), but the current system demonstrates that meaningful preference separation is achievable with purely linear methods.

---

## N-Step Orthogonal Decomposition Algorithm

#### **Initialization**

1.  **Input Data**: The raw embedding matrix $E \in \mathbb{R}^{N \times d}$, where $N$ is the number of samples and $d$ is the embedding dimension (e.g., 1536).
2.  **Initial Residual**: Set the first residual matrix $R^{(0)}$ to be the input data.
    $$R^{(0)} = E$$

#### **Iterative Decomposition Loop**

For each step $i = 1, 2, ..., n$:

1.  **Decompose the Residual**: Perform Singular Value Decomposition (SVD) on the residual from the previous step, $R^{(i-1)}$.
    $$U^{(i)}, \Sigma^{(i)}, (V^{(i)})^T = \text{SVD}(R^{(i-1)})$$
2.  **Determine Rank**: Analyze the singular values in $\Sigma^{(i)}$ (e.g., using an elbow plot or cumulative variance threshold) to determine the optimal rank $k_i$ for the current layer of meaning (e.g., "Topic," "Tone," "Formality," etc.).
3.  **Extract Basis Vectors**: The basis for this layer, $A^{(i)}$, consists of the top $k_i$ right singular vectors from $V^{(i)}$.
    $$A^{(i)} = V^{(i)}[1:k_i, :] \in \mathbb{R}^{k_i \times d}$$
4.  **Create Projection Matrix**: Construct the idempotent projection matrix $P^{(i)}$ for the current subspace.
    $$P^{(i)} = (A^{(i)})^T A^{(i)} \in \mathbb{R}^{d \times d}$$
5.  **Isolate Component**: Project the previous residual $R^{(i-1)}$ onto this new subspace to get the component matrix $E^{(i)}$ for the current layer. This is the "pure" representation of this layer's meaning for each sample.
    $$E^{(i)} = R^{(i-1)} \cdot P^{(i)}$$
6.  **Calculate Next Residual**: Subtract the isolated component from the previous residual to prepare for the next iteration. This new residual $R^{(i)}$ is now, by definition, orthogonal to the subspace defined by $A^{(i)}$.
    $$R^{(i)} = R^{(i-1)} - E^{(i)}$$

#### **Final Output**

After $n$ steps, the decomposition yields:

* A set of **mutually orthogonal component matrices**: $\{E^{(1)}, E^{(2)}, ..., E^{(n)}\}$, where the row space of any $E^{(i)}$ is orthogonal to the row space of any $E^{(j)}$ for $i \neq j$.
* A **final residual matrix** $R^{(n)}$, representing unexplained variance.

The original embedding matrix is the sum of these components:
$$E = E^{(1)} + E^{(2)} + \cdots + E^{(n)} + R^{(n)}$$

This additive decomposition corresponds to a **direct sum** of the underlying vector subspaces. Let $S_i$ be the row space of $E^{(i)}$ and $S_{\text{res}}$ be the row space of the final residual $R^{(n)}$. The original embedding space $S$ (the row space of $E$) can then be expressed as:
$$S = S_1 \oplus S_2 \oplus \cdots \oplus S_n \oplus S_{\text{res}}$$

If the process continues until the residual $R^{(n)}$ is a zero matrix (or numerically close), the decomposition is complete. The dimension of a direct sum is the sum of the dimensions of its subspaces. Since the rank $k_i$ is the dimension of the subspace $S_i$, the sum of the ranks equals the dimension of the original embedding space (assuming $E$ has full rank).
$$\sum_{i=1}^{n} k_i = \dim(S) = d$$

For example:
$$(160 \text{ topics}) + (130 \text{ tone}) + \cdots + (k_n \text{ noise}) = 1536$$

---

## Multi-Layer Preference Scoring Algorithm

This algorithm takes a small set of items a user likes and produces a detailed "preference profile" broken down by each of the decomposed layers (topic, tone, etc.).

#### **Phase 1: Pre-computation (Done Once, Offline)**

Before any user interacts with the system, the N-Step Decomposition Algorithm must be completed. The essential outputs required from that process are the **basis vectors** for each layer:
* $A^{(1)} \in \mathbb{R}^{k_1 \times d}$ (Topic Basis)
* $A^{(2)} \in \mathbb{R}^{k_2 \times d}$ (Tone Basis)
* ...
* $A^{(n)} \in \mathbb{R}^{k_n \times d}$ (Final Layer Basis)

These matrices serve as the measurement framework for user preferences.

#### **Phase 2: Preference Scoring (Done Per User, Real-time)**

**1. Collect User Input**
* A user selects a small number, $m$, of content pieces they like (e.g., $m=5$).
* Retrieve their corresponding raw embeddings to form a matrix $E_{\text{user}} \in \mathbb{R}^{m \times d}$.

**2. Calculate Global Preference Vector ($w_{\text{global}}$)**
* Average the embeddings of the user's chosen items to create a single **global preference vector**. This vector represents the center of the user's taste in the high-dimensional embedding space.
    $$w_{\text{global}} = \frac{1}{m} \sum_{j=1}^{m} E_{\text{user}}[j, :]$$
* The result is a single vector $w_{\text{global}} \in \mathbb{R}^{d}$.

**3. Project Preference Vector onto Each Layer**
* Measure how this global preference vector aligns with each of the predefined layers. For each layer $i = 1, 2, ..., n$:
    * Calculate the **preference scores** for that layer, $s^{(i)}$, by taking the dot product of the global preference vector with the transpose of that layer's basis matrix, $A^{(i)}$.
        $$s^{(i)} = w_{\text{global}} \cdot (A^{(i)})^T$$
* This step decomposes the user's single preference vector into a set of score vectors.

#### **Final Output**

The output is a **user preference profile**, which is a set of score vectors: $\{s^{(1)}, s^{(2)}, ..., s^{(n)}\}$.
* $s^{(1)} \in \mathbb{R}^{k_1}$: The user's scores for each of the $k_1$ topic axes.
* $s^{(2)} \in \mathbb{R}^{k_2}$: The user's scores for each of the $k_2$ tone axes.
* ...and so on.

#### **Content Recommendation**

Once a user's preference profile is established, it can be used to recommend new content:

**1. Score New Content**: For any new post with embedding $e_{\text{new}}$, calculate its scores for each layer:
    * Topic Scores: $c^{(1)} = e_{\text{new}} \cdot (A^{(1)})^T$
    * Tone Scores: $c^{(2)} = e_{\text{new}} \cdot (A^{(2)})^T$
    * ...and so on.

**2. Measure Similarity**: Recommend content where the content's score profile $\{c^{(1)}, c^{(2)}, ...\}$ is most similar to the user's preference profile $\{s^{(1)}, s^{(2)}, ...\}$. Similarity can be calculated for each layer (e.g., using cosine similarity between $c^{(1)}$ and $s^{(1)}$) and then combined into an overall recommendation score.

This approach enables explainable recommendations by identifying which specific layers (topic, tone, etc.) drive the recommendation.

---

## Future Direction: Per-Subspace VAEs

A potential extension is to train individual VAEs (Variational Autoencoders) per decomposed subspace. Each of the $k$ VAEs would learn a nonlinear representation of the same embedding vector within its respective subspace, enabling richer semantic matching beyond what linear projection provides. This remains exploratory and is not part of the current system.
