# N-Step Orthogonal Decomposition Algorithm

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
