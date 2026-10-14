"""Synthetic geometry check; no corpus, service calls or semantic guarantees."""
import numpy as np


def main():
    rng = np.random.default_rng(42)
    left, _ = np.linalg.qr(rng.normal(size=(20, 6)))
    right, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    embeddings = left @ np.diag([9., 7., 5., 3., 2., 1.]) @ right.T
    _, _, original_vt = np.linalg.svd(embeddings, full_matrices=False)
    residual = embeddings.copy()
    components, bases = [], []
    offset = 0
    for rank in (2, 1, 2):
        _, _, vt = np.linalg.svd(residual, full_matrices=False)
        basis = vt[:rank]
        component = residual @ basis.T @ basis
        reference = original_vt[offset:offset + rank]
        np.testing.assert_allclose(component, embeddings @ reference.T @ reference, atol=1e-12)
        for previous in bases:
            np.testing.assert_allclose(basis @ previous.T, 0, atol=1e-12)
        residual -= component
        np.testing.assert_allclose(residual @ basis.T, 0, atol=1e-12)
        components.append(component)
        bases.append(basis)
        offset += rank
    np.testing.assert_allclose(sum(components) + residual, embeddings, atol=1e-12)
    print("PASS: reconstruction, orthogonality, and equivalence to one-SVD blocks")

    # Balanced, independent synthetic topic/tone labels. Both SVD directions mix them.
    topic = np.array([-1., -1., 1., 1.])
    tone = np.array([-1., 1., -1., 1.])
    mixed = np.column_stack((2 * (topic + tone), topic - tone))
    _, _, vt = np.linalg.svd(mixed, full_matrices=False)
    scores = mixed @ vt.T
    np.testing.assert_allclose(vt @ vt.T, np.eye(2), atol=1e-12)
    semantic_correlations = np.corrcoef(np.column_stack((scores, topic, tone)), rowvar=False)[:2, 2:]
    np.testing.assert_allclose(np.abs(semantic_correlations), np.sqrt(.5), atol=1e-12)
    zero_a, zero_b = np.isclose(scores[:, 0], 0), np.isclose(scores[:, 1], 0)
    joint = np.mean(zero_a & zero_b)
    independent_product = np.mean(zero_a) * np.mean(zero_b)
    assert not np.isclose(joint, independent_product)
    print(f"COUNTEREXAMPLE: joint probability {joint:.2f} != marginal product {independent_product:.2f}")
    print("Orthogonal components still mix topic/tone and are statistically dependent.")
    print("This checks geometry, not human style preference accuracy.")


if __name__ == "__main__":
    main()
