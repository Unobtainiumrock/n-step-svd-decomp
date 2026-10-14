> Historical working notes, preserved for context. Claims and commands below are not verified handoff instructions. Use the root README and docs/HANDOFF.md; relative links may refer to the original layout.

# �� Mathematical Reference - Quick Access

## 🔗 **Complete Mathematical Documentation**
**Location**: `scripts/legacy/RESIDUALIZED_TONE_SPACE_IMPLEMENTATION.md`

## 🎯 **Quick Mathematical Overview**

### Core Equations
- **Full Embedding Space**: $E = \mathbb{R}^{1536}$
- **Topic Subspace**: $T \subset E$ (160 SVD components)
- **Tone Subspace**: $R = E \ominus T$ (130 SVD components)
- **Residual Vector**: $\vec{r}_i = \vec{e}_i - \text{Proj}_T(\vec{e}_i)$
- **Tone Score**: $\text{tone\_score} = \vec{w}_r^{(s)} \cdot \vec{r}_i$

### Key Concepts
1. **Residualized Approach**: Separate topic from tone
2. **SVD Decomposition**: Optimal rank selection via elbow method
3. **Orthogonal Complement**: $R = E \ominus T$
4. **Convergent Search**: Strategic navigation in embedding space

## 📚 **What's in the Full Documentation**
- Complete mathematical foundations
- Step-by-step implementation guide
- Code examples and algorithms
- Theoretical background and proofs
- Performance analysis and optimization

## 🚀 **Quick Access**
```bash
# View the complete mathematical documentation
cat scripts/legacy/RESIDUALIZED_TONE_SPACE_IMPLEMENTATION.md

# Or open in your editor
code scripts/legacy/RESIDUALIZED_TONE_SPACE_IMPLEMENTATION.md
```

---
*This is a quick reference. For complete mathematical details, see the full documentation linked above.*
