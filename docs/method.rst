Method
======

Each subject is one attributed graph: DTI tractography defines the
propagation topology, the functional connectivity profile of each region is
the signal propagated over it.

1. **WL embedding** (:func:`wwl_connectomics.embedding.wl_embedding`).
   With ``W`` the row-normalised structural matrix and ``a⁰ = FC``:

   .. math::

      a^{t+1} = \tfrac12 \left(a^{t} + W a^{t}\right), \qquad t = 0,\dots,H-1

   The embedding is the concatenation :math:`[a^0 | a^1 | \dots | a^H]`,
   an ``(N, N·(H+1))`` point cloud per subject.

2. **Fractional variant**
   (:func:`wwl_connectomics.embedding.wl_embedding_fractional`). ``W`` is
   built from the regularized fractional graph of Filippo & Mazza (2026):
   real structural edges keep their weight, every other pair gets
   :math:`\beta\, w^{\alpha}_{ij}` (power-law decay). :math:`\alpha\to1`
   recovers plain DTI propagation; :math:`\alpha\to0` is maximally non-local.

3. **Distance** (:func:`wwl_connectomics.distances.build_D`): pairwise
   Wasserstein-1 distance between embeddings (optimal one-to-one matching).

4. **Kernel** (:func:`wwl_connectomics.kernels.build_K`):
   :math:`K = \exp(-\lambda D)`; :math:`\lambda` by ``1/mu``, CV, joint
   :math:`(\lambda, C)` search, or Fisher separability.

5. **Nested cross-validation**
   (:func:`wwl_connectomics.crossval.nested_cv_kernel`): :math:`\lambda` and
   :math:`C` tuned on each training fold only; balanced accuracy reported.

References
----------

* Togninalli, Ghisu, Llinares-López, Rieck, Borgwardt. *Wasserstein
  Weisfeiler-Lehman Graph Kernels.* NeurIPS 2019.
* Filippo, Mazza. *Spectral and computational aspects of a regularized
  fractional Laplacian for non-local diffusion on graphs.* J. Numer. Math.
  2026, doi:10.1515/jnma-2026-0007.
