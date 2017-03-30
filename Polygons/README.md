Pretty sure we can make this differentiable.

Similar to how the deformable conv works.
Using an indicator function $$ x(p) = \sum G(q,p) cdot x(q)$$ to select which
values contribute to the weighted sum $$y(p_0) = \sum w(p_n) · x(p_0 + p_n +
\Delta p_n)$$.

Could then be used for a) jaccard loss b) more flexible region proposals c) a new type of spatial transformer.
