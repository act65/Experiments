Dense prediction of masks is expensive.
The storage and reading of masks is expensive.
<!-- Need a better story here -->

Is there a cheaper way? We would like a low dimensional representation of
a mask. But what should the representation be? We will also want two representations
to be comparable so we can minimize their different when training.


There are all ready existing low dimensional representations of masks which are
commonly used in practice. Bounding boxes, polygons,

### Bounding boxes

Relationship between deformable roi pooling and region proposal networks.
What is actually gained? Dont need to use policy gradients.

### Polygons

Polygons can give greated accuracy than bounding boxes.

We could easily imagine interpreting the outputs of a neural network as the
points of a (multi)polygon, but how can we create a loss from these points?

Let `p' = CNN(x)` or `= RNN(x)` (a polygon as output)

but, we need a way to learn the CNN/RNN.

1. `L: ploy' x poly -> loss`. This requires that; the
polygons have the same number of points, and are in the same order. `|| p - p' ||`
(so the net also needs to learn the right output ordering)
* `L: f(poly') x mask -> loss`. We meed a differentiable mapping from polygons
to masks (for use at training time) so we could minimise the
jaccard loss between ground truth mask and generated mask (`min jaccard(D(p), mask)`)
  *  construct a differentiable based on: [winding](), [interploation](),
  * learn `min || D(p) - cv2.fillPoly(p) ||`


Analysis
1. is not great because;
  * the absolute positions of polygons are often arbitrary.
  * we often need different numbers of points for different shapes.
* We are still training with masks which will be more expensive (although,
  what is the difference between the sorts of information that we can get from the gradients?!)
* The learned representation will not be as small, and it will not have the nice
 properties of polygons (which are?!?) (unless we have a seqential hidden space and regularised to look similar to polygons?!?)


##### Winding


##### Interpolation

Similar to how the deformable conv works.
Using an indicator function $$ x(p) = \sum G(q,p) cdot x(q)$$ to select which
values contribute to the weighted sum $$y(p_0) = \sum w(p_n) · x(p_0 + p_n +
\Delta p_n)$$.



### A learned representation

`L: h x h -> loss` Learn D, E in an autoencoder fashion. `min || D(E(mask)) - mask ||`.
[Straight to shapes](https://arxiv.org/abs/1611.07932) and [Grass](https://arxiv.org/pdf/1705.02090v1.pdf) even represents 3D models.
