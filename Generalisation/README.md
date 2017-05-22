__What are the benefits of depth vs width?__

this review by [poggio et al.](https://arxiv.org/abs/1611.00740) should have pretty much everything. a couple of other seemingly important papers that I have stumbled across (which they also cite) are [montufar - number of linear regions](https://arxiv.org/abs/1402.1869) and [telgarsky - high freq oscillations](https://arxiv.org/pdf/1602.04485.pdf).
despite this work, i havent seen conclusive proof that deep is 'better' than wide. depth does seem to allow the representation of more complex functions with fewer parameters (through a kind of factorisation). but, the interaction between width/depth, learning dynamics and generalisation doesnt seem clear to (at least) me.

Subsubsection "6.4.1 Universal Approximation Properties and Depth" of the book http://www.deeplearningbook.org/contents/mlp.html relevant to the width-depth discussion.
They conclude:
"In summary, a feedforward network with a single layer is suﬃcient to representany function, but the layer may be infeasibly large and may fail to learn andgeneralize correctly. In many circumstances, using deeper models can reduce thenumber of units required to represent the desired function and can reduce theamount of generalization error"


I would explain the reduction of generalisation error, via depth, as the result of having fewer parameters to overfit with (rather than the many more a wide net would have needed to learn the target function).
Although, I am not so sure now that I think about it. as both the wide net, and the smaller deep net should be able to represent functions of the same order of complexity.
probably my favourite point they make just under the section you quoted (from the DL book)
> Choosing a deep model encodes a very general belief that the function we want to learn should involve composition of several simpler functions.
depth is sequential/hierarchical composition, width is parallel composition. and in conv nets the composition is local. so depth will help in some cases, but only when it reflects the target function.

also. at the end of the section they say;
>Empirically, greater depth does seem to result in better generalization for a wide variety of tasks (Bengio et al., 2007; Erhan et al., 2009; Bengio, 2009;Mesnil et al., 2011; Ciresan et al., 2012; Krizhevsky et al., 2012; Sermanet et al.,2013; Farabet et al., 2013; Couprie et al., 2013; Kahou et al., 2013; Goodfellowet al., 2014d; Szegedy et al., 2014a).

but, i dont think any of these papers actually show that depth gives better generalisation than width?? (maybe i am wrong?)
time to test.


### Experiments.

Hypothesis. Wide nets should do better than deep nets when that prior matches the structure of the data.

Experiment. Generate 1000 orthogonal vectors in $\mathbb R^{784}$ and sample from them by adding some noise (or doing some other time of transform - which would make sense?).

Discussion. These 'images' should share no ?!?!? and thus should suit a wide net better than a deep net.
