Testing out some alternative data augmentation methods.

Why do we do this?
How does this make labels go further?
Assumptions:
	- want invariance to certain things
	- small pertubations should still retain the same label.

* Peturbing in feature space
	* Use SVD to find principle component of batch.
	* ?
* Rotations?
* For audio?


Adding noise. Where makes the most sense?
(why do we add noise?)
Want an estimate of the gradient, but if it is too accurate then we can overfit easily.
Can add to;
- inputs
- hidden features
- gradients
- labels

key question. how to balance noise and parallelism.
need noise for generalisatiob?! and exploration,
but need parallelism for low bias grad estimates, faster learning?, 
