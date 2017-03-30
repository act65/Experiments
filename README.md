Want to test different ways to <u>share</u> parameters in neural nets.
The relation between parameter sharing schemes and tensor factorisations is especially
interesting.

* How can we structure our networks to have an inductive bias that matches the data?
* How can we build invariance to the right 'dimensions'?
* Which features can be reused/transfered between different tasks?

What theory is there around these concepts?

### TODO.

Test;

* atrous conv (sharing over scale)
* TT decompositions (sharing over ?!?)
* random recycling (sharing in depth or ??)
* local patchwork (sharing over space -- in a more unusual way?)
* ?

Want: a nice way of visualising, thinking about, and implementing shared variables.
