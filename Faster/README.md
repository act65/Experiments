NNs can fit noise, yet they still generalise well. What gives?

Hypothesis: General patterns are learned first because they give 'stronger' gradients.

Because we are summing over a batch of inputs, it is likely than any patterns common to the examples in the batch will each have a component in the same direction.

Experiment: Clip weak/select strong components of the gradients of a batch. This should increase generalisation/reduce overfitting. Alternatively, Clip strong/select weak, this should lead to faster memorisation/less generalisation.

Results: ...






TODOs
* Find some references
* Figure out the directions for the components
* run some experiments...
