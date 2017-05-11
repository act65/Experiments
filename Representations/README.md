
http://www.inference.vc/unsupervised-learning-by-predicting-noise-an-information-maximization-view-2/


TODO. rework train script into class.
TODO. use different training methods. ls vs gd.
TODO. what if we do orthogonal outputs and orthogonal weights?
TODO. semisupervised


3 different settings
- freeze net and train classifier. (are these features good?)
- train end to end and restore net after (is this set of features close?)
- give small subset of labels as net trains. ()
  - two was to look at it;
  - inserting a bunch of random crap to small label problem. does it help generalise? (reminicent of dataaugmentation)
  - adding a couple of labels into training, how does this effect representations learnt?


***

What makes a good representation/code?

__Disentangled factors of variation__. If we have seen even a single example where
there was A (say a road) but not B (say cars) then we know that A and B are
distinct objects and thus we can disentangle them.
__Low dimensional__.
__Maximises MI__. Not actually sure this is something we want to do. We can to throw away info, not keep it.
__Boundaries__. Caputres the boundaries between classes, (but this requires labels?!)
