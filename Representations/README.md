
http://www.inference.vc/unsupervised-learning-by-predicting-noise-an-information-maximization-view-2/


TODO. rework train script into class.
use different training methods. ls vs gd.
different ways to validate.



TODO. actualy cross entropy. what actual discriminative loss...
TODO. what if we do orthogonal outputs and orthogonal weights?


3 different settings
- freeze net and train classifier. (are these features good?)
- train end to end and restore net after (is this set of features close?)
- give small subset of labels as net trains. ()
  - two was to look at it;
  - inserting a bunch of random crap to small label problem. does it help generalise? (reminicent of dataaugmentation)
  - adding a couple of labels into training, how does this effect representations learnt?
