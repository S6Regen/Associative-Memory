# Associative-Memory
Associative Memory using the fast Walsh Hadamard transform, random projection, Locality Sensitive Hashing
(Press 1 to start And stop training):
http://gamespace.eu5.org/associativememory/index.html

Further information (Ie. the preconditions the variance equation for linear combinations of random variables imposes for the weighted sum to act as a general associative memory, and how to meet those preconditions):
https://ai462qqq.blogspot.com/2019/11/artificial-neural-networks.html

New version (No WHT Info but better code quality):
https://editor.p5js.org/siobhan.491/present/k7UePTA4H
Code:
https://editor.p5js.org/siobhan.491/sketches/k7UePTA4H

If you did normal full hashing you can think about how the weighted sum would act as an associative memory in that situation.
Remembering that random vectors in higher dimensional space are aapproximately orthongonal.  Then downgrade to the mild hashing case.
You can think about the geometry as well, in particular how the vector length will increase as you add more associations, and how that interacts with the variance equation for linear combinations of random variables.
In practice it just tells you that undercapacity you get recall with error correction (reduction of variance), at capacity recall with no error correction, over capacity recall with Gaussian noise.
Two use cases might be:

A/ One shot learning with a pretrained neural network acting as a feature extractor.

B/ As memory for a Neural Turing Machine.  That is a neural network with an external (associative) memory bank.

