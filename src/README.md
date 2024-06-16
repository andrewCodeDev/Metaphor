
## Metaphor Core

The core of Metaphor is comprised of two main branches:

  - algorithms
  - operations

Algorithms are functions that perform a forward transformation on data. This includes sorting, filling, sequencing, and more. Operations are forward transformations that have effect the gradient. This includes arithmetic, activation, scaling, loss, etc.

As such, Operations are just Algorithms that also have a reverse function that calculates the gradient.


