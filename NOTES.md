## TODO

### General Permutation

- the kernel is finished, just need the parser

### Optimizers

- Momentum
- RMSProp
- ADAM
- these optimizers require hash tables...
- hashing: (GraphID, WgtIdx) -> Tensor

### Interface

- Dimensional inference... need to identify which ops need this.
- String parsing - "_ij,_jk->_ik" means optional 3rd dimension.

### Loss

- Need Rank-2 variants (row-wise)
- Mean Squared Error
- Binary-Cross Entropy

### Activation

- Row-, column-wise Softmax (see dimensional inference)
- Sigmoid (why not?)
- More...

### Reduction

- Takes "ijk->ij" sum over k dimension and return ij
- Derivative is to broadcast d_ij -> ijk (k copies of derivative)

### Broadcast

- mentioned above, copies some value across a dimension
- What parameters to take? Maybe Rank(N){ m, n, ... }

### Linear

- General inner-product? Probably.
- Finish Rank-2 variants.

### Samplers

- Given a tensor, randomly pick a value within parameters.
- Temperature, Top-K... some combination there-in

### Randomize

- Give different randomization types? Gaussian, uniform... etc?
- Move to GPU? Probably? How many times do we call this?
