# GPRwithUS
Gaussian process regression with uncertainty sampling

## Overview
Implementing Gaussian process regression using only numpy.

For more information on the Gaussian process, see http://www.gaussianprocess.org/gpml/chapters/RW.pdf

Uncertainty sampling (US) is one of the active learning methods to improve the accuracy of a model efficiently. The next observation point is the point with the maximum variance.

## Plot
![iteration 1](https://github.com/SK-tklab/GPRwithUS/blob/main/image/iterarion1.png)
![iteration 3](https://github.com/SK-tklab/GPRwithUS/blob/main/image/iterarion3.png)
![iteration 4](https://github.com/SK-tklab/GPRwithUS/blob/main/image/iterarion4.png)
![iteration 8](https://github.com/SK-tklab/GPRwithUS/blob/main/image/iterarion8.png)

## Experimetal Setting
- noise variance: 1e-4
- kernel: RBF kernel
  - length scale = 0.4
  - variance = 1
