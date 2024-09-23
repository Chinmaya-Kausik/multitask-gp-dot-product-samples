# multitask-gp-dot-product-samples
A library for performing exact inference with multitask GPs using samples of dot products with given vectors, built on top of gpytorch. Open sourced from my internship at Microsoft Ads, see project page [here](https://chinmaya-kausik.github.io/projects/multitask_gps/). 

While the project includes code for updating a GP using dot product samples, it is shipped together with code for representing a function $y(x_1, x_2) = \sum_i f_i(x_1) \alpha_i(x_2)$, where $\alpah_i$ is the $i^{th}$ output of our multitask/vector-valued GP. The code also allows one to update the function $y$. In the future, you can expect:
1. A cleaner separation between the "function-representation" code and the base GP update code, splitting them into different folders.
2. A Likelihood class for dot product samples.

More details and a setup.py file to be added.
