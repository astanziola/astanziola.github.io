---
layout: post
date: 2023-02-16 00:00:00-0400
title: The paper describing <code>jwave</code> is available on SoftwareX
inline: false
---


The paper describing the main functionalities of [`jwave`](https://github.com/ucl-bug/jwave) is [out on SoftwareX](https://linkinghub.elsevier.com/retrieve/pii/S2352711023000341)!

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/jwave-example.png" title="Time harmonic simulation in jwave" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Comparison of a transcranial simulation in j-Wave against the widely used k-Wave. Adapted from <a href="https://arxiv.org/pdf/2202.04552.pdf">[Aubry et al.]</a>
</div>


j-Wave is a simulator based on [JAX](https://github.com/google/jax), that can solve time-varying and time-harmonic acoustic problems. 

It supports automatic differentiation (and all of [JAX fancy composable program transformations](https://github.com/google/jax)), making it a valuable tool for machine learning and scientific computing. 

j-Wave is composed of modular components that can be easily customized and reused. while being compatible with some of the most popular machine learning libraries, such as JAX and TensorFlow. The accuracy of the simulator is evaluated against the widely used k-Wave toolbox and [a cohort of acoustic simulation software](https://arxiv.org/pdf/2202.04552.pdf). 

