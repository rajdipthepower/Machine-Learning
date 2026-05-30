# Building a Convolutional Neural Network from Scratch with PyTorch Autograd

## Overview

This project was an experiment to understand what actually happens behind the scenes when PyTorch performs automatic differentiation and backpropagation.

Instead of relying on high-level modules such as `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Linear`, or built-in loss functions, I manually implemented the forward pass of a small Convolutional Neural Network using PyTorch tensors and Autograd.

The primary goal was not achieving high accuracy, but understanding:

* Computational Graphs
* Automatic Differentiation
* Backpropagation
* Gradient Flow
* Tensor Construction
* In-Place Operations and their impact on Autograd

---

## Components Implemented Manually

The following components were implemented from scratch:

### Convolution Layer

* Multi-channel convolution
* Custom kernel initialization
* Trainable weights and biases

### ReLU Activation

Implemented using:

```python
torch.clamp(x, min=0)
```

instead of manually modifying tensor values.

### Max Pooling

* Custom sliding window implementation
* Configurable pooling size
* Configurable stride

### Fully Connected Layer

* Manual flattening
* Matrix multiplication using `torch.matmul`

### Cross Entropy Loss

Implemented directly from logits using the log-sum-exp formulation for numerical stability.

### Gradient Descent

Parameter updates performed manually using:

```python
with torch.no_grad():
    parameter -= learning_rate * parameter.grad
```

without using any optimizer classes.

---

## Dataset

To keep the experiment lightweight, I trained the model on only:

* 2 Cat Images
* 2 Dog Images

The images were resized to:

```python
128 x 128
```

and converted into normalized RGB tensors.

---

## Results

As expected, the network quickly overfit the training data.

This was not surprising because:

* The dataset was extremely small.
* The model had enough capacity to memorize the training examples.
* There was no regularization.
* There was no train-validation split.

The purpose of the project was educational rather than achieving generalization.

---

## A Key Lesson: Preserving the Computational Graph

One of the most interesting discoveries during implementation was learning how easily Autograd can be disrupted.

At first glance, writing values directly into an output tensor seems perfectly reasonable:

```python
temp_array[kernel, i, j, k] = value
```

However, direct tensor assignment and certain in-place modifications can interfere with PyTorch's ability to track the mathematical history required for gradient computation.

Since backpropagation relies on this history, preserving the computational graph becomes critical.

### Solution Used

Instead of writing values directly into tensors:

1. Intermediate results were stored in a dictionary.
2. Coordinates were used as dictionary keys.
3. The final tensor was reconstructed afterward.

Example concept:

```python
temp_dict[kernel, i, j, k] = convolution_result
```

followed by rebuilding tensors using:

```python
torch.stack(...)
```

---

## Why `torch.stack()`?

One of the most important implementation choices in this project was the use of `torch.stack()`.

Instead of modifying an existing tensor repeatedly, I reconstructed entire tensors from previously computed values.

Example:

```python
rows.append(torch.stack(cols))
```

and later:

```python
output = torch.stack(kernels)
```

This approach allows PyTorch to preserve the chain of operations connecting the final loss back to every trainable parameter.

In other words:

* `torch.stack()` creates a new tensor from tracked operations.
* Autograd knows exactly how each element was produced.
* The computational graph remains intact.

---

## Why `torch.clamp()` for ReLU?

A similar idea applies to ReLU.

Instead of manually overwriting negative values:

```python
x[x < 0] = 0
```

I used:

```python
torch.clamp(x, min=0)
```

This performs the operation while preserving the graph that Autograd needs for backpropagation.

---

## Visualizing the Computational Graph

After building the network, I used `torchviz` to visualize the complete computational graph.

```python
from torchviz import make_dot
```

The graph reveals the Directed Acyclic Graph (DAG) created by PyTorch's Autograd engine.

Each node represents a mathematical operation.

Each edge represents tensor dependencies.

During:

```python
loss.backward()
```

PyTorch traverses this graph in reverse order to compute gradients for every trainable parameter.

Seeing this graph was one of the most valuable parts of the project because it transformed backpropagation from an abstract concept into something visible.

---

## Example Predictions

After training:

| Image            | Prediction    |
| ---------------- | ------------- |
| Cat Image        | Cat           |
| Dog Image        | Dog           |
| Unseen Cat Image | Cat           |
| Unseen Dog Image | Misclassified |

The results further demonstrated the impact of training on an extremely small dataset.

---

## Technologies Used

* Python
* PyTorch
* NumPy
* PIL
* TorchViz

---

## Key Takeaways

This project reinforced several important ideas:

* Deep learning frameworks are building mathematical graphs, not just executing code.
* Backpropagation depends on preserving those graphs.
* Tensor construction strategies matter.
* In-place operations require careful handling.
* Understanding Autograd becomes much easier when implementing components manually.
* Visualizing the computational graph provides valuable intuition about gradient flow.

---

## Future Improvements

Potential extensions include:

* Mini-batch training
* Multiple convolutional layers
* Better weight initialization experiments
* Batch normalization
* Dropout
* Validation dataset
* Optimizers such as SGD and Adam
* Comparison with PyTorch's built-in implementations

---

## Computational Graph Preview

A visualization generated using TorchViz is included in this repository.

It illustrates the complete computational graph produced by the custom CNN and serves as a visual representation of how PyTorch's Autograd engine tracks operations during forward propagation and computes gradients during backpropagation.

---

### Final Thought

Building neural networks from scratch is rarely the most efficient approach.

However, it is one of the best ways to understand how modern deep learning frameworks work internally.

Sometimes the fastest way to learn is to remove the abstractions.
