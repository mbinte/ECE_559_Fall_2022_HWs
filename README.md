# ECE-599 · Neural Network

**Fall 2022 coursework**

Seven assignments cover threshold networks, perceptron learning, backpropagation, and neural networks for images and text. Each homework folder pairs the original assignment with a worked solution report. Programming assignments also include source code and experiment outputs.

The original handouts carry the course designation **ECE/CS 559**.

## Homework index

| Homework | Topic | Assignment | Solution |
| --- | --- | --- | --- |
| [HW01](hw01-logic-networks/) | Boolean logic with threshold networks | [Prompt](hw01-logic-networks/assignment.pdf) | [Derivation and diagrams](hw01-logic-networks/solution-report.pdf) |
| [HW02](hw02-perceptron/) | Binary perceptron and learning-rate experiments | [Prompt](hw02-perceptron/assignment.pdf) | [Report](hw02-perceptron/solution-report.pdf) |
| [HW03](hw03-mnist-perceptron/) | Multicategory perceptron on MNIST | [Prompt](hw03-mnist-perceptron/assignment.pdf) | [Report](hw03-mnist-perceptron/solution-report.pdf) |
| [HW04](hw04-curve-fitting/) | Curve fitting with online backpropagation | [Prompt](hw04-curve-fitting/assignment.pdf) | [Report](hw04-curve-fitting/solution-report.pdf) |
| [HW05](hw05-shape-classification/) | Neural classification of nine geometric shapes | [Prompt](hw05-shape-classification/assignment.pdf) | [Report](hw05-shape-classification/solution-report.pdf) |
| [HW06](hw06-denoising-autoencoder/) | Denoising autoencoder and k-means clustering | [Prompt](hw06-denoising-autoencoder/assignment.pdf) | [Report](hw06-denoising-autoencoder/solution-report.pdf) |
| [HW07](hw07-name-generation/) | Character-level name generation with an LSTM | [Prompt](hw07-name-generation/assignment.pdf) | [Report](hw07-name-generation/solution-report.pdf) |

## Run the solutions

Run the commands from the corresponding homework folder. HW01 is a written solution.

| Homework | Entry point | Requirements |
| --- | --- | --- |
| HW02 | `python3 02-BinteHashem.py` | NumPy, Matplotlib |
| HW03 | `python3 03-BinteHashem.py` | NumPy, Matplotlib, C compiler available as `cc` |
| HW04 | `python3 04-BinteHashem.py` | NumPy, Matplotlib |
| HW05 | [Training and inference instructions](hw05-shape-classification/README.md) | NumPy, Matplotlib, Pillow |
| HW06 | `python3 starter.py`, then `python3 06-BinteHashem.py` | PyTorch, torchvision, NumPy, SciPy, Matplotlib |
| HW07 | `python3 0701-BinteHashem.py` to train, or `python3 0703-BinteHashem.py a` to generate names with the included model | PyTorch, Matplotlib |

HW07 inference accepts a single starting letter and prints 20 names. Use `--seed 559 --temperature 0.85` to reproduce the samples in the report.

## Datasets

| Dataset | Used in | Included data |
| --- | --- | --- |
| [MNIST](datasets/mnist/MNIST/raw/) | HW03, HW06 | 60,000 training images and 10,000 test images, with labels |
| [2D geometric shapes](hw05-shape-classification/data/geometry-90000.zip) | HW05 | 90,000 RGB images across nine classes |
| [Names](hw07-name-generation/data/names.txt) | HW07 | 2,000 names provided with the assignment |

HW02 and HW04 generate their own samples as part of the experiment. The geometry dataset is credited to [Anas El Korchi, Mendeley Data](https://doi.org/10.17632/wzr2yv7r53.1).
