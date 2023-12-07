# Distributed Deep Learning with TensorFlow: Simple Multi-GPU Training Example

## Overview

This repository contains a simple implementation of distributed deep learning using TensorFlow, specifically designed for multi-GPU training. The provided script defines a basic neural network using the Keras API, preprocesses the MNIST dataset, and employs the MirroredStrategy for synchronous training across multiple GPUs.


## Overview

The script demonstrates a straightforward example of leveraging distributed computing to enhance efficiency in deep learning tasks. The key components include:

1. **Model Definition:** A simple neural network model is defined using the Keras API. The model consists of a flattening layer, a dense layer with ReLU activation, a dropout layer, and a final dense layer.

2. **Data Preprocessing:** The MNIST dataset is used as an example. The script loads and preprocesses the dataset by scaling pixel values to the range [0, 1].

3. **MirroredStrategy:** TensorFlow's `tf.distribute.MirroredStrategy` is employed to perform synchronous training across multiple GPUs. The strategy scope is used to create and compile the model within the distributed training context.

4. **Training and Evaluation:** The model is trained using the `fit` method within the strategy scope. After training, the script evaluates the model on the test data and prints the test accuracy.

## Prerequisites

To run the script, ensure you have the following dependencies installed:

- TensorFlow (version compatible with the script)
- Python 3.x

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/distributed-deep-learning-tensorflow.git
   cd distributed-deep-learning-tensorflow
2. Install the required dependencies:
    pip install tensorflow
3. Run the script:
    python Multi_Gpu_DDL.ipynb
## Results
   ...
Test accuracy: 0.9768999814987183

## License
This project is licensed under the MIT License.

## Acknowledgements
Special thanks to the TensorFlow team for providing powerful tools for distributed deep learning.
Feel free to modify and extend the code for your own projects. If you encounter any issues or have suggestions, please open an issue or submit a pull request.

