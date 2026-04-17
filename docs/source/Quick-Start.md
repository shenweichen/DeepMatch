# Quick-Start

## Installation Guide
Now `deepmatch` supports Python `>=3.7` and is tested with TensorFlow `1.15` and TensorFlow `2.x`.

DeepMatch does not pin or install TensorFlow for you. Install a TensorFlow build that matches your Python, NumPy, CPU/GPU, and operating system first, then install DeepMatch:

```bash
$ pip install tensorflow
$ pip install deepmatch
```

For GPU environments, install the TensorFlow package recommended for your CUDA, cuDNN, and platform combination, then install `deepmatch`.

For Python `>=3.9`, DeepMatch and its dependencies allow modern `h5py` releases with `h5py>=3.7.0`. If TensorFlow reports a NumPy conflict, follow the TensorFlow requirement for your selected TensorFlow release, for example using `numpy<2` when required by TensorFlow.

Use public `tensorflow.keras` APIs in your own code. Avoid mixing `tensorflow.python.keras` with `tensorflow.keras`, because `tensorflow.python.*` is private TensorFlow API and can break model serialization or optimizer/metric loading across TensorFlow versions.

### Install from source

```bash
$ git clone https://github.com/shenweichen/DeepMatch.git
$ cd DeepMatch
$ pip install .
```
## Run examples !!

- [Run models on MovieLen1M in Google colab](./Examples.html#run-models-on-movielen1m-in-google-colab)

- [YoutubeDNN/MIND with sampled softmax](./Examples.html#youtubednn-mind-with-sampled-softmax)
- [SDM with sampled softmax](./Examples.html#sdm-with-sampled-softmax)
- [DSSM with in batch softmax](./Examples.html#dssm-with-in-batch-softmax)
- [DSSM with negative sampling](./Examples.html#dssm-with-negative-sampling)
