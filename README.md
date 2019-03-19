# hierarchical-attention-for-document-classfication-with-keras

This repository uses Keras to implement the hierachical attention network presented in Hierarchical Attention Networks for Document Classification. [link](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

## How to use the package

1. Clone the repository.
2. In the root of repo, run the `pip install .` to install all packages required.
3. Import and initialize the class:

```python
from han.model import HAN

han = HAN(embedding_matrix)
```

⋅⋅⋅You would like to change value of parameters during the initialization, for instance:

```python
han = HAN(embedding_matrix, max_sent_length=150, max_sent_num=15)
```
4. When you initialize the `HAN`, the models are also set, so you could print the summary to check layers:
```python
han.print_summary()
```
5. Train the model simply with:
```python
han.train_model(checkpoint_path, X_train, y_train, X_test, y_test)
```
And you could also tune the value of parameters.
