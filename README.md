# Interpretable-han-for-document-classfication-with-keras

This repository uses Keras to implement the hierachical attention network presented in Hierarchical Attention Networks for Document Classification. [link](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

## How to use the package

1. Clone the repository.
2. In the root of repo, run the `pip install .` to install all packages required.
3. Import and initialize the class:

```python
from han.model import HAN

han = HAN(embedding_matrix)
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You would like to change value of parameters during the initialization, for instance:

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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;And you could also tune the value of parameters.

6. Show the attention weights for word level:
```python
han.show_word_attention(X)
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`X` is the embedded matrix vector for one review.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Show the attention weights for sentence level:
```python
han.show_sent_attention(X)
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`X` is the embedded matrix vector for reviews (could be multiple reviews).

7. Truncate attention weights based on sentence length and number, and transform them into dataframe to make the result easily understandable:
```python
han.word_att_to_df(sent_tokenized_review, word_att)

han.sent_att_to_df(sent_tokenized_reviews, sent_att)
```
