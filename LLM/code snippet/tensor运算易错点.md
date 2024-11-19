```python
token_logits[0,[7], :] 返回的结果会有额外的维度，形状是 (1, vocab_size)。
token_logits[0, 7, :] 直接返回形状为 (vocab_size,) 的张量。
```
