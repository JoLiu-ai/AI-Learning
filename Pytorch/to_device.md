```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m2 = m1.to(device)
b = a.to(device) # a tensor
```

>- a 在cpu上，b在gpu上
>- m1 和 m2在gpu上，idential
