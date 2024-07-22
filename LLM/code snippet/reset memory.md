```
del tokenizer, model, pipe # 这个操作会删除这些变量名，但不一定会立即释放这些对象占用的内存

# Empty VRAM cache
import torch
import gc
gc.collect() # 清理那些不再被引用的对象，释放它们占用的内存,主要用于管理CPU内存。它会尝试释放所有不再使用的Python对象，包括那些可能占用大量内存的对象。
torch.cuda.empty_cache() # 专门用于管理GPU内存。它不会释放正在使用的内存，只会清空那些已经分配但当前未使用的缓存。
```

### `gc.collect()` vs `torch.cuda.empty_cache(`

作用范围：gc.collect() 主要作用于CPU内存，而 torch.cuda.empty_cache() 作用于GPU内存。
释放机制：gc.collect() 通过垃圾收集机制释放内存，而 torch.cuda.empty_cache() 通过清空缓存释放GPU内存。
使用场景：gc.collect() 用于确保所有不再使用的Python对象都被回收；torch.cuda.empty_cache() 用于在深度学习训练或推理过程中管理GPU内存，防止内存不足。
