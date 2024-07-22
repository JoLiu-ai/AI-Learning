```
# Delete any models previously created
del model, tokenizer, pipe

# Empty VRAM cache
import torch
torch.cuda.empty_cache()
```
