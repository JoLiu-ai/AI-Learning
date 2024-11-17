

![image](https://github.com/user-attachments/assets/9edd67cb-451f-4436-8eda-1ba3c16d930d)


```python
from peft import  get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit

tuning_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, #This type indicates the model will generate text.
    prompt_tuning_init=PromptTuningInit.RANDOM,  #The added virtual tokens are initializad with random numbers
    num_virtual_tokens=4, #Number of virtual tokens to be added and trained.
    tokenizer_name_or_path=model_name
)

peft_model = get_peft_model(model, tuning_config)
```
