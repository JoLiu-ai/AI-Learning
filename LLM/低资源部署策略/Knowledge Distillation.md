# What is knowledge distillation(KD)

- transfer knowledge from a large, complex model (teacher) to a smaller, simpler model (student)
- the student model learns by mimicking the teacher’s predictions, typically in the form of soft labels or probabilities instead of hard labels (i.e., categorical outputs like 0 or 1).

![image](https://github.com/user-attachments/assets/c23bc0fc-1ad8-4498-9151-619f90120588)


## Key Components of Knowledge Distillation:
- a teacher model
- a student model
- Soft Labels

## loss function
- Distillation Loss(Soft target loss): The **KL divergence** between the student model's output and the teacher model's soft labels.
- Student Loss( Hard target loss): The **cross-entropy loss** between the student model's output and the true labels.
```python
# Soft target loss (KL divergence)
Distillation_Loss = KL_div(student_soft_output, teacher_soft_output)

# Hard target loss (cross-entropy)
Student_Loss = cross_entropy(student_output, true_labels)
Loss = α * Distillation_Loss(soft_target_loss) + (1-α) * Student_Loss
```

# Why Do We Need Knowledge Distillation?
drawbacks of large models:
- High computational requirements
- Large memory footprint
- Slow inference time
- High energy consumption
- Difficulty in deploying to edge devices

# The Core Concept
## Hard vs Soft Targets
### Hard Labels: 
A hard label is a discrete label where one class is marked as "1" (the correct class), and all others are marked as "0". This is often called a one-hot encoding.
- Example: Hard Targets	[0, 1, 0] (dog)
- Characteristics:
  - Contains only binary information (correct or incorrect).
  - Does not contain similarity information between categories.
  - Supervision signal is sparse.
  
### Soft Labels: 
A soft label is the probability distribution predicted by a model (teacher), typically before applying a hard decision (such as the argmax function). These are continuous values that sum to 1 and reflect the teacher model’s "belief" in each class.
- Example: Soft Targets: [0.1, 0.8, 0.1] (dog)
- Characteristics:
  - Contains relative relationships between classes.
  - Reflects the model's uncertainty.
  - Provides a richer supervision signal.

### Why both types of labels are needed:
Hard labels ensure that the model learns the correct decision boundaries.
Soft labels help the model learn the subtle relationships between classes.

# Advanced Techniques and Best Practices

##  Temperature Tuning
- **function** :adjusting the teacher model's "soft targets" distribution
```python
def softmax_with_temperature(logits, temperature):
    return torch.softmax(logits / temperature, dim=1)
```
- When T = 1: The distribution is relatively "sharp," with the probability of the main class close to 1.
- As T increases: The distribution becomes smoother, and the differences between class probabilities decrease.
- As T → ∞: The distribution approaches a uniform distribution."

![image](https://github.com/user-attachments/assets/2ffecdaf-94bf-4823-a64c-4e09e7b2b315)
- C is the number of classes. T is the temperature, which is normally set to 1.
-  logits Z_i



# match 方法
## 1. Outputlogits
```python
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    
    distill_loss = F.kl_div(student_probs, 
                           soft_targets.detach(), 
                           reduction='batchmean')
    return temperature * temperature * distill_loss

# 完整训练循环
def train_step(student, teacher, optimizer, data, labels):
    optimizer.zero_grad()
    
    # 获取教师和学生的输出
    with torch.no_grad():
        teacher_logits = teacher(data)
    student_logits = student(data)
    
    # 计算蒸馏损失和硬标签损失
    distill_loss = distillation_loss(student_logits, teacher_logits)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 总损失为两者的加权和
    total_loss = 0.7 * distill_loss + 0.3 * hard_loss
    total_loss.backward()
    optimizer.step()
```
## 2.Intermediateweights
> ⚠️the teacher model and the student model, their weight dimension is different.
![image](https://github.com/user-attachments/assets/26d2d62f-8265-4ef0-bdf1-87b750a0e0fd)
![image](https://github.com/user-attachments/assets/82b55fa9-7830-4cb7-9658-f0f8626702ec)


## 3.Intermediatefeatures
![image](https://github.com/user-attachments/assets/d0a4620d-d5b3-4588-bd79-7ef1def0ccad)

## 4.Gradients
## 5.Sparsitypatterns
## 6.Relationalinformation


# Reference
1. EfficientML.ai Lecture 9 - Knowledge Distillation (MIT 6.5940, Fall 2024)
2. 
