1. self-attention和attention有什么区别
  * self-attention 主要关注序列内部的依赖关系
  * 其他 attention 主要关注序列之间的依赖关系。 模型关注输入序列的一个部分，然后利用这个关注的信息来执行某种操作，比如在序列到序列的任务中，解码器关注编码器输出的某些部分来生成输出序列。
2. attention scores v.s. fully-connected layer
* the attention weights :
  * * dynamic and input dependent. （change with input）
  * * They are useful for capturing relationships and dependencies within the input data.
* a convolutional or fully-connected layer weights:
  * * fixed after training
