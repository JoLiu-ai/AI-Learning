

# Broadcast
![image](https://github.com/user-attachments/assets/9cc8db01-2e9d-45c4-9783-af7a69d326c3)

# Reduce
![image](https://github.com/user-attachments/assets/b6f0fa2f-6f9c-4da5-abbc-fcc495d5be9c)

# All-Reduce
将多个处理单元的数据进行归约（如求和），并将结果分发到所有处理单元  
![image](https://github.com/user-attachments/assets/3202e676-0475-440d-ac72-f19e46dd44bf)

# AllGather
将所有处理单元的数据聚集到每个处理单元上  
![image](https://github.com/user-attachments/assets/5a068646-629c-49e6-aa26-deff9463ebed)


# ReduceScatter
服务器将自己的数据分为同等大小的数据块，每个服务器将根据index得到的数据做一个Reduce操作，即先做Scatter再做Reduce  
![image](https://github.com/user-attachments/assets/6a4fab76-c263-4928-90e8-275d08eeeffd)


# ring AllReduce
![image](https://github.com/user-attachments/assets/0fd66194-be46-4930-90c2-b692e85f738a)

reduce-scatter (a–d) and all-gather (e–g) in ring AllReduce


# 来源
1.[Collective Operation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
