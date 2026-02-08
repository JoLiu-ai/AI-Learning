
## asyncio 的层级结构
```
                    asyncio 模块
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    Event Loop      协程支持        同步原语
        │               │               │
    ├─ run()        ├─ Task          ├─ Lock
    ├─ create_task  ├─ Future        ├─ Semaphore
    ├─ call_later   ├─ gather()      ├─ Event
    └─ run_forever  ├─ wait()        ├─ Queue
                    └─ timeout()     └─ Condition
                        │
                   协程(底层概念)
                        │
                ┌───────┴────────┐
            async def        await
```

- `await X` ：
>让当前协程暂停，把执行权交给外部，并且告诉外部：
>“等 X 好了，再叫我回来”。

- `send(None)`
>协程恢复执行

-`Task`
>Task = 自动在正确时机调用 send(None) 的东西

```
while not coro.done():
    coro.send(None)
```



```
async def modern_coroutine():
    print("开始")
    await asyncio.sleep(1)  # ← 语义更清晰
    print("结束")

# 使用
asyncio.run(modern_coroutine())  # Python 3.7+ 的简化方式
```

## 执行时间线
```
async def demo():
    print("A")
    await asyncio.sleep(1)
    print("B")
```


```
Task 调用 send(None)
  → 打印 A
  → 遇到 await
  → 协程 yield 出 Future（sleep）
Task 停止调这个协程
Event Loop 等 1 秒
Event Loop 通知 Task：Future 完成
Task 再次 send(None)
  → 打印 B
  → 结束
```

>await 让协程停下
>Task 让协程继续

```
协程内部:         await
                    ↓
            （yield 控制权）
                    ↓
Task:        send(None) 恢复
                    ↓
            协程继续执行
```


## 🎬 异步编程的禅意

> *同步是顺序的美,异步是并发的诗。*  
> *线程是操作系统的恩赐,进程是隔离的代价。*  
> *GIL 是历史的枷锁,Event Loop 是未来的钥匙。*

如果把程序比作交响乐团:
- **同步** = 独奏(一个人演奏完整首曲子)
- **线程** = 多人轮流用一把小提琴(GIL)
- **进程** = 每人一把琴,各拉各的
- **asyncio** = 一个指挥家(Event Loop)协调所有乐器

---

**彩蛋**: 观察每个任务的线程名,你会发现 asyncio 的所有任务都跑在 `MainThread`,而 threading 会看到 `Thread-1`, `Thread-2`... 这就是协程 vs 线程的本质差异!  
