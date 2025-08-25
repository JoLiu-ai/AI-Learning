# 🚀 LLM Agent 工程黄金法则 - 完整版踩坑指南

> **TL;DR**: 一份来自生产环境血泪教训的 Agent 开发避坑指南，20 条黄金法则帮你构建稳定、可靠的 LLM Agent 系统。

## 📋 目录
- [核心法则对照表](#核心法则对照表)
- [详细实践指南](#详细实践指南)
- [完整代码示例](#完整代码示例)
- [检查清单](#检查清单)
- [常见问题 FAQ](#常见问题-faq)

---

## 💥 核心法则对照表

| #  | 法则名称                           | 常见坑                           | 实战建议 / 示例                                                                         |
| -- | ------------------------------ | ----------------------------- | --------------------------------------------------------------------------------- |
| 1  | ✅ Natural Language → Tool Call | 直接靠模型自由输出 tool 调用，结果字段混乱、接口错位 | 给工具定义 **明确 schema（pydantic / JSON）**，用结构化输出工具（如 OpenAI functions / tool\_use）引导模型 |
| 2  | ✅ Own your Prompt              | prompt 像小说，一改业务就崩             | 将 prompt 组织为模块化组件，保持 prompt 与功能对齐；重要语义用占位变量替代硬编码                                  |
| 3  | ✅ Own your Context Window      | 把所有聊天拼进去，context 一溢就忘历史       | 使用 **context summarization + memory key**，保留必要状态，丢弃噪声；使用 chunked memory 模型        |
| 4  | ✅ Tools = Structured Output    | 工具返回 HTML、自然语言等，二次解析出错        | 工具输出强制为 **JSON schema**；返回字段要统一、简洁                                                |
| 5  | ✅ Unify Exec & Business State  | Agent 内部执行状态混乱，不知道是第几步了       | 建立一个 **StateManager / FSM**，统一追踪每一步工具调用、执行流程及业务状态                                 |
| 6  | ✅ Pause / Resume with APIs     | Agent 中断后"记忆全失"，无法恢复          | 建立**resume 接口**，从数据库或存储恢复 context 和 state，结合 Factor 5 实现                          |
| 7  | ✅ Ask Human via Tool Call      | 无权限操作时仍尝试执行，高风险动作直接跑了         | 遇到高风险操作（如下单、发票等）前插入 ask\_human 工具调用，请用户确认                                         |
| 8  | ✅ Own your Control Flow        | 控制流全靠模型自猜，流程混乱                | 明确设计 control flow：如 Finite State Machine、plan → execute、RAG + guardrail 结构        |
| 9  | ✅ Compact Errors into Context  | 错误信息散落日志中，模型无法"知道失败过"         | 将错误 compact 成自然语言注入 context，或用 structured error trace + retry trigger             |
| 10 | ✅ Small, Focused Agents        | 一个 Agent 包办所有任务，逻辑复杂难 debug   | 每个 Agent 负责单一职责（search\_flight / book\_ticket / ask\_human），由上层调度协调               |
| 11 | ✅ Trigger Anywhere             | Agent 只绑定在一个 web 前端，渠道死板      | 支持在不同渠道触发（网页、API、Slack、移动端），Agent 逻辑应解耦前端                                         |
| 12 | ✅ Stateless Reducer            | Agent 状态多样，结果不可预测             | 每次输入都用状态 + 用户输入推导新状态，保证确定性输出；可类比 Redux reducer 思路                                 |
| 13 | ✅ Logging and Replay           | 没日志，出错无法复现                    | 实现完整日志追踪（包括 tool 输入输出、context 变化），可用 trace 工具或 OpenTelemetry                      |
| 14 | ✅ Manage Memory Budget         | 模型突然输出乱码、性能抖动                 | 明确上下文长度预算，使用**token 估算器**，必要时截断无效内容；上下文规划前置                                       |
| 15 | ✅ Modular Plan → Execute       | plan 阶段和 execute 阶段混杂，无法复用    | 将计划（step plan）和执行（tool 调用）拆开，Plan = 结构化 list，逐项执行、易 debug、可 resume                |
| 16 | ✅ Security Boundaries          | Agent 有过度权限，可能执行危险操作          | 实现权限分级、敏感操作白名单、rate limiting、sandbox 执行                                            |
| 17 | ✅ Test Strategy                | Agent 行为不确定，难以写单元测试          | 使用 mock tools、golden dataset、regression testing、A/B testing                       |
| 18 | ✅ Cost Management              | 长对话导致 token 消耗爆炸              | 智能 context 压缩、缓存机制、成本监控、预算告警                                                      |
| 19 | ✅ Multimodal Handling          | 不同模态数据处理方式不一致                 | 统一的媒体处理接口、格式标准化、模态转换管道                                                          |
| 20 | ✅ Graceful Degradation         | 外部服务挂了，Agent 直接崩溃            | 实现 fallback 机制、优雅降级、服务熔断、用户友好的错误提示                                                |

---

## 🔍 详细实践指南

### 📝 Factor 1: Natural Language → Tool Call

**问题场景**：
```python
# ❌ 错误示例
user_input = "帮我查一下明天北京到上海的机票"
response = llm.generate(f"用户说：{user_input}，请调用工具")
# 模型输出：search_flight(from="北京", to="上海", date="明天")
# 结果：日期格式错误，参数不规范
```

**正确做法**：
```python
# ✅ 正确示例
from pydantic import BaseModel
from typing import List
from datetime import datetime

class FlightSearchTool(BaseModel):
    origin: str  # 出发城市IATA代码
    destination: str  # 目的地IATA代码
    departure_date: str  # YYYY-MM-DD格式
    passengers: int = 1  # 乘客数量

# 使用 OpenAI Function Calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flight",
            "description": "搜索机票信息",
            "parameters": FlightSearchTool.model_json_schema()
        }
    }
]

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": user_input}],
    tools=tools,
    tool_choice="auto"
)
```

### 📝 Factor 2: Own your Prompt

**问题场景**：
```python
# ❌ 错误示例 - 硬编码 prompt
prompt = """
你是一个旅行助手，帮助用户预订机票。
用户可能会问机票价格、时间、航空公司等信息。
请礼貌回答，如果需要更多信息请询问。
用户说：{user_input}
"""
```

**正确做法**：
```python
# ✅ 正确示例 - 模块化 prompt
class PromptTemplate:
    def __init__(self):
        self.system_role = "你是一个专业的旅行助手"
        self.capabilities = [
            "搜索机票信息",
            "比较价格和时间",
            "提供航空公司建议"
        ]
        self.constraints = [
            "需要用户确认后才能预订",
            "价格信息可能存在变化",
            "遵守航空公司政策"
        ]
    
    def build_prompt(self, context: dict) -> str:
        return f"""
{self.system_role}

## 能力范围
{chr(10).join(f"- {cap}" for cap in self.capabilities)}

## 约束条件
{chr(10).join(f"- {constraint}" for constraint in self.constraints)}

## 当前上下文
- 用户位置：{context.get('location', '未知')}
- 历史偏好：{context.get('preferences', '无')}

## 工具使用规范
使用structured output格式调用工具，确保参数完整准确。
"""

# 使用
prompt_builder = PromptTemplate()
context = {"location": "北京", "preferences": "经济舱"}
system_prompt = prompt_builder.build_prompt(context)
```

### 📝 Factor 3: Own your Context Window

**问题场景**：
```python
# ❌ 错误示例
def build_context(chat_history):
    # 把所有历史消息都塞进去
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
```

**正确做法**：
```python
# ✅ 正确示例
class ContextManager:
    def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens
        self.memory_keys = ["user_preferences", "current_task", "recent_errors"]
    
    def compress_context(self, chat_history: List[dict]) -> dict:
        # 保留系统状态
        system_state = self.extract_system_state(chat_history)
        
        # 压缩历史对话
        recent_messages = chat_history[-5:]  # 最近5条
        summary = self.summarize_old_messages(chat_history[:-5])
        
        return {
            "system_state": system_state,
            "conversation_summary": summary,
            "recent_messages": recent_messages,
            "memory_keys": self.get_memory_snapshot()
        }
    
    def estimate_tokens(self, text: str) -> int:
        # 简单估算，1 token ≈ 4 characters
        return len(text) // 4
    
    def get_memory_snapshot(self) -> dict:
        return {
            "user_preferences": {"class": "economy", "airline": "preferred_none"},
            "current_task": "flight_search",
            "recent_errors": []
        }
```

### 📝 Factor 8: Own your Control Flow

**问题场景**：
```python
# ❌ 错误示例 - 控制流混乱
def process_request(user_input):
    response = llm.generate(user_input)
    # 不知道下一步该做什么，全靠模型自己决定
    return response
```

**正确做法**：
```python
# ✅ 正确示例 - 清晰的控制流
from enum import Enum

class AgentState(Enum):
    INIT = "init"
    PLANNING = "planning"
    TOOL_CALLING = "tool_calling"
    WAITING_HUMAN = "waiting_human"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"

class FlightBookingAgent:
    def __init__(self):
        self.state = AgentState.INIT
        self.plan = []
        self.current_step = 0
    
    def process(self, user_input: str) -> dict:
        if self.state == AgentState.INIT:
            return self.handle_init(user_input)
        elif self.state == AgentState.PLANNING:
            return self.handle_planning(user_input)
        elif self.state == AgentState.TOOL_CALLING:
            return self.handle_tool_calling(user_input)
        elif self.state == AgentState.WAITING_HUMAN:
            return self.handle_human_input(user_input)
        # ... 其他状态处理
    
    def handle_init(self, user_input: str) -> dict:
        # 解析用户需求，生成计划
        self.plan = self.generate_plan(user_input)
        self.state = AgentState.PLANNING
        return {"message": "正在为您制定搜索计划...", "state": self.state.value}
    
    def generate_plan(self, user_input: str) -> List[dict]:
        # 生成结构化执行计划
        return [
            {"step": "parse_travel_info", "params": {"input": user_input}},
            {"step": "search_flights", "params": {}},
            {"step": "ask_human_confirmation", "params": {}},
            {"step": "book_ticket", "params": {}}
        ]
```

---

## 💻 完整代码示例

### 旅行助手 Agent 完整实现

```python
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum
import json
import logging

class AgentState(Enum):
    INIT = "init"
    PLANNING = "planning"
    SEARCHING = "searching"
    CONFIRMING = "confirming"
    BOOKING = "booking"
    COMPLETED = "completed"
    ERROR = "error"

class FlightSearchParams(BaseModel):
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str] = None
    passengers: int = 1
    cabin_class: str = "economy"

class FlightResult(BaseModel):
    flight_id: str
    airline: str
    departure_time: str
    arrival_time: str
    price: float
    duration: str

class TravelAgent:
    def __init__(self):
        self.state = AgentState.INIT
        self.context = {}
        self.plan = []
        self.current_step = 0
        self.logger = logging.getLogger(__name__)
        
    def process(self, user_input: str) -> dict:
        """主处理流程"""
        try:
            if self.state == AgentState.INIT:
                return self._handle_init(user_input)
            elif self.state == AgentState.PLANNING:
                return self._handle_planning()
            elif self.state == AgentState.SEARCHING:
                return self._handle_searching()
            elif self.state == AgentState.CONFIRMING:
                return self._handle_confirming(user_input)
            elif self.state == AgentState.BOOKING:
                return self._handle_booking()
            else:
                return self._handle_error("未知状态")
                
        except Exception as e:
            self.logger.error(f"处理请求时出错: {str(e)}")
            return self._handle_error(str(e))
    
    def _handle_init(self, user_input: str) -> dict:
        """初始化处理"""
        # 解析用户需求
        travel_info = self._parse_travel_request(user_input)
        if not travel_info:
            return {
                "message": "请提供更详细的旅行信息，比如出发地、目的地和日期",
                "state": self.state.value,
                "needs_input": True
            }
        
        self.context["travel_info"] = travel_info
        self.state = AgentState.PLANNING
        return {
            "message": f"正在为您搜索从{travel_info['origin']}到{travel_info['destination']}的机票...",
            "state": self.state.value
        }
    
    def _parse_travel_request(self, user_input: str) -> Optional[dict]:
        """解析旅行请求 - 实际项目中使用 LLM"""
        # 简化示例，实际应该用 LLM + structured output
        if "北京" in user_input and "上海" in user_input:
            return {
                "origin": "北京",
                "destination": "上海",
                "departure_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                "passengers": 1
            }
        return None
    
    def _handle_planning(self) -> dict:
        """计划处理"""
        self.plan = [
            {"action": "search_flights", "status": "pending"},
            {"action": "present_options", "status": "pending"},
            {"action": "confirm_booking", "status": "pending"}
        ]
        self.state = AgentState.SEARCHING
        return self._handle_searching()
    
    def _handle_searching(self) -> dict:
        """搜索处理"""
        travel_info = self.context["travel_info"]
        
        # 模拟搜索结果
        flights = [
            FlightResult(
                flight_id="CA1234",
                airline="中国国际航空",
                departure_time="08:00",
                arrival_time="10:30",
                price=580.0,
                duration="2h30m"
            ),
            FlightResult(
                flight_id="MU5678",
                airline="中国东方航空",
                departure_time="14:00",
                arrival_time="16:20",
                price=520.0,
                duration="2h20m"
            )
        ]
        
        self.context["search_results"] = flights
        self.state = AgentState.CONFIRMING
        
        # 格式化展示结果
        flight_options = []
        for i, flight in enumerate(flights, 1):
            flight_options.append(
                f"{i}. {flight.airline} {flight.flight_id}\n"
                f"   出发: {flight.departure_time} 到达: {flight.arrival_time}\n"
                f"   价格: ¥{flight.price} 飞行时长: {flight.duration}"
            )
        
        return {
            "message": f"找到以下航班选项：\n\n" + "\n\n".join(flight_options) + "\n\n请选择航班编号(1-2)，或输入'取消':",
            "state": self.state.value,
            "needs_input": True,
            "options": [f.dict() for f in flights]
        }
    
    def _handle_confirming(self, user_input: str) -> dict:
        """确认处理"""
        if user_input.lower() in ['取消', 'cancel']:
            self.state = AgentState.INIT
            return {"message": "已取消预订", "state": self.state.value}
        
        try:
            choice = int(user_input) - 1
            flights = self.context["search_results"]
            if 0 <= choice < len(flights):
                selected_flight = flights[choice]
                self.context["selected_flight"] = selected_flight
                
                return {
                    "message": f"您选择了 {selected_flight.airline} {selected_flight.flight_id}，"
                              f"价格 ¥{selected_flight.price}。\n\n"
                              f"⚠️ 确认预订吗？这将产生实际费用。(输入'确认'或'取消')",
                    "state": self.state.value,
                    "needs_input": True,
                    "requires_confirmation": True
                }
            else:
                return {
                    "message": "无效选择，请输入1-2之间的数字",
                    "state": self.state.value,
                    "needs_input": True
                }
        except ValueError:
            if user_input.lower() in ['确认', 'confirm', 'yes']:
                self.state = AgentState.BOOKING
                return self._handle_booking()
            else:
                return {
                    "message": "请输入有效的选择（1-2）或'确认'/'取消'",
                    "state": self.state.value,
                    "needs_input": True
                }
    
    def _handle_booking(self) -> dict:
        """预订处理"""
        selected_flight = self.context["selected_flight"]
        
        # 模拟预订流程
        booking_result = {
            "booking_id": "BK" + str(int(datetime.now().timestamp())),
            "status": "confirmed",
            "flight": selected_flight.dict(),
            "passenger_info": "需要后续完善"
        }
        
        self.context["booking_result"] = booking_result
        self.state = AgentState.COMPLETED
        
        return {
            "message": f"🎉 预订成功！\n\n"
                      f"预订编号: {booking_result['booking_id']}\n"
                      f"航班: {selected_flight.airline} {selected_flight.flight_id}\n"
                      f"价格: ¥{selected_flight.price}\n\n"
                      f"请保存预订编号，稍后会发送确认邮件。",
            "state": self.state.value,
            "booking_id": booking_result["booking_id"]
        }
    
    def _handle_error(self, error_msg: str) -> dict:
        """错误处理"""
        self.state = AgentState.ERROR
        return {
            "message": f"抱歉，处理请求时出现错误：{error_msg}",
            "state": self.state.value,
            "error": True
        }
    
    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            "state": self.state.value,
            "context": self.context,
            "plan": self.plan,
            "current_step": self.current_step
        }
    
    def resume_from_state(self, state_data: dict):
        """从状态恢复"""
        self.state = AgentState(state_data["state"])
        self.context = state_data["context"]
        self.plan = state_data.get("plan", [])
        self.current_step = state_data.get("current_step", 0)

# 使用示例
def main():
    agent = TravelAgent()
    
    # 模拟对话流程
    print("=== 旅行助手 Agent 演示 ===")
    
    # 第一轮：初始请求
    result = agent.process("我要订明天从北京到上海的机票")
    print(f"助手: {result['message']}")
    
    # 第二轮：显示选项
    result = agent.process("")  # 触发搜索
    print(f"助手: {result['message']}")
    
    # 第三轮：用户选择
    result = agent.process("1")
    print(f"助手: {result['message']}")
    
    # 第四轮：确认预订
    result = agent.process("确认")
    print(f"助手: {result['message']}")
    
    # 显示最终状态
    print(f"\n最终状态: {agent.get_state()}")

if __name__ == "__main__":
    main()
```

---

## ✅ 检查清单

在部署 Agent 前，请确保以下所有项目都已完成：

### 🔧 技术架构
- [ ] 所有工具都有明确的 JSON Schema 定义
- [ ] Prompt 采用模块化设计，支持动态组装
- [ ] 实现了 Context Window 管理和压缩机制
- [ ] 建立了状态管理系统（FSM 或类似）
- [ ] 支持 pause/resume 功能

### 🛡️ 安全性
- [ ] 高风险操作需要人工确认
- [ ] 实现了权限分级和操作白名单
- [ ] 有 rate limiting 和防滥用机制
- [ ] 敏感信息不会被日志记录
- [ ] 外部 API 调用有超时和重试机制

### 🔄 控制流
- [ ] 明确定义了 Agent 的状态转换逻辑
- [ ] 实现了错误处理和重试机制
- [ ] 支持多路径决策和回退
- [ ] 有循环检测和终止条件

### 📊 监控和测试
- [ ] 完整的日志记录（输入、输出、状态变化）
- [ ] 实现了 trace 和 replay 功能
- [ ] 有自动化测试用例
- [ ] 成本监控和告警机制

### 🚀 生产就绪
- [ ] 支持多渠道部署（API、Web、移动端）
- [ ] 实现了优雅降级和 fallback
- [ ] 有性能监控和告警
- [ ] 文档完整，团队成员可以快速上手

---

## 🙋 常见问题 FAQ

### Q1: Agent 经常"忘记"之前的对话，怎么办？
**A**: 这是 Factor 3 的问题。实现智能的 context 管理：
- 保留关键信息（用户偏好、当前任务状态）
- 压缩历史对话为摘要
- 使用外部存储保存长期记忆

### Q2: Agent 调用工具时参数总是错误？
**A**: 这是 Factor 1 的问题。解决方案：
- 使用严格的 JSON Schema 定义
- 采用 Function Calling 而非自然语言输出
- 在 prompt 中提供工具使用示例

### Q3: Agent 的行为不可预测，难以调试？
**A**: 这涉及多个因素：
- Factor 8: 实现明确的控制流
- Factor 12: 使用 stateless reducer 模式
- Factor 13: 完整的日志和 replay 功能

### Q4: 如何控制 Agent 的运行成本？
**A**: Factor 18 提供了完整方案：
- 实现智能的 context 压缩
- 使用缓存减少重复调用
- 设置预算告警和自动停止

### Q5: Agent 在生产环境中经常崩溃？
**A**: 这是 Factor 20 的问题：
- 实现优雅降级机制
- 添加服务熔断器
- 为所有外部依赖添加 fallback

### Q6: 如何测试 Agent 的行为？
**A**: Factor 17 提供了测试策略：
- 使用 mock 工具进行单元测试
- 建立 golden dataset 进行回归测试
- 实现 A/B 测试框架

### Q7: Agent 处理多模态内容时混乱？
**A**: Factor 19 的解决方案：
- 统一的媒体处理接口
- 标准化的格式转换
- 模态间的一致性检查

---

## 🎯 最佳实践总结

1. **从简单开始**：不要试图一次性实现所有功能，先做一个能工作的最小版本
2. **状态优先**：Agent 的状态管理是核心，其他功能都围绕状态设计
3. **工具标准化**：所有工具的输入输出都要标准化，这能解决80%的问题
4. **测试驱动**：每个功能都要有对应的测试用例
5. **监控完备**：生产环境中监控比功能更重要

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

- 📝 **文档改进**：发现错误或有更好的表达方式
- 💻 **代码示例**：提供更完整的实现示例
- 🐛 **问题反馈**：分享你在实践中遇到的问题
- 💡 **最佳实践**：分享你的生产环境经验

---

## 📄 许可证

MIT License - 自由使用，欢迎分享和修改。

---

⭐ 如果这个指南对你有帮助，请给个 Star 支持一下！

**最后更新**: 2025-07-11
