from typing import Any, Dict, List, Mapping, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.router import MultiPromptChain, RouterChain
from pydantic import BaseModel
from langchain.chains.base import Chain

# Define different task prompts and LLMChains
# 1. Math calculation task
math_prompt_template = "你是一个数学专家。请回答以下数学问题：{input}"
math_prompt = PromptTemplate(template=math_prompt_template, input_variables=["input"])
math_chain = LLMChain(llm=llm, prompt=math_prompt)

# # 2. Knowledge task
# knowledge_prompt_template = "你是一个百科全书。请回答以下关于常识性问题：{input}"
# knowledge_prompt = PromptTemplate(template=knowledge_prompt_template, input_variables=["input"])
# knowledge_chain = LLMChain(llm=llm, prompt=knowledge_prompt)

# 3. Text summarization task
summary_prompt_template = "你是一个精炼的文本摘要工具。请为以下内容生成摘要：{input}"
summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["input"])
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Define default handling task
default_prompt_template = "这是一个常规问题，请提供合适的答案：{input}"
default_prompt = PromptTemplate(template=default_prompt_template, input_variables=["input"])
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# Define task information including descriptions and keywords
prompt_infos = [
    {
        "name": "math",
        "description": "回答数学问题",
        "prompt_template": math_prompt_template,
        "keywords": ["计算", "等于", "平方", "除以", "乘以", "加上", "减去", "数学", "根"]
    },
#     {
#         "name": "knowledge",
#         "description": "回答常识性问题",
#         "prompt_template": knowledge_prompt_template,
#         "keywords": ["是什么", "为什么", "怎么", "哪个", "谁", "何时", "在哪里", "多少"]
#     },
    {
        "name": "summary",
        "description": "生成文本摘要",
        "prompt_template": summary_prompt_template,
        "keywords": ["总结", "概括", "摘要", "归纳", "总的来说"]
    }
]

# Create destination_chains
destination_chains = {
    info["name"]: LLMChain(llm=llm, prompt=PromptTemplate(template=info["prompt_template"], input_variables=["input"]))
    for info in prompt_infos
}

class SimpleRouterChain(RouterChain):
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        return {"destination": "default", "next_inputs": inputs}

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["destination", "next_inputs"]

class CustomMultiPromptChain(Chain):
    router_chain: RouterChain
    destination_chains: Mapping[str, Chain]
    default_chain: Chain
    prompt_infos: List[Dict]

    class Config:
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["response", "router_name"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        input_text = inputs["input"].lower()
        router_name = "default"
        response = None
        
        # 检查每个任务类型的关键词
        for prompt_info in self.prompt_infos:
            if any(keyword in input_text for keyword in prompt_info["keywords"]):
                router_name = prompt_info["name"]
                chain = self.destination_chains[router_name]
                response = chain.run(inputs["input"])
                break
        
        # 如果没有匹配到任何任务，使用默认链
        if response is None:
            response = self.default_chain.run(inputs["input"])

        return {
            "router_name": router_name,
            "response": response
        }

# 创建自定义链
custom_multi_prompt_chain = CustomMultiPromptChain(
    router_chain=SimpleRouterChain(),
    destination_chains=destination_chains,
    default_chain=default_chain,
    prompt_infos=prompt_infos
)

# 测试不同类型的输入
test_inputs = [
    {"input": "12的平方根是多少？"},            # Math task
    {"input": "太阳系中最大的行星是什么？"},    # Knowledge task
    {"input": "总结一下人工智能的应用领域"},    # Summary task
    {"input": "你觉得天气怎么样？"}             # Default task
]

# 运行测试
for input_data in test_inputs:
    output = custom_multi_prompt_chain(input_data)
    print("\nInput:", input_data["input"])
    print("Router:", output["router_name"])
    print("Response:", output["response"])
