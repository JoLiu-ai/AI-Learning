from typing import Any, Dict, List, Mapping, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.router import MultiPromptChain, RouterChain
from pydantic import BaseModel
from langchain.chains.base import Chain

# Initialize LLM
llm = ChatOpenAI(temperature=0)

# Define different task prompts and LLMChains
# 1. Math calculation task
math_prompt_template = "你是一个数学专家。请回答以下数学问题：{input}"
math_prompt = PromptTemplate(template=math_prompt_template, input_variables=["input"])
math_chain = LLMChain(llm=llm, prompt=math_prompt)

# 2. Knowledge task
knowledge_prompt_template = "你是一个百科全书。请回答以下关于常识性问题：{input}"
knowledge_prompt = PromptTemplate(template=knowledge_prompt_template, input_variables=["input"])
knowledge_chain = LLMChain(llm=llm, prompt=knowledge_prompt)

# 3. Text summarization task
summary_prompt_template = "你是一个精炼的文本摘要工具。请为以下内容生成摘要：{input}"
summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["input"])
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Define default handling task
default_prompt_template = "这是一个常规问题，请提供合适的答案：{input}"
default_prompt = PromptTemplate(template=default_prompt_template, input_variables=["input"])
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# Define task information
prompt_infos = [
    {
        "name": "math",
        "description": "适用于数学计算、方程式、数学问题等",
    },
    {
        "name": "knowledge",
        "description": "适用于常识性问题、科学知识、历史、地理等知识性问题",
    },
    {
        "name": "summary",
        "description": "适用于需要总结、概括、提炼要点的文本内容",
    }
]

# Create destination_chains
destination_chains = {
    "math": math_chain,
    "knowledge": knowledge_chain,
    "summary": summary_chain
}

# Create a router prompt template
router_template = """作为一个智能路由系统，你需要将用户的问题分配给最合适的专家来回答。

可选的专家类型有：
{destinations}

用户问题: {input}

请你仔细分析这个问题，然后只回答一个最合适的专家类型（math/knowledge/summary/default）。
注意：如果都不太合适，请回答 default。

你的回答："""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input", "destinations"],
)

# Create a router chain that uses LLM to decide
class LLMRouterChain(RouterChain, BaseModel):
    llm_chain: LLMChain
    destination_chains: Dict[str, Chain]
    
    @classmethod
    def from_llm(cls, llm, prompt, destination_chains):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, destination_chains=destination_chains)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        # 构建专家描述文本
        destinations = "\n".join([
            f"- {info['name']}: {info['description']}"
            for info in prompt_infos
        ])
        
        # 让模型决定路由
        router_result = self.llm_chain.run(
            input=inputs["input"],
            destinations=destinations
        ).strip().lower()
        
        # 确保返回有效的目标
        if router_result not in ["math", "knowledge", "summary", "default"]:
            router_result = "default"
            
        return {
            "destination": router_result,
            "next_inputs": inputs
        }

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["destination", "next_inputs"]

# Create the router chain
router_chain = LLMRouterChain.from_llm(
    llm=llm,
    prompt=router_prompt,
    destination_chains=destination_chains
)

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
        # 使用路由链来决定目标
        route_output = self.router_chain._call(inputs)
        router_name = route_output["destination"]
        
        # 根据路由结果选择相应的chain
        if router_name in self.destination_chains:
            chain = self.destination_chains[router_name]
        else:
            chain = self.default_chain
            router_name = "default"
        
        # 运行选中的chain
        response = chain.run(inputs["input"])
        
        return {
            "router_name": router_name,
            "response": response
        }

# 创建主链
custom_multi_prompt_chain = CustomMultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    prompt_infos=prompt_infos
)

# 测试用例
test_inputs = [
    {"input": "12的平方根是多少？"},            # 应该路由到 math
    {"input": "太阳系中最大的行星是什么？"},    # 应该路由到 knowledge
    {"input": "总结一下人工智能的应用领域"},    # 应该路由到 summary
    {"input": "你觉得天气怎么样？"}             # 应该路由到 default
]

# 运行测试
for input_data in test_inputs:
    output = custom_multi_prompt_chain(input_data)
    print("\nInput:", input_data["input"])
    print("Router:", output["router_name"])
    print("Response:", output["response"])
