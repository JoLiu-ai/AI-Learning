"""
modify code from https://learn.deeplearning.ai/courses/langchain/lesson/4/chains and give more detailed output
# Multi-Prompt Chain for Task Routing
## Functionality
This code implements a multi-prompt chain that routes user inputs to specific language model tasks:

## Input and Output
### Example Input
```json
{"input": "12的平方根是多少？"}  // Math task
```

### Example Output
```json
{
    "input": "12的平方根是多少？",
    "router_name": "math",
    "response": "3.464"  // Response from the math chain
}
```
"""


import json
from typing import Any, Dict, List, Mapping
from langchain.chains.router import RouterChain
from pydantic import BaseModel
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.prompts import PromptTemplate

llm_model = "gpt-3.5-turbo"
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""

computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>


"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template
    },
    {
        "name": "History",
        "description": "Good for answering history questions",
        "prompt_template": history_template
    },
    {
        "name": "computer science",
        "description": "Good for answering computer science questions",
        "prompt_template": computerscience_template
    }
]
router_dict_ = [p['name'].lower() for p in prompt_infos]


# Create a router chain that uses LLM to decide
class LLMRouterChain(RouterChain, BaseModel):
    llm_chain: LLMChain
    destination_chains: Dict[str, Chain]

    @classmethod
    def from_llm(cls, llm, prompt, destination_chains):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, destination_chains=destination_chains)

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        destinations = "\n".join([
            f"- {info['name']}: {info['description']}"
            for info in prompt_infos
        ])

        router_result = self.llm_chain.run(
            input=inputs["input"],
            destinations=destinations
        ).strip()

        try:
            json_str = router_result[router_result.find('{'):router_result.rfind('}') + 1]
            # print("Extracted JSON:", json_str)  # 调试输出
            result_dict = json.loads(json_str)
            destination = result_dict["destination"].lower()
            # print("Parsed destination:", destination)  # 调试输出

            if destination not in router_dict_:
                print(f"Invalid destination: {destination}")  # 调试输出
                destination = "default"

            return {
                "destination": destination,
                "next_inputs": inputs
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing result: {e}")  # 调试输出
            return {
                "destination": "default",
                "next_inputs": inputs
            }

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return ["destination", "next_inputs"]


destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
llm = ChatOpenAI(temperature=0, model=llm_model)
# router_chain = LLMRouterChain.from_llm(llm, router_prompt)

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

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

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)
# 创建主链
custom_multi_prompt_chain = CustomMultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    prompt_infos=prompt_infos
)

custom_multi_prompt_chain("What is black body radiation?")
