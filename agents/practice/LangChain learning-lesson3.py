"""
from : learn.deeplearning.ai 
Lession 
https://learn.deeplearning.ai/courses/langchain/lesson/4/chains
LangChain for LLM Application Development

fix problem about: Lesson 3: Chains, invalid chain name “biology”
"""

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
    "destination": string \  "DEFAULT" or name of the prompt to use in {destinations}
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: 
"destination" -if the input is not well-suited for any of the candidate prompts, set it as "DEFAULT".

REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""
chain.run("Why does every cell in our body contain DNA?")

"""
output:
> Entering new MultiPromptChain chain...
None: {'input': 'Why does every cell in our body contain DNA?'}
> Finished chain.
'Every cell in our body contains DNA because DNA carries the genetic information that determines the characteristics and functions of an organism. ...
"""
