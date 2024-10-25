"""
based on https://learn.deeplearning.ai/courses/langchain/lesson/4/chains
My updated version of chains
"""

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# First prompt: Translate review to English
template1 = PromptTemplate(
    input_variables=["Review"],
    template="Translate the following review to English:\n\n{Review}"
)
first_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate(prompt=template1)
])

chain_one = LLMChain(
    llm=llm,
    prompt=first_prompt,
    output_key="English_Review"
)

# Second prompt: Generate tags and comments
template2 = PromptTemplate(
    input_variables=["English_Review"],
    template=(
        "Product review: {English_Review}\n\n"
        "Create tags and detailed comments about this product.\n"
        "Structure: List of objects with tag and comment fields.\n"
        "Make sure to format as valid JSON array."
    )
)
second_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate(prompt=template2)
])

chain_two = LLMChain(
    llm=llm,
    prompt=second_prompt,
    output_key="tags_with_comments"
)

# Third prompt: Final recommendation
template3 = PromptTemplate(
    input_variables=["tags_with_comments"],
    template=(
        "Analysis: {tags_with_comments}\n\n"
        "Based on this analysis, provide:\n"
        "1. Recommendation\n"
        "2. Positives\n"
        "3. Areas for improvement"
    )
)
third_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate(prompt=template3)
])

chain_three = LLMChain(
    llm=llm,
    prompt=third_prompt,
    output_key="recommendation"
)

# Combine chains
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three],
    input_variables=["Review"],
    output_variables=["English_Review", "tags_with_comments", "recommendation"],
    verbose=True
)
