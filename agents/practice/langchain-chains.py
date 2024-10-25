"""
based on https://learn.deeplearning.ai/courses/langchain/lesson/4/chains
My updated version of chains
"""

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to English:"
    "\n\n{Review}"
)
# Chain 1: Input = Review and output = English_Review
chain_one = LLMChain(
    llm=llm,
    prompt=first_prompt,
    output_key="English_Review"
)
# Second prompt: Generate tags and comments based on English review
second_prompt = ChatPromptTemplate.from_template(
       "Product review: {English_Review}\n\n"
        "Create tags and detailed comments about this product.\n"
        "Structure: List of objects with tag and comment fields.\n"
        "Make sure to format as valid JSON array."
)
# Chain 2: Input = English_Review and output = tags_with_comments
chain_two = LLMChain(
    llm=llm,
    prompt=second_prompt,
    output_key="tags_with_comments"  # This will be a JSON string of an array of dicts
)
template3 = """
Given the following product comments and tags: {tags_with_comments}

Please analyze the comments and provide an evaluation with the following structure:
1. A score out of 10
2. A recommendation (yes/no)
3. A detailed reason for the score and recommendation

Format your response as a valid JSON object with the following keys:
- score: number between 0-10 (score/10)
- is_recommend: boolean
- reason: string explaining the evaluation


Example response format:
{
    "score": 3,
    "is_recommend": false,
    "reason": "The comments indicate significant dissatisfaction with the product, particularly regarding its taste and foam quality. Users report that the taste is poor compared to similar products, and the foam does not hold up well. Additionally, concerns about potential counterfeit or old batches further diminish the product's credibility. Given these issues, a recommendation against purchasing is warranted."
}

Please ensure the response is a properly formatted JSON object.

"""

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# Third prompt: Evaluate comments and provide recommendation
third_prompt = ChatPromptTemplate.from_template("Analysis: {tags_with_comments}\n\n"
        "Based on this analysis, provide:\n"
        "1. Recommendation\n"
        "2. Positives\n"
        "3. drawbacks\n"
        "4. Areas for improvement\n"
        "5. recommend score: a score out of 10")


# Chain 3: Input = tags_with_comments (which is expected to be JSON) and output = scores
chain_three = LLMChain(
    llm=llm,
    prompt=third_prompt,
    output_key="evaluation_result"
)
# Combine all chains into a sequential chain
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three],
    input_variables=["Review"],
    output_variables=["English_Review", "tags_with_comments", "evaluation_result"],
    verbose=True
)



结果：
```{
  "Review": "Je trouve le goût médiocre. La mousse ne tient pas, c'est bizarre. J'achète les mêmes dans le commerce et le goût est bien meilleur...\nVieux lot ou contrefaçon !?",
  "English_Review": "I find the taste mediocre. The foam doesn't hold, it's strange. I buy the same ones in stores and the taste is much better... Old batch or counterfeit!?",
  "tags_with_comments": [
    {
      "product": "Foam Cup",
      "tags": {
        "taste": "mediocre",
        "foam quality": "poor",
        "authenticity": "potentially counterfeit"
      },
      "comments": {
        "taste": "The taste of the foam cup is mediocre at best, not as good as similar products bought in stores.",
        "foam quality": "The foam doesn't hold well, which is strange and suggests low quality material.",
        "authenticity": "There's a possibility that this batch may be counterfeit, as the taste and foam quality do not meet expectations."
      }
    }
  ],
  "evaluation_result": {
    "1. Recommendation": "It is recommended to avoid purchasing or using the Foam Cup due to its mediocre taste, poor foam quality, and potentially counterfeit nature.",
    "2. Positives": "None mentioned in the analysis.",
    "3. Drawbacks": [
      "The taste is described as mediocre.",
      "The foam quality is poor and does not hold well.",
      "There is a concern about the authenticity of the product, potentially being counterfeit."
    ],
    "4. Areas for Improvement": [
      "Improve the taste of the foam cup to meet customers' expectations.",
      "Enhance the quality of the foam material used to ensure it holds well.",
      "Address any potential counterfeit issues to maintain trust with customers."
    ],
    "5. Recommend Score": "2 out of 10"
  }
}
```
