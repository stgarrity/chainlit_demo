from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable, evaluate

from dotenv import load_dotenv
load_dotenv()

client = wrap_openai(OpenAI())

TEST_PROMPT_V1='given the email message below, tell me how urgent it is that i reply to it? Do I need to reply within one hour, four hours, one day, or two days? please reply with only the time frame, and do not include your reasoning. "one hour", "four hours", "one day" or "two days" are the only acceptable replies'

@traceable
def leetcode_agent(inputs: dict) -> dict:
    messages = [
        {"role": "system", "content": TEST_PROMPT_V1},
        *inputs["messages"]
    ]

    result = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )

    return {
        "message": {
            "role": "assistant",
            "content": result.choices[0].message.content
        }
    }

# The name or UUID of the LangSmith dataset to evaluate on.
data = "week1milestone3"

# A string to prefix the experiment name with.
experiment_prefix = "week1milestone4"

def correctness_evaluator(run, example) -> dict:
    """
    Evaluates the correctness of generated unit tests
    
    Args:
        run: Contains the run information including inputs and outputs
        example: Contains the reference example if available
    
    Returns:
        Dictionary with score (0-1) and explanation
    """
    # Extract the original LeetCode problem from inputs
    leetcode_problem = run.inputs["inputs"]["messages"][-1]["content"]
    
    # Extract the model's generated tests
    generated_tests = run.outputs["message"]["content"]
    
    # Rest of the evaluation logic remains the same
    evaluation_prompt = f"""
    Given this email:
    {leetcode_problem}

    Evaluate this response for correctly determining the urgency of the response required
    {generated_tests}
    
    Score from 0-4:
    2 = The answer is of the correct format and matches the urgency required of the email.
    1 = The answer is of the correct format, but does not make sense given the context of the email
    0 = The answer is not one of the strings "one hour", "four hours", "one day", or "two days"

    Return only the number (0-2).
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a test evaluation assistant. Respond only with a number 0-2."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0
    )
    
    try:
        score = int(response.choices[0].message.content.strip())
        return {
            "key": "correctness score",
            "score": score / 2,  # Normalize to 0-1
            "explanation": f"Test correctness score: {score}/2"
        }
    except ValueError:
        return {
            "key": "correctness score",
            "score": 0,
            "explanation": "Failed to parse score"
        }

# List of evaluators to score the outputs of target task
evaluators = [
    correctness_evaluator
]

# Evaluate the target task
results = evaluate(
    leetcode_agent,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix
)
