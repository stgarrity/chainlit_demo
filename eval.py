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
    email = run.inputs["inputs"]["messages"][-1]["content"]
    
    # Extract the model's generated tests
    generated_tests = run.outputs["message"]["content"]
    
    # Rest of the evaluation logic remains the same
    evaluation_prompt = f"""
    Given this email:
    {email}

    Evaluate this response for correctly determining the urgency of the response required
    {generated_tests}
    
    Evaluate the response for its accuracy in determining the urgency required for a reply. Use the following criteria to score the response:
	1.	Correct Format: The response must be one of the strings: "one hour", "four hours", "one day", or "two days". The strings may or may not be quoted.
	2.	Contextual Match: The response should align with the urgency implied by the email’s content.

    Score the response as follows:
	4: The response is in the correct format and perfectly matches the urgency required by the email.
	3: The response is in the correct format and matches the urgency but misses slight nuances.
	2: The response is in the correct format but does not match the urgency of the email.
	1: The response is in the correct format but is entirely illogical for the email’s context.
	0: The response is not in one of the accepted formats (“one hour”, “four hours”, “one day”, or “two days”, with the quotes being optional).

    Return only the score as a number (0-4).
"""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a test evaluation assistant. Respond only with a number 0-4."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0
    )
    
    try:
        score = int(response.choices[0].message.content.strip())
        return {
            "key": "correctness score",
            "score": score / 4,  # Normalize to 0-1
            "explanation": f"Test correctness score: {score}/4"
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
