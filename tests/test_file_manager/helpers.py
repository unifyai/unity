from unity.common.llm_client import new_llm_client


async def ask_judge(
    instruction: str,
    response: str,
    before_state: dict | None = None,
    after_state: dict | None = None,
    file_content: str | dict | None = None,
) -> str:
    """Asks an LLM judge to evaluate if a file manager operation was successful."""
    client = new_llm_client(stateful=False)

    prompt = "You are a test evaluator. Your task is to determine if a file management operation was successful based on the provided information.\n\n"
    prompt += f'Initial instruction:\n"{instruction}"\n\n'

    if file_content:
        prompt += f"Content of the relevant file(s):\n{file_content}\n\n"

    if before_state is not None:
        prompt += f"State of file system before operation:\n{before_state}\n\n"

    if after_state is not None:
        prompt += f"State of file system after operation:\n{after_state}\n\n"

    prompt += f'Final response from the file manager:\n"{response}"\n\n'

    prompt += "Based on all the information, was the operation performed correctly and plausibly?\n"
    prompt += 'Answer with a single word: "Correct" or "Incorrect", followed by a brief explanation.'

    verdict = await client.generate(prompt)
    return verdict
