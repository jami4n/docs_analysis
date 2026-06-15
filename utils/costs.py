"""
costs.py

Cost estimation helpers.
"""


def estimate_gemini_cost(
    prompt_tokens,
    completion_tokens
):
    """
    Rough Gemini Flash estimate.

    Update these numbers later if pricing changes.
    """

    INPUT_COST_PER_MILLION = 0.30
    OUTPUT_COST_PER_MILLION = 2.50

    input_cost = (
        prompt_tokens / 1_000_000
    ) * INPUT_COST_PER_MILLION

    output_cost = (
        completion_tokens / 1_000_000
    ) * OUTPUT_COST_PER_MILLION

    return input_cost + output_cost