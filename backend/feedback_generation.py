# Provide feedback and suggestions based on the predicted class label
def feedback_generator(squat_class,llm):
    # Prompt template for AI trainer feedback
    template = """
    Prompt: You are an AI trainer analyzing squats. Your task is to provide feedback on the incorrectness of a squat and suggest corrections. The possible incorrectness categories are:

    1. Good: The squat technique is correct.
    2. Bad Head: The head is misaligned, either looking up excessively or looking down.
    3. Bad Back Warp: The back excessively arches or hyperextends during the squat.
    4. Bad Back Round: The back is rounded instead of maintaining a neutral spine.
    5. Bad Inner Thigh: The knees collapse inward during the squat.
    6. Bad Toe: The weight shifts to the toes instead of being balanced on the whole foot.
    7. Bad Shallow: The squat depth is insufficient, not going low enough.
    
    Given an input describing the incorrectness category, your AI should provide feedback and suggest corrections to improve the squat technique.

    Example Input: "bad_inner_thigh"

    Example Output:
    Stand with your feet slightly wider than shoulder-width apart and point your toes slightly outwards to ensure better alignment during the squat.
    Below is the type of incorrectness: {squat_class}

    Only if the class is "good" congratulate them and suggest something.
    make sure to always tell what is done wrong before providing the feedback.

    YOUR RESPONSE:
    """
    prompt_with_squat_class = template.format(squat_class=squat_class)
    trainer_feedback = llm(prompt_with_squat_class)
    return trainer_feedback
