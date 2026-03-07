# prompts.py
# Stores prompt templates for each summary style.
# Each function receives the input text and desired word limit,
# and returns a fully-formed prompt string for the LLM.

# Map focus areas (used in the UI) to specific instructions
FOCUS_AREA_MAP = {
    "All (Balanced)": "Provide a balanced summary covering the entire paper equally.",
    "Abstract / Introduction": "Focus primarily on the abstract, background, and introduction of the paper. Pay special attention to the core problem and motivations.",
    "Methodology": "Focus primarily on the methodology, experimental design, and theoretical framework used in the paper. Detail the methods rather than the conclusions.",
    "Results / Findings": "Focus primarily on the results, numerical data, and findings of the experiments. Highlight what was discovered rather than how it was done.",
    "Conclusion": "Focus primarily on the conclusions, implications, and future work discussed at the end of the paper.",
}

# --------------------------------------------------------------------------- #

def get_tldr_prompt(text: str, word_limit: int, focus_instruction: str) -> str:
    """
    Returns a prompt that asks the model for a short TLDR summary.
    """
    return (
        f"You are a helpful assistant that summarizes papers clearly and concisely.\n\n"
        f"Read the following text and write a TLDR (Too Long; Didn't Read) summary "
        f"in approximately {word_limit} words. "
        f"Be direct and capture only the single most important insight.\n\n"
        f"Focus Instruction: {focus_instruction}\n"
        f"(If the requested focus area is not clearly present in the text, do your best to summarize what is available, but state briefly that the specific section was not found.)\n\n"
        f"Text:\n{text}\n\n"
        f"TLDR Summary:"
    )


def get_bullet_prompt(text: str, word_limit: int, focus_instruction: str) -> str:
    """
    Returns a prompt that asks the model for a bullet-point summary.
    """
    return (
        f"You are a helpful assistant that summarizes papers clearly and concisely.\n\n"
        f"Read the following text and produce a bullet-point summary "
        f"in approximately {word_limit} words total. "
        f"Use 4–6 concise bullet points (•). Each bullet should capture one key idea.\n\n"
        f"Focus Instruction: {focus_instruction}\n"
        f"(If the requested focus area is not clearly present in the text, do your best to summarize what is available, but state briefly that the specific section was not found.)\n\n"
        f"Text:\n{text}\n\n"
        f"Bullet-Point Summary:"
    )


def get_detailed_prompt(text: str, word_limit: int, focus_instruction: str) -> str:
    """
    Returns a prompt that asks the model for a detailed explanation.
    """
    return (
        f"You are a helpful assistant that explains complex papers in plain language.\n\n"
        f"Read the following text and write a detailed explanation "
        f"in approximately {word_limit} words. "
        f"Cover the main topic, key findings or arguments, methodology (if any), "
        f"and conclusions. Write in clear, readable paragraphs.\n\n"
        f"Focus Instruction: {focus_instruction}\n"
        f"(If the requested focus area is not clearly present in the text, do your best to summarize what is available, but state briefly that the specific section was not found.)\n\n"
        f"Text:\n{text}\n\n"
        f"Detailed Explanation:"
    )


def get_technical_prompt(text: str, word_limit: int, focus_instruction: str) -> str:
    """
    Returns a prompt that asks the model for a technical explanation
    including mathematical equations in LaTeX notation.
    """
    return (
        f"You are an expert AI research assistant with deep knowledge of mathematics "
        f"and science. Read the following text and write a thorough technical explanation "
        f"in approximately {word_limit} words.\n\n"
        f"Requirements:\n"
        f"- Explain all key concepts in depth\n"
        f"- Include ALL important mathematical equations, formulas, and derivations "
        f"using LaTeX notation: use $...$ for inline math and $$...$$ for block equations\n"
        f"- Define every variable and symbol used in the equations\n"
        f"- Describe the methodology, architecture, or algorithm in detail\n"
        f"- Explain the significance of each formula in context\n"
        f"- Include key results and what they mean mathematically\n\n"
        f"Focus Instruction: {focus_instruction}\n"
        f"(If the requested focus area is not clearly present in the text, focus on the closest available technical details and note the absence of the requested section.)\n\n"
        f"Text:\n{text}\n\n"
        f"Technical Explanation with Equations:"
    )


# Map style names (used in the UI) to the corresponding prompt builder function
PROMPT_MAP = {
    "Short Summary (TLDR)":        get_tldr_prompt,
    "Bullet Point Summary":        get_bullet_prompt,
    "Detailed Explanation":        get_detailed_prompt,
    "Technical with Equations":    get_technical_prompt,
}
