from textwrap import dedent

PROMPT_TEMPLATE = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""


def get_instructions(target_language):
    """
    Generate translation instructions dynamically based on the target language.
    """
    instruction_report = (
        f"Translate the following software security content into {target_language} "
        "while preserving technical accuracy and meaning."
    )
    instruction_general = (
        instruction_report
        + " Translate the following text while strictly maintaining its original structure, "
        "including all formatting elements. The text contains markdown and HTML formatting "
        "created in a WYSIWYG/text editor. Ensure the following during translation:\n"
        "1. Preserve all leading and trailing spaces, tabs, and indentations exactly as they are.\n"
        "2. Retain all special characters, markdown syntax, and HTML tags without modification.\n"
        "3. Do not add, remove, or alter any formatting elements, even if they seem unnecessary.\n"
        "4. Focus only on translating human-readable content while leaving code, tags, and formatting untouched."
    )
    return instruction_report, instruction_general


