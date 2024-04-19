from stitch.pydantic_objects import Template


ROLE = """You will be given a question, as well as 8 potential answers to the question. You must pick which one of the 8 options is the correct answer."""
INSTRUCTIONS = """Think step-by-step and choose the correct likely answer. After your reasoning, you will write "Answer: ", followed by the letter of the answer you've chosen. You must always pick one answer, and you must always end your message with just "Answer: your one chosen answer letter"."""

XML_Template = Template(
    name="xml",
    doc_separator_start="<doc id={doc_id}>\n",
    doc_separator_end="\n</doc>",
    role_block="<role> " + ROLE + " </role>",
    document_block="<documents>\n{documents}\n</documents>",
    question_block="<question> {question} </question>",
    answers_block="<potential_answers>\n{answers}\n</potential_answers>",
    instruction_block="<instructions> " + INSTRUCTIONS + "</instructions>",
)

Text_Template = Template(
    name="text",
    doc_separator_start="=== Document {doc_id} ===\n",
    role_block="=== Role ===\n" + ROLE,
    document_block="=== Documents ===\n{documents}",
    question_block="=== Question ===\n{question}",
    answers_block="=== Potential Answers ===\n{answers}",
    instruction_block="=== Instructions ===\n" + INSTRUCTIONS,
)

Markdown_Template = Template(
    name="markdown",
    doc_separator_start="## Document {doc_id}\n",
    role_block="# Role \n" + ROLE,
    document_block="# Documents \n{documents}",
    question_block="# Question \n{question}",
    answers_block="# Potential Answers \n{answers}",
    instruction_block="# Instructions \n" + INSTRUCTIONS,
)
