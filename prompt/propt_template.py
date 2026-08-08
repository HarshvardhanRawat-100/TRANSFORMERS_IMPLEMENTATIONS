from langchain_core.prompts import PromptTemplate

# Load prompt template

template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True ,
    template="""
Summarize the research paper titled "{paper_input}".

Explanation Style: {style_input}
Explanation Length: {length_input}

Instructions:
- Explain clearly based on known knowledge of the paper
- Include key ideas, architecture, and contributions
- Add mathematical intuition if relevant
- Use analogies to simplify concepts

Do NOT say "insufficient information".
"""
)

template.save('template.json')