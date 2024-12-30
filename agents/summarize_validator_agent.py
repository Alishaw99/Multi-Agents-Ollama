# agents/summarize_validator_agent.py

from .agent_base import AgentBase

class SummarizeValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SummarizeValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, original_text, summary):
        system_message = "You are an Qaulatative data Analyst  and AI assistant that validates that correct themes have been extracted from the texts."
        user_content = (
            "Given the original text and assess whether the text has been properly analyzed and correct and precise themes are captured from the original text.\n"
            "Provide a brief analysis and rate the themes on a scale of 1 to 5, where 5 indicates excellent quality.\n\n"
            f"Original Text:\n{original_text}\n\n"
            f"Summary:\n{summary}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        validation = self.call_llama(messages, max_tokens=512)
        return validation
