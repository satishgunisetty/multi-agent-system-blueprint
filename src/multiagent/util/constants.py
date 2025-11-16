# LLM_ENDPOINT_NAME = "azure-gpt4o-mini-testing"
# LLM_ENDPOINT_NAME = "model-daia-dev-genai-platform-o4-mini"
LLM_ENDPOINT_NAME = "databricks-gpt-5"
HELP_INTENT_PATTERNS = [
    r"\bhow\s+can\s+you\s+help\b",
    r"\bwhat\s+can\s+you\s+do\b",
    r"\bcan\s+you\s+help\b",
    r"\bi\s+need\s+assistance\b",
    r"\bwhat\s+help\s+do\s+you\s+provide\b",
    r"\bhow\s+do\s+you\s+support\b",
]

SUPPORT_KEYWORDS = ["help", "support", "assist", "issue", "problem", "trouble"]

YES_KEYWORDS = ["yes", "yeah", "yep", "ok", "sure"]

NO_KEYWORDS = ["no", "nope", "nah", "not really", "not now"]
