# src/templates/prompts.py


GEN_PROMPT_TEMP = (
    "You are a creative assistant suggesting domain names.\n\n"
        "Business: {desc}\n\n"
        "Rules:\n"
        "- lowercase only\n"
        "- 3-10 letters before the TLD\n"
        "- no numbers, no leading/trailing hyphens, no profanity\n"
        "- prefer .com, .io, .org, .net\n\n"
        "Return exactly {n} domain names as a JSON array of strings.\n"
        'Example: ["brandly.com", "neocafe.io", "greenbrew.org"]')

SFT_PROMPT_TEMP = (
    "You are a helpful assistant that suggests short, brandable domain names.\n"
    "Rules: lowercase, avoid numbers, avoid leading/trailing hyphens, avoid profanity, 3-10 letters before TLD.\n"
    "Return ONLY a JSON array of domain strings.\n\n"
    "Business: {desc}"
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent judge for domain name suggestions. "
    "Return only valid JSON.\n\n"
    "Scores (0.0–1.0): relevance, memorability, readability, safety.\n"
    "Compute 'overall' as weighted average using provided weights."
)
JUDGE_USER_PROMPT_TEMP = (
    "Evaluate domain suggestions for the business.\n\n"
    "Business:\n{business}\n\n"
    "Suggestions (JSON array of strings):\n{suggestions}\n\n"
    "Weights (JSON):\n{weights}\n\n"
    "Return a JSON array like:\n"
    '[{{"domain":"...", "relevance":0.8, "memorability":0.7, "readability":0.9, "safety":1.0, "overall":0.84}}]'
)
