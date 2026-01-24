CLAIM_EXTRACTION_PROMPT = """You are an information extraction system.

Task:
Extract all atomic, factual claims from the following answer.

Rules:
- Each claim must be a single, verifiable fact.
- Do NOT include opinions, explanations, or vague statements.
- Do NOT include speculative language (e.g., might, could, possibly).
- Do NOT repeat claims.
- If no factual claims are present, return an empty list.
- Output ONLY valid JSON.
- JSON must be a list of strings.

Answer:
"{LLM_ANSWER}"
"""
