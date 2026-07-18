# judges/azure_double_judge.py
from concurrent.futures import ThreadPoolExecutor

def double_judge(chunk_text: str) -> JudgeVerdict:
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_a = ex.submit(ai_foundry_invoke, "gpt-4o",       JUDGE_PROMPT, chunk_text)
        f_b = ex.submit(ai_foundry_invoke, "llama-judge",  JUDGE_PROMPT, chunk_text)
        a, b = f_a.result(), f_b.result()
    if a.verdict != b.verdict:
        return escalate_to_human(chunk_text, a, b)   # <- discrepancia ⇒ humano
    return a