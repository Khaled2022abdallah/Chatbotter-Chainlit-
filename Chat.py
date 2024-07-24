from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)
def get_prompt(instruction: str, history: list[str] = None) -> str:
    system = " you are a helpful AI assistant. You answer the question in a short and concise way"
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is conversation history:{''.join(history)} Now answer the question"
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


history = []
question = "which city is the capital of India?"

answer = ""
for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = " And which is of Lebanon"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
