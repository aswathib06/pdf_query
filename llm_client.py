from utils.embedder import load_embedder, retrieve_top_k
import requests
import openai

def call_openai(api_key, system_prompt, prompt, model_name):
    openai.api_key = api_key
    res = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content

def call_hf(api_key, model_name, prompt):
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}

    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()[0]["generated_text"]

def answer_with_context(question, chunks, embedder, embeddings, nn, provider, api_key, model_name):
    q_emb = embedder.encode(question, convert_to_numpy=True)
    idxs = retrieve_top_k(q_emb, nn, k=4)

    context = "\n\n---\n\n".join(chunks[i] for i in idxs)
    system_prompt = "You are an assistant that answers only from the given context."
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    if provider == "openai":
        return call_openai(api_key, system_prompt, prompt, model_name)
    else:
        return call_hf(api_key, model_name, prompt)
