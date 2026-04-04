from groq import Groq

client = Groq()

def ask_question(db, query):
    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a highly accurate financial document assistant designed for Chartered Accountants.

STRICT INSTRUCTIONS:
- Answer ONLY using the provided context.
- Do NOT guess or add information not present in the context.
- If the answer is not found, say: "Information not found in the document."
- Be precise and extract exact values when possible.
- Always include the page number(s) where the answer was found.
- If multiple values exist, list them clearly.
- Keep the answer concise and professional.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER FORMAT:
- Answer: <your answer>
- Page(s): <page numbers>
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    return answer, docs