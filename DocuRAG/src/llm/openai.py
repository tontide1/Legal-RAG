def get_openai_response(prompt, api_key):
    import openai

    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message['content']


def get_openai_embedding(text, api_key):
    import openai

    openai.api_key = api_key

    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )

    return response['data'][0]['embedding']