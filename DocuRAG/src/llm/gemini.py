def get_gemini_response(api_key, prompt):
    import requests

    url = "https://api.gemini.com/v1/ask"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 150
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json().get("response")
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

def process_gemini_response(response):
    # Process the response from the Gemini API as needed
    return response.strip() if response else "No response received."