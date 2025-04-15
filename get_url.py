import requests

def get_privacy_policy_url(api_key, input):

    url = "https://google.serper.dev/search"

    payload = {"q": f"{input} privacy policy"}

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)

    data = response.json()

    try:
        privacy_policy_url = data['organic'][0]['link']
    except (KeyError, IndexError):
        privacy_policy_url = None
    return privacy_policy_url