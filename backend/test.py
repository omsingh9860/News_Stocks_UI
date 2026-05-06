import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

TRADINGVIEW_URL = "https://in.tradingview.com/markets/stocks-india/ideas/"

response = requests.get(TRADINGVIEW_URL, timeout=15)
response.encoding = "utf-8"

print("Status Code:", response.status_code)
print("Content Length:", len(response.text))

soup = BeautifulSoup(response.text, "html.parser")

# OPTIONAL DEBUG: check container exists
container = soup.find("div", class_="listContainer-smEfFVbv")
print("\nContainer found:", container is not None)

# Your selectors (current version)
all_ideas = soup.find_all(
    "a",
    class_="title-tkslJwxl line-clamp-tkslJwxl stretched-outline-tkslJwxl"
)

all_conditions = soup.find_all(
    "span",
    class_="visuallyHiddenLabel-cYxls04V"
)

print("\nIdeas found:", len(all_ideas))
print("Conditions found:", len(all_conditions))

print("\n--- SAMPLE IDEAS ---\n")

for i in range(min(10, len(all_ideas))):
    try:
        idea_tag = all_ideas[i]
        condition_tag = all_conditions[i] if i < len(all_conditions) else None

        href = idea_tag.get("href", "")
        full_link = urljoin("https://in.tradingview.com", href)

        text = idea_tag.get_text(strip=True)

        condition_text = condition_tag.get_text(strip=True) if condition_tag else "N/A"

        print(f"{i+1}. {text}")
        print("   Link:", full_link)
        print("   Condition:", condition_text)
        print("-" * 50)

    except Exception as e:
        print("[ERROR]", e)