import requests
import feedparser
import csv

# === CONFIG ===
# First group (LLM terms)
group1 = [
    "large language model",
    "LLM",
    "foundation models",
    "transformers"
]

# Second group (code quality terms)
group2 = [
    "code smells",
    "code defects",
    "prompt smells",
    "code quality",
    "best practices",
    "technical debt",
    "common coding mistakes",
    "anti-pattern",
    "inference quality",
    "failures"
]

# === BUILD QUERY ===
def build_group_query(group, fields=("ti", "abs")):
    terms = []
    for term in group:
        for field in fields:
            terms.append(f'{field}:"{term}"')
    return "(" + " OR ".join(terms) + ")"

query_part1 = build_group_query(group1)
query_part2 = build_group_query(group2)

final_query = f"{query_part1} AND {query_part2}"
print("Query:", final_query)

# === FETCH RESULTS ===
base_url = "http://export.arxiv.org/api/query"
max_results = 2000  # API allows up to 2000
params = {
    "search_query": final_query,
    "start": 0,
    "max_results": max_results,
    "sortBy": "submittedDate",
    "sortOrder": "descending"
}

response = requests.get(base_url, params=params)
feed = feedparser.parse(response.text)

# === DISPLAY COUNT ===
entries = feed.entries
print(f"\nFound {len(entries)} results on arXiv.\n")

# === ASK TO SAVE CSV ===
save = input("Do you want to save the results as a CSV for Rayyan? (y/n): ").strip().lower()

if save == "y":
    filename = "arxiv_results.csv"
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Rayyan-compatible columns: Title, Abstract, Authors, Link, Published
        writer.writerow(["Title", "Abstract", "Authors", "Link", "Published"])
        for e in entries:
            title = e.title.replace("\n", " ").strip()
            abstract = e.summary.replace("\n", " ").strip()
            authors = ", ".join([a.name for a in e.authors])
            link = e.link
            published = e.published
            writer.writerow([title, abstract, authors, link, published])
    print(f"\n✅ Saved {len(entries)} results to {filename}")

else:
    print("Skipped saving.")

