import json
import requests
import numpy as np
import os
import csv

AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]

def embedAndStore():
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    phaseA = {
        "A1": "Install uv (if required) and run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/datagen.py with ${user.email} as the only argument.",
        "A2": "Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place",
        "A3": "The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to /data/dates-wednesdays.txt",
        "A4": "Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json",
        "A5": "Write the first line of the 10 most recent .log file in /data/logs/ to /data/logs-recent.txt, most recent first",
        "A6": "Find all Markdown (.md) files in /data/docs/. For each file, extract the first occurrance of each H1 (i.e. a line starting with # ). Create an index file /data/docs/index.json that maps each filename (without the /data/docs/ prefix) to its title (e.g. {\"README.md\": \"Home\", \"path/to/large-language-models.md\": \"Large Language Models\", ...})",
        "A7": "/data/email.txt contains an email message. Pass the content to an LLM with instructions to extract the sender's email address, and write just the email address to /data/email-sender.txt",
        "A8": "/data/credit-card.png contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to /data/credit-card.txt",
        "A9": "/data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line",
        "A10": "The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type? Write the number in /data/ticket-sales-gold.txt"
    }

    tasks = list(phaseA.values())

    payload = {
        "model": "text-embedding-3-small",
        "input": tasks
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        embeddings = {}

        for task, embs in zip(tasks, response.json()["data"]):
            embeddings[task] = np.array(embs["embedding"])

        print(len(response.json()["data"][0]["embedding"]))

        with open("pA_task_embs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task"] + [f"dim_{i}" for i in range(1536)])  # Header row
            
            for task, embs in zip(tasks, response.json()["data"]):
                embedding = np.array(embs["embedding"])
                writer.writerow([task] + embedding.tolist())

        return
    else:
        raise Exception(f"OpenAI API Error: {response.json()}")

embedAndStore()