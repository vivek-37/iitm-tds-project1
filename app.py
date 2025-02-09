from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import requests
import numpy as np
import os
import csv
import subprocess

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])   

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

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

functions = {
    "Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place": "format_md_with_prettier(version:str, source:str)",
    "The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to /data/dates-wednesdays.txt": "count_weekdays(weekday: str, source: str, destination: str)",
    "Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json": "sort_contacts(source: str, destination: str)",
    "Write the first line of the 10 most recent .log file in /data/logs/ to /data/logs-recent.txt, most recent first": "write_recent_logs(no_of_logs: str, source: str, destination: str)",
    "Find all Markdown (.md) files in /data/docs/. For each file, extract the first occurrance of each H1 (i.e. a line starting with # ). Create an index file /data/docs/index.json that maps each filename (without the /data/docs/ prefix) to its title (e.g. {\"README.md\": \"Home\", \"path/to/large-language-models.md\": \"Large Language Models\", ...})": "extract_h1(source: str, index_path: str, destination: str)",
    "/data/email.txt contains an email message. Pass the content to an LLM with instructions to extract the sender's email address, and write just the email address to /data/email-sender.txt": "extract_email(source: str, destination: str)",
    "/data/credit-card.png contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to /data/credit-card.txt": "extract_credit_card(source: str, destination: str)",
    "/data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line": "find_similar_comments(source: str, destination: str)",
    "The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type? Write the number in /data/ticket-sales-gold.txt": "total_sales(ticket_type: str, source: str, destination: str)"
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "format_md_with_prettier",
            "parameters": {
                type: "object",
                "properties": {
                    "version": {"type": "string"},
                    "source": {"type": "string"}
                },
                "required": ["version", "source"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_weekdays",
            "parameters": {
                type: "object",
                "properties": {
                    "weekday": {"type": "string"},
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["weekday", "source", "destination"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_contacts",
            "parameters": {
                type: "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["source", "destination"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_recent_logs",
            "parameters": {
                type: "object",
                "properties": {
                    "no_of_logs": {"type": "string"},
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["no_of_logs", "source", "destination"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_h1",
            "parameters": {
                type: "object",
                "properties": {
                    "source": {"type": "string"},
                    "index_path": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["source", "index_path", "destination"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_email",
            "parameters": {
                type: "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["source", "destination"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_credit_card",
            "parameters": {
                type: "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["source", "destination"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_comments",
            "parameters": {
                type: "object",
                "properties": {
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["source", "destination"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "total_sales",
            "parameters": {
                type: "object",
                "properties": {
                    "ticket_type": {"type": "string"},
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["ticket_type", "source", "destination"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

def extractParams(task_desc, matched_task):
    ''' 
    Extract the parameters from the task description.
    '''
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system","content": "Extract parameters for function execution and respond in JSON"},
            {"role": "user", "content": task_desc}
        ],
        "tools": tools,
        "tool_choice": "required",
        "response_format": {"type": "json_object"}
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()['choices'][0]['message']['tool_calls'][0]['function']

@app.get("/read")
async def read(path: str = Query(None, alias="path")):
    ''' 
    Read the path and return the file content, iff the path does not access the any other path except '/data'.
    Return the file content.
    '''
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, "r") as file:
        content = file.read()
    return content 

# Find a way to store embeddings of the tasks instead of calling the API each time the program runs.

@app.post("/run")
async def run(task_desc: str = Query(None, alias="task")):
    ''' 
    Identify the task by embedding and comparing similarity with the tasks defined.
    After identifying the task, call the specific function to execute the task.
    '''

    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = { 
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": task_desc
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        query_embedding = np.array(response.json()["data"][0]["embedding"])

        with open("pA_task_embs.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row 
            max_similarity = 0
            for row in reader:
                task = row[0]
                task_embedding = np.array(row[1:], dtype=np.float32)  # Convert to float
                similarity = np.dot(query_embedding, task_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(task_embedding))
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_task = task
            print(matched_task)
            print(task_desc)
            print(max_similarity)

        if matched_task == phaseA["A1"]:
            subprocess.run(["pip", "install", "uv"])
            subprocess.run(["uv", "run", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/datagen.py", "23f3001745@ds.study.iitm.ac.in"])
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity})
        elif matched_task == phaseA["A2"]:
            # Call the function to execute the task
            
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity})
        elif matched_task == phaseA["A3"]:
            # Call the function to execute the task
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity})
        elif matched_task == phaseA["A4"]:
            # Call the function to execute the task
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity})
        elif matched_task == phaseA["A5"]:
            # Call the function to execute the task
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity})
        elif matched_task == phaseA["A6"]:
            # Call the function to execute the task
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity})
        elif matched_task == phaseA["A7"]:
            # Call the function to execute the task
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity})
        elif matched_task == phaseA["A8"]:
            # Call the function to execute the task
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity})
        elif matched_task == phaseA["A9"]:
            # Call the function to execute the task
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity})
        elif matched_task == phaseA["A10"]:
            # Call the function to execute the task
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity})
    return