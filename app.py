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
                task_embedding = np.array(row[1:])
                similarity = np.dot(query_embedding, task_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(task_embedding))
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_task = task
            print(matched_task)
            print(task_desc)
            print(max_similarity)
    return