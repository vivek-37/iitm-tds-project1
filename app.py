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
    "Install uv (if required) and run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/datagen.py with ${user.email} as the only argument.": "install_and_run(url:str, email:str)",
    "Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place": "format_md_with_prettier(version:str, source:str)",
    "The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to /data/dates-wednesdays.txt": "count_weekdays(weekday: str, source: str, destination: str)",
    "Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json": "sort_contacts(source: str, destination: str)",
    "Write the first line of the 10 most recent .log file in /data/logs/ to /data/logs-recent.txt, most recent first": "write_recent_logs(no_of_logs: str, source: str, destination: str)",
    "Find all Markdown (.md) files in /data/docs/. For each file, extract the first occurrance of each H1 (i.e. a line starting with # ). Create an index file /data/docs/index.json that maps each filename (without the /data/docs/ prefix) to its title (e.g. {\"README.md\": \"Home\", \"path/to/large-language-models.md\": \"Large Language Models\", ...})": "extract_h1(source: str, destination: str)",
    "/data/email.txt contains an email message. Pass the content to an LLM with instructions to extract the sender's email address, and write just the email address to /data/email-sender.txt": "extract_email(source: str, destination: str)",
    "/data/credit-card.png contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to /data/credit-card.txt": "extract_credit_card(source: str, destination: str)",
    "/data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line": "find_similar_comments(source: str, destination: str)",
    "The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type? Write the number in /data/ticket-sales-gold.txt": "total_sales(sales_of_ticket_type: str, source: str, destination: str)"
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "install_and_run",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "email": {"type": "string"}
                },
                "required": ["url", "email"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_md_with_prettier",
            "parameters": {
                "type": "object",
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
                "type": "object",
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
            "name": "sort_json_by_keys",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {"type": "array", "items": {"type": "string"}},
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
                "type": "object",
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
            "name": "extract_specific_header",
            "parameters": {
                "type": "object",
                "properties": {
                    "header_type": {"type": "string"},
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
            "name": "extract_key",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
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
                "type": "object",
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
                "type": "object",
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
                "type": "object",
                "properties": {
                    "sales_of_ticket_type": {"type": "string"},
                    "source": {"type": "string"},
                    "destination": {"type": "string"}
                },
                "required": ["sales_of_ticket_type", "source", "destination"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

def extractParams(task_desc):
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
    print(response.json())
    print(response.json()['choices'][0]['message']['tool_calls'][0]['function'])
    return response.json()['choices'][0]['message']['tool_calls'][0]['function']

# needs to be post not get
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

    print("Calling the openAI API")
    response = requests.post(url, json=payload, headers=headers)


    if response.status_code == 200:
        query_embedding = np.array(response.json()["data"][0]["embedding"])

        with open("pA_task_embs.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row 
            max_similarity = 0
            matched_task = None
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


        params = extractParams(task_desc)
        print(params)
        '''
        {'name': 'install_and_run', 'arguments': '{"url":"https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py","email":"23f3001745@study.iitm.ac.in"}'}
        '''
        # convert the arguments to a dictionary
        params['arguments'] = json.loads(params['arguments'])
        print(params)

        # convert the parameters to a list
        params_list = []
        for key, value in params['arguments'].items():
            params_list.append(value)
        print(params_list)

        # convert the parameters into a dictionary as function : parameter list
        function_params = {params['name']: params_list}
        print(function_params)

        if matched_task == phaseA["A1"]:
            # check if uv is installed, run datagen.py
            subprocess.run(["pip", "install", "uv"])
            subprocess.run(["uv", "run", params_list[0], params_list[1]])
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity, "function": function_params})

        elif matched_task == phaseA["A2"]:
            # format the unformatted markdown file with prettier
            subprocess.run("apt update && apt install -y npm", shell=True, check=True)
            subprocess.run(["npx", "prettier@"+params_list[0], "--write", params_list[1]])
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity, "function": function_params})

        elif matched_task == phaseA["A3"]:
            from datetime import datetime

            # Define supported date formats
            formats = [
                "%Y-%m-%d",       # 2024-03-14
                "%d-%b-%Y",       # 14-Mar-2024
                "%b %d, %Y",      # Mar 14, 2024
                "%Y/%m/%d %H:%M:%S"  # 2024/03/14 15:30:45
            ]

            def parse_date(date_str):
                """Attempt to parse a date string using multiple formats."""
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str.strip(), fmt).weekday()  # Returns weekday as int (0=Monday, 6=Sunday)
                    except ValueError:
                        continue  # Try next format
                return None  # If no format matches

            # Read input parameters
            weekday_map = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
            
            target_weekday = params_list[0].strip().lower()
            input_file = params_list[1]
            output_file = params_list[2]

            if target_weekday not in weekday_map:
                return JSONResponse(content={"error": f"Invalid weekday: {target_weekday}"})

            target_weekday_num = weekday_map[target_weekday]
            day_count = 0

            # Read the file and process dates
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    weekday = parse_date(line)
                    if weekday is not None and weekday == target_weekday_num:
                        day_count += 1

            # Write the result to output file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(str(day_count) + "\n")

            return JSONResponse(content={"task": matched_task, "similarity": max_similarity, "function": function_params})

        elif matched_task == phaseA["A4"]:
            # sort the json by last name and first name (keys: key1 then key2), source and destination files are json files
            keys = params_list[0]
            with open(params_list[1], "r") as f:
                contacts = json.load(f)
                # Sort the contacts by keys[0] then keys[1] till keys[n]
                contacts.sort(key=lambda x: tuple(x[key] for key in keys))

                with open(params_list[2], "w") as f:
                    json.dump(contacts, f, indent=4)

            return JSONResponse(content={"task": matched_task, "similarity": max_similarity, "function": function_params})

        elif matched_task == phaseA["A5"]:
            # write the first line of the n most recent .log files to a new file
            from pathlib import Path

            n = int(params_list[0])
            log_dir = Path(params_list[1])
            output_file = params_list[2]

            # Get all .log files sorted by modification time (newest first)
            log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)

            # Take the 10 most recent log files
            recent_logs = log_files[:n]

            first_lines = []

            # Read the first line from each log file
            for log_file in recent_logs:
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                        first_lines.append(first_line)
                except Exception as e:
                    first_lines.append(f"[Error reading {log_file.name}: {str(e)}]")

            # Write to the output file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(first_lines) + "\n")

            print(f"Extracted first lines from {len(recent_logs)} log files into {output_file}")

            return JSONResponse(content={"task": matched_task, "similarity": max_similarity, "function": function_params})

        elif matched_task == phaseA["A6"]:

            def extract_first_header_type(file_path, header_type):
                header_map = {
                    "h1": "#",
                    "h2": "##",
                    "h3": "###",
                    "h4": "####",
                    "h5": "#####",
                    "h6": "######"
                }
                if header_type not in header_map:
                    return None

                prefix = header_map[header_type]

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()                        
                        if line.startswith(prefix+" "):  # Check for header_type
                            return line[len(prefix)+1:].strip()  # Remove '# ' and extra spaces
                return ""  # Default if no header_type found

            # extract the first occurance of each header_type in the markdown files
            header_type = params_list[0]
            docs_dir = params_list[1]
            index = {}

            # Walk through /data/docs/ to find .md files
            for root, _, files in os.walk(docs_dir):
                for file in files:
                    if file.endswith(".md"):
                        file_path = os.path.join(root, file)
                        title = extract_first_header_type(file_path)

                        # Store in index without /data/docs/ prefix
                        relative_path = os.path.relpath(file_path, docs_dir)
                        index[relative_path] = title

            # Save index to /data/docs/index.json
            index_file = params_list[2]

            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=4)
            
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity, "function": function_params})

        elif matched_task == phaseA["A7"]:
            # extract the key from the email
            key = params_list[0]
            email_file = params_list[1]
            output_file = params_list[2]

            with open(email_file, "r", encoding="utf-8") as f:
                email_content = f.read()

            # Call the LLM to extract the email address
            email_tool = [{
                "type": "function",
                "function": {
                    "name": "extract_"+key,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            key: {"type": "string"}
                        },
                        "required": ["source", "destination"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }]

            url_email = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {AIPROXY_TOKEN}",
                "Content-Type": "application/json"
            }

            email_payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system","content": "Extract parameters for function execution and respond in JSON"},
                    {"role": "user", "content": email_content}
                ],
                "tools": email_tool,
                "tool_choice": "required",
                "response_format": {"type": "json_object"}
            }

            email_response = requests.post(url_email, json=email_payload, headers=headers)
            email = email_response.json()['choices'][0]['message']['tool_calls'][0]['function']['arguments']['sender\'s email']
            email = email.strip()
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(email)

            return JSONResponse(content={"task": matched_task, "similarity": max_similarity, "function": function_params})

        elif matched_task == phaseA["A8"]:
            # extract the credit card number from the image
            import base64
            image_file = params_list[0]
            output_file = params_list[1]

            # Extract and validate card number
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                b64_image = base64.b64encode(image_data).decode('utf-8')

                url_img = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

                img_payload = {
                    "model": "gpt-4-turbo",
                    "messages": [
                        {"role": "system", "content": "Extract only the credit card number from the given image. Do not include spaces or any other characters."},
                        {
                            "role": "user", "content": [
                            {
                                "type": "text",
                                "text": "Extract the credit card digits from this image."
                            }, 
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"} 
                            }
                            ]
                        }
                    ]
                }
                response = requests.post(url_img, headers, json=img_payload)
                extracted_number = response["choices"][0]["message"]["content"].strip()
                # Remove spaces and dashes to ensure proper formatting
                formatted_number = extracted_number.replace(" ", "").replace("-", "")

            card_number = formatted_number

            # Write extracted card number to output file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(card_number)
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity, "function": function_params})

        elif matched_task == phaseA["A9"]:
            # find the most similar pair of comments
            input_file = params_list[0]
            output_file = params_list[1]

            with open(input_file, "r", encoding="utf-8") as f:
                comments = [line.strip() for line in f.readlines() if line.strip()]
            
            # Calculate embeddings for each comment
            payload_comm = {
                "model": "text-embedding-3-small",
                "input": comments
            }

            url_comm = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"

            response_comm = requests.post(url_comm, json=payload_comm, headers=headers)

            if response_comm.status_code == 200:
                embeddings = {}

                for comment, embs in zip(comments, response.json()["data"]):
                    embeddings[comment] = np.array(embs["embedding"])

                similarities = {}
                for comment1 in comments:
                    for comment2 in comments:
                        if comment1 == comment2:
                            continue        # skip same comments
                        similarity = np.dot(comment1, comment2) / (np.linalg.norm(comment1) * np.linalg.norm(comment2))
                        similarities[(comment1,comment2)] = similarity

                # Find the most similar pair
                most_similar_pair = max(similarities, key=similarities.get)
                comment1, comment2 = most_similar_pair

                # Write the most similar pair to the output file
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"{comment1}\n{comment2}\n")
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity, "function": function_params})

        elif matched_task == phaseA["A10"]:
            # find the total sales of the "Gold" ticket type
            '''
            The SQLite database file /data/ticket-sales.db has a tickets with columns type, units, and price.
            Each row is a customer bid for a concert ticket. What is the total sales of all the items in the “Gold” ticket type?
            Write the number in /data/ticket-sales-gold.txt
            '''
            import sqlite3
            db_file = params_list[1]
            output_file = params_list[2]
            ticket_type = params_list[0]

            # Connect to the SQLite database
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            # Query the total sales of the ticket type
            cursor.execute(f"SELECT SUM(units * price) FROM tickets WHERE type = '{ticket_type}'")
            total_sales = cursor.fetchone()[0]

            # Write the total sales to the output file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(str(total_sales))

            # Close the database connection
            conn.close()
            
            return JSONResponse(content={"task": matched_task, "similarity": max_similarity, "function": function_params})
    else:
        return JSONResponse(content={"error": "Error in calling the OpenAI API"})