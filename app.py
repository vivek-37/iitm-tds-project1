from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import requests
import numpy as np


app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])   


@app.get("/read")
async def read(path: str = Query(None, alias="path")):
    ''' 
    Read the path and return the file content, iff the path does not access the any other path except '/data'.
    Return the file content.
    '''
    return 

@app.post("/run")
async def run(task_desc: str = Query(None, alias="task")):
    ''' 
    Identify the task by embedding and comparing similarity with the tasks defined.
    After identifying the task, call the specific function to execute the task.
    '''
    return