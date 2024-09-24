import tempfile
from fastapi import FastAPI, File, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel
import requests
import json
import os
app = FastAPI()

class Questions(BaseModel):
    questions: list[str]


def load_document(file: UploadFile):
    if file.filename.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
    
        loader = PyPDFLoader(tmp_path)
        document = loader.load()
        text = " ".join([page.page_content for page in document])
        os.remove(tmp_path)
    
    elif file.filename.endswith(".json"):
        content = json.load(file.file)
        text = " ".join([entry["text"] for entry in content])
    else:
        raise ValueError("Unsupported file type")
    
    return text

def query_ollama_stream(prompt, model="tinyllama"):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response_content = ""
    with requests.post(url, data=json.dumps(payload), headers=headers, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                response_content += data['message']['content']
    
    return response_content

@app.post("/answer")
async def answer_questions(questions_file: UploadFile, document_file: UploadFile):
    questions_data = await questions_file.read()
    questions = json.loads(questions_data)["questions"]
    document_text = load_document(document_file)
    responses = []
    for question in questions:
        prompt = f"Answer the question based on the following document:\nDocument: {document_text}\nQuestion: {question}"
        answer = query_ollama_stream(prompt, model="tinyllama")
        responses.append({"question": question, "answer": answer})
    
    return responses

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
