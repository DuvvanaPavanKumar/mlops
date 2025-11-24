from python:3.9-slim
workdir /app
#install dependencies
copy requirements.txt .
run pip install -r requirements.txt
#copy rest of the code
copy . .
#command to run the model training script 
cmd ["python", "src/train.py"]