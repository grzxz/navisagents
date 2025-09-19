import os
import shutil
import boto3
from botocore.config import Config
from huggingface_hub import login
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# Akave O3 API Keys
AKAVE_O3_PUBLIC_KEY = "O3_xxx"
AKAVE_O3_SECRET_KEY = "0Pe_xxx"

# Huggingface API Keys
HUGGINGFACE_API_KEY = "hf_ic_xxx"

# Initialize Akave storage
s3 = boto3.client(
    's3',
    aws_access_key_id='AKAVE_O3_PUBLIC_KEY',
    aws_secret_access_key='AKAVE_O3_SECRET_KEY',
    region_name='us-east-1',
    endpoint_url='https://o3-rc3.akave.xyz',
    config=Config(request_checksum_calculation="when_required", response_checksum_validation="when_required")
)

def download_model_from_huggingface():
    # Downloads model from huggingface
    my_token = HUGGINGFACE_API_KEY

    # Call the login function and pass your token
    login(token=my_token)

    # Download gemma-3-270m from huggingface transformers
    # Specify the model name
    model_name = "google/gemma-3-270m"

    # Download the tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


def upload_model_to_akave(directory, bucket, destination):
    # Uploads model to Akave
    for root, dirs, files in os.walk(directory):
        for filename in files:
            
            local_path = os.path.join(root, filename)
            
            relative_path = os.path.relpath(local_path, directory)
            s3_key = os.path.join(destination_prefix, relative_path).replace("\\", "/")
            print(f'Uploading {local_path} to s3://{bucket}/{s3_key}')
            s3.upload_file(local_path, bucket, s3_key)

def delete_model(path):
    # Function to delete Huggingface downloaded model locally to test model download from Akave
    expanded_path = os.path.expanduser(path)
    shutil.rmtree(expanded_path)

def download_model_from_akave(bucket_name, prefix='', local_dir='gemma'):
    os.makedirs(local_dir, exist_ok=True)
    downloaded_files = []

    # List all objects
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith('/'):
            continue

        local_path = os.path.join(local_dir, os.path.relpath(key, prefix))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            # Use get_object instead of download_file
            response = s3.get_object(Bucket=bucket_name, Key=key)
            with open(local_path, 'wb') as f:
                f.write(response['Body'].read())
            downloaded_files.append(local_path)
            print(f"Downloaded: {key}")
        except Exception as e:
            print(f"Failed to download {key}: {e}")

    return downloaded_files

def run_inference():
    # Run model inference loading down
    model_dir = 'gemma_model/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

    # Run inference on downloaded model from Filebase
    prompt = "What is capital of Argentina?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=30)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    return response



# Download Gemma3 model from Huggingface 
download_model_from_huggingface()

local_directory = os.path.expanduser('~/.cache/huggingface/hub/models--google--gemma-3-270m')
bucket = 'navis'
destination_prefix = 'gemma270m'

# Uploads to Akave O3
upload_model_to_akave(local_directory, bucket, destination_prefix)
delete_model("~/.cache/huggingface/hub/")

# Download model from Akave O3 and run inference as test
download_model_from_akave("navis", "gemma270m", "gemma_model")
run_inference()
