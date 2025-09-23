This report provides a comprehensive guide to running the FRA AI Fusion System, detailing the process from setup to deployment. The analysis is based on the provided source code, documentation, and configuration files.

1. How to Run the System
There are three primary ways to run the system, catering to different needs from a quick demonstration to a full production-like setup.

Option 1: Quick Start with the Automated Runner (Recommended)
This method uses the main orchestration script run.py to handle all steps automatically.

Prerequisites:

Python 3.8+

Git

Around 50GB of free disk space for models and datasets.

An NVIDIA GPU is recommended for training.

Setup Environment:

Clone the repository: git clone <repository-url>

Navigate to the project directory: cd sih

Create and activate a Python virtual environment:

Bash

python3 -m venv fra_env
source fra_env/bin/activate
Install dependencies from Full prototype/requirements.txt.

Configure Environment Variables:

Navigate to the Full prototype directory: cd "Full prototype"

Copy the example environment file: cp .env.example .env

Edit the .env file to add your Hugging Face token and any other necessary configurations:

Code snippet

HF_TOKEN="your_huggingface_token_here"
Run the Complete Pipeline:

From the Full prototype directory, execute the following command:

Bash

python run.py --complete
This single command will trigger the entire process: setup, model/data downloads, data processing, model training, and finally, it will start the API server.

Option 2: Docker Deployment
For a containerized and isolated environment, you can use Docker.

Prerequisites:

Docker and Docker Compose installed.

Build and Run:

From the root sih directory, run:

Bash

docker-compose up --build
This will build the Docker image as defined in the Dockerfile and start all services, including the main application, Redis, and PostgreSQL, as configured in docker-compose.yml.

Option 3: Single-File Executable
The fra_ai_complete_system.py script is a self-contained solution that handles everything from dependency installation to serving the API.

Navigate to the directory:

Bash

cd "sih_-main/Full prototype"
Run the complete pipeline:

Bash

python fra_ai_complete_system.py --action all
This will:

Auto-install all dependencies.

Download all AI models and datasets.

Fine-tune the models.

Perform knowledge distillation.

Set up a SQLite database with sample data.

Start the production API server on http://localhost:8000.

2. The Automated Process: What Happens When You Run --complete
When you execute python run.py --complete, a series of automated steps are performed in a specific order to set up, train, and deploy the system.

Environment Setup: The script first initializes the environment by creating necessary directories for data, models, logs, and outputs (data/raw, data/processed, models, logs, outputs). It also installs all Python packages listed in requirements.txt.

Model and Data Download: The system then proceeds to download all required AI models and datasets. This is orchestrated by scripts/download_models.py and scripts/download_data.py, which read the model_sources and data_sources from configs/config.json.

Data Processing: Once the raw data is downloaded, the 1_data_processing/data_pipeline.py script is executed. This pipeline performs several tasks:

OCR and Text Extraction: Scanned FRA documents are processed to extract raw text.

Satellite Imagery Preprocessing: Satellite images are prepared for analysis.

Geospatial Data Integration: GIS layers and other spatial data are integrated.

Training Data Generation: The processed data is transformed into a structured format suitable for training the AI models and saved as training_data.json.

Model Training: With the processed data ready, the training pipeline, defined in 2_model_fusion/train_fusion.py, is initiated. This is a multi-stage process designed to build a powerful, unified model.

Knowledge Distillation: After the main model is trained, a smaller, more efficient version is created through knowledge distillation, as detailed in 2_model_fusion/distillation.py.

API Server Launch: Finally, the script starts the FastAPI web server, defined in 3_webgis_backend/api.py. The server exposes several endpoints for interacting with the trained models, processing documents, and accessing geospatial data.

3. Models and Datasets to be Downloaded
The system is configured to download a comprehensive set of models and datasets to cover all its functionalities. The exact list is defined in configs/config.json.

Models Download Order
The models are downloaded based on a priority system (essential, standard, advanced) defined in scripts/download_models.py. The essential models are downloaded first.

Essential Models (Downloaded First):

LLM: mistralai/Mistral-7B-Instruct-v0.1 (Primary LLM for NLP tasks)

OCR/Layout: microsoft/layoutlmv3-base and microsoft/trocr-base-stage1 (For document understanding and text extraction)

Vision: facebook/deeplabv3-resnet50-ade (For satellite image segmentation), facebook/detr-resnet-50 (For object detection), and openai/clip-vit-base-patch32 (For vision-language tasks)

NER: ai4bharat/indic-bert and opennyaiorg/en_legal_ner_trf (For named entity recognition in Indian languages and legal text)

Other Models (Standard and Advanced): The system will also download larger and more specialized models like Llama-2-7b-chat-hf, falcon-7b-instruct, layoutlmv3-large, trocr-large, sam-vit-huge (Segment Anything Model), and the ibm-nasa-geospatial/Prithvi-100M foundation model.

Datasets to be Downloaded
The scripts/download_data.py script downloads datasets from various sources as specified in the configuration.

FRA and Legal Datasets:

AI4Bharat IndicNLP Corpus: For large-scale Indic language text.

InLegalNER: Annotated NER datasets for Indian legal texts from Hugging Face.

OCR Datasets:

ICDAR 2019 MLT: For multilingual text detection and recognition.

IIIT Hindi OCR: Hindi OCR datasets from IIIT Hyderabad.

Satellite Imagery:

IndiaSAT Dataset: Indian satellite imagery for land cover classification.

BHUVAN Free Data: Satellite data from NRSC India.

Geospatial Boundaries:

OpenStreetMap India: Administrative boundaries and other map data.

DataMeet Boundaries: Indian administrative boundary shapefiles.

4. Fine-Tuning, Merging, and Training Process
The core of this system is its sophisticated, multi-stage training pipeline that creates a single, unified AI model capable of handling diverse tasks. The process is detailed in stepbystepprocess.md and implemented in train_fusion.py.

Fine-Tuning and Merging Order
Digitization & NER (Foundation First): The process starts by fine-tuning models for Optical Character Recognition (OCR) and Named Entity Recognition (NER). A LayoutLM-style model is fine-tuned on annotated FRA forms to extract structured information like village names, patta holders, and coordinates. This is done first because it produces the structured text and labels needed for all subsequent stages.

Geospatial Segmentation: Concurrently, computer vision models like DeepLabV3+ or U-Net are trained on satellite imagery to identify and segment land cover types (forests, agriculture, water bodies). This creates the foundational geospatial layers for the system.

Cross-Modal Alignment: This is a crucial step where the textual and visual models are aligned. A contrastive learning approach is used, where the model learns to associate the text from an FRA claim with the corresponding satellite image of the claimed area. This creates a unified embedding space where text and images with similar meanings are close to each other.

LLM Tool Skills: The primary LLM (Mistral-7B) is then fine-tuned to interact with the geospatial database (PostGIS). It learns to generate SQL queries from natural language questions (e.g., "Show all pending claims in Odisha"). This is achieved through supervised fine-tuning on pairs of natural language questions and their corresponding SQL queries.

DSS and Policy Fusion: In the final stage, the model is trained to provide Decision Support System (DSS) capabilities. It learns to recommend government schemes by correlating the data from FRA claims and satellite analysis with the eligibility criteria of schemes like PM-KISAN and Jal Jeevan Mission.

Training Process
Training begins after the data processing pipeline has created the training_data.json file. The training is divided into five stages, as defined in configs/config.json and executed by train_fusion.py:

Stage 0 - Multimodal Pretraining (15 epochs): This is the initial and most critical stage where the model learns to create a unified representation of text, images, and geospatial data. It uses contrastive learning and other self-supervised objectives to align the different modalities.

Stage 1 - Foundation Training (10 epochs): The model is fine-tuned on specific tasks like NER and land cover classification using the labeled data.

Stage 2 - Alignment Training (8 epochs): The model's outputs are further refined to align with human preferences and expectations.

Stage 3 - Tool Skills (5 epochs): The model is trained to generate SQL queries and interact with other tools.

Stage 4 - DSS Fine-tuning (5 epochs): The model is specialized for decision support tasks, learning to make accurate and relevant recommendations.

5. Knowledge Distillation
After the large, multi-stage training is complete, the system performs knowledge distillation to create a smaller, faster, and more efficient model suitable for deployment.

When and How: Distillation is the final step after the main training pipeline is complete. The process is handled by 2_model_fusion/distillation.py. It uses a teacher-student approach where the large, fully-trained model (the "teacher") is used to train a smaller model (the "student").

Models Involved:

Teacher Model: The final_model.pth produced by the multi-stage training pipeline. This is a powerful but large model.

Student Model: A smaller version of the same architecture. The configuration for the student model is created by reducing the hidden size and other parameters of the teacher model. For instance, Mistral-7B-Instruct could be distilled from a larger teacher model like Llama-3-8B-Instruct.

The distillation process involves using a special loss function (DistillationLoss) that encourages the student model to mimic the outputs of the teacher model. This allows the smaller student model to learn the complex patterns captured by the larger teacher, resulting in a compact yet powerful model for production use.
