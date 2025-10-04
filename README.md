---
🎯 SLP SimuCase Generator
An AI-powered tool for Speech-Language Pathologists, students, and educators to generate realistic, simulated student case files for training and educational purposes. 

✨ Features

Customizable Scenarios: Specify grade levels, disorder types (e.g., Articulation, Language, Fluency), and the number of students to generate. 


Powered by Advanced AI: Choose from leading models like GPT-4o, Gemini 2.5 Pro, and Claude 3.5 Sonnet to create diverse and nuanced profiles. 


Comprehensive Output: Each case includes a detailed student profile, background history, annual IEP goals, and sample session notes. 


RAG Architecture: Leverages a knowledge base of clinical documents to ensure generated content is realistic and context-aware. 


Cost Tracking: Includes built-in functionality to monitor and log estimated API costs for each generation. 

🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Python 3.9+

Git

API keys for OpenAI, Google AI, and Anthropic

Installation
Clone the repository:

Bash

git clone https://github.com/your-username/slp-simucase-generator.git
cd slp-simucase-generator
Install dependencies:
It's recommended to use a virtual environment.

Bash

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
Set up your environment variables:
Create a file named .env in the root of the project folder and add your API keys:

Code snippet

OPENAI_API_KEY="sk-..."
GOOGLE_API_KEY="..."
ANTHROPIC_API_KEY="sk-ant-..."
Build the Vector Database:
Place your clinical knowledge documents (.pdf, .docx) inside the data/slp_knowledge_base/ folder. Then, run the setup script to create the vector store:

Bash

python setup_kb.py
Usage
Once the installation is complete and the vector database is built, launch the Gradio application:

Bash

python app.py
Open your web browser and navigate to the local URL provided in the terminal (usually http://127.0.0.1:7860).

📂 Project Structure

app.py: The main Gradio application script that contains the UI and core generation logic. 

setup_kb.py: A script to process local documents and build the ChromaDB vector store.

requirements.txt: A list of all the Python libraries required for the project.


data/slp_knowledge_base/: The directory where you should place your source documents for the knowledge base. 

⚠️ Disclaimer
All generated student profiles and data are entirely fictional and created by an AI. This tool is intended for educational and simulation purposes only and should not be used for real-world clinical decision-making.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
