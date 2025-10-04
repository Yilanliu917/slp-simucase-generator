import os
import json
from dotenv import load_dotenv
from typing import List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma # <-- UPDATED IMPORT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- 1. CONFIGURATION and ENVIRONMENT LOADING ---
load_dotenv()
DB_PATH = "slp_knowledge_base/"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- 2. DEFINE THE OUTPUT STRUCTURE ---
class BackgroundInfo(BaseModel):
    medical_history: str = Field(description="Relevant medical history (e.g., 'None reported', 'History of otitis media','Cerebral Palsy', 'Autism Spectrum Disorder','ADHD').")
    parent_concerns: str = Field(description="Summary of concerns reported by parents regarding the student's communication.")
    teacher_concerns: str = Field(description="Summary of concerns reported by teachers regarding classroom communication.")

class StudentProfile(BaseModel):
    name: str = Field(description="A realistic, anonymized student name.")
    age: int = Field(description="The student's age.")
    grade_level: str = Field(description="The student's grade level.")
    gender: str = Field(description="The student's gender.")
    background: BackgroundInfo = Field(description="Detailed background information for the student.")

class SimuCaseFile(BaseModel):
    """The overall structure for the generated case file."""
    student_profile: StudentProfile = Field(description="Comprehensive profile of the student.")
    annual_goals: List[str] = Field(description="A list of 2-3 specific, measurable annual IEP goals for speech therapy.")
    latest_session_notes: List[str] = Field(
        description="A list of the latest 3 session notes as single-paragraph performance summaries for one specific goal. Each summary combines objective data (O) and assessment (A). "
                    "Example 1: 'John produced /s/ sounds at the word level with 55% accuracy with 1 visual and 1 verbal prompt.' "
                    "Example 2: 'Emily used a pacing board as a fluency strategy to slow down her speech in 3 out of 5 trials with 1 visual cue.'"
    )
    generated_by_model: str = Field(description="The specific language model used to generate this profile.")

# --- 3. FUNCTION TO INITIALIZE THE LLM ---
def initialize_llm(model_choice: str):
    """Initializes the chosen language model and returns the model object and its name."""
    print(f"Initializing model: {model_choice}...")
    if model_choice == "openai":
        model_name = "gpt-4o"
        llm = ChatOpenAI(model_name=model_name, temperature=0.7)
    elif model_choice == "google":
        model_name = "gemini-2.5-pro"
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7)
    elif model_choice == "anthropic":
        model_name = "claude-opus-4-1-20250805"
        llm = ChatAnthropic(model=model_name, temperature=0.7)
    else:
        raise ValueError("Invalid model selected.")
    
    # Bind the Pydantic schema directly to the model
    structured_llm = llm.with_structured_output(SimuCaseFile)
    return structured_llm, model_name

# --- 4. MAIN APPLICATION LOGIC ---
if __name__ == "__main__":
    print("--- SLP SimuCase Generator ---")

    while True:
        model_selection = input("Choose a model to use (openai, google, or anthropic): ").lower()
        if model_selection in ["openai", "google", "anthropic"]:
            llm, specific_model_name = initialize_llm(model_selection)
            break
        else:
            print("Invalid choice. Please try again.")
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    template = """
    You are an expert school-based CCC-SLP creating a simulation case file based on a user's request.
    Use the following retrieved clinical context to generate a comprehensive, ethical, and realistic case file.
    Structure your output to match the requested schema.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    grade_level = input("Enter the student's grade level (e.g., '1st grade'): ")
    disorder_type = input("Enter the communication disorder (e.g., 'stuttering', 'articulation disorder'): ")
    
    question = f"Generate a case file for a {grade_level} student with a {disorder_type}."
    print(f"\nProcessing query: '{question}'...")

    response = rag_chain.invoke(question)
    response.generated_by_model = specific_model_name

    print("\n--- ✅ SIMUCASE FILE GENERATED ---")
    profile = response.student_profile
    print(f"\n## Student Profile")
    print(f"Name: {profile.name} | Age: {profile.age} | Grade: {profile.grade_level} | Gender: {profile.gender}")
    print("\n  Background:")
    print(f"    - Medical History: {profile.background.medical_history}")
    print(f"    - Parent Concerns: {profile.background.parent_concerns}")
    print(f"    - Teacher Concerns: {profile.background.teacher_concerns}")

    print(f"\n## Annual IEP Goals")
    for i, goal in enumerate(response.annual_goals, 1):
        print(f"  {i}. {goal}")

    print(f"\n## Latest 3 Session Notes (Performance Summaries)")
    for i, note in enumerate(response.latest_session_notes, 1):
        print(f"  Session {i}: {note}")
    
    print(f"\n(Generated by: {response.generated_by_model})")
    print("\n----------------------------------")

    confirmation = input("\nDo you want to save this profile as a JSON file? (y/n): ").lower()
    if confirmation == 'y':
        filename = f"{profile.name.replace(' ', '_')}_profile.json"
        profile_dict = response.model_dump()
        with open(filename, 'w') as f:
            json.dump(profile_dict, f, indent=4)
        print(f"✅ Profile saved as '{filename}'")
    else:
        print("Profile not saved.")