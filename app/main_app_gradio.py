import os
import json
from dotenv import load_dotenv
from typing import List, Dict
from datetime import datetime
from collections import defaultdict

import gradio as gr
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- 1. CONFIGURATION and SETUP ---
load_dotenv()
DB_PATH = "data/slp_knowledge_base/"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CONDITIONS = 5
DEFAULT_OUTPUT_PATH = "generated_case_files/"
COST_LOG_FILE = "cost_tracking.json"

# Create default output directory if it doesn't exist
os.makedirs(DEFAULT_OUTPUT_PATH, exist_ok=True)

# API Cost per 1M tokens (based on official pricing)
API_COSTS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},  # per 1M tokens
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},  # <=200k tokens, >200k: input $2.50, output $15.00
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00}
}

# Estimated tokens per generation (average)
ESTIMATED_TOKENS = {
    "input": 2000,  # Context + prompt
    "output": 1500  # Generated case file
}

# --- Pydantic Models ---
class BackgroundInfo(BaseModel):
    medical_history: str = Field(description="Relevant medical history.")
    parent_concerns: str = Field(description="Summary of concerns from parents.")
    teacher_concerns: str = Field(description="Summary of concerns from teachers.")

class StudentProfile(BaseModel):
    name: str = Field(description="A realistic, anonymized student name.")
    age: int = Field(description="The student's age.")
    grade_level: str = Field(description="The student's grade level.")
    gender: str = Field(description="The student's gender.")
    background: BackgroundInfo = Field(description="Detailed background information.")

class SimuCaseFile(BaseModel):
    student_profile: StudentProfile
    annual_goals: List[str]
    latest_session_notes: List[str]

# --- Cost Tracking Functions ---
def calculate_cost(model: str) -> float:
    """Calculate estimated cost for one generation."""
    costs = API_COSTS.get(model, {"input": 0, "output": 0})
    input_cost = (ESTIMATED_TOKENS["input"] / 1_000_000) * costs["input"]
    output_cost = (ESTIMATED_TOKENS["output"] / 1_000_000) * costs["output"]
    return input_cost + output_cost

def load_cost_log() -> Dict:
    """Load cost tracking data from file."""
    if os.path.exists(COST_LOG_FILE):
        with open(COST_LOG_FILE, 'r') as f:
            return json.load(f)
    return {"daily_costs": {}, "total_cost": 0.0}

def save_cost_log(cost_data: Dict):
    """Save cost tracking data to file."""
    with open(COST_LOG_FILE, 'w') as f:
        json.dump(cost_data, f, indent=2)

def update_cost_tracking(run_cost: float, model_breakdown: Dict) -> tuple:
    """Update cost log and return current stats."""
    cost_log = load_cost_log()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Update daily costs
    if today not in cost_log["daily_costs"]:
        cost_log["daily_costs"][today] = {"total": 0.0, "models": {}}
    
    cost_log["daily_costs"][today]["total"] += run_cost
    
    # Update per-model costs
    for model, cost in model_breakdown.items():
        if model not in cost_log["daily_costs"][today]["models"]:
            cost_log["daily_costs"][today]["models"][model] = 0.0
        cost_log["daily_costs"][today]["models"][model] += cost
    
    # Update total cost
    cost_log["total_cost"] += run_cost
    
    save_cost_log(cost_log)
    
    return cost_log["daily_costs"][today]["total"], cost_log["total_cost"]

def get_today_cost() -> float:
    """Get today's total cost."""
    cost_log = load_cost_log()
    today = datetime.now().strftime("%Y-%m-%d")
    return cost_log["daily_costs"].get(today, {}).get("total", 0.0)

# --- 2. REFACTORED CORE LOGIC ---
initialized_models = {}

def get_llm(model_choice: str):
    """Initializes or retrieves a cached LLM."""
    if model_choice in initialized_models:
        return initialized_models[model_choice]

    print(f"Initializing model: {model_choice}...")
    
    model_map = {
        "gpt-4o": ChatOpenAI,
        "gemini-2.5-pro": ChatGoogleGenerativeAI,
        "claude-3-opus": ChatAnthropic,
        "claude-3.5-sonnet": ChatAnthropic
    }
    model_name_map = {
        "gpt-4o": "gpt-4o",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20240620"
    }
    
    model_class = model_map.get(model_choice)
    model_name = model_name_map.get(model_choice)
    
    if not model_class:
        raise ValueError(f"Invalid model selected: {model_choice}")
        
    llm = model_class(model=model_name, temperature=0.7).with_structured_output(SimuCaseFile)
    initialized_models[model_choice] = llm
    return llm

def save_case_files(content: str, output_path: str) -> str:
    """Saves generated case files to the specified path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"case_files_{timestamp}.md"
    full_path = os.path.join(output_path, filename)
    
    os.makedirs(output_path, exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return full_path

def process_generation_request(mode, *args):
    """Gathers all UI inputs, builds a task list, and generates profiles."""
    
    # Extract output_path (last argument)
    output_path = args[-1]
    
    tasks = []
    if mode == "Single Condition":
        num_students, model, grade, disorders = args[0:4]
        for _ in range(int(num_students)):
            tasks.append({"grade": grade, "disorders": disorders, "model": model})
    else: # Multiple Conditions
        # Extract visible_count
        visible_count = args[4]
        for i in range(visible_count):
            grade = args[5 + i * 4]
            disorders = args[6 + i * 4]
            num = args[7 + i * 4]
            model = args[8 + i * 4]
            for _ in range(int(num)):
                tasks.append({"grade": grade, "disorders": disorders, "model": model})

    if not tasks:
        raise gr.Error("No generation tasks specified. Please add conditions and student numbers.")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    template = """
    You are an expert school-based CCC-SLP creating a simulation case file based on a user's request.
    Use the following retrieved clinical context to generate a comprehensive, ethical, and realistic case file.
    Structure your output to perfectly match the requested schema.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    all_profiles_md = f"# SLP SimuCase Files\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Track costs
    model_costs = defaultdict(float)
    total_run_cost = 0.0
    
    # Use yield for progress updates
    total_tasks = len(tasks)
    
    for i, task in enumerate(tasks):
        disorder_string = ", ".join(task['disorders'])
        question = f"Generate a case file for a {task['grade']} student with: {disorder_string}."
        
        # Calculate progress
        progress_pct = int((i / total_tasks) * 100)
        status_msg = f"### ⚙️ Progress: {progress_pct}%\nGenerating profile {i+1}/{total_tasks} with **{task['model']}**..."
        
        # Yield progress update
        yield (
            gr.update(value=all_profiles_md if i > 0 else "🚀 Starting generation..."),
            gr.update(value="⚙️ Processing...", variant="secondary", interactive=False),
            gr.update(value=status_msg, visible=True),
            gr.update(value="Calculating...", visible=True)
        )
        
        llm = get_llm(task['model'])
        rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
        response = rag_chain.invoke(question)
        
        # Calculate cost for this generation
        generation_cost = calculate_cost(task['model'])
        model_costs[task['model']] += generation_cost
        total_run_cost += generation_cost
        
        profile = response.student_profile
        profile_md = f"""
---
### **Profile {i+1}: {profile.name}** (Generated by: {task['model']})
**Age:** {profile.age} | **Grade:** {profile.grade_level} | **Gender:** {profile.gender}

#### Background
- **Medical History:** {profile.background.medical_history}
- **Parent Concerns:** {profile.background.parent_concerns}
- **Teacher Concerns:** {profile.background.teacher_concerns}

#### Annual IEP Goals
""" + "\n".join([f"{j+1}. {goal}" for j, goal in enumerate(response.annual_goals)])
        
        profile_md += "\n\n#### Latest 3 Session Notes\n" + "\n".join([f"**Session {j+1}:** {note}" for j, note in enumerate(response.latest_session_notes)])
        all_profiles_md += profile_md + "\n"

    # Save to file
    saved_path = save_case_files(all_profiles_md, output_path)
    
    # Update cost tracking
    daily_cost, total_cost = update_cost_tracking(total_run_cost, dict(model_costs))
    
    # Create cost breakdown
    cost_breakdown = "\n\n---\n### 💰 Cost Analysis\n"
    cost_breakdown += f"**This Run:** ${total_run_cost:.4f}\n\n"
    cost_breakdown += "**Model Breakdown:**\n"
    for model, cost in model_costs.items():
        cost_breakdown += f"- {model}: ${cost:.4f}\n"
    cost_breakdown += f"\n**Today's Total:** ${daily_cost:.4f}\n"
    cost_breakdown += f"**All-Time Total:** ${total_cost:.4f}\n"
    
    # Add save confirmation to output
    save_message = f"\n\n---\n✅ **Case files saved to:** `{saved_path}`"
    
    final_output = all_profiles_md + cost_breakdown + save_message
    
    # Final yield with all results
    yield (
        gr.update(value=final_output),
        gr.update(value="✓ Generated", variant="secondary", interactive=True),
        gr.update(value="### ✅ Complete!", visible=True),
        gr.update(value=f"💰 **Today:** ${daily_cost:.4f} | **Total:** ${total_cost:.4f}", visible=True)
    )

# --- 4. CREATE THE GRADIO UI ---
if __name__ == "__main__":
    grade_levels = ["Pre-K", "Kindergarten", "1st Grade", "2nd Grade", "3rd Grade", "4th Grade", "5th Grade", 
                    "6th Grade", "7th Grade", "8th Grade", "9th Grade", "10th Grade", "11th Grade", "12th Grade"]
    disorder_types = ["Speech Sound", "Articulation", "Phonology", "Fluency", 
                      "Expressive Language", "Receptive Language", "Language", "Voice"]
    model_choices = ["gpt-4o", "gemini-2.5-pro", "claude-3-opus", "claude-3.5-sonnet"]

    with gr.Blocks(theme=gr.themes.Soft(), title="SLP SimuCase Generator", css="""
        .generate-btn {
            font-size: 14px !important;
            padding: 8px 16px !important;
            min-height: 38px !important;
        }
        .cost-display {
            font-weight: bold;
            color: #2e7d32;
            padding: 8px;
            background-color: #e8f5e9;
            border-radius: 4px;
            text-align: center;
        }
    """) as ui:
        gr.Markdown("# 🎯 SLP SimuCase Generator")
        gr.Markdown("Generate realistic speech-language pathology case files using AI models")
        
        # Cost display at the top
        with gr.Row():
            cost_display = gr.Markdown(
                value=f"💰 **Today's Cost:** ${get_today_cost():.4f}",
                visible=True,
                elem_classes="cost-display"
            )
        
        gen_mode = gr.Radio(
            ["Single Condition", "Multiple Conditions"], 
            label="Select Generation Mode", 
            value="Single Condition"
        )

        with gr.Group(visible=True) as single_condition_group:
            with gr.Row():
                num_students_single = gr.Number(label="Number of Students", value=1, minimum=1, step=1)
                model_single = gr.Dropdown(choices=model_choices, label="AI Model", value="gpt-4o")
            single_grade = gr.Dropdown(choices=grade_levels, label="Grade Level", value="1st Grade")
            single_disorders = gr.Dropdown(
                choices=disorder_types, 
                label="Disorder(s)", 
                value=["Articulation"], 
                multiselect=True
            )

        multi_condition_rows = []
        multi_condition_row_components = []
        
        with gr.Group(visible=False) as multi_condition_group:
            visible_rows = gr.State(1)
            
            for i in range(MAX_CONDITIONS):
                with gr.Row(visible=(i==0)) as row:
                    grade = gr.Dropdown(choices=grade_levels, label=f"Grade (Set {i+1})", value="1st Grade")
                    disorders = gr.Dropdown(
                        choices=disorder_types, 
                        label="Disorder(s)", 
                        value=["Articulation"], 
                        multiselect=True
                    )
                    num = gr.Number(label="# Students", value=1, minimum=1, step=1)
                    model = gr.Dropdown(choices=model_choices, label="Model", value="gpt-4o")
                    multi_condition_rows.append([grade, disorders, num, model])
                    multi_condition_row_components.append(row)
            
            with gr.Row():
                add_button = gr.Button("➕ Add Condition Set", size="sm")
                remove_button = gr.Button("➖ Remove Last Set", size="sm")
        
        # Output settings section
        with gr.Accordion("📁 Output Settings", open=True):
            output_path = gr.Textbox(
                label="Save Location",
                value=DEFAULT_OUTPUT_PATH,
                placeholder="Enter folder path to save generated files",
                info=f"Default: {DEFAULT_OUTPUT_PATH}"
            )
        
        # Generate button with custom class
        with gr.Row():
            generate_button = gr.Button(
                "🚀 Generate Case Files", 
                variant="primary", 
                size="sm",
                elem_classes="generate-btn"
            )
        
        # Progress indicator using Markdown
        progress_status = gr.Markdown(value="", visible=False)
        
        output_display = gr.Markdown(label="Generated Case Files")
        
        def update_visibility(mode):
            return gr.update(visible=(mode == "Single Condition")), gr.update(visible=(mode == "Multiple Conditions"))
        
        gen_mode.change(
            fn=update_visibility, 
            inputs=gen_mode, 
            outputs=[single_condition_group, multi_condition_group]
        )
        
        def update_rows(count):
            """Updates visibility of all condition rows based on count."""
            return [gr.update(visible=(i < count)) for i in range(MAX_CONDITIONS)]

        def add_row(count):
            """Increment the visible row count."""
            new_count = min(count + 1, MAX_CONDITIONS)
            return new_count

        def remove_row(count):
            """Decrement the visible row count."""
            new_count = max(count - 1, 1)
            return new_count

        # Connect add/remove buttons
        add_button.click(
            fn=add_row, 
            inputs=[visible_rows], 
            outputs=[visible_rows]
        ).then(
            fn=update_rows,
            inputs=[visible_rows],
            outputs=multi_condition_row_components
        )
        
        remove_button.click(
            fn=remove_row, 
            inputs=[visible_rows], 
            outputs=[visible_rows]
        ).then(
            fn=update_rows,
            inputs=[visible_rows],
            outputs=multi_condition_row_components
        )

        # Flatten all multi-condition inputs for the generate button
        all_multi_inputs = [item for sublist in multi_condition_rows for item in sublist]
        
        # Generate button click handler
        generate_button.click(
            fn=process_generation_request,
            inputs=[gen_mode, num_students_single, model_single, single_grade, single_disorders, visible_rows] + all_multi_inputs + [output_path],
            outputs=[output_display, generate_button, progress_status, cost_display]
        )
    
    ui.launch()