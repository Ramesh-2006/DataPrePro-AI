from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import json
import os
from groq import Groq
import tempfile
# Using fpdf2 as it's more actively maintained and recommended over older fpdf
from fpdf import FPDF # Keep this import for now, but ensure fpdf2 is installed if issues arise
import base64
import chardet
from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Optional
import uvicorn
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
import secrets
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DataPreProAI API",
    description="API for AI-powered data preprocessing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files and setup templates (primarily for server-side rendering, less critical for pure React frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORS configuration
# IMPORTANT: Add any ports your frontend might run on during development
origins = [
     "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3004", # Add this line
    "http://localhost:5000",
    "https://your-production-domain.com" # Replace with your actual production domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Get API keys from environment
load_dotenv()  # Load environment variables from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("Missing GROQ_API_KEY environment variable")

if not GOOGLE_CLIENT_ID:
    logger.warning("GOOGLE_CLIENT_ID not found - Google Sign-In will be disabled")

# Session management
sessions: Dict[str, Dict] = {}

# Pydantic models
class CleaningStep(BaseModel):
    action: str
    column: str
    parameters: Dict
    reason: str

class ChatRequest(BaseModel):
    question: str

class VisualizationRequest(BaseModel):
    request: str

class GoogleAuthRequest(BaseModel):
    token: str

# New Pydantic model for email/password login
class LoginRequest(BaseModel):
    email: str
    password: str

# Helper functions
def detect_encoding(file_content: bytes) -> str:
    result = chardet.detect(file_content)
    return result['encoding']

def generate_unique_session_id() -> str:
    """Generate a secure random session ID"""
    return secrets.token_urlsafe(32)

def store_user_session(session_id: str, user_data: Dict) -> None:
    """Store user session with expiration"""
    sessions[session_id] = {
        **user_data,
        'expires_at': datetime.now() + timedelta(hours=24)
    }

def validate_session(session_id: str) -> Dict:
    """Validate and return session if exists and not expired"""
    cleanup_sessions()
    if session_id not in sessions:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    return sessions[session_id]

def cleanup_sessions() -> None:
    """Remove expired sessions"""
    now = datetime.now()
    expired = [sid for sid, session in sessions.items()
              if session.get('expires_at', now) < now]
    for sid in expired:
        # Before deleting, consider if there's any cleanup needed for associated files
        # For now, just delete the session data
        del sessions[sid]
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # This route is typically for serving a server-rendered index.html.
    # For a React frontend, you'll usually serve the React build directly.
    # If you're running React on a different port, this route might not be hit by the browser.
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/status")
async def get_status():
    cleanup_sessions()
    return {
        "status": "running",
        "version": app.version,
        "timestamp": datetime.now().isoformat(),
        "sessions_active": len(sessions)
    }

@app.post("/api/google-auth")
async def google_auth(auth_request: GoogleAuthRequest):
    try:
        if not GOOGLE_CLIENT_ID:
            raise HTTPException(status_code=501, detail="Google Sign-In not configured")

        token = auth_request.token
        if not token:
            raise HTTPException(status_code=400, detail="Token missing")

        # Verify Google token
        id_info = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        # Create new session
        session_id = generate_unique_session_id()
        store_user_session(session_id, {
            'user_info': {
                'email': id_info.get('email'),
                'name': id_info.get('name'),
                'picture': id_info.get('picture')
            },
            'df': None,
            'cleaned_df': None,
            'cleaning_steps': [],
            'initial_analysis': None,
            'original_filename': None
        })

        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "user": {
                "email": id_info.get('email'),
                "name": id_info.get('name'),
                "picture": id_info.get('picture') # Include picture for frontend display
            }
        })

    except ValueError as e:
        logger.error(f"Google token verification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        logger.error(f"Google auth error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Authentication failed")

# NEW: Email/Password Login Endpoint (Placeholder)
@app.post("/api/login")
async def email_login(login_request: LoginRequest):
    # This is a placeholder. In a real application, you would:
    # 1. Look up the user by email in your database.
    # 2. Verify the password (e.g., using bcrypt.checkpw).
    # 3. Handle successful login or invalid credentials.

    # For demonstration, we'll just simulate a successful login for any input
    # and create a new session.
    logger.info(f"Attempting login for: {login_request.email}")
    if login_request.email and login_request.password:
        session_id = generate_unique_session_id()
        user_info = {
            'email': login_request.email,
            'name': login_request.email.split('@')[0], # Simple name from email
            'picture': None # No picture for email login by default
        }
        store_user_session(session_id, {
            'user_info': user_info,
            'df': None,
            'cleaned_df': None,
            'cleaning_steps': [],
            'initial_analysis': None,
            'original_filename': None
        })
        return JSONResponse({
            "status": "success",
            "session_id": session_id,
            "user": user_info,
            "message": "Login successful (placeholder)"
        })
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials (placeholder)")


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(..., max_size=50_000_000), session_id: Optional[str] = Form(None)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")

        # Create new session if none provided, or validate existing
        if not session_id:
            session_id = generate_unique_session_id()
            store_user_session(session_id, {
                'user_info': None, # User info will be added if authenticated later
                'df': None,
                'cleaned_df': None,
                'cleaning_steps': [],
                'initial_analysis': None,
                'original_filename': file.filename
            })
        else:
            session = validate_session(session_id)
            session['original_filename'] = file.filename
            # Update session expiration on activity
            session['expires_at'] = datetime.now() + timedelta(hours=24)


        file_content = await file.read()

        try:
            if file.filename.lower().endswith('.csv'):
                encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                df = None

                for encoding in encodings_to_try:
                    try:
                        file_obj = StringIO(file_content.decode(encoding))
                        df = pd.read_csv(file_obj)
                        break
                    except UnicodeDecodeError:
                        continue

                if df is None:
                    # Fallback to chardet if common encodings fail
                    encoding = detect_encoding(file_content)
                    file_obj = StringIO(file_content.decode(encoding))
                    df = pd.read_csv(file_obj)

            elif file.filename.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(BytesIO(file_content))
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Supported formats: CSV, Excel (XLSX/XLS)"
                )

            if df.empty:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty or couldn't be parsed"
                )

            sessions[session_id]['df'] = df.to_json(orient='split')
            sessions[session_id]['cleaned_df'] = df.to_json(orient='split') # Initialize cleaned_df

            return {
                "session_id": session_id,
                "filename": file.filename,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "columns_list": df.columns.tolist(),
                "session_expires": sessions[session_id]['expires_at'].isoformat()
            }

        except HTTPException:
            raise # Re-raise FastAPI HTTPExceptions
        except Exception as e:
            logger.error(f"File processing error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail="Error processing file. Please check the file format or content."
            )

    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during file upload"
        )

@app.get("/api/dataset-info/{session_id}")
async def get_dataset_info(session_id: str):
    try:
        session = validate_session(session_id)
        df = pd.read_json(StringIO(session['df']), orient='split')

        info = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "columns_list": df.columns.tolist(),
            "memory_usage": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "missing_values": int(df.isnull().sum().sum()),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "df_head_json": df.head(10).to_json(orient='split', index=False) # Added for frontend preview
        }

        missing_values = df.isnull().sum()
        info["missing_values_per_column"] = {
            col: int(count) for col, count in missing_values.items() if count > 0
        }

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            info["numeric_stats"] = json.loads(df[numeric_cols].describe().to_json())

        return info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error retrieving dataset information"
        )

@app.get("/api/recommendations/{session_id}")
async def get_cleaning_recommendations(session_id: str):
    try:
        session = validate_session(session_id)
        df = pd.read_json(StringIO(session['cleaned_df']), orient='split')
        client = Groq(api_key="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")

        # Prepare a concise summary of the DataFrame for the LLM
        df_summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values_count": df.isnull().sum().to_dict(),
            "first_5_rows": df.head(5).to_dict(orient='records')
        }

        context = f"""
        Current Dataset State:
        {json.dumps(df_summary, indent=2)}
        """

        prompt = """
        Analyze this dataset's current state and provide a list of recommended cleaning steps.
        Return the recommendations in JSON format with the following structure:
        {
            "steps": [
                {
                    "action": "action_name",
                    "column": "column_name",
                    "parameters": {},
                    "reason": "explanation why this step is needed"
                }
            ]
        }

        Possible actions and their common parameters:
        - "handle_missing": {"method": "drop" | "mean" | "median" | "mode" | "ffill" | "bfill" | "interpolate"}
        - "convert_type": {"type": "numeric" | "datetime" | "category"}
        - "remove_outliers": {"method": "iqr" | "zscore"}
        - "normalize": {"method": "minmax" | "zscore"}
        - "encode_categorical": {"method": "onehot" | "label"}
        - "drop_column": {}

        Focus on data quality improvements (missing values, incorrect types, outliers, etc.).
        Be specific about columns and parameters. If no cleaning is needed, return an empty list for "steps".
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert data cleaning assistant. Analyze datasets and provide specific, actionable cleaning recommendations in JSON format."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {prompt}"
                }
            ],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        response = json.loads(chat_completion.choices[0].message.content)
        # Store recommendations in session for potential future use or reporting
        session['cleaning_recommendations'] = response.get('steps', [])

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cleaning recommendations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error generating cleaning recommendations"
        )

@app.post("/api/apply-step/{session_id}")
async def apply_cleaning_step(session_id: str, step: CleaningStep):
    try:
        session = validate_session(session_id)
        df = pd.read_json(StringIO(session['cleaned_df']), orient='split')
        col = step.column
        params = step.parameters

        # Apply the cleaning action
        if step.action == 'handle_missing':
            method = params.get('method', 'drop')
            if method == 'drop':
                df.dropna(subset=[col], inplace=True)
            elif method == 'mean':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    raise ValueError(f"Cannot fill mean on non-numeric column: {col}")
            elif method == 'median':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    raise ValueError(f"Cannot fill median on non-numeric column: {col}")
            elif method == 'mode':
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == 'ffill':
                df[col].fillna(method='ffill', inplace=True)
            elif method == 'bfill':
                df[col].fillna(method='bfill', inplace=True)
            elif method == 'interpolate':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].interpolate(inplace=True)
                else:
                    raise ValueError(f"Cannot interpolate on non-numeric column: {col}")

        elif step.action == 'convert_type':
            new_type = params.get('type', 'str')
            if new_type == 'numeric':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif new_type == 'datetime':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif new_type == 'category':
                df[col] = df[col].astype('category')

        elif step.action == 'remove_outliers':
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Outlier removal only applicable to numeric columns: {col}")
            method = params.get('method', 'iqr')
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # Cap outliers rather than removing rows, to maintain shape
                df[col] = np.where(df[col] < lower_bound, lower_bound,
                                   np.where(df[col] > upper_bound, upper_bound, df[col]))
            elif method == 'zscore':
                # Replace outliers with median
                z_scores = (df[col] - df[col].mean()) / df[col].std()
                df[col] = np.where(np.abs(z_scores) > 3, df[col].median(), df[col])

        elif step.action == 'normalize':
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Normalization only applicable to numeric columns: {col}")
            method = params.get('method', 'minmax')
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if (max_val - min_val) != 0:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else: # Handle case where all values are the same
                    df[col] = 0.0
            elif method == 'zscore':
                std_val = df[col].std()
                if std_val != 0:
                    df[col] = (df[col] - df[col].mean()) / std_val
                else: # Handle case where all values are the same
                    df[col] = 0.0

        elif step.action == 'encode_categorical':
            if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                raise ValueError(f"Categorical encoding only applicable to string/categorical columns: {col}")
            method = params.get('method', 'onehot')
            if method == 'onehot':
                encoded = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(col, axis=1), encoded], axis=1)
            elif method == 'label':
                # Ensure scikit-learn is installed: pip install scikit-learn
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str)) # Convert to string to handle mixed types safely

        elif step.action == 'drop_column':
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            df.drop(col, axis=1, inplace=True)

        else:
            raise HTTPException(status_code=400, detail=f"Unknown cleaning action: {step.action}")

        session['cleaned_df'] = df.to_json(orient='split')
        step_description = f"{step.action} applied to {col}"
        if step.reason:
            step_description += f" ({step.reason})"
        session['cleaning_steps'].append(step_description)

        # Update session expiration on activity
        session['expires_at'] = datetime.now() + timedelta(hours=24)

        return {
            "message": "Step applied successfully",
            "step_description": step_description,
            "new_shape": df.shape,
            "session_expires": session['expires_at'].isoformat()
        }

    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except ValueError as e:
        logger.error(f"Validation error applying cleaning step: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid operation: {str(e)}")
    except Exception as e:
        logger.error(f"Error applying cleaning step: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error applying data cleaning step"
        )

@app.post("/api/chat/{session_id}")
async def chat_with_ai(session_id: str, chat_request: ChatRequest):
    try:
        session = validate_session(session_id)
        df = pd.read_json(StringIO(session['cleaned_df']), orient='split')
        client = Groq(api_key=GROQ_API_KEY)

        # Prepare a concise summary of the DataFrame for the LLM
        df_summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values_count": df.isnull().sum().to_dict(),
            "first_5_rows": df.head(5).to_dict(orient='records')
        }

        context = f"""
        Current Dataset State:
        {json.dumps(df_summary, indent=2)}
        """

        if session['cleaning_steps']:
            context += f"\nApplied Cleaning Steps:\n" + "\n".join(session['cleaning_steps'])

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful data science assistant. Analyze the provided dataset and answer questions about it concisely. If a question is about cleaning, refer to the 'cleaning_recommendations' endpoint. If it's about visualization, refer to the 'generate-code' endpoint. Do not generate code or lists of cleaning steps here, just provide insights or guidance."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {chat_request.question}"
                }
            ],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.6
        )

        response = chat_completion.choices[0].message.content

        # Store initial analysis if it's the first chat or a significant one
        if not session.get('initial_analysis'):
            session['initial_analysis'] = response
        # Update session expiration on activity
        session['expires_at'] = datetime.now() + timedelta(hours=24)

        return {"response": response}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in AI chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing your question"
        )

@app.post("/api/generate-code/{session_id}")
async def generate_visualization_code(session_id: str, vis_request: VisualizationRequest):
    try:
        session = validate_session(session_id)
        df = pd.read_json(StringIO(session['cleaned_df']), orient='split')
        client = Groq(api_key=GROQ_API_KEY)

        # Convert Timestamp objects to string format for JSON serialization
        # This creates a copy to avoid modifying the original DataFrame in session directly
        df_for_summary = df.copy()
        for col in df_for_summary.select_dtypes(include=['datetime64']).columns:
            df_for_summary[col] = df_for_summary[col].dt.isoformat() if not df_for_summary[col].isnull().all() else None
        
        # Prepare a concise summary of the DataFrame for the LLM
        # Limit the number of columns and rows in the summary to save tokens
        df_summary = {
            "shape": df_for_summary.shape,
            "columns": df_for_summary.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df_for_summary.dtypes.items()},
            "numeric_columns": df_for_summary.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            "categorical_columns": df_for_summary.select_dtypes(include=['object', 'category']).columns.tolist(),
            # Only include a very small head, or none if context is too long
            "first_3_rows_sample": df_for_summary.head(3).to_dict(orient='records')
        }

        context = f"""
        Current Dataset Schema and Sample:
        {json.dumps(df_summary, indent=2)}
        """

        prompt = f"""
        Generate concise Python code for data visualization based on this dataset and the user's request.
        User Request: "{vis_request.request}"

        Requirements for the Python code:
        1. Use `matplotlib.pyplot` (as `plt`) and/or `seaborn` (as `sns`).
        2. The code MUST be self-contained and directly executable.
        3. Use the DataFrame variable named `df`.
        4. Include `plt.title()`, `plt.xlabel()`, `plt.ylabel()` for clarity.
        5. Call `plt.tight_layout()` and `plt.close()` after `plt.savefig()`.
        6. Save the plot to a BytesIO object as PNG and encode it as base64.
        7. The final base64 string MUST be assigned to a variable named `plot_base64`.
        8. Keep the code as CONCISE as possible to avoid token limits. Do not add excessive comments or verbose explanations in the code itself.
        9. If plotting datetime columns, ensure they are handled appropriately (e.g., converted to numeric or plotted as time series).

        Example of saving and encoding:
        ```python
        import matplotlib.pyplot as plt
        import seaborn as sns
        from io import BytesIO
        import base64
        # Plotting code here, e.g., sns.histplot(df['column'])
        plt.title('Plot Title')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        ```
        Return a JSON response with ONLY "code" and "plot" keys.
        {{
            "code": "your_generated_python_code_string",
            "plot": "base64_encoded_image_string"
        }}
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful data visualization code generator. Generate CONCISE Python plotting code based on user requests and dataset context. Prioritize brevity and directness in the code output."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {prompt}"
                }
            ],
            model="deepseek-r1-distill-llama-70b",
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        response_content = chat_completion.choices[0].message.content
        response_dict = json.loads(response_content)
        generated_code = response_dict.get('code', '')

        # Execute the generated code to produce the plot
        exec_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'df': df, # Pass the original DataFrame to the execution environment for plotting
            'BytesIO': BytesIO,
            'base64': base64
        }

        # Clear existing plots to prevent them from stacking
        plt.clf()
        plt.close('all')

        exec(generated_code, exec_globals)

        plot_base64 = exec_globals.get('plot_base64') # Retrieve the base64 plot from exec_globals

        if not plot_base64:
            raise ValueError("Generated code did not produce a 'plot_base64' variable.")

        # Update session expiration on activity
        session['expires_at'] = datetime.now() + timedelta(hours=24)

        return {
            "code": generated_code,
            "plot": plot_base64
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating or executing visualization code: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating visualization: {str(e)}. Please try a different request or check data types."
        )

@app.get("/api/generate-report/{session_id}")
async def generate_report(session_id: str):
    try:
        session = validate_session(session_id)
        df = pd.read_json(StringIO(session['cleaned_df']), orient='split')
        initial_df = pd.read_json(StringIO(session['df']), orient='split')

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=10)
        pdf.add_page()

        # Report header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Data Analysis Report", 0, 1, 'C')
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        pdf.cell(0, 5, f"Original file: {session['original_filename']}", 0, 1, 'C')
        pdf.ln(10)

        # Dataset Overview
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "1. Dataset Overview", 0, 1)
        pdf.set_font("Arial", '', 10)

        col_width = pdf.w / 2.5
        overview_data = [
            ["Number of Rows (Cleaned)", str(df.shape[0])],
            ["Number of Columns (Cleaned)", str(df.shape[1])],
            ["Memory Usage (Cleaned)", f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"],
            ["Duplicate Rows (Cleaned)", str(df.duplicated().sum())]
        ]

        for row in overview_data:
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(col_width, 8, row[0], border=1, fill=True)
            pdf.cell(col_width, 8, row[1], border=1)
            pdf.ln()

        pdf.ln(5)

        # Data Quality Assessment (on cleaned data)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "2. Data Quality Assessment (Cleaned Data)", 0, 1)

        missing_values = df.isnull().sum()[df.isnull().sum() > 0]
        if len(missing_values) > 0:
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 8, "Missing Values:", 0, 1)

            pdf.set_font("Arial", 'B', 10)
            pdf.cell(100, 8, "Column", border=1, fill=True)
            pdf.cell(40, 8, "Missing Count", border=1, fill=True)
            pdf.cell(40, 8, "Percentage", border=1, fill=True)
            pdf.ln()

            pdf.set_font("Arial", '', 10)
            for col, count in missing_values.items():
                percentage = (count / df.shape[0]) * 100
                pdf.cell(100, 8, col, border=1)
                pdf.cell(40, 8, str(count), border=1)
                pdf.cell(40, 8, f"{percentage:.1f}%", border=1)
                pdf.ln()
        else:
            pdf.cell(0, 8, "No missing values found in cleaned data.", 0, 1)

        pdf.ln(5)

        # Data Cleaning Steps
        if session['cleaning_steps']:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "3. Data Cleaning Steps Applied", 0, 1)
            pdf.set_font("Arial", '', 10)

            for i, step in enumerate(session['cleaning_steps'], 1):
                pdf.multi_cell(0, 8, f"{i}. {step}")
            pdf.ln(5)
        else:
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 8, "No cleaning steps were applied.", 0, 1)
            pdf.ln(5)


        # Summary Statistics (on cleaned data)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "4. Summary Statistics (Cleaned Data)", 0, 1)

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            pdf.set_font("Arial", '', 8)
            stats = df[numeric_cols].describe().round(2).transpose()

            # Add page if content is too long
            if pdf.get_y() + (len(stats) * 8) + 20 > pdf.h - 20: # Estimate space needed
                pdf.add_page()
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "4. Summary Statistics (Cleaned Data) - Continued", 0, 1)
                pdf.set_font("Arial", '', 8)


            pdf.set_fill_color(200, 220, 255)
            # Adjust column widths dynamically based on content or fixed for readability
            col_widths = [40] + [20] * (len(stats.columns)) # Reduced width for stats columns
            headers = ["Column"] + list(stats.columns)

            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 8, header, border=1, fill=True)
            pdf.ln()

            for idx, row in stats.iterrows():
                # Check for page break before adding a row
                if pdf.get_y() + 8 > pdf.h - 10: # If current row won't fit within 10mm from bottom
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 10) # Repeat header on new page
                    for i, header in enumerate(headers):
                        pdf.cell(col_widths[i], 8, header, border=1, fill=True)
                    pdf.ln()
                    pdf.set_font("Arial", '', 8) # Reset font after header

                pdf.cell(col_widths[0], 8, str(idx), border=1)
                for i, val in enumerate(row):
                    pdf.cell(col_widths[i+1], 8, str(val), border=1)
                pdf.ln()
        else:
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 8, "No numeric columns for statistics in cleaned data.", 0, 1)

        pdf.ln(8)

        # Data Distributions (on cleaned data)
        if len(numeric_cols) > 0:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "5. Data Distributions (Cleaned Data)", 0, 1)

            temp_dir = tempfile.mkdtemp()
            img_paths = []

            # Generate and add plots to PDF
            for i, col in enumerate(numeric_cols[:6]):   # Limit to 6 plots for report size
                # Check if enough space for plot, otherwise add new page
                if pdf.get_y() + 60 > pdf.h - 20: # 60mm for plot height + some margin
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, f"5. Data Distributions (Cleaned Data) - {col}", 0, 1)
                    pdf.set_font("Arial", '', 10)

                plt.figure(figsize=(5, 2.5)) # Smaller figure size for PDF
                sns.histplot(df[col].dropna(), kde=True) # Dropna for plotting
                plt.title(f"Distribution of {col}", fontsize=8)
                plt.xlabel(col, fontsize=7)
                plt.ylabel("Count", fontsize=7)
                plt.xticks(fontsize=6)
                plt.yticks(fontsize=6)
                plt.tight_layout()

                img_path = os.path.join(temp_dir, f"{col}_dist.png").replace('\\', '/')
                plt.savefig(img_path, dpi=150, bbox_inches='tight') # Higher DPI for better quality
                plt.close() # Important to close plot to free memory
                img_paths.append(img_path)

                # Add image to PDF
                # x_pos = 10 + (i % 2) * 95 # Two plots per row
                # y_pos = pdf.get_y()
                pdf.image(img_path, x=pdf.get_x(), y=pdf.get_y(), w=90) # Adjust width as needed
                pdf.ln(50) # Move down after image

        else:
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 8, "No numeric columns to plot distributions for in cleaned data.", 0, 1)

        pdf.ln(8)

        # AI Analysis Summary
        if session.get('initial_analysis'):
            # Add page if content is too long
            if pdf.get_y() + 20 > pdf.h - 20:
                pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "6. AI Analysis Summary", 0, 1)
            pdf.set_font("Arial", '', 10)

            analysis_text = "\n".join([line for line in session['initial_analysis'].split("\n") if line.strip()])
            pdf.multi_cell(0, 8, analysis_text)
            pdf.ln(5)
        else:
            pdf.set_font("Arial", '', 10)
            pdf.cell(0, 8, "No AI analysis summary available yet.", 0, 1)
            pdf.ln(5)

        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 10, "Report generated by DataPreProAI", 0, 1, 'C')

        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_io = BytesIO()
        pdf_io.write(pdf_bytes)
        pdf_io.seek(0)

        # Clean up temporary image files
        for img_path in img_paths:
            if os.path.exists(img_path):
                os.unlink(img_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

        return StreamingResponse(
            pdf_io,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=data_report.pdf",
                "Session-Expires": session['expires_at'].isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating analysis report: {str(e)}"
        )

@app.get("/api/export-data/{session_id}")
async def export_data(
    session_id: str,
    format_type: str = 'csv',
    filename: str = 'cleaned_data'
):
    try:
        session = validate_session(session_id)
        df = pd.read_json(StringIO(session['cleaned_df']), orient='split')

        if format_type == 'csv':
            csv_data = df.to_csv(index=False)
            return StreamingResponse(
                BytesIO(csv_data.encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}.csv",
                    "Session-Expires": session['expires_at'].isoformat()
                }
            )
        elif format_type == 'excel':
            excel_io = BytesIO()
            df.to_excel(excel_io, index=False)
            excel_io.seek(0)
            return StreamingResponse(
                excel_io,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}.xlsx",
                    "Session-Expires": session['expires_at'].isoformat()
                }
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Supported formats: csv, excel"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error exporting dataset"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info",
        timeout_keep_alive=60
    )