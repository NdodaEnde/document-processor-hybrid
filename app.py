# app.py
import os
import io
import time
import json
import base64
import tempfile
import concurrent.futures
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv

# Set SDK parallelism to maximum recommended values
os.environ["BATCH_SIZE"] = "20"  # Higher batch size for processing multiple documents
os.environ["MAX_WORKERS"] = "5"  # Max workers per document processing
os.environ["MAX_RETRIES"] = "100"  # Maximum retry attempts
os.environ["RETRY_LOGGING_STYLE"] = "inline_block"  # More compact logging

# Import the Agentic Doc SDK
from agentic_doc.parse import parse_documents
from agentic_doc.common import ChunkType

# Create FastAPI app
app = FastAPI(title="Document Processing Microservice")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
api_key = os.getenv("VISION_AGENT_API_KEY")
if not api_key:
    print("WARNING: VISION_AGENT_API_KEY not set in .env!")

# Data models
class DocumentResult(BaseModel):
    document_id: Optional[str] = None
    document_type: str
    structured_data: Dict[str, Any]
    raw_response: Dict[str, Any]
    bounding_boxes: Dict[str, List]
    processing_time: float

# Parse PDFs with the agentic-doc SDK
def parse_pdf_agentic(file_content, filename):
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        # Use the SDK to process the document
        start_time = time.time()
        parse_results = parse_documents([tmp_path])
        processing_time = time.time() - start_time
        
        if not parse_results:
            return {}, {}, processing_time
            
        parsed_doc = parse_results[0]
        page_map = {}
        
        # Extract raw text and structured content
        all_text = ""
        
        for chunk in parsed_doc.chunks:
            if chunk.chunk_type == "error":
                continue
                
            if chunk.text:
                all_text += chunk.text + "\n"
                
            # A single chunk can have multiple groundings (boxes)
            for grounding in chunk.grounding:
                # grounding.page is 0-based
                page_idx = grounding.page + 1  # convert to 1-based
                # Convert integer key to string
                page_key = str(page_idx)
                
                if page_key not in page_map:
                    page_map[page_key] = []
                    
                # Get bounding box
                box = grounding.box
                x1, y1 = box.l, box.t
                w, h = box.r - box.l, box.b - box.t
                
                # Add to page_map structure
                page_map[page_key].append({
                    "bboxes": [[x1, y1, w, h]],
                    "captions": [chunk.text],
                })
        
        # Extract key information from markdown
        patient_name = "Unknown"
        patient_id = ""
        company_name = ""
        examination_date = ""
        expiry_date = ""
        job_title = ""
        is_pre_employment = False
        is_periodical = False
        is_exit = False
        is_fit = False
        is_fit_with_restriction = False
        is_fit_with_condition = False
        is_temporarily_unfit = False
        is_unfit = False
        
        # Extract from markdown
        for line in all_text.split('\n'):
            if "**Initials & Surname**:" in line:
                patient_name = line.split("**Initials & Surname**:")[1].strip()
            elif "**ID No**:" in line:
                patient_id = line.split("**ID No**:")[1].strip()
            elif "**Company Name**:" in line:
                company_name = line.split("**Company Name**:")[1].strip()
            elif "**Date of Examination**:" in line:
                examination_date = line.split("**Date of Examination**:")[1].strip()
            elif "**Expiry Date**:" in line:
                expiry_date = line.split("**Expiry Date**:")[1].strip()
            elif "**Job Title**:" in line:
                job_title = line.split("**Job Title**:")[1].strip()
            elif "Pre-Employment**: [x]" in line or "Pre-Employment**:[x]" in line:
                is_pre_employment = True
            elif "Periodical**: [x]" in line or "Periodical**:[x]" in line:
                is_periodical = True
            elif "Exit**: [x]" in line or "Exit**:[x]" in line:
                is_exit = True
            elif "FIT**: [x]" in line or "FIT**:[x]" in line:
                is_fit = True
            elif "Fit with Restriction**: [x]" in line or "Fit with Restriction**:[x]" in line:
                is_fit_with_restriction = True
            elif "Fit with Condition**: [x]" in line or "Fit with Condition**:[x]" in line:
                is_fit_with_condition = True
            elif "Temporary Unfit**: [x]" in line or "Temporary Unfit**:[x]" in line:
                is_temporarily_unfit = True
            elif "UNFIT**: [x]" in line or "UNFIT**:[x]" in line:
                is_unfit = True
        
        # Create a checkbox based on the full text
        checkboxes = []
        if all_text:
            checkboxes.append({
                "text": all_text,
                "checked": True,
                "page": 1,
                "bbox": [0.05625, 0.1626179875333927, 0.91375, 0.7255943900267141]
            })
        
        # Build the structure that matches your API response
        structured_content = {
            "document_text": all_text,
            "tables": [],
            "form_fields": {},
            "checkboxes": checkboxes,
            "full_text": all_text,
            # Add these critical fields
            "patient": {
                "name": patient_name,
                "id_number": patient_id,
                "company": company_name,
                "occupation": job_title,
                "date_of_birth": "",
                "employee_id": patient_id
            },
            "examination_results": {
                "date": examination_date,
                "type": {
                    "pre_employment": is_pre_employment,
                    "periodical": is_periodical,
                    "exit": is_exit
                },
                "test_results": {}
            },
            "certification": {
                "examination_date": examination_date,
                "valid_until": expiry_date,
                "fit": is_fit,
                "fit_with_restrictions": is_fit_with_restriction,
                "fit_with_condition": is_fit_with_condition,
                "temporarily_unfit": is_temporarily_unfit,
                "unfit": is_unfit,
                "comments": ""
            },
            "restrictions": {}
        }
        
        # Extract test results
        test_structure = {
            "bloods": {"done": False, "results": ""},
            "far_near_vision": {"done": False, "results": ""},
            "side_depth": {"done": False, "results": ""},
            "night_vision": {"done": False, "results": ""},
            "hearing": {"done": False, "results": ""},
            "heights": {"done": False, "results": ""},
            "lung_function": {"done": False, "results": ""},
            "x_ray": {"done": False, "results": ""},
            "drug_screen": {"done": False, "results": ""}
        }
        
        # Check for test results in the markdown
        if "BLOODS" in all_text and "[x]" in all_text:
            test_structure["bloods"]["done"] = True
        if "FAR, NEAR VISION" in all_text and "[x]" in all_text:
            test_structure["far_near_vision"]["done"] = True
        if "SIDE & DEPTH" in all_text and "[x]" in all_text:
            test_structure["side_depth"]["done"] = True
        if "NIGHT VISION" in all_text and "[x]" in all_text:
            test_structure["night_vision"]["done"] = True
        if "Hearing" in all_text and "[x]" in all_text:
            test_structure["hearing"]["done"] = True
        if "Working at Heights" in all_text and "[x]" in all_text:
            test_structure["heights"]["done"] = True
        if "Lung Function" in all_text and "[x]" in all_text:
            test_structure["lung_function"]["done"] = True
        if "X-Ray" in all_text and "[x]" in all_text:
            test_structure["x_ray"]["done"] = True
        if "Drug Screen" in all_text and "[x]" in all_text:
            test_structure["drug_screen"]["done"] = True
            
        # Add test results to structured data
        for test_key, test_info in test_structure.items():
            structured_content["examination_results"]["test_results"][f"{test_key}_done"] = test_info["done"]
            structured_content["examination_results"]["test_results"][f"{test_key}_results"] = test_info["results"]
        
        # Create raw response with the same structure
        raw_response = {
            "document_analysis": {
                "text": all_text,
                "tables": []
            },
            "data": {
                "markdown": all_text
            }
        }
        
        return structured_content, page_map, processing_time
    
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    document_id: Optional[str] = Form(None)
):
    try:
        # Read the file content
        file_content = await file.read()
        filename = file.filename
        
        # Process the document with SDK
        structured_content, bounding_boxes, processing_time = parse_pdf_agentic(file_content, filename)
        
        # Skip Pydantic validation and return JSONResponse directly
        return JSONResponse(
            status_code=200,
            content={
                "document_id": document_id,
                "document_type": document_type,
                "structured_data": structured_content,
                "raw_response": {
                    "document_analysis": {
                        "text": structured_content.get("full_text", ""),
                        "tables": structured_content.get("tables", [])
                    },
                    "data": {
                        "markdown": structured_content.get("full_text", "")
                    }
                }
            }
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"PROCESSING ERROR: {str(e)}")
        print(f"TRACEBACK: {error_trace}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

@app.post("/process-documents", response_model=List[DocumentResult])
async def process_multiple_documents(
    files: List[UploadFile] = File(...),
    document_type: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Process multiple documents in parallel with the agentic-doc SDK.
    Returns structured data and bounding box information for each document.
    """
    results = []
    temp_paths = []
    
    try:
        # Write files to disk for processing
        for file in files:
            content = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                temp_paths.append((tmp.name, file.filename, content))
        
        # Process all documents
        start_time = time.time()
        path_list = [path for path, _, _ in temp_paths]
        
        # Use the SDK to process documents in parallel
        parse_results = parse_documents(path_list)
        total_time = time.time() - start_time
        
        # Process results
        for i, (tmp_path, filename, content) in enumerate(temp_paths):
            if i < len(parse_results):
                parsed_doc = parse_results[i]
                
                # Extract data (similar to the single document process)
                structured_content = {
                    "document_text": "",
                    "tables": [],
                    "form_fields": {},
                    "checkboxes": []
                }
                
                page_map = {}
                all_text = ""
                
                for chunk in parsed_doc.chunks:
                    if chunk.chunk_type == "error":
                        continue
                        
                    if chunk.text:
                        all_text += chunk.text + "\n"
                        
                    if chunk.chunk_type == "table":
                        structured_content["tables"].append(chunk.text)
                    elif chunk.chunk_type == "text":
                        structured_content["document_text"] += chunk.text + "\n"
                        
                    # Process bounding boxes
                    for grounding in chunk.grounding:
                        page_idx = grounding.page + 1
                        page_key = str(page_idx)
                        if page_key not in page_map:
                            page_map[page_key] = []
                            
                        box = grounding.box
                        x1, y1 = box.l, box.t
                        w, h = box.r - box.l, box.b - box.t
                        
                        page_map[page_key].append({
                            "bboxes": [[x1, y1, w, h]],
                            "captions": [chunk.text],
                        })
                        
                        # Check for form fields and checkboxes as before
                        if ":" in chunk.text and len(chunk.text.split(":")) == 2:
                            key, value = chunk.text.split(":", 1)
                            structured_content["form_fields"][key.strip()] = value.strip()
                        
                        if "[X]" in chunk.text or "[x]" in chunk.text or "☑" in chunk.text:
                            structured_content["checkboxes"].append({
                                "text": chunk.text,
                                "checked": True,
                                "page": page_idx,
                                "bbox": [x1, y1, w, h]
                            })
                        elif "[ ]" in chunk.text or "☐" in chunk.text:
                            structured_content["checkboxes"].append({
                                "text": chunk.text,
                                "checked": False,
                                "page": page_idx,
                                "bbox": [x1, y1, w, h]
                            })
                
                # Add full text
                structured_content["full_text"] = all_text
                
                # Create the result object
                result = DocumentResult(
                    document_id=None,  # Client can assign this
                    document_type=document_type,
                    structured_data=structured_content,
                    raw_response={
                        "document_analysis": {
                            "text": all_text,
                            "tables": structured_content["tables"]
                        },
                        "data": {
                            "markdown": all_text
                        }
                    },
                    bounding_boxes=page_map,
                    processing_time=total_time / len(temp_paths)  # Approximate time per document
                )
                
                results.append(result)
            
        return results
        
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Batch processing failed: {str(e)}"}
        )
    finally:
        # Clean up temp files
        for path, _, _ in temp_paths:
            try:
                os.unlink(path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
