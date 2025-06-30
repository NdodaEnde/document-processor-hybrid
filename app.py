"""
Medical Document Processing Microservice - Production Ready
===========================================================

Features:
- Direct bytes processing (no temp files)
- Pydantic-based structured extraction
- Concurrent batch processing
- Comprehensive error handling
- Memory optimization for Render
- Real-time progress tracking
- Certificate-specific field extraction
"""

from flask import Flask, request, jsonify
import os
import uuid
import json
import time
import tracemalloc
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any, Tuple
import queue
from io import BytesIO

# Pydantic imports for structured extraction
from pydantic import BaseModel, Field
from flask_cors import CORS

# Start memory tracing
tracemalloc.start()

# =============================================================================
# CONFIGURATION & ENVIRONMENT
# =============================================================================

# Version check and SDK import
def check_and_import_agentic_doc():
    """Check agentic-doc version and import appropriate functions"""
    try:
        import pkg_resources
        agentic_doc_version = pkg_resources.get_distribution("agentic-doc").version
        version_parts = agentic_doc_version.split('.')
        
        # Check version requirements
        major, minor = int(version_parts[0]), int(version_parts[1])
        has_parse_function = major > 0 or (major == 0 and minor >= 2)
        has_bytes_support = major > 0 or (major == 0 and minor >= 2 and len(version_parts) > 2 and int(version_parts[2]) >= 4)
        
        print(f"[INIT] agentic-doc version: {agentic_doc_version}")
        print(f"[INIT] Parse function available: {has_parse_function}")
        print(f"[INIT] Bytes support available: {has_bytes_support}")
        
        # Import functions
        if has_parse_function:
            from agentic_doc.parse import parse, parse_documents
            return True, True, parse, parse_documents, agentic_doc_version
        else:
            from agentic_doc.parse import parse_documents
            return True, False, None, parse_documents, agentic_doc_version
            
    except ImportError as e:
        print(f"[INIT] agentic-doc not available: {e}")
        return False, False, None, None, None

# Initialize SDK
AGENTIC_DOC_AVAILABLE, PARSE_FUNCTION_AVAILABLE, parse_func, parse_documents_func, sdk_version = check_and_import_agentic_doc()

# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED EXTRACTION
# =============================================================================

class EmployeeInfo(BaseModel):
    """Employee information from certificate header"""
    full_name: str = Field(description="Complete employee name from 'Name:' or 'Initials & Surname:' field")
    company_name: str = Field(description="Employer company name from 'Company Name:' field")
    id_number: str = Field(description="South African 13-digit ID number from 'ID NO:' field")
    job_title: str = Field(description="Employee position from 'Job Title:' or 'Occupation:' field")

class MedicalTest(BaseModel):
    """Individual medical test result"""
    performed: bool = Field(description="Whether test was completed - look for checkmarks or 'YES' indicators")
    result: str = Field(description="Test outcome from 'Results' column (e.g., 'NORMAL', '20/20', 'PASS')")

class MedicalTests(BaseModel):
    """Medical examination test results table"""
    vision_test: Optional[MedicalTest] = Field(description="Vision/eyesight test from examination table")
    hearing_test: Optional[MedicalTest] = Field(description="Hearing/audiometry test from examination table")
    blood_test: Optional[MedicalTest] = Field(description="Blood work/laboratory test from examination table")
    lung_function: Optional[MedicalTest] = Field(description="Spirometry/lung function test from examination table")
    x_ray: Optional[MedicalTest] = Field(description="Chest X-ray test from examination table")
    drug_screen: Optional[MedicalTest] = Field(description="Drug/substance screening test from examination table")

class MedicalExamination(BaseModel):
    """Medical examination details and results"""
    examination_date: str = Field(description="Date of medical exam in DD.MM.YYYY format (e.g., '15.03.2024')")
    expiry_date: str = Field(description="Certificate expiry date in DD.MM.YYYY format")
    examination_type: str = Field(description="Type: 'PRE-EMPLOYMENT', 'PERIODICAL', or 'EXIT' examination")
    fitness_status: str = Field(description="Medical fitness result: 'FIT', 'UNFIT', 'Fit with Restriction', etc.")
    restrictions: List[str] = Field(description="Any work restrictions or limitations listed")
    comments: Optional[str] = Field(description="Additional medical comments or notes")

class MedicalPractitioner(BaseModel):
    """Medical practitioner information"""
    doctor_name: str = Field(description="Examining doctor's name from header or signature area")
    practice_number: str = Field(description="Medical practice registration number")
    signature_present: bool = Field(description="Whether doctor's signature is visible on certificate")
    stamp_present: bool = Field(description="Whether official medical practice stamp is present")

class CertificateOfFitness(BaseModel):
    """Complete Certificate of Fitness for structured extraction"""
    document_classification: str = Field(description="Document type - should be 'certificate_of_fitness'")
    employee_info: EmployeeInfo = Field(description="Employee personal and employment details")
    medical_examination: MedicalExamination = Field(description="Medical examination results and dates")
    medical_tests: MedicalTests = Field(description="Individual test results from examination table")
    medical_practitioner: MedicalPractitioner = Field(description="Examining doctor and practice information")

class QuestionnaireResponse(BaseModel):
    """Medical questionnaire response data"""
    document_classification: str = Field(description="Document type classification")
    patient_info: EmployeeInfo = Field(description="Patient demographic information")
    medical_history: Dict[str, Any] = Field(description="Medical history responses")
    current_medications: List[str] = Field(description="List of current medications")
    allergies: List[str] = Field(description="Known allergies and reactions")
    symptoms: List[str] = Field(description="Current symptoms or concerns")

class TestResults(BaseModel):
    """Standalone medical test results"""
    document_classification: str = Field(description="Document type classification")
    patient_info: EmployeeInfo = Field(description="Patient identification")
    test_results: Dict[str, str] = Field(description="Laboratory or diagnostic test results")
    test_date: str = Field(description="Date tests were performed")
    reference_ranges: Dict[str, str] = Field(description="Normal reference values")

# =============================================================================
# PROGRESS TRACKING & MEMORY MANAGEMENT
# =============================================================================

@dataclass
class BatchProgress:
    """Track batch processing progress with memory efficiency"""
    batch_id: str
    total_files: int
    processed_files: int
    failed_files: int
    status: str  # 'processing', 'completed', 'failed'
    start_time: float
    estimated_completion: Optional[float] = None
    current_file: Optional[str] = None
    errors: List[str] = None
    memory_usage: Dict[str, float] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.memory_usage is None:
            self.memory_usage = {}

    @property
    def progress_percentage(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100

    @property
    def processing_time(self) -> float:
        return time.time() - self.start_time

    def update_memory_usage(self):
        """Update current memory usage metrics"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            self.memory_usage = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "cpu_percent": process.cpu_percent()
            }
        except Exception as e:
            print(f"[MEMORY] Error updating memory usage: {e}")

    def to_dict(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "status": self.status,
            "progress_percentage": self.progress_percentage,
            "processing_time": self.processing_time,
            "estimated_completion": self.estimated_completion,
            "current_file": self.current_file,
            "errors": self.errors[-5:],  # Last 5 errors only
            "memory_usage": self.memory_usage
        }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_extraction_model(document_type: str) -> type:
    """Get appropriate Pydantic model for document type"""
    type_mapping = {
        'certificate-fitness': CertificateOfFitness,
        'certificate': CertificateOfFitness,
        'medical-questionnaire': QuestionnaireResponse,
        'questionnaire': QuestionnaireResponse,
        'test-results': TestResults,
        'lab-results': TestResults,
        'audiogram': TestResults,
        'spirometry': TestResults,
        'x-ray-report': TestResults
    }
    return type_mapping.get(document_type.lower(), CertificateOfFitness)

def calculate_confidence_score(extracted_data: Dict[str, Any]) -> float:
    """Calculate extraction confidence based on data completeness"""
    def count_fields(obj, depth=0):
        if depth > 3:  # Prevent infinite recursion
            return 0, 0
        
        total, filled = 0, 0
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    sub_total, sub_filled = count_fields(value, depth + 1)
                    total += sub_total
                    filled += sub_filled
                else:
                    total += 1
                    if value and str(value).strip() and str(value).lower() not in ['n/a', 'none', 'null', '']:
                        filled += 1
        elif isinstance(obj, list):
            for item in obj:
                sub_total, sub_filled = count_fields(item, depth + 1)
                total += sub_total
                filled += sub_filled
        
        return total, filled
    
    total_fields, filled_fields = count_fields(extracted_data)
    
    if total_fields == 0:
        return 0.0
    
    base_confidence = filled_fields / total_fields
    
    # Bonus for critical fields
    critical_bonus = 0
    if extracted_data.get('employee_info', {}).get('full_name'):
        critical_bonus += 0.1
    if extracted_data.get('employee_info', {}).get('id_number'):
        critical_bonus += 0.1
    if extracted_data.get('medical_examination', {}).get('examination_date'):
        critical_bonus += 0.1
    
    return min(1.0, base_confidence + critical_bonus)

def log_memory_usage(context: str = ""):
    """Enhanced memory logging for production monitoring"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        print(f"[MEMORY] {context}: RSS={memory_mb:.2f}MB, VMS={memory_info.vms / 1024 / 1024:.2f}MB, CPU={process.cpu_percent():.1f}%")
        
        # Log top memory consumers (limited to reduce log spam)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        if top_stats:
            print(f"[MEMORY] Top consumer: {top_stats[0]}")
            
    except Exception as e:
        print(f"[MEMORY] Error logging memory usage: {e}")

def force_garbage_collection():
    """Aggressive garbage collection for memory-constrained environments"""
    try:
        collected = []
        for generation in range(3):
            collected.append(gc.collect(generation))
        
        total_collected = sum(collected)
        if total_collected > 0:
            print(f"[GC] Collected {total_collected} objects across {len(collected)} generations")
        
        return total_collected
    except Exception as e:
        print(f"[GC] Error during garbage collection: {e}")
        return 0

def validate_file_bytes(file_bytes: bytes, filename: str, max_size_mb: int = 25) -> Tuple[bool, str]:
    """Validate file bytes before processing"""
    try:
        # Check file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.2f}MB (max {max_size_mb}MB)"
        
        # Check if file has content
        if len(file_bytes) == 0:
            return False, "File is empty"
        
        # Basic file type validation (check magic bytes)
        pdf_signature = file_bytes[:4] == b'%PDF'
        png_signature = file_bytes[:8] == b'\x89PNG\r\n\x1a\n'
        jpg_signature = file_bytes[:3] == b'\xff\xd8\xff'
        
        if not (pdf_signature or png_signature or jpg_signature):
            return False, "Unsupported file type (only PDF, PNG, JPG supported)"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def process_document_bytes(file_bytes: bytes, filename: str, document_type: str, 
                          extraction_method: str, batch_id: str) -> Dict[str, Any]:
    """Process document from bytes using agentic-doc with structured extraction"""
    
    start_time = time.time()
    
    try:
        print(f"[BATCH {batch_id}] Processing {filename} ({len(file_bytes)} bytes, type: {document_type}, method: {extraction_method})")
        
        # Validate file bytes
        is_valid, message = validate_file_bytes(file_bytes, filename)
        if not is_valid:
            raise ValueError(f"File validation failed: {message}")
        
        if extraction_method == 'structured' and PARSE_FUNCTION_AVAILABLE:
            # Use structured extraction with Pydantic models
            extraction_model = get_extraction_model(document_type)
            
            print(f"[STRUCTURED] Using {extraction_model.__name__} for {filename}")
            
            # Direct bytes processing - no temp files!
            results = parse_func(file_bytes, extraction_model=extraction_model)
            
            if not results or len(results) == 0:
                raise Exception("No results from structured extraction")
            
            result = results[0]
            extracted_data = result.extraction.dict() if hasattr(result.extraction, 'dict') else result.extraction
            
            processing_time = time.time() - start_time
        log_memory_usage("AFTER batch processing")
        
        # Store results in memory
        processed_docs[batch_id] = {
            "result": batch_result["successful_results"],
            "failed_results": batch_result["failed_results"],
            "processed_at": time.time(),
            "processing_stats": {
                "total_files": batch_result["total_files"],
                "successful_count": batch_result["successful_count"],
                "failed_count": batch_result["failed_count"],
                "total_processing_time": batch_result["total_processing_time"],
                "extraction_method": batch_result["extraction_method"]
            }
        }
        
        # Force garbage collection after batch completion
        force_garbage_collection()
        log_memory_usage("END batch processing")
        
        # Build response
        response = {
            "batch_id": batch_id,
            "document_count": len(files_data),
            "successful_count": batch_result["successful_count"],
            "failed_count": batch_result["failed_count"],
            "processing_time_seconds": processing_time,
            "status": "success",
            "extraction_method": extraction_method,
            "structured_extraction_used": extraction_method == 'structured' and PARSE_FUNCTION_AVAILABLE,
            "bytes_processing_used": True,
            "temp_files_created": 0,  # No temp files with bytes processing!
            "concurrent_processing": True,
            "warnings": []
        }
        
        if invalid_files:
            response["invalid_files"] = invalid_files
            response["warnings"].append(f"{len(invalid_files)} files were invalid and skipped")
        
        if batch_result["failed_count"] > 0:
            response["warnings"].append(f"{batch_result['failed_count']} files failed processing")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")
        log_memory_usage("ERROR state")
        
        # Emergency cleanup
        force_garbage_collection()
        
        return jsonify({
            "error": f"Batch processing failed: {str(e)}",
            "batch_id": batch_id,
            "partial_results": False
        }), 500

@app.route('/get-document-data/<batch_id>', methods=['GET'])
def get_document_data(batch_id):
    """Retrieve processed document data by batch ID"""
    log_memory_usage(f"GET document data for batch {batch_id}")
    
    if batch_id not in processed_docs:
        return jsonify({"error": "Batch ID not found"}), 404
    
    try:
        batch_data = processed_docs[batch_id]
        
        response = {
            "batch_id": batch_id,
            "result": batch_data["result"],
            "failed_results": batch_data.get("failed_results", []),
            "processed_at": batch_data["processed_at"],
            "processing_stats": batch_data.get("processing_stats", {}),
            "total_documents": len(batch_data["result"]) + len(batch_data.get("failed_results", [])),
            "extraction_summary": {
                "structured_extractions": len([r for r in batch_data["result"] if r.get("extraction_method") == "structured_pydantic"]),
                "ocr_extractions": len([r for r in batch_data["result"] if r.get("extraction_method") == "ocr_only"]),
                "average_confidence": sum(r.get("confidence_score", 0) for r in batch_data["result"]) / max(len(batch_data["result"]), 1)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"[ERROR] Error retrieving document data: {e}")
        return jsonify({"error": f"Error retrieving document data: {str(e)}"}), 500

@app.route('/process-single-document', methods=['POST'])
def process_single_document():
    """Process a single document with direct bytes handling"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    document_type = request.form.get('document_type', 'certificate-fitness')
    extraction_method = request.form.get('extraction_method', 'structured')
    
    if extraction_method not in ['structured', 'ocr']:
        return jsonify({"error": "Invalid extraction_method"}), 400
    
    try:
        # Read file bytes directly (no temp file!)
        file_bytes = file.read()
        
        # Validate file
        is_valid, message = validate_file_bytes(file_bytes, file.filename)
        if not is_valid:
            return jsonify({"error": f"File validation failed: {message}"}), 400
        
        # Process document
        batch_id = f"single_{uuid.uuid4()}"
        result = process_document_bytes(
            file_bytes, file.filename, document_type, extraction_method, batch_id
        )
        
        if result["status"] == "error":
            return jsonify({"error": result["error"]}), 500
        
        return jsonify({
            "status": "success",
            "result": result,
            "processing_method": "direct_bytes",
            "temp_files_created": 0
        })
        
    except Exception as e:
        print(f"[ERROR] Single document processing failed: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/debug-extraction', methods=['POST'])
def debug_extraction():
    """Debug endpoint to show raw OCR vs structured extraction"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Read file bytes
        file_bytes = file.read()
        
        # Validate file
        is_valid, message = validate_file_bytes(file_bytes, file.filename)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        debug_info = {
            "filename": file.filename,
            "file_size_mb": len(file_bytes) / (1024 * 1024),
            "sdk_available": AGENTIC_DOC_AVAILABLE,
            "structured_extraction_available": PARSE_FUNCTION_AVAILABLE,
            "bytes_processing": True
        }
        
        # Try OCR extraction
        try:
            if AGENTIC_DOC_AVAILABLE:
                ocr_results = parse_documents_func([file_bytes])
                if ocr_results:
                    ocr_data = serialize_parsed_document(ocr_results[0])
                    debug_info["ocr_extraction"] = {
                        "status": "success",
                        "markdown_length": len(ocr_data.get('markdown', '')),
                        "chunks_count": len(ocr_data.get('chunks', [])),
                        "sample_text": ocr_data.get('markdown', '')[:500] + "..." if len(ocr_data.get('markdown', '')) > 500 else ocr_data.get('markdown', '')
                    }
                else:
                    debug_info["ocr_extraction"] = {"status": "no_results"}
            else:
                debug_info["ocr_extraction"] = {"status": "sdk_not_available"}
        except Exception as e:
            debug_info["ocr_extraction"] = {"status": "error", "error": str(e)}
        
        # Try structured extraction
        try:
            if PARSE_FUNCTION_AVAILABLE:
                extraction_model = CertificateOfFitness
                structured_results = parse_func(file_bytes, extraction_model=extraction_model)
                if structured_results:
                    extracted_data = structured_results[0].extraction.dict()
                    debug_info["structured_extraction"] = {
                        "status": "success",
                        "model_used": "CertificateOfFitness",
                        "fields_extracted": len(str(extracted_data)),
                        "confidence_score": calculate_confidence_score(extracted_data),
                        "sample_data": {k: v for k, v in extracted_data.items() if k in ['document_classification', 'employee_info']}
                    }
                else:
                    debug_info["structured_extraction"] = {"status": "no_results"}
            else:
                debug_info["structured_extraction"] = {"status": "function_not_available"}
        except Exception as e:
            debug_info["structured_extraction"] = {"status": "error", "error": str(e)}
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({"error": f"Debug extraction failed: {str(e)}"}), 500

@app.route('/cleanup/<batch_id>', methods=['DELETE'])
def cleanup_batch(batch_id):
    """Enhanced cleanup for batch processing (memory only - no temp files!)"""
    if batch_id not in processed_docs:
        return jsonify({"error": "Batch ID not found"}), 404
    
    log_memory_usage(f"BEFORE cleanup batch {batch_id}")
    
    try:
        batch_data = processed_docs[batch_id]
        
        # Remove from processed docs
        del processed_docs[batch_id]
        
        # Remove from progress tracking
        with processing_lock:
            if batch_id in batch_progress:
                del batch_progress[batch_id]
        
        # Force garbage collection
        collected = force_garbage_collection()
        
        log_memory_usage(f"AFTER cleanup batch {batch_id}")
        
        return jsonify({
            "status": "success",
            "message": "Batch cleaned up successfully",
            "documents_cleaned": len(batch_data["result"]) + len(batch_data.get("failed_results", [])),
            "temp_files_removed": 0,  # No temp files with bytes processing!
            "memory_objects_collected": collected,
            "note": "Using bytes processing - no temporary files were created"
        })
        
    except Exception as e:
        print(f"[CLEANUP] Error during cleanup: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/system-info', methods=['GET'])
def system_info():
    """Get detailed system information for monitoring"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return jsonify({
            "system": {
                "memory_rss_mb": memory_info.rss / 1024 / 1024,
                "memory_vms_mb": memory_info.vms / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads()
            },
            "application": {
                "active_batches": len(processed_docs),
                "processing_batches": len([p for p in batch_progress.values() if p.status == "processing"]),
                "sdk_version": sdk_version,
                "structured_extraction": PARSE_FUNCTION_AVAILABLE,
                "bytes_processing": True
            },
            "memory_top_consumers": [
                {
                    "file": stat.traceback.format()[0] if stat.traceback.format() else "unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                }
                for stat in top_stats[:3]
            ] if top_stats else []
        })
        
    except Exception as e:
        return jsonify({"error": f"System info error: {str(e)}"}), 500

@app.route('/force-gc', methods=['POST'])
def force_gc():
    """Force garbage collection - useful for memory management"""
    try:
        log_memory_usage("BEFORE forced GC")
        collected = force_garbage_collection()
        log_memory_usage("AFTER forced GC")
        
        return jsonify({
            "status": "success",
            "objects_collected": collected,
            "message": f"Garbage collection completed, collected {collected} objects"
        })
        
    except Exception as e:
        return jsonify({"error": f"Garbage collection failed: {str(e)}"}), 500

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({
        "error": "File too large",
        "message": "The uploaded file exceeds the maximum size limit of 100MB",
        "max_size_mb": 100
    }), 413

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    print(f"[ERROR] Unhandled exception: {e}")
    log_memory_usage("EXCEPTION state")
    
    # Force cleanup on errors
    force_garbage_collection()
    
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "type": type(e).__name__
    }), 500

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == '__main__':
    log_memory_usage("Application startup")
    
    print("="*80)
    print("🏥 MEDICAL DOCUMENT PROCESSING MICROSERVICE v2.0")
    print("="*80)
    print(f"📦 agentic-doc version: {sdk_version}")
    print(f"🔧 SDK available: {AGENTIC_DOC_AVAILABLE}")
    print(f"⚡ Structured extraction: {PARSE_FUNCTION_AVAILABLE}")
    print(f"📄 Bytes processing: ✅ (no temp files)")
    print(f"🧠 Memory optimization: ✅")
    print(f"🔄 Concurrent processing: ✅ (max 3 workers)")
    print(f"📊 Real-time progress: ✅")
    print("")
    print("🎯 Supported document types:")
    for doc_type in ["certificate-fitness", "medical-questionnaire", "test-results", "audiogram", "spirometry"]:
        print(f"   • {doc_type}")
    print("")
    print("🚀 Extraction methods:")
    print("   • structured (Pydantic models)")
    print("   • ocr (text extraction only)")
    print("")
    print("💾 Memory features:")
    print("   • No temporary file creation")
    print("   • Direct bytes processing")
    print("   • Automatic garbage collection")
    print("   • Real-time memory monitoring")
    print("="*80)
    
    # Use Render's PORT environment variable
    port = int(os.environ.get('PORT', 5001))
    print(f"🌐 Starting server on port {port}")
    print("✨ Ready to process medical documents!")
    
    app.run(host='0.0.0.0', port=port, debug=False)_time = time.time() - start_time
            confidence_score = calculate_confidence_score(extracted_data)
            
            return {
                "status": "success",
                "filename": filename,
                "extraction_method": "structured_pydantic",
                "structured_data": extracted_data,
                "raw_data": None,
                "confidence_score": confidence_score,
                "processing_time": processing_time,
                "document_type": document_type,
                "extraction_metadata": getattr(result, 'extraction_metadata', {}),
                "extraction_error": None,
                "file_size_mb": len(file_bytes) / (1024 * 1024)
            }
            
        else:
            # Fall back to OCR-only processing
            print(f"[OCR] Using OCR-only processing for {filename}")
            
            if AGENTIC_DOC_AVAILABLE:
                # Use bytes processing if available
                results = parse_documents_func([file_bytes])
            else:
                # Mock processing for development
                results = [create_mock_parsed_document(filename)]
            
            if not results or len(results) == 0:
                raise Exception("No results from OCR processing")
            
            processing_time = time.time() - start_time
            raw_data = serialize_parsed_document(results[0])
            
            return {
                "status": "success",
                "filename": filename,
                "extraction_method": "ocr_only",
                "structured_data": None,
                "raw_data": raw_data,
                "confidence_score": 0.5,  # Medium confidence for OCR-only
                "processing_time": processing_time,
                "document_type": document_type,
                "extraction_error": None,
                "file_size_mb": len(file_bytes) / (1024 * 1024)
            }
            
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Processing failed for {filename}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        
        return {
            "status": "error",
            "filename": filename,
            "error": error_msg,
            "processing_time": processing_time,
            "document_type": document_type,
            "file_size_mb": len(file_bytes) / (1024 * 1024) if file_bytes else 0
        }

def create_mock_parsed_document(filename: str):
    """Create mock parsed document for development/testing"""
    return type('MockParsedDocument', (), {
        'markdown': f'Mock OCR content for {filename}\n\nEmployee Name: Mock Employee\nID Number: 1234567890123\nCompany: Mock Company',
        'chunks': [
            {
                'type': 'text',
                'content': f'Mock content from {filename}',
                'page': 1,
                'chunk_id': str(uuid.uuid4()),
                'grounding': [],
                'metadata': {}
            }
        ],
        'errors': [],
        'processing_time': 1.0
    })()

def serialize_parsed_document(parsed_doc) -> Dict[str, Any]:
    """Convert ParsedDocument to JSON-serializable format"""
    try:
        return {
            "markdown": getattr(parsed_doc, 'markdown', ''),
            "chunks": serialize_chunks(getattr(parsed_doc, 'chunks', [])),
            "errors": serialize_errors(getattr(parsed_doc, 'errors', [])),
            "processing_time": getattr(parsed_doc, 'processing_time', 0)
        }
    except Exception as e:
        print(f"[SERIALIZE] Error: {e}")
        return {
            "markdown": str(parsed_doc) if parsed_doc else '',
            "chunks": [],
            "errors": [{"message": f"Serialization error: {str(e)}", "page": 0}],
            "processing_time": 0
        }

def serialize_chunks(chunks) -> List[Dict[str, Any]]:
    """Serialize document chunks safely"""
    if not chunks:
        return []
    
    serialized = []
    for chunk in chunks:
        try:
            serialized.append({
                "type": getattr(chunk, 'type', 'unknown'),
                "content": getattr(chunk, 'content', ''),
                "page": getattr(chunk, 'page', 0),
                "chunk_id": getattr(chunk, 'chunk_id', str(uuid.uuid4())),
                "grounding": [],
                "metadata": getattr(chunk, 'metadata', {})
            })
        except Exception as e:
            print(f"[SERIALIZE] Chunk error: {e}")
            serialized.append({
                "type": "error",
                "content": f"Chunk serialization error: {str(e)}",
                "page": 0,
                "chunk_id": str(uuid.uuid4()),
                "grounding": [],
                "metadata": {}
            })
    
    return serialized

def serialize_errors(errors) -> List[Dict[str, Any]]:
    """Serialize processing errors safely"""
    serialized = []
    for error in errors:
        try:
            serialized.append({
                "message": getattr(error, 'message', str(error)),
                "page": getattr(error, 'page', 0),
                "error_code": getattr(error, 'error_code', 'unknown')
            })
        except Exception as e:
            serialized.append({
                "message": str(error),
                "page": 0,
                "error_code": "serialization_error"
            })
    
    return serialized

# =============================================================================
# BATCH PROCESSING ENGINE
# =============================================================================

def process_batch_concurrent(files_data: List[Tuple[bytes, str, str]], 
                           document_types: List[str], 
                           extraction_method: str, 
                           batch_id: str) -> Dict[str, Any]:
    """Concurrent batch processing with memory optimization"""
    
    print(f"[BATCH {batch_id}] Starting concurrent processing of {len(files_data)} files")
    
    # Initialize progress tracking
    progress = BatchProgress(
        batch_id=batch_id,
        total_files=len(files_data),
        processed_files=0,
        failed_files=0,
        status="processing",
        start_time=time.time()
    )
    
    batch_progress[batch_id] = progress
    
    # Process files concurrently with optimized thread pool
    max_workers = min(3, len(files_data))  # Limit workers for memory efficiency
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"Batch-{batch_id}") as executor:
        # Submit all files for processing
        future_to_index = {}
        for i, (file_bytes, filename, original_filename) in enumerate(files_data):
            document_type = document_types[i] if i < len(document_types) else 'certificate-fitness'
            
            future = executor.submit(
                process_document_bytes,
                file_bytes, filename, document_type, extraction_method, batch_id
            )
            future_to_index[future] = i
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            try:
                result = future.result()
                results.append(result)
                
                # Update progress
                with processing_lock:
                    if result["status"] == "success":
                        progress.processed_files += 1
                    else:
                        progress.failed_files += 1
                        progress.errors.append(result.get("error", "Unknown error"))
                    
                    progress.current_file = result["filename"]
                    progress.update_memory_usage()
                    
                    # Force garbage collection periodically
                    if (progress.processed_files + progress.failed_files) % 5 == 0:
                        force_garbage_collection()
                
                print(f"[BATCH {batch_id}] Progress: {progress.processed_files + progress.failed_files}/{progress.total_files}")
                
            except Exception as e:
                print(f"[BATCH {batch_id}] Future execution error: {e}")
                results.append({
                    "status": "error",
                    "filename": "unknown",
                    "error": f"Future execution error: {str(e)}",
                    "processing_time": 0
                })
                
                with processing_lock:
                    progress.failed_files += 1
                    progress.errors.append(str(e))
    
    # Finalize batch status
    with processing_lock:
        total_processed = progress.processed_files + progress.failed_files
        if total_processed == progress.total_files:
            progress.status = "completed" if progress.failed_files == 0 else "completed_with_errors"
        else:
            progress.status = "failed"
        
        progress.current_file = None
        progress.update_memory_usage()
    
    # Separate successful and failed results
    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [{"filename": r["filename"], "error": r["error"]} for r in results if r["status"] == "error"]
    
    # Final garbage collection
    force_garbage_collection()
    
    return {
        "successful_results": successful_results,
        "failed_results": failed_results,
        "total_files": len(files_data),
        "successful_count": len(successful_results),
        "failed_count": len(failed_results),
        "total_processing_time": time.time() - progress.start_time,
        "extraction_method": extraction_method,
        "batch_id": batch_id
    }

# =============================================================================
# FLASK APPLICATION SETUP
# =============================================================================

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload

# Global storage
processed_docs = {}
batch_progress = {}
processing_lock = threading.Lock()

# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with comprehensive system info"""
    log_memory_usage("Health check")
    
    with processing_lock:
        active_batches = len(batch_progress)
        processing_batches = len([p for p in batch_progress.values() if p.status == "processing"])
    
    # Memory statistics
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return jsonify({
        "status": "healthy",
        "service": "medical-document-processor",
        "version": "2.0.0",
        "agentic_doc_version": sdk_version,
        "agentic_doc_available": AGENTIC_DOC_AVAILABLE,
        "structured_extraction_available": PARSE_FUNCTION_AVAILABLE,
        "bytes_processing_enabled": True,
        "active_batches": len(processed_docs),
        "processing_batches": processing_batches,
        "total_tracked_batches": active_batches,
        "concurrent_processing": True,
        "max_workers": 3,
        "supported_document_types": [
            "certificate-fitness", "medical-questionnaire", "test-results",
            "audiogram", "spirometry", "x-ray-report", "lab-results"
        ],
        "extraction_methods": ["structured", "ocr"],
        "system_info": {
            "memory_rss_mb": memory_info.rss / 1024 / 1024,
            "memory_vms_mb": memory_info.vms / 1024 / 1024,
            "cpu_percent": process.cpu_percent()
        },
        "features": {
            "no_temp_files": True,
            "memory_optimized": True,
            "real_time_progress": True,
            "certificate_specific_extraction": True
        }
    })

@app.route('/batch-status/<batch_id>', methods=['GET'])
def get_batch_status(batch_id):
    """Get real-time batch processing status with memory info"""
    with processing_lock:
        if batch_id not in batch_progress:
            return jsonify({"error": "Batch ID not found"}), 404
        
        progress = batch_progress[batch_id]
        progress.update_memory_usage()  # Update memory stats
        return jsonify(progress.to_dict())

@app.route('/process-documents', methods=['POST'])
def process_documents():
    """Enhanced document processing with direct bytes handling"""
    log_memory_usage("START batch processing")
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No files selected"}), 400
    
    # Extract parameters
    extraction_method = request.form.get('extraction_method', 'structured')
    document_types_json = request.form.get('document_types', '[]')
    
    try:
        document_types = json.loads(document_types_json)
    except:
        document_types = []
    
    # Validate extraction method
    if extraction_method not in ['structured', 'ocr']:
        return jsonify({"error": "Invalid extraction_method. Use 'structured' or 'ocr'"}), 400
    
    # Fallback if structured extraction not available
    if extraction_method == 'structured' and not PARSE_FUNCTION_AVAILABLE:
        print(f"[WARNING] Structured extraction requested but not available. Using OCR.")
        extraction_method = 'ocr'
    
    # Process files directly from bytes (no temp files!)
    files_data = []
    invalid_files = []
    
    for file in files:
        if file and file.filename:
            try:
                # Read file bytes directly
                file_bytes = file.read()
                
                # Validate file
                is_valid, message = validate_file_bytes(file_bytes, file.filename)
                if not is_valid:
                    invalid_files.append({"filename": file.filename, "error": message})
                    continue
                
                # Generate unique filename for processing
                unique_filename = f"{uuid.uuid4()}_{file.filename}"
                files_data.append((file_bytes, unique_filename, file.filename))
                
            except Exception as e:
                invalid_files.append({"filename": file.filename, "error": f"File read error: {str(e)}"})
    
    if not files_data:
        error_msg = "No valid files to process"
        if invalid_files:
            error_msg += f". Invalid files: {len(invalid_files)}"
        return jsonify({"error": error_msg, "invalid_files": invalid_files}), 400
    
    log_memory_usage(f"AFTER processing {len(files_data)} files from bytes")
    
    # Generate batch ID and process
    batch_id = str(uuid.uuid4())
    
    try:
        start_time = time.time()
        
        # Process batch with bytes (no temp files!)
        batch_result = process_batch_concurrent(
            files_data, document_types, extraction_method, batch_id
        )
        
        processing
