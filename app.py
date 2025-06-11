from flask import Flask, request, jsonify, send_from_directory
import os
import tempfile
import uuid
import json
import sys
import time
import tracemalloc
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from werkzeug.utils import secure_filename
from flask_cors import CORS
from dataclasses import dataclass
from typing import List, Dict, Optional
import queue

# Start memory tracing
tracemalloc.start()

# Add parent directory to path to access the Mock SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Version check - Before trying to import the SDK
import pkg_resources

@dataclass
class BatchProgress:
    batch_id: str
    total_files: int
    processed_files: int
    failed_files: int
    status: str  # 'processing', 'completed', 'failed'
    start_time: float
    estimated_completion: Optional[float] = None
    current_file: Optional[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def progress_percentage(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100

    @property
    def processing_time(self) -> float:
        return time.time() - self.start_time

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
            "errors": self.errors[-5:]  # Only return last 5 errors
        }

def log_memory_usage(context=""):
    """Log current memory usage and top consumers"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"[MEMORY] {context}: {memory_mb:.2f} MB RSS, {memory_info.vms / 1024 / 1024:.2f} MB VMS")
        
        # Get top memory consumers (reduced to 2 to avoid log spam)
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print(f"[MEMORY] Top 2 memory consumers:")
        for index, stat in enumerate(top_stats[:2], 1):
            print(f"  {index}. {stat}")
            
    except Exception as e:
        print(f"[MEMORY] Error getting memory info: {e}")

def force_garbage_collection():
    """Force garbage collection and log results"""
    try:
        collected = gc.collect()
        print(f"[GC] Collected {collected} objects")
        gc.collect(1)  # Collect generation 1
        gc.collect(2)  # Collect generation 2
    except Exception as e:
        print(f"[GC] Error during garbage collection: {e}")

def check_agentic_doc_version():
    try:
        agentic_doc_version = pkg_resources.get_distribution("agentic-doc").version
        version_parts = agentic_doc_version.split('.')
        
        if int(version_parts[0]) == 0 and int(version_parts[1]) == 0 and int(version_parts[2]) < 13:
            print(f"WARNING: agentic-doc version {agentic_doc_version} is too old and will stop working after May 22!")
            print("Please upgrade to at least version 0.2.0 with: pip install --upgrade agentic-doc==0.2.1")
        elif int(version_parts[0]) == 0 and int(version_parts[1]) < 2:
            print(f"WARNING: agentic-doc version {agentic_doc_version} uses legacy chunk types.")
            print("It's recommended to upgrade to version 0.2.0 or later.")
        else:
            print(f"Using agentic-doc version {agentic_doc_version} with new chunk types.")
        
        return agentic_doc_version
    except pkg_resources.DistributionNotFound:
        print("agentic-doc not found. Will use mock SDK.")
        return None

# Run the version check
agentic_doc_version = check_agentic_doc_version()

try:
    from agentic_doc.parse import parse_documents
    print("Using real agentic_doc SDK")
except ImportError:
    try:
        from next_pdf_app.backend.mock_sdk import parse_documents
        print("Using mock SDK from next-pdf-app")
    except ImportError:
        from mock_sdk import parse_documents
        print("Using local mock SDK")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Enhanced Configuration for Batch Processing
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'doc_processor_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB for batch uploads

# Optimized SDK Configuration for Concurrent Processing
os.environ.setdefault('BATCH_SIZE', '5')        # Process 5 files concurrently
os.environ.setdefault('MAX_WORKERS', '3')       # 3 worker threads
os.environ.setdefault('MAX_RETRIES', '50')      # Reduced retries for faster failure
os.environ.setdefault('MAX_RETRY_WAIT_TIME', '30')  # Reduced wait time
os.environ.setdefault('RETRY_LOGGING_STYLE', 'log_msg')

# Global storage
processed_docs = {}
batch_progress = {}  # Track batch processing progress
processing_lock = threading.Lock()

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="DocProcessor")

def check_file_size(file_path, max_size_mb=25):
    """Check if file is too large for processing"""
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.2f}MB (max {max_size_mb}MB)"
        return True, "OK"
    except Exception as e:
        return False, f"Error checking file size: {e}"

def estimate_completion_time(progress: BatchProgress) -> Optional[float]:
    """Estimate completion time based on current progress"""
    if progress.processed_files == 0:
        return None
    
    avg_time_per_file = progress.processing_time / progress.processed_files
    remaining_files = progress.total_files - progress.processed_files
    estimated_remaining_time = remaining_files * avg_time_per_file
    
    return time.time() + estimated_remaining_time

def process_single_file(file_path: str, batch_id: str, include_marginalia: bool, 
                       include_metadata: bool, save_groundings: bool, grounding_dir: str) -> Dict:
    """Process a single file and return the result"""
    filename = os.path.basename(file_path)
    
    try:
        print(f"[BATCH {batch_id}] Processing file: {filename}")
        
        # Update progress
        with processing_lock:
            if batch_id in batch_progress:
                batch_progress[batch_id].current_file = filename
        
        # Process single file with SDK
        start_time = time.time()
        result = parse_documents(
            [file_path],
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata,
            grounding_save_dir=grounding_dir if save_groundings else None
        )
        processing_time = time.time() - start_time
        
        # Serialize result
        if result and len(result) > 0:
            serialized_doc = serialize_parsed_document(result[0])
            
            # Limit markdown size for memory efficiency
            if len(serialized_doc.get('markdown', '')) > 512 * 1024:  # 512KB limit per file
                serialized_doc['markdown'] = serialized_doc['markdown'][:512*1024] + "... [TRUNCATED]"
            
            serialized_doc['processing_time'] = processing_time
            serialized_doc['filename'] = filename
            
            print(f"[BATCH {batch_id}] ✅ Completed {filename} in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "filename": filename,
                "data": serialized_doc,
                "processing_time": processing_time
            }
        else:
            raise Exception("No result returned from SDK")
            
    except Exception as e:
        error_msg = f"Failed to process {filename}: {str(e)}"
        print(f"[BATCH {batch_id}] ❌ {error_msg}")
        
        return {
            "status": "error",
            "filename": filename,
            "error": error_msg,
            "processing_time": 0
        }

def process_batch_concurrent(saved_files: List[str], batch_id: str, include_marginalia: bool,
                           include_metadata: bool, save_groundings: bool) -> Dict:
    """Process multiple files concurrently"""
    
    print(f"[BATCH {batch_id}] Starting concurrent processing of {len(saved_files)} files")
    
    # Initialize progress tracking
    progress = BatchProgress(
        batch_id=batch_id,
        total_files=len(saved_files),
        processed_files=0,
        failed_files=0,
        status="processing",
        start_time=time.time()
    )
    
    with processing_lock:
        batch_progress[batch_id] = progress
    
    # Create grounding directory if needed
    grounding_dir = None
    if save_groundings:
        grounding_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"groundings_{batch_id}")
        os.makedirs(grounding_dir, exist_ok=True)
    
    # Process files concurrently
    results = []
    futures = []
    
    # Submit all files for processing
    for file_path in saved_files:
        future = executor.submit(
            process_single_file,
            file_path, batch_id, include_marginalia, 
            include_metadata, save_groundings, grounding_dir
        )
        futures.append(future)
    
    # Collect results as they complete
    for future in as_completed(futures):
        try:
            result = future.result()
            results.append(result)
            
            # Update progress
            with processing_lock:
                if batch_id in batch_progress:
                    if result["status"] == "success":
                        batch_progress[batch_id].processed_files += 1
                    else:
                        batch_progress[batch_id].failed_files += 1
                        batch_progress[batch_id].errors.append(result.get("error", "Unknown error"))
                    
                    # Update estimated completion time
                    batch_progress[batch_id].estimated_completion = estimate_completion_time(batch_progress[batch_id])
                    
                    print(f"[BATCH {batch_id}] Progress: {batch_progress[batch_id].processed_files + batch_progress[batch_id].failed_files}/{batch_progress[batch_id].total_files}")
            
        except Exception as e:
            print(f"[BATCH {batch_id}] Future execution error: {e}")
            results.append({
                "status": "error",
                "filename": "unknown",
                "error": f"Future execution error: {str(e)}",
                "processing_time": 0
            })
            
            with processing_lock:
                if batch_id in batch_progress:
                    batch_progress[batch_id].failed_files += 1
                    batch_progress[batch_id].errors.append(str(e))
    
    # Update final status
    with processing_lock:
        if batch_id in batch_progress:
            total_processed = batch_progress[batch_id].processed_files + batch_progress[batch_id].failed_files
            if total_processed == batch_progress[batch_id].total_files:
                if batch_progress[batch_id].failed_files == 0:
                    batch_progress[batch_id].status = "completed"
                else:
                    batch_progress[batch_id].status = "completed_with_errors"
            else:
                batch_progress[batch_id].status = "failed"
            
            batch_progress[batch_id].current_file = None
    
    # Separate successful and failed results
    successful_results = [r["data"] for r in results if r["status"] == "success"]
    failed_results = [{"filename": r["filename"], "error": r["error"]} for r in results if r["status"] == "error"]
    
    total_processing_time = time.time() - progress.start_time
    
    print(f"[BATCH {batch_id}] Completed: {len(successful_results)} successful, {len(failed_results)} failed in {total_processing_time:.2f}s")
    
    return {
        "successful_results": successful_results,
        "failed_results": failed_results,
        "total_files": len(saved_files),
        "successful_count": len(successful_results),
        "failed_count": len(failed_results),
        "total_processing_time": total_processing_time,
        "grounding_dir": grounding_dir
    }

# Include all the existing serialization functions (keeping them the same)
def serialize_parsed_document(parsed_doc):
    """Convert ParsedDocument object to JSON-serializable dictionary"""
    try:
        if hasattr(parsed_doc, '__dict__'):
            return {
                "markdown": getattr(parsed_doc, 'markdown', ''),
                "chunks": serialize_chunks(getattr(parsed_doc, 'chunks', [])),
                "errors": serialize_errors(getattr(parsed_doc, 'errors', [])),
                "processing_time": getattr(parsed_doc, 'processing_time', 0)
            }
        elif isinstance(parsed_doc, dict):
            return {
                "markdown": parsed_doc.get('markdown', ''),
                "chunks": serialize_chunks(parsed_doc.get('chunks', [])),
                "errors": serialize_errors(parsed_doc.get('errors', [])),
                "processing_time": parsed_doc.get('processing_time', 0)
            }
        else:
            return {
                "markdown": str(parsed_doc) if parsed_doc else '',
                "chunks": [],
                "errors": [{"message": f"Unknown document type: {type(parsed_doc)}", "page": 0}],
                "processing_time": 0
            }
    except Exception as e:
        print(f"Error serializing parsed document: {e}, doc type: {type(parsed_doc)}")
        return {
            "markdown": str(parsed_doc) if parsed_doc else '',
            "chunks": [],
            "errors": [{"message": f"Serialization error: {str(e)}", "page": 0}],
            "processing_time": 0
        }

def serialize_chunks(chunks):
    """Convert chunks to JSON-serializable format"""
    if not chunks:
        return []
    
    serialized_chunks = []
    for chunk in chunks:
        try:
            if hasattr(chunk, '__dict__'):
                serialized_chunk = {
                    "type": getattr(chunk, 'type', 'unknown'),
                    "content": getattr(chunk, 'content', ''),
                    "page": getattr(chunk, 'page', 0),
                    "chunk_id": getattr(chunk, 'chunk_id', str(uuid.uuid4())),
                    "grounding": serialize_grounding(getattr(chunk, 'grounding', [])),
                    "metadata": serialize_metadata(getattr(chunk, 'metadata', {}))
                }
            elif isinstance(chunk, dict):
                serialized_chunk = {
                    "type": chunk.get('type', 'unknown'),
                    "content": chunk.get('content', ''),
                    "page": chunk.get('page', 0),
                    "chunk_id": chunk.get('chunk_id', str(uuid.uuid4())),
                    "grounding": serialize_grounding(chunk.get('grounding', [])),
                    "metadata": serialize_metadata(chunk.get('metadata', {}))
                }
            else:
                serialized_chunk = {
                    "type": "unknown",
                    "content": str(chunk),
                    "page": 0,
                    "chunk_id": str(uuid.uuid4()),
                    "grounding": [],
                    "metadata": {}
                }
            
            serialized_chunks.append(serialized_chunk)
            
        except Exception as e:
            print(f"Error serializing chunk: {e}, chunk type: {type(chunk)}")
            serialized_chunks.append({
                "type": "error",
                "content": f"Error serializing chunk: {str(e)}",
                "page": 0,
                "chunk_id": str(uuid.uuid4()),
                "grounding": [],
                "metadata": {"serialization_error": str(e)}
            })
    
    return serialized_chunks

def serialize_grounding(grounding):
    """Convert grounding objects to JSON-serializable format"""
    if not grounding:
        return []
    
    serialized_grounding = []
    for ground in grounding:
        try:
            if hasattr(ground, '__dict__'):
                serialized_ground = {}
                
                if hasattr(ground, 'box'):
                    box = getattr(ground, 'box')
                    if isinstance(box, (list, tuple)):
                        serialized_ground["box"] = list(box)
                    else:
                        serialized_ground["box"] = serialize_box_object(box)
                else:
                    serialized_ground["box"] = [0, 0, 0, 0]
                
                serialized_ground["page"] = getattr(ground, 'page', 0)
                serialized_ground["confidence"] = getattr(ground, 'confidence', 0.0)
                serialized_ground["image_path"] = getattr(ground, 'image_path', None)
                
            elif isinstance(ground, dict):
                serialized_ground = {
                    "box": ground.get('box', [0, 0, 0, 0]),
                    "page": ground.get('page', 0),
                    "confidence": ground.get('confidence', 0.0),
                    "image_path": ground.get('image_path', None)
                }
            else:
                serialized_ground = {
                    "box": [0, 0, 0, 0],
                    "page": 0,
                    "confidence": 0.0,
                    "image_path": None,
                    "raw": str(ground)
                }
            
            serialized_grounding.append(serialized_ground)
            
        except Exception as e:
            print(f"Error serializing grounding item: {e}, type: {type(ground)}")
            serialized_grounding.append({
                "box": [0, 0, 0, 0],
                "page": 0,
                "confidence": 0.0,
                "image_path": None,
                "error": str(e)
            })
    
    return serialized_grounding

def serialize_box_object(box):
    """Convert box objects to JSON-serializable format"""
    try:
        if hasattr(box, '__dict__'):
            if hasattr(box, 'x') and hasattr(box, 'y') and hasattr(box, 'width') and hasattr(box, 'height'):
                x1 = getattr(box, 'x', 0)
                y1 = getattr(box, 'y', 0)
                width = getattr(box, 'width', 0)
                height = getattr(box, 'height', 0)
                return [x1, y1, x1 + width, y1 + height]
            elif hasattr(box, 'x1') and hasattr(box, 'y1') and hasattr(box, 'x2') and hasattr(box, 'y2'):
                return [getattr(box, 'x1', 0), getattr(box, 'y1', 0), getattr(box, 'x2', 0), getattr(box, 'y2', 0)]
            elif hasattr(box, 'left') and hasattr(box, 'top') and hasattr(box, 'right') and hasattr(box, 'bottom'):
                return [getattr(box, 'left', 0), getattr(box, 'top', 0), getattr(box, 'right', 0), getattr(box, 'bottom', 0)]
            else:
                if hasattr(box, '__iter__') and not isinstance(box, str):
                    return list(box)
                else:
                    attrs = [getattr(box, attr) for attr in dir(box) if not attr.startswith('_') and isinstance(getattr(box, attr), (int, float))]
                    return attrs[:4] if len(attrs) >= 4 else [0, 0, 0, 0]
        elif isinstance(box, (list, tuple)):
            return list(box)
        else:
            return [0, 0, 0, 0]
    except Exception as e:
        print(f"Error serializing box object: {e}, type: {type(box)}")
        return [0, 0, 0, 0]

def serialize_errors(errors):
    """Convert error objects to JSON-serializable format"""
    serialized_errors = []
    for error in errors:
        try:
            serialized_error = {
                "message": getattr(error, 'message', str(error)),
                "page": getattr(error, 'page', 0),
                "error_code": getattr(error, 'error_code', 'unknown')
            }
            serialized_errors.append(serialized_error)
        except Exception as e:
            print(f"Error serializing error: {e}")
            serialized_errors.append({
                "message": str(error),
                "page": 0,
                "error_code": "serialization_error"
            })
    return serialized_errors

def serialize_metadata(metadata):
    """Convert metadata to JSON-serializable format"""
    if metadata is None:
        return {}
    
    try:
        if isinstance(metadata, dict):
            return {str(k): serialize_value(v) for k, v in metadata.items()}
        else:
            return {"raw": str(metadata)}
    except Exception as e:
        print(f"Error serializing metadata: {e}")
        return {"error": f"Metadata serialization error: {str(e)}"}

def serialize_value(value):
    """Convert individual values to JSON-serializable format"""
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    else:
        return str(value)

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with batch processing info"""
    log_memory_usage("Health check")
    
    with processing_lock:
        active_batches = len(batch_progress)
        processing_batches = len([p for p in batch_progress.values() if p.status == "processing"])
    
    return jsonify({
        "status": "healthy", 
        "service": "document-processor",
        "agentic_doc_version": agentic_doc_version,
        "active_batches": len(processed_docs),
        "processing_batches": processing_batches,
        "total_tracked_batches": active_batches,
        "concurrent_processing": True,
        "max_workers": 3,
        "batch_size": 5
    })

@app.route('/batch-status/<batch_id>', methods=['GET'])
def get_batch_status(batch_id):
    """Get real-time batch processing status"""
    with processing_lock:
        if batch_id not in batch_progress:
            return jsonify({"error": "Batch ID not found"}), 404
        
        progress = batch_progress[batch_id]
        return jsonify(progress.to_dict())

@app.route('/process-documents', methods=['POST'])
def process_documents():
    """Enhanced process_documents with concurrent batch processing"""
    log_memory_usage("START of batch process_documents")
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No files selected"}), 400
    
    # Get optional parameters
    include_marginalia = request.form.get('include_marginalia', 'true').lower() == 'true'
    include_metadata = request.form.get('include_metadata', 'true').lower() == 'true'
    save_groundings = request.form.get('save_groundings', 'false').lower() == 'true'
    
    print(f"[BATCH] Processing {len(files)} files concurrently")
    log_memory_usage("AFTER parameter extraction")
    
    # Save uploaded files
    saved_files = []
    invalid_files = []
    
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
            
            # Save file in chunks
            with open(temp_path, 'wb') as f:
                while True:
                    chunk = file.stream.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Check file size
            is_valid, message = check_file_size(temp_path)
            if not is_valid:
                try:
                    os.remove(temp_path)
                except:
                    pass
                invalid_files.append({"filename": filename, "error": message})
                continue
            
            saved_files.append(temp_path)
    
    if not saved_files:
        error_msg = "No valid files to process"
        if invalid_files:
            error_msg += f". Invalid files: {len(invalid_files)}"
        return jsonify({"error": error_msg, "invalid_files": invalid_files}), 400
    
    log_memory_usage(f"AFTER saving {len(saved_files)} files")
    
    # Generate batch ID
    batch_id = str(uuid.uuid4())
    
    try:
        # Process files concurrently
        force_garbage_collection()
        log_memory_usage("BEFORE concurrent processing")
        
        start_time = time.time()
        
        # Process batch concurrently
        batch_result = process_batch_concurrent(
            saved_files, batch_id, include_marginalia, 
            include_metadata, save_groundings
        )
        
        processing_time = time.time() - start_time
        log_memory_usage("AFTER concurrent processing")
        
        # Store results
        processed_docs[batch_id] = {
            "result": batch_result["successful_results"],
            "files": saved_files,
            "processed_at": time.time(),
            "groundings_dir": batch_result.get("grounding_dir"),
            "failed_results": batch_result["failed_results"],
            "processing_stats": {
                "total_files": batch_result["total_files"],
                "successful_count": batch_result["successful_count"],
                "failed_count": batch_result["failed_count"],
                "total_processing_time": batch_result["total_processing_time"]
            }
        }
        
        force_garbage_collection()
        log_memory_usage("END of batch process_documents")
        
        # Response
        response = {
            "batch_id": batch_id,
            "document_count": len(saved_files),
            "successful_count": batch_result["successful_count"],
            "failed_count": batch_result["failed_count"],
            "processing_time_seconds": processing_time,
            "status": "success",
            "grounding_images_saved": save_groundings,
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
        print(f"Error in batch processing: {e}")
        log_memory_usage("ERROR state")
        
        # Emergency cleanup
        for file_path in saved_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        
        force_garbage_collection()
        
        return jsonify({"error": str(e)}), 500

@app.route('/get-document-data/<batch_id>', methods=['GET'])
def get_document_data(batch_id):
    """Retrieve processed document data by batch ID with stats"""
    log_memory_usage(f"GET document data for batch {batch_id}")
    
    if batch_id not in processed_docs:
        return jsonify({"error": "Batch ID not found"}), 404
    
    try:
        batch_data = processed_docs[batch_id]
        
        response = {
            "batch_id": batch_id,
            "result": batch_data["result"],
            "files": [os.path.basename(f) for f in batch_data["files"]],
            "processed_at": batch_data["processed_at"],
            "groundings_dir": batch_data.get("groundings_dir"),
            "failed_results": batch_data.get("failed_results", []),
            "processing_stats": batch_data.get("processing_stats", {})
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error retrieving document data: {e}")
        return jsonify({"error": f"Error retrieving document data: {str(e)}"}), 500

@app.route('/ask-question', methods=['POST'])
def ask_question():
    """Answer a question about processed documents"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    if "batch_id" not in data or "question" not in data:
        return jsonify({"error": "Missing required fields: batch_id and question"}), 400
    
    batch_id = data["batch_id"]
    question = data["question"]
    
    if batch_id not in processed_docs:
        return jsonify({"error": "Batch ID not found"}), 404
    
    try:
        evidence = processed_docs[batch_id]["result"]
        
        response = {
            "answer": f"This is a mock answer to the question: {question}",
            "reasoning": "This is placeholder reasoning. Real integration would use OpenAI.",
            "evidence": [
                {"text": "Sample evidence text", "score": 0.95}
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/cleanup/<batch_id>', methods=['DELETE'])
def cleanup_batch(batch_id):
    """Enhanced cleanup for batch processing"""
    if batch_id not in processed_docs:
        return jsonify({"error": "Batch ID not found"}), 404
    
    log_memory_usage(f"BEFORE cleanup batch {batch_id}")
    
    try:
        batch_data = processed_docs[batch_id]
        
        # Delete temporary files
        for file_path in batch_data["files"]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[CLEANUP] Removed file: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"[CLEANUP] Error removing file {file_path}: {e}")
        
        # Clean up grounding directory
        grounding_dir = batch_data.get("groundings_dir")
        if grounding_dir and os.path.exists(grounding_dir):
            try:
                import shutil
                shutil.rmtree(grounding_dir)
                print(f"[CLEANUP] Removed grounding directory: {grounding_dir}")
            except Exception as e:
                print(f"[CLEANUP] Error removing grounding directory: {e}")
        
        # Remove from processed docs
        del processed_docs[batch_id]
        
        # Remove from progress tracking
        with processing_lock:
            if batch_id in batch_progress:
                del batch_progress[batch_id]
        
        # Force garbage collection
        force_garbage_collection()
        
        log_memory_usage(f"AFTER cleanup batch {batch_id}")
        
        return jsonify({
            "status": "success", 
            "message": "Batch cleaned up successfully",
            "files_cleaned": len(batch_data["files"])
        })
        
    except Exception as e:
        print(f"[CLEANUP] Error during cleanup: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/list-batches', methods=['GET'])
def list_batches():
    """List all active batches with their status"""
    try:
        batches = []
        
        # Add completed batches from processed_docs
        for batch_id, batch_data in processed_docs.items():
            stats = batch_data.get("processing_stats", {})
            batches.append({
                "batch_id": batch_id,
                "status": "completed",
                "total_files": stats.get("total_files", len(batch_data.get("files", []))),
                "successful_count": stats.get("successful_count", len(batch_data.get("result", []))),
                "failed_count": stats.get("failed_count", len(batch_data.get("failed_results", []))),
                "processed_at": batch_data.get("processed_at"),
                "processing_time": stats.get("total_processing_time", 0)
            })
        
        # Add in-progress batches from batch_progress
        with processing_lock:
            for batch_id, progress in batch_progress.items():
                if batch_id not in processed_docs:  # Don't duplicate completed batches
                    batches.append({
                        "batch_id": batch_id,
                        "status": progress.status,
                        "total_files": progress.total_files,
                        "processed_files": progress.processed_files,
                        "failed_files": progress.failed_files,
                        "progress_percentage": progress.progress_percentage,
                        "processing_time": progress.processing_time,
                        "current_file": progress.current_file
                    })
        
        return jsonify({
            "batches": batches,
            "total_batches": len(batches)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/memory-status', methods=['GET'])
def memory_status():
    """Enhanced memory status with batch processing info"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        with processing_lock:
            active_batches = len(batch_progress)
            processing_batches = len([p for p in batch_progress.values() if p.status == "processing"])
        
        return jsonify({
            "memory_mb": memory_info.rss / 1024 / 1024,
            "virtual_memory_mb": memory_info.vms / 1024 / 1024,
            "completed_batches": len(processed_docs),
            "active_batches": active_batches,
            "processing_batches": processing_batches,
            "thread_pool_active": True,
            "max_workers": 4,
            "top_memory_consumers": [str(stat) for stat in top_stats[:5]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch-stats', methods=['GET'])
def batch_stats():
    """Get comprehensive batch processing statistics"""
    try:
        # Calculate statistics from processed batches
        total_files_processed = 0
        total_successful = 0
        total_failed = 0
        total_processing_time = 0
        batch_count = len(processed_docs)
        
        for batch_data in processed_docs.values():
            stats = batch_data.get("processing_stats", {})
            total_files_processed += stats.get("total_files", 0)
            total_successful += stats.get("successful_count", 0)
            total_failed += stats.get("failed_count", 0)
            total_processing_time += stats.get("total_processing_time", 0)
        
        # Calculate averages
        avg_files_per_batch = total_files_processed / batch_count if batch_count > 0 else 0
        avg_processing_time_per_batch = total_processing_time / batch_count if batch_count > 0 else 0
        avg_processing_time_per_file = total_processing_time / total_files_processed if total_files_processed > 0 else 0
        success_rate = (total_successful / total_files_processed * 100) if total_files_processed > 0 else 0
        
        with processing_lock:
            currently_processing = len([p for p in batch_progress.values() if p.status == "processing"])
        
        return jsonify({
            "summary": {
                "total_batches_processed": batch_count,
                "total_files_processed": total_files_processed,
                "total_successful_files": total_successful,
                "total_failed_files": total_failed,
                "overall_success_rate": round(success_rate, 2),
                "currently_processing_batches": currently_processing
            },
            "averages": {
                "files_per_batch": round(avg_files_per_batch, 2),
                "processing_time_per_batch_seconds": round(avg_processing_time_per_batch, 2),
                "processing_time_per_file_seconds": round(avg_processing_time_per_file, 2)
            },
            "performance": {
                "concurrent_processing": True,
                "max_workers": 4,
                "batch_size": 5,
                "estimated_throughput_files_per_hour": round(3600 / avg_processing_time_per_file, 0) if avg_processing_time_per_file > 0 else 0
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    log_memory_usage("Application startup")
    print("[STARTUP] Enhanced batch processing microservice starting...")
    print(f"[STARTUP] Concurrent processing enabled: max_workers=4, batch_size=5")
    print(f"[STARTUP] File size limit: 25MB per file, 100MB total upload")
    
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug for production
