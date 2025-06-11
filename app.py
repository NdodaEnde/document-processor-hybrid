from flask import Flask, request, jsonify, send_from_directory
import os
import tempfile
import uuid
import json
import sys
import time
import threading
from functools import wraps
from werkzeug.utils import secure_filename
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
import psutil

# Add parent directory to path to access the Mock SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Version check - Before trying to import the SDK
import pkg_resources

def check_agentic_doc_version():
    try:
        # Check the installed version of agentic-doc
        agentic_doc_version = pkg_resources.get_distribution("agentic-doc").version
        version_parts = agentic_doc_version.split('.')
        
        # Check if version is too old (before 0.0.13)
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
    # Try to import the real SDK if available
    from agentic_doc.parse import parse_documents
    print("Using real agentic_doc SDK")
except ImportError:
    # If real SDK not available, use mock SDK
    try:
        from next_pdf_app.backend.mock_sdk import parse_documents
        print("Using mock SDK from next-pdf-app")
    except ImportError:
        # Fallback to local mock implementation
        from mock_sdk import parse_documents
        print("Using local mock SDK")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ENHANCED CONFIGURATION
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'doc_processor_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Enhanced resource limits
MAX_BATCH_SIZE = 50  # Maximum files per batch
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB per file
MAX_TOTAL_BATCH_SIZE = 100 * 1024 * 1024  # 100MB total per batch
MEMORY_LIMIT = 85  # Stop processing if memory > 85%
PROCESSING_TIMEOUT = 180  # 3 minutes max per batch
CHUNK_SIZE = 10  # Process 10 files at a time
CLEANUP_INTERVAL = 300  # Clean up every 5 minutes

# SDK Configuration options loaded from environment variables
os.environ.setdefault('BATCH_SIZE', '4')
os.environ.setdefault('MAX_WORKERS', '3')  # Reduced for stability
os.environ.setdefault('MAX_RETRIES', '50')  # Reduced retries
os.environ.setdefault('MAX_RETRY_WAIT_TIME', '30')  # Reduced wait time
os.environ.setdefault('RETRY_LOGGING_STYLE', 'log_msg')

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=3)

# Storage for processed documents and processing status
processed_docs = {}
processing_status = {}
storage_lock = threading.Lock()

def validate_batch_request(files):
    """Validate batch processing request"""
    if len(files) > MAX_BATCH_SIZE:
        return False, f"Too many files. Maximum: {MAX_BATCH_SIZE}"
    
    total_size = 0
    for file in files:
        if file.filename:
            # Check file size
            file.seek(0, 2)  # Seek to end
            size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if size > MAX_FILE_SIZE:
                return False, f"File {file.filename} too large. Maximum: {MAX_FILE_SIZE/1024/1024:.1f}MB"
            
            if size == 0:
                return False, f"File {file.filename} is empty"
            
            total_size += size
    
    if total_size > MAX_TOTAL_BATCH_SIZE:
        return False, f"Batch too large. Total maximum: {MAX_TOTAL_BATCH_SIZE/1024/1024:.1f}MB"
    
    # Memory check
    try:
        if psutil.virtual_memory().percent > MEMORY_LIMIT:
            return False, "Server overloaded. Please try again later."
    except:
        pass  # psutil might not be available
    
    return True, "Valid"

def process_batch_async(batch_id, saved_files, include_marginalia, include_metadata, grounding_dir):
    """Process batch in background thread with chunking"""
    try:
        print(f"Starting async processing for batch {batch_id} with {len(saved_files)} files")
        
        with storage_lock:
            processing_status[batch_id] = {
                "status": "processing",
                "progress": 0,
                "total_files": len(saved_files),
                "completed_files": 0,
                "start_time": time.time(),
                "current_chunk": 0,
                "total_chunks": (len(saved_files) + CHUNK_SIZE - 1) // CHUNK_SIZE
            }
        
        # Process in chunks to prevent memory issues
        all_results = []
        chunk_number = 0
        
        for i in range(0, len(saved_files), CHUNK_SIZE):
            chunk_files = saved_files[i:i + CHUNK_SIZE]
            chunk_number += 1
            
            print(f"Batch {batch_id}: Processing chunk {chunk_number} with {len(chunk_files)} files")
            
            try:
                # Update status
                with storage_lock:
                    if batch_id in processing_status:
                        processing_status[batch_id]["current_chunk"] = chunk_number
                        processing_status[batch_id]["status"] = f"processing_chunk_{chunk_number}"
                
                # Process chunk with timeout protection
                chunk_start_time = time.time()
                chunk_result = parse_documents(
                    chunk_files,
                    include_marginalia=include_marginalia,
                    include_metadata_in_markdown=include_metadata,
                    grounding_save_dir=grounding_dir
                )
                chunk_processing_time = time.time() - chunk_start_time
                
                print(f"Batch {batch_id}: Chunk {chunk_number} processed in {chunk_processing_time:.2f}s")
                
                # Serialize chunk results immediately to free memory
                for doc in chunk_result:
                    try:
                        serialized_doc = serialize_parsed_document(doc)
                        all_results.append(serialized_doc)
                    except Exception as serialize_error:
                        print(f"Error serializing document in batch {batch_id}: {serialize_error}")
                        all_results.append({
                            "markdown": f"Error serializing document: {serialize_error}",
                            "chunks": [],
                            "errors": [{"message": str(serialize_error), "page": 0}],
                            "processing_time": 0
                        })
                
                # Update progress
                completed = i + len(chunk_files)
                progress = int((completed / len(saved_files)) * 100)
                
                with storage_lock:
                    if batch_id in processing_status:
                        processing_status[batch_id]["completed_files"] = completed
                        processing_status[batch_id]["progress"] = progress
                
                print(f"Batch {batch_id}: {completed}/{len(saved_files)} files processed ({progress}%)")
                
                # Memory and cleanup between chunks
                try:
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 90:
                        print(f"High memory usage ({memory_percent}%), pausing between chunks...")
                        time.sleep(2)
                except:
                    pass
                
                # Clean up chunk files immediately after processing
                for file_path in chunk_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as cleanup_error:
                        print(f"Error cleaning up chunk file {file_path}: {cleanup_error}")
                        
            except Exception as chunk_error:
                print(f"Error processing chunk {chunk_number} in batch {batch_id}: {chunk_error}")
                
                # Add error documents for failed chunk
                for _ in chunk_files:
                    all_results.append({
                        "markdown": f"Error processing chunk: {chunk_error}",
                        "chunks": [],
                        "errors": [{"message": str(chunk_error), "page": 0}],
                        "processing_time": 0
                    })
                
                # Continue with next chunk instead of failing entire batch
                continue
        
        # Store final results
        total_processing_time = time.time() - processing_status[batch_id]["start_time"]
        
        with storage_lock:
            processed_docs[batch_id] = {
                "result": all_results,
                "files": [],  # Files already cleaned up
                "processed_at": time.time(),
                "groundings_dir": grounding_dir,
                "processing_time": total_processing_time,
                "file_count": len(saved_files),
                "success_count": len([r for r in all_results if not r.get("errors")])
            }
            
            processing_status[batch_id]["status"] = "completed"
            processing_status[batch_id]["progress"] = 100
            processing_status[batch_id]["completed_at"] = time.time()
        
        print(f"Batch {batch_id} completed successfully in {total_processing_time:.2f}s")
        
    except Exception as e:
        print(f"Batch {batch_id} failed with error: {e}")
        
        with storage_lock:
            if batch_id in processing_status:
                processing_status[batch_id]["status"] = "failed"
                processing_status[batch_id]["error"] = str(e)
                processing_status[batch_id]["completed_at"] = time.time()
        
        # Clean up files on failure
        for file_path in saved_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass

def serialize_parsed_document(parsed_doc):
    """
    Convert ParsedDocument object to JSON-serializable dictionary
    """
    try:
        if hasattr(parsed_doc, '__dict__'):
            return {
                "markdown": getattr(parsed_doc, 'markdown', ''),
                "chunks": serialize_chunks(getattr(parsed_doc, 'chunks', [])),
                "errors": serialize_errors(getattr(parsed_doc, 'errors', [])),
                "processing_time": getattr(parsed_doc, 'processing_time', 0)
            }
        elif isinstance(parsed_doc, dict):
            # If it's already a dictionary
            return {
                "markdown": parsed_doc.get('markdown', ''),
                "chunks": serialize_chunks(parsed_doc.get('chunks', [])),
                "errors": serialize_errors(parsed_doc.get('errors', [])),
                "processing_time": parsed_doc.get('processing_time', 0)
            }
        else:
            # Fallback for unknown document types
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
    """
    Convert chunks to JSON-serializable format
    """
    if not chunks:
        return []
    
    serialized_chunks = []
    # Limit chunks per document to prevent memory issues
    for chunk in chunks[:50]:  # Maximum 50 chunks per document
        try:
            # Handle different chunk object types
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
                # If chunk is already a dictionary
                serialized_chunk = {
                    "type": chunk.get('type', 'unknown'),
                    "content": chunk.get('content', ''),
                    "page": chunk.get('page', 0),
                    "chunk_id": chunk.get('chunk_id', str(uuid.uuid4())),
                    "grounding": serialize_grounding(chunk.get('grounding', [])),
                    "metadata": serialize_metadata(chunk.get('metadata', {}))
                }
            else:
                # Fallback for unknown chunk types
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
            # Add a fallback chunk with error info
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
    """
    Convert grounding objects to JSON-serializable format
    """
    if not grounding:
        return []
    
    serialized_grounding = []
    for ground in grounding[:10]:  # Limit grounding objects
        try:
            # Handle different types of grounding objects
            if hasattr(ground, '__dict__'):
                # If it's an object with attributes, extract them
                serialized_ground = {}
                
                # Common attributes to look for
                if hasattr(ground, 'box'):
                    box = getattr(ground, 'box')
                    if isinstance(box, (list, tuple)):
                        serialized_ground["box"] = list(box)
                    else:
                        # Handle ChunkGroundingBox or similar objects
                        serialized_ground["box"] = serialize_box_object(box)
                else:
                    serialized_ground["box"] = [0, 0, 0, 0]
                
                serialized_ground["page"] = getattr(ground, 'page', 0)
                serialized_ground["confidence"] = getattr(ground, 'confidence', 0.0)
                serialized_ground["image_path"] = getattr(ground, 'image_path', None)
                
            elif isinstance(ground, dict):
                # If it's already a dictionary
                serialized_ground = {
                    "box": ground.get('box', [0, 0, 0, 0]),
                    "page": ground.get('page', 0),
                    "confidence": ground.get('confidence', 0.0),
                    "image_path": ground.get('image_path', None)
                }
            else:
                # Fallback for unknown types
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
    """
    Convert box objects (like ChunkGroundingBox) to JSON-serializable format
    """
    try:
        if hasattr(box, '__dict__'):
            # Try to extract coordinates from the object
            if hasattr(box, 'x') and hasattr(box, 'y') and hasattr(box, 'width') and hasattr(box, 'height'):
                # Convert from x,y,width,height to x1,y1,x2,y2 format
                x1 = getattr(box, 'x', 0)
                y1 = getattr(box, 'y', 0)
                width = getattr(box, 'width', 0)
                height = getattr(box, 'height', 0)
                return [x1, y1, x1 + width, y1 + height]
            elif hasattr(box, 'x1') and hasattr(box, 'y1') and hasattr(box, 'x2') and hasattr(box, 'y2'):
                # Already in x1,y1,x2,y2 format
                return [getattr(box, 'x1', 0), getattr(box, 'y1', 0), getattr(box, 'x2', 0), getattr(box, 'y2', 0)]
            elif hasattr(box, 'left') and hasattr(box, 'top') and hasattr(box, 'right') and hasattr(box, 'bottom'):
                # left,top,right,bottom format
                return [getattr(box, 'left', 0), getattr(box, 'top', 0), getattr(box, 'right', 0), getattr(box, 'bottom', 0)]
            else:
                # Try to convert the entire object to a list/array
                if hasattr(box, '__iter__') and not isinstance(box, str):
                    return list(box)
                else:
                    # Last resort - extract all numeric attributes
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
    """
    Convert error objects to JSON-serializable format
    """
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
    """
    Convert metadata to JSON-serializable format
    """
    if metadata is None:
        return {}
    
    try:
        # Handle different types of metadata
        if isinstance(metadata, dict):
            return {str(k): serialize_value(v) for k, v in metadata.items()}
        else:
            return {"raw": str(metadata)}
    except Exception as e:
        print(f"Error serializing metadata: {e}")
        return {"error": f"Metadata serialization error: {str(e)}"}

def serialize_value(value):
    """
    Convert individual values to JSON-serializable format
    """
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

def cleanup_old_results():
    """Enhanced cleanup with memory pressure detection"""
    current_time = time.time()
    cutoff_time = current_time - 1800  # 30 minutes
    
    with storage_lock:
        # Clean processed docs
        keys_to_remove = []
        for batch_id, data in processed_docs.items():
            if data.get('processed_at', 0) < cutoff_time:
                keys_to_remove.append(batch_id)
        
        for key in keys_to_remove:
            try:
                batch_data = processed_docs[key]
                # Clean up files
                for file_path in batch_data.get('files', []):
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                # Clean up grounding directory
                grounding_dir = batch_data.get('groundings_dir')
                if grounding_dir and os.path.exists(grounding_dir):
                    import shutil
                    shutil.rmtree(grounding_dir)
                
                del processed_docs[key]
                print(f"Cleaned up old batch: {key}")
            except Exception as e:
                print(f"Error cleaning up batch {key}: {e}")
        
        # Clean processing status
        status_keys_to_remove = []
        for batch_id, status in processing_status.items():
            if status.get('start_time', 0) < cutoff_time:
                status_keys_to_remove.append(batch_id)
        
        for key in status_keys_to_remove:
            del processing_status[key]
        
        if keys_to_remove or status_keys_to_remove:
            print(f"Cleanup completed: removed {len(keys_to_remove)} batches, {len(status_keys_to_remove)} status entries")

def cleanup_loop():
    """Background cleanup thread"""
    while True:
        time.sleep(CLEANUP_INTERVAL)
        try:
            cleanup_old_results()
        except Exception as e:
            print(f"Error in cleanup loop: {e}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
cleanup_thread.start()

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with resource info"""
    try:
        memory_info = psutil.virtual_memory()
        memory_usage = f"{memory_info.percent}%"
    except:
        memory_usage = "Unknown"
    
    with storage_lock:
        active_batches = len(processing_status)
        completed_batches = len(processed_docs)
    
    return jsonify({
        "status": "healthy", 
        "service": "document-processor-enhanced",
        "agentic_doc_version": agentic_doc_version,
        "memory_usage": memory_usage,
        "active_batches": active_batches,
        "completed_batches": completed_batches,
        "limits": {
            "max_batch_size": MAX_BATCH_SIZE,
            "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
            "max_total_batch_size_mb": MAX_TOTAL_BATCH_SIZE / 1024 / 1024,
            "memory_limit": f"{MEMORY_LIMIT}%",
            "processing_timeout": PROCESSING_TIMEOUT,
            "chunk_size": CHUNK_SIZE
        }
    })

@app.route('/process-documents', methods=['POST'])
def process_documents():
    """
    Enhanced document processing with async support for large batches
    """
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No files selected"}), 400
    
    # Validate batch
    is_valid, message = validate_batch_request(files)
    if not is_valid:
        return jsonify({"error": message}), 400
    
    # Get optional parameters from request
    include_marginalia = request.form.get('include_marginalia', 'true').lower() == 'true'
    include_metadata = request.form.get('include_metadata', 'true').lower() == 'true'
    save_groundings = request.form.get('save_groundings', 'false').lower() == 'true'
    force_async = request.form.get('async', 'false').lower() == 'true'
    
    print(f"Processing batch: {len(files)} files, async={force_async}")
    
    # Save uploaded files to temp directory
    saved_files = []
    batch_id = str(uuid.uuid4())
    total_size = 0
    
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{batch_id}_{filename}")
            file.save(temp_path)
            saved_files.append(temp_path)
            total_size += os.path.getsize(temp_path)
    
    if not saved_files:
        return jsonify({"error": "No valid files uploaded"}), 400
    
    try:
        # Create grounding directory if needed
        grounding_dir = None
        if save_groundings:
            grounding_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"groundings_{batch_id}")
            os.makedirs(grounding_dir, exist_ok=True)
        
        # Decision: Use async processing for large batches or if forced
        use_async = force_async or len(files) > 10 or total_size > 20 * 1024 * 1024  # 20MB
        
        if use_async:
            # Submit to background processing
            executor.submit(process_batch_async, batch_id, saved_files, include_marginalia, include_metadata, grounding_dir)
            
            return jsonify({
                "batch_id": batch_id,
                "status": "processing",
                "document_count": len(saved_files),
                "total_size_mb": total_size / 1024 / 1024,
                "async": True,
                "estimated_time_minutes": len(saved_files) * 0.5,  # 30 seconds per file estimate
                "message": "Batch submitted for background processing"
            })
        
        else:
            # Synchronous processing for small batches (original behavior)
            start_time = time.time()
            result = parse_documents(
                saved_files,
                include_marginalia=include_marginalia,
                include_metadata_in_markdown=include_metadata,
                grounding_save_dir=grounding_dir if save_groundings else None
            )
            processing_time = time.time() - start_time
            
            # Serialize the result before storing
            serialized_result = []
            for doc in result:
                serialized_result.append(serialize_parsed_document(doc))
            
            with storage_lock:
                processed_docs[batch_id] = {
                    "result": serialized_result,
                    "files": saved_files,
                    "processed_at": time.time(),
                    "groundings_dir": grounding_dir if save_groundings else None,
                    "processing_time": processing_time,
                    "file_count": len(saved_files),
                    "success_count": len(serialized_result)
                }
            
            return jsonify({
                "batch_id": batch_id,
                "document_count": len(saved_files),
                "processing_time_seconds": processing_time,
                "total_size_mb": total_size / 1024 / 1024,
                "status": "completed",
                "async": False,
                "grounding_images_saved": save_groundings
            })
    
    except Exception as e:
        print(f"Error processing documents: {e}")
        
        # Clean up files on error
        for file_path in saved_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as cleanup_error:
                print(f"Error cleaning up file {file_path}: {cleanup_error}")
        
        return jsonify({"error": str(e)}), 500

@app.route('/batch-status/<batch_id>', methods=['GET'])
def get_batch_status(batch_id):
    """Get processing status for async batches"""
    with storage_lock:
        if batch_id in processing_status:
            status = processing_status[batch_id].copy()
            if "start_time" in status:
                status["elapsed_time"] = time.time() - status["start_time"]
            return jsonify(status)
        elif batch_id in processed_docs:
            batch_data = processed_docs[batch_id]
            return jsonify({
                "status": "completed", 
                "progress": 100,
                "file_count": batch_data.get("file_count", 0),
                "success_count": batch_data.get("success_count", 0),
                "processing_time": batch_data.get("processing_time", 0)
            })
        else:
            return jsonify({"error": "Batch not found"}), 404

@app.route('/get-document-data/<batch_id>', methods=['GET'])
def get_document_data(batch_id):
    """
    Retrieve processed document data by batch ID (legacy endpoint)
    """
    with storage_lock:
        if batch_id not in processed_docs:
            # Check if it's still processing
            if batch_id in processing_status:
                status = processing_status[batch_id]
                return jsonify({
                    "error": "Batch still processing",
                    "status": status.get("status", "processing"),
                    "progress": status.get("progress", 0)
                }), 202
            return jsonify({"error": "Batch ID not found"}), 404
    
    try:
        batch_data = processed_docs[batch_id]
        return jsonify({
            "batch_id": batch_id,
            "result": batch_data["result"],
            "files": [os.path.basename(f) for f in batch_data.get("files", [])],
            "processed_at": batch_data["processed_at"],
            "processing_time": batch_data.get("processing_time", 0),
            "file_count": batch_data.get("file_count", 0),
            "success_count": batch_data.get("success_count", 0),
            "groundings_dir": batch_data.get("groundings_dir")
        })
    except Exception as e:
        print(f"Error retrieving document data: {e}")
        return jsonify({"error": f"Error retrieving document data: {str(e)}"}), 500

@app.route('/batch-results/<batch_id>', methods=['GET'])
def get_batch_results(batch_id):
    """Get batch processing results (new endpoint)"""
    with storage_lock:
        if batch_id not in processed_docs:
            # Check if it's still processing
            if batch_id in processing_status:
                status = processing_status[batch_id]
                return jsonify({
                    "error": "Batch not completed",
                    "status": status.get("status", "processing"),
                    "progress": status.get("progress", 0),
                    "completed_files": status.get("completed_files", 0),
                    "total_files": status.get("total_files", 0)
                }), 400
            return jsonify({"error": "Batch not found"}), 404
    
    try:
        batch_data = processed_docs[batch_id]
        return jsonify({
            "batch_id": batch_id,
            "status": "completed",
            "results": batch_data["result"],
            "file_count": batch_data.get("file_count", 0),
            "success_count": batch_data.get("success_count", 0),
            "processing_time": batch_data.get("processing_time", 0),
            "processed_at": batch_data["processed_at"]
        })
    except Exception as e:
        print(f"Error retrieving batch results: {e}")
        return jsonify({"error": f"Error retrieving batch results: {str(e)}"}), 500

@app.route('/ask-question', methods=['POST'])
def ask_question():
    """
    Answer a question about processed documents
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Check required fields
    if "batch_id" not in data or "question" not in data:
        return jsonify({"error": "Missing required fields: batch_id and question"}), 400
    
    batch_id = data["batch_id"]
    question = data["question"]
    
    # Check if the batch exists
    with storage_lock:
        if batch_id not in processed_docs:
            return jsonify({"error": "Batch ID not found"}), 404
    
    try:
        # Get document evidence
        evidence = processed_docs[batch_id]["result"]
        
        # For now, return a mock response
        # TODO: Implement actual OpenAI integration here
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
    """
    Clean up temporary files for a specific batch
    """
    with storage_lock:
        # Check processed docs
        if batch_id in processed_docs:
            batch_data = processed_docs[batch_id]
            
            # Delete the temporary files
            for file_path in batch_data.get("files", []):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            
            # Clean up grounding directory if it exists
            grounding_dir = batch_data.get("groundings_dir")
            if grounding_dir and os.path.exists(grounding_dir):
                try:
                    import shutil
                    shutil.rmtree(grounding_dir)
                except Exception as e:
                    print(f"Error removing grounding directory {grounding_dir}: {e}")
            
            # Remove from the processed docs dictionary
            del processed_docs[batch_id]
            cleanup_success = True
        else:
            cleanup_success = False
        
        # Also clean up processing status
        if batch_id in processing_status:
            del processing_status[batch_id]
    
    if cleanup_success:
        return jsonify({"status": "success", "message": "Batch cleaned up successfully"})
    else:
        return jsonify({"error": "Batch ID not found"}), 404

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        memory_info = psutil.virtual_memory()
        
        with storage_lock:
            active_count = len(processing_status)
            completed_count = len(processed_docs)
            
            # Calculate processing stats
            total_files_processed = sum(
                batch.get("file_count", 0) for batch in processed_docs.values()
            )
            total_success_files = sum(
                batch.get("success_count", 0) for batch in processed_docs.values()
            )
            
            # Get active batch details
            active_batches = []
            for batch_id, status in processing_status.items():
                active_batches.append({
                    "batch_id": batch_id,
                    "status": status.get("status", "unknown"),
                    "progress": status.get("progress", 0),
                    "completed_files": status.get("completed_files", 0),
                    "total_files": status.get("total_files", 0),
                    "elapsed_time": time.time() - status.get("start_time", time.time())
                })
        
        return jsonify({
            "system": {
                "memory_usage_percent": memory_info.percent,
                "memory_available_gb": memory_info.available / (1024**3),
                "uptime_seconds": time.time() - app.start_time if hasattr(app, 'start_time') else 0
            },
            "processing": {
                "active_batches": active_count,
                "completed_batches": completed_count,
                "total_files_processed": total_files_processed,
                "total_success_files": total_success_files,
                "success_rate": (total_success_files / total_files_processed * 100) if total_files_processed > 0 else 100
            },
            "active_batches": active_batches,
            "limits": {
                "max_batch_size": MAX_BATCH_SIZE,
                "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
                "memory_limit_percent": MEMORY_LIMIT,
                "processing_timeout_seconds": PROCESSING_TIMEOUT
            }
        })
    except Exception as e:
        return jsonify({"error": f"Error getting stats: {str(e)}"}), 500

# Add startup time for uptime calculation
app.start_time = time.time()

if __name__ == '__main__':
    # Set the port based on environment variable or default to 5001
    port = int(os.environ.get('PORT', 5001))
    
    print(f"Starting enhanced document processor on port {port}")
    print(f"Configuration:")
    print(f"  - Max batch size: {MAX_BATCH_SIZE} files")
    print(f"  - Max file size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB")
    print(f"  - Max total batch size: {MAX_TOTAL_BATCH_SIZE / 1024 / 1024:.1f}MB")
    print(f"  - Memory limit: {MEMORY_LIMIT}%")
    print(f"  - Processing timeout: {PROCESSING_TIMEOUT}s")
    print(f"  - Chunk size: {CHUNK_SIZE} files")
    print(f"  - Cleanup interval: {CLEANUP_INTERVAL}s")
    
    # Production vs Development configuration
    if os.environ.get('FLASK_ENV') == 'production':
        print("Running in production mode")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        print("Running in development mode")
        app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
