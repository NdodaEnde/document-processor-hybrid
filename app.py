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
from functools import wraps
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Start memory tracing
tracemalloc.start()

# Add parent directory to path to access the Mock SDK
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Version check - Before trying to import the SDK
import pkg_resources

def log_memory_usage(context=""):
    """Log current memory usage and top consumers"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"[MEMORY] {context}: {memory_mb:.2f} MB RSS, {memory_info.vms / 1024 / 1024:.2f} MB VMS")
        
        # Get top memory consumers
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print(f"[MEMORY] Top 3 memory consumers:")
        for index, stat in enumerate(top_stats[:3], 1):
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

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        temp_dir = tempfile.gettempdir()
        for filename in os.listdir(temp_dir):
            if filename.startswith(('tmp', 'doc_processor', 'agentic')):
                try:
                    filepath = os.path.join(temp_dir, filename)
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                        print(f"[CLEANUP] Removed temp file: {filename}")
                    elif os.path.isdir(filepath):
                        import shutil
                        shutil.rmtree(filepath)
                        print(f"[CLEANUP] Removed temp dir: {filename}")
                except Exception as e:
                    print(f"[CLEANUP] Error removing {filename}: {e}")
    except Exception as e:
        print(f"[CLEANUP] Error during cleanup: {e}")

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

# Configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'doc_processor_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# SDK Configuration options loaded from environment variables
os.environ.setdefault('BATCH_SIZE', '1')  # Reduced from 4 to 1
os.environ.setdefault('MAX_WORKERS', '1')  # Reduced from 5 to 1
os.environ.setdefault('MAX_RETRIES', '100')
os.environ.setdefault('MAX_RETRY_WAIT_TIME', '60')
os.environ.setdefault('RETRY_LOGGING_STYLE', 'log_msg')

# Storage for processed documents
processed_docs = {}

def check_file_size(file_path):
    """Check if file is too large for processing"""
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > 10:  # 10MB limit
            return False, f"File too large: {size_mb:.2f}MB (max 10MB)"
        return True, "OK"
    except Exception as e:
        return False, f"Error checking file size: {e}"

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
    for chunk in chunks:
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
    for ground in grounding:
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    log_memory_usage("Health check")
    return jsonify({
        "status": "healthy", 
        "service": "document-processor",
        "agentic_doc_version": agentic_doc_version,
        "active_batches": len(processed_docs)
    })

@app.route('/memory-status', methods=['GET'])
def memory_status():
    """Get current memory status"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return jsonify({
            "memory_mb": memory_info.rss / 1024 / 1024,
            "virtual_memory_mb": memory_info.vms / 1024 / 1024,
            "active_batches": len(processed_docs),
            "top_memory_consumers": [str(stat) for stat in top_stats[:5]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process-documents', methods=['POST'])
def process_documents():
    """
    Process uploaded documents using the SDK with enhanced memory management
    """
    log_memory_usage("START of process_documents")
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No files selected"}), 400
    
    # Get optional parameters from request
    include_marginalia = request.form.get('include_marginalia', 'true').lower() == 'true'
    include_metadata = request.form.get('include_metadata', 'true').lower() == 'true'
    save_groundings = request.form.get('save_groundings', 'false').lower() == 'true'
    
    log_memory_usage("AFTER parameter extraction")
    
    # Save uploaded files to temp directory
    saved_files = []
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
            
            # Save file in chunks to avoid loading entire file in memory
            with open(temp_path, 'wb') as f:
                while True:
                    chunk = file.stream.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
            
            # Check file size
            is_valid, message = check_file_size(temp_path)
            if not is_valid:
                # Clean up the file we just saved
                try:
                    os.remove(temp_path)
                except:
                    pass
                return jsonify({"error": message}), 413
            
            saved_files.append(temp_path)
            log_memory_usage(f"AFTER saving file {filename}")
    
    if not saved_files:
        return jsonify({"error": "No valid files uploaded"}), 400
    
    # Force cleanup before processing
    force_garbage_collection()
    cleanup_temp_files()
    log_memory_usage("BEFORE SDK processing")
    
    try:
        # Create grounding directory if needed
        grounding_dir = None
        if save_groundings:
            grounding_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"groundings_{uuid.uuid4()}")
            os.makedirs(grounding_dir, exist_ok=True)
            
        # Process documents using SDK with memory optimization
        start_time = time.time()
        
        # Process only one file at a time to reduce memory usage
        result = parse_documents(
            saved_files[:1],  # Process only the first file
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata,
            grounding_save_dir=grounding_dir if save_groundings else None
        )
        
        processing_time = time.time() - start_time
        log_memory_usage("AFTER SDK processing")
        
        # Generate a unique ID for this batch of documents
        batch_id = str(uuid.uuid4())
        
        # Serialize the result before storing with memory monitoring
        log_memory_usage("BEFORE serialization")
        serialized_result = []
        
        for i, doc in enumerate(result):
            try:
                serialized_doc = serialize_parsed_document(doc)
                
                # Limit markdown size to prevent memory issues
                if len(serialized_doc.get('markdown', '')) > 1024 * 1024:  # 1MB limit
                    serialized_doc['markdown'] = serialized_doc['markdown'][:1024*1024] + "... [TRUNCATED]"
                
                serialized_result.append(serialized_doc)
                
                # Clear reference to help with memory
                doc = None
                
                log_memory_usage(f"SERIALIZED document {i+1}")
                
            except Exception as serialize_error:
                print(f"[ERROR] Serialization failed for doc {i}: {serialize_error}")
                serialized_result.append({
                    "markdown": f"Serialization failed: {str(serialize_error)}",
                    "chunks": [],
                    "errors": [{"message": str(serialize_error), "page": 0}],
                    "processing_time": 0
                })
        
        # Clear the original result to free memory
        result = None
        force_garbage_collection()
        log_memory_usage("AFTER serialization and GC")
        
        processed_docs[batch_id] = {
            "result": serialized_result,
            "files": saved_files,
            "processed_at": time.time(),
            "groundings_dir": grounding_dir if save_groundings else None
        }
        
        # Clear serialized_result reference
        serialized_result = None
        force_garbage_collection()
        
        log_memory_usage("END of process_documents")
        
        # Format the response
        formatted_result = {
            "batch_id": batch_id,
            "document_count": len(saved_files),
            "processing_time_seconds": processing_time,
            "status": "success",
            "grounding_images_saved": save_groundings,
            "warnings": []
        }
        
        # Add warning about chunk type changes if using old version
        if agentic_doc_version and (agentic_doc_version.startswith("0.0.") or agentic_doc_version.startswith("0.1.")):
            formatted_result["warnings"].append(
                "IMPORTANT: Chunk types are changing as of May 22, 2025. Please upgrade to agentic-doc v0.2.0 or later."
            )
        
        return jsonify(formatted_result)
    
    except Exception as e:
        print(f"Error processing documents: {e}")
        log_memory_usage("ERROR state")
        
        # Emergency cleanup
        for file_path in saved_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        
        cleanup_temp_files()
        force_garbage_collection()
        
        return jsonify({"error": str(e)}), 500

@app.route('/get-document-data/<batch_id>', methods=['GET'])
def get_document_data(batch_id):
    """
    Retrieve processed document data by batch ID
    """
    log_memory_usage(f"GET document data for batch {batch_id}")
    
    if batch_id not in processed_docs:
        return jsonify({"error": "Batch ID not found"}), 404
    
    try:
        # Return the processed document data (already serialized)
        return jsonify({
            "batch_id": batch_id,
            "result": processed_docs[batch_id]["result"],
            "files": [os.path.basename(f) for f in processed_docs[batch_id]["files"]],
            "processed_at": processed_docs[batch_id]["processed_at"],
            "groundings_dir": processed_docs[batch_id].get("groundings_dir")
        })
    except Exception as e:
        print(f"Error retrieving document data: {e}")
        return jsonify({"error": f"Error retrieving document data: {str(e)}"}), 500

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
    Clean up temporary files for a specific batch with enhanced memory management
    """
    if batch_id not in processed_docs:
        return jsonify({"error": "Batch ID not found"}), 404
    
    log_memory_usage(f"BEFORE cleanup batch {batch_id}")
    
    try:
        # Delete the temporary files
        for file_path in processed_docs[batch_id]["files"]:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[CLEANUP] Removed file: {file_path}")
            except Exception as e:
                print(f"[CLEANUP] Error removing file {file_path}: {e}")
        
        # Clean up grounding directory if it exists
        grounding_dir = processed_docs[batch_id].get("groundings_dir")
        if grounding_dir and os.path.exists(grounding_dir):
            try:
                import shutil
                shutil.rmtree(grounding_dir)
                print(f"[CLEANUP] Removed grounding directory: {grounding_dir}")
            except Exception as e:
                print(f"[CLEANUP] Error removing grounding directory {grounding_dir}: {e}")
        
        # Remove from the processed docs dictionary
        del processed_docs[batch_id]
        
        # Force garbage collection
        force_garbage_collection()
        
        # Additional temp file cleanup
        cleanup_temp_files()
        
        log_memory_usage(f"AFTER cleanup batch {batch_id}")
        
        return jsonify({"status": "success", "message": "Batch cleaned up successfully"})
        
    except Exception as e:
        print(f"[CLEANUP] Error during cleanup: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Log startup memory
    log_memory_usage("Application startup")
    
    # Set the port based on environment variable or default to 5001
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
