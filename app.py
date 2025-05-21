from flask import Flask, request, jsonify
import os
import tempfile
import uuid
import json
import sys
import time
from werkzeug.utils import secure_filename
from flask_cors import CORS

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
            print("Please upgrade to at least version 0.2.0 with: pip install --upgrade agentic-doc==0.2.0")
            # You can either exit or continue with a warning
            # sys.exit(1)  # Uncomment to force exit if version is incompatible
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
# These will be picked up automatically by the SDK
os.environ.setdefault('BATCH_SIZE', '4')
os.environ.setdefault('MAX_WORKERS', '5')
os.environ.setdefault('MAX_RETRIES', '100')
os.environ.setdefault('MAX_RETRY_WAIT_TIME', '60')
os.environ.setdefault('RETRY_LOGGING_STYLE', 'log_msg')

# Storage for processed documents
processed_docs = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "service": "document-processor",
        "agentic_doc_version": agentic_doc_version
    })

@app.route('/process-documents', methods=['POST'])
def process_documents():
    """
    Process uploaded documents using the SDK
    """
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No files selected"}), 400
    
    # Get optional parameters from request
    include_marginalia = request.form.get('include_marginalia', 'true').lower() == 'true'
    include_metadata = request.form.get('include_metadata', 'true').lower() == 'true'
    save_groundings = request.form.get('save_groundings', 'false').lower() == 'true'
    
    # Save uploaded files to temp directory
    saved_files = []
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
            file.save(temp_path)
            saved_files.append(temp_path)
    
    if not saved_files:
        return jsonify({"error": "No valid files uploaded"}), 400
    
    try:
        # Create grounding directory if needed
        grounding_dir = None
        if save_groundings:
            grounding_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"groundings_{uuid.uuid4()}")
            os.makedirs(grounding_dir, exist_ok=True)
            
        # Process documents using SDK with new parameters
        start_time = time.time()
        result = parse_documents(
            saved_files,
            include_marginalia=include_marginalia,
            include_metadata_in_markdown=include_metadata,
            grounding_save_dir=grounding_dir if save_groundings else None
        )
        processing_time = time.time() - start_time
        
        # Generate a unique ID for this batch of documents
        batch_id = str(uuid.uuid4())
        processed_docs[batch_id] = {
            "result": result,
            "files": saved_files,
            "processed_at": time.time(),
            "groundings_dir": grounding_dir if save_groundings else None
        }
        
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
        if agentic_doc_version and agentic_doc_version.startswith("0.0.") or agentic_doc_version and agentic_doc_version.startswith("0.1."):
            formatted_result["warnings"].append(
                "IMPORTANT: Chunk types are changing as of May 22, 2025. Please upgrade to agentic-doc v0.2.0 or later."
            )
        
        return jsonify(formatted_result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-document-data/<batch_id>', methods=['GET'])
def get_document_data(batch_id):
    """
    Retrieve processed document data by batch ID
    """
    if batch_id not in processed_docs:
        return jsonify({"error": "Batch ID not found"}), 404
    
    # Return the processed document data
    return jsonify({
        "batch_id": batch_id,
        "result": processed_docs[batch_id]["result"],
        "files": [os.path.basename(f) for f in processed_docs[batch_id]["files"]],
        "processed_at": processed_docs[batch_id]["processed_at"],
        "groundings_dir": processed_docs[batch_id].get("groundings_dir")
    })

@app.route('/visualize-document/<batch_id>', methods=['GET'])
def visualize_document(batch_id):
    """
    Create visualizations of the document parsing results
    """
    if batch_id not in processed_docs:
        return jsonify({"error": "Batch ID not found"}), 404
    
    try:
        # Import visualization utilities from the SDK
        try:
            from agentic_doc.utils import viz_parsed_document
        except ImportError:
            return jsonify({"error": "Visualization feature not available in the current SDK version"}), 501
        
        # Get the document data
        doc_data = processed_docs[batch_id]
        file_paths = doc_data["files"]
        results = doc_data["result"]
        
        # Create visualization directory
        viz_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"viz_{batch_id}")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate visualizations for each document
        viz_paths = []
        for i, file_path in enumerate(file_paths):
            if i < len(results):
                output_images = viz_parsed_document(
                    file_path,
                    results[i],
                    output_dir=viz_dir
                )
                
                # Save paths of visualization images
                for j, img in enumerate(output_images):
                    viz_filename = f"{os.path.basename(file_path)}_viz_page_{j}.png"
                    viz_path = os.path.join(viz_dir, viz_filename)
                    img.save(viz_path)
                    viz_paths.append(viz_path)
        
        return jsonify({
            "status": "success",
            "visualization_paths": viz_paths,
            "batch_id": batch_id
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    Clean up temporary files for a specific batch
    """
    if batch_id not in processed_docs:
        return jsonify({"error": "Batch ID not found"}), 404
    
    # Delete the temporary files
    for file_path in processed_docs[batch_id]["files"]:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")
    
    # Clean up grounding directory if it exists
    grounding_dir = processed_docs[batch_id].get("groundings_dir")
    if grounding_dir and os.path.exists(grounding_dir):
        try:
            import shutil
            shutil.rmtree(grounding_dir)
        except Exception as e:
            print(f"Error removing grounding directory {grounding_dir}: {e}")
    
    # Remove from the processed docs dictionary
    del processed_docs[batch_id]
    
    return jsonify({"status": "success", "message": "Batch cleaned up successfully"})

if __name__ == '__main__':
    # Set the port based on environment variable or default to 5001
    # Using 5001 instead of 5000 to avoid conflicts with AirPlay on macOS
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
