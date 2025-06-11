# Document Processing Microservice SDK

A Flask-based microservice for processing medical documents using the Agentic Document AI SDK. This service provides document parsing, text extraction, and structured data analysis capabilities specifically designed for medical certificate processing workflows.

## Features

- **Document Processing**: Parse PDFs and images using Agentic Document AI SDK
- **Medical Document Support**: Specialized handling for certificates of fitness and medical questionnaires
- **Signature & Stamp Detection**: Automatic detection of signatures and stamps in medical documents
- **Structured Data Extraction**: Extract patient information, examination results, and certification details
- **Template Auto-Detection**: Automatically determines if documents are modern workflow or historical filed records
- **Batch Processing**: Process multiple documents simultaneously
- **REST API**: Easy integration with web applications
- **CORS Support**: Cross-origin requests enabled for web frontend integration

## Medical Document Types Supported

- **Certificate of Fitness**: Pre-employment, periodical, and exit medical examinations
- **Medical Questionnaires**: Patient health history and assessment forms
- **Medical Certificates**: General medical certification documents

## API Endpoints

### Health Check
```
GET /health
```
Returns service status and SDK version information.

### Process Documents
```
POST /process-documents
```
Upload and process documents using the Agentic Document AI SDK.

**Parameters:**
- `files`: Document files (PDF, images)
- `include_marginalia`: Include marginalia in processing (default: true)
- `include_metadata`: Include metadata in markdown (default: true)
- `save_groundings`: Save grounding images (default: false)

**Response:**
```json
{
  "batch_id": "uuid",
  "document_count": 1,
  "processing_time_seconds": 2.5,
  "status": "success",
  "grounding_images_saved": false,
  "warnings": []
}
```

### Get Document Data
```
GET /get-document-data/{batch_id}
```
Retrieve processed document data using the batch ID.

**Response:**
```json
{
  "batch_id": "uuid",
  "result": [...],
  "files": ["filename.pdf"],
  "processed_at": 1640995200,
  "groundings_dir": "/path/to/groundings"
}
```

### Cleanup
```
DELETE /cleanup/{batch_id}
```
Clean up temporary files and free memory for a processed batch.

## Medical Document Processing Features

### Certificate of Fitness Processing
- **Patient Information**: Name, ID number, company, job title
- **Examination Details**: Date, physician, examination type (pre-employment/periodical/exit)
- **Medical Tests**: Blood work, vision tests, hearing, lung function, X-ray, drug screening
- **Fitness Assessment**: Fit, fit with restrictions, fit with condition, temporarily unfit, unfit
- **Restrictions**: Heights, dust exposure, chemical exposure, hearing protection, etc.
- **Certification Dates**: Examination date and expiry date

### Signature & Stamp Detection
The service automatically detects:
- **Digital/Scanned Signatures**: Handwriting detection in figure chunks
- **Medical Practice Stamps**: Practice numbers and official stamps
- **Template Classification**: Modern documents (current workflow) vs Historical documents (filed records)

### Data Serialization
All document objects are automatically serialized to JSON-compatible format:
- **Chunks**: Text, tables, figures, forms with grounding coordinates
- **Metadata**: Document properties and processing information
- **Errors**: Processing errors with page references
- **Grounding**: Bounding box coordinates for visual elements

## Installation

### Prerequisites
- Python 3.8+
- Flask
- Agentic Document AI SDK (v0.2.0+ recommended)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd document-processing-microservice

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export BATCH_SIZE=4
export MAX_WORKERS=5
export MAX_RETRIES=100
export MAX_RETRY_WAIT_TIME=60
export RETRY_LOGGING_STYLE=log_msg

# Run the service
python app.py
```

### Environment Variables
- `PORT`: Service port (default: 5001)
- `BATCH_SIZE`: Document processing batch size (default: 4)
- `MAX_WORKERS`: Maximum worker threads (default: 5)
- `MAX_RETRIES`: Maximum retry attempts (default: 100)
- `MAX_RETRY_WAIT_TIME`: Maximum retry wait time in seconds (default: 60)

## Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5001

CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t document-processor .
docker run -p 5001:5001 document-processor
```

## Integration Example

### Frontend Integration (JavaScript)
```javascript
// Upload and process document
const formData = new FormData();
formData.append('files', fileInput.files[0]);
formData.append('include_marginalia', 'true');
formData.append('save_groundings', 'false');

const response = await fetch('http://localhost:5001/process-documents', {
  method: 'POST',
  body: formData
});

const result = await response.json();
const batchId = result.batch_id;

// Retrieve processed data
const dataResponse = await fetch(`http://localhost:5001/get-document-data/${batchId}`);
const documentData = await dataResponse.json();

// Clean up when done
await fetch(`http://localhost:5001/cleanup/${batchId}`, { method: 'DELETE' });
```

### Supabase Edge Function Integration
```typescript
// Create form data for microservice
const forwardFormData = new FormData();
forwardFormData.append('files', file);

// Process document
const response = await fetch(`${microserviceUrl}/process-documents`, {
  method: 'POST',
  body: forwardFormData
});

const result = await response.json();

// Get processed data
const dataResponse = await fetch(`${microserviceUrl}/get-document-data/${result.batch_id}`);
const documentData = await dataResponse.json();
```

## SDK Version Compatibility

The service automatically detects and warns about Agentic Document AI SDK versions:
- **v0.2.0+**: Latest chunk types and features (recommended)
- **v0.1.x**: Legacy chunk types (deprecated)
- **v0.0.x**: Old versions (will stop working after May 2025)

## Medical Data Security

- **Temporary Processing**: Files are automatically cleaned up after processing
- **No Persistent Storage**: Documents are not permanently stored on the microservice
- **Memory Management**: Processed data is cleared when cleanup endpoint is called
- **CORS Configuration**: Configurable for secure frontend integration

## Error Handling

The service provides comprehensive error handling:
- **File Upload Errors**: Invalid files, size limits, format validation
- **Processing Errors**: SDK failures, parsing errors, serialization issues
- **Medical Data Validation**: Missing required fields, invalid date formats
- **Graceful Degradation**: Fallback processing when SDK is unavailable

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature/new-feature`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
1. Check the [Issues](../../issues) page
2. Review the API documentation above
3. Ensure you're using a compatible Agentic Document AI SDK version

## Changelog

### v1.0.0
- Initial release with medical document processing
- Signature and stamp detection
- Certificate of fitness template support
- Batch processing and cleanup functionality
