import json
import time
import asyncio
import logging
from datetime import datetime
from flask import render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
from app import app, db
from models import QueryRequest, PerformanceStats, DocumentCache
from text_processor import AdvancedTextProcessor
from document_handler import DocumentHandler
from llm_client import LLMClient, FallbackLLMClient

logger = logging.getLogger(__name__)

# Initialize processors
text_processor = AdvancedTextProcessor()
document_handler = DocumentHandler()
llm_client = LLMClient()
fallback_client = FallbackLLMClient()


@app.route('/')
def index():
    """Main page for document query interface"""
    try:
        stats = PerformanceStats.get_stats()
        return render_template('index.html', stats=stats)
    except Exception as e:
        logger.error(f"Error loading index page: {e}")
        return render_template('index.html', stats=None)


@app.route('/history')
def history():
    """View query history"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 20
        
        queries = QueryRequest.query.order_by(
            QueryRequest.created_at.desc()
        ).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        return render_template('history.html', queries=queries)
    except Exception as e:
        logger.error(f"Error loading history page: {e}")
        flash("Error loading query history", "error")
        return redirect(url_for('index'))


@app.route('/query', methods=['POST'])
def process_query():
    """Process document query with enhanced accuracy"""
    start_time = time.time()
    
    try:
        # Get form data
        questions_text = request.form.get('questions', '').strip()
        document_url = request.form.get('document_url', '').strip()
        uploaded_file = request.files.get('document_file')
        
        # Validate input
        if not questions_text:
            flash("Please provide at least one question", "error")
            return redirect(url_for('index'))
        
        if not document_url and not uploaded_file:
            flash("Please provide either a document URL or upload a file", "error")
            return redirect(url_for('index'))
        
        # Parse questions
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        if not questions:
            flash("Please provide valid questions", "error")
            return redirect(url_for('index'))
        
        # Process document and generate answers
        result = asyncio.run(process_query_async(
            questions, document_url, uploaded_file
        ))
        
        if result.get('error'):
            flash(result['error'], "error")
            return redirect(url_for('index'))
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save to database
        save_query_to_db(
            document_url=document_url,
            filename=result.get('filename'),
            doc_hash=result['doc_hash'],
            questions=questions,
            answers=result['answers'],
            processing_time=processing_time,
            document_length=result['document_length'],
            chunks_created=result['chunks_created']
        )
        
        # Update performance stats
        stats = PerformanceStats.get_stats()
        stats.update_stats(
            requests=1,
            questions=len(questions),
            response_time=processing_time
        )
        
        # Return results
        return render_template('index.html', 
                             questions=questions,
                             answers=result['answers'],
                             processing_time=processing_time,
                             stats=stats,
                             success=True)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        flash(f"Error processing query: {str(e)}", "error")
        return redirect(url_for('index'))


async def process_query_async(questions, document_url, uploaded_file):
    """Async processing of document query"""
    try:
        # Process document
        if uploaded_file and uploaded_file.filename:
            filename = secure_filename(uploaded_file.filename)
            file_content = uploaded_file.read()
            
            if not file_content:
                return {"error": "Uploaded file is empty"}
            
            document_text, doc_hash = await document_handler.process_uploaded_file(
                file_content, filename
            )
            filename_info = filename
        else:
            document_text, doc_hash = await document_handler.download_and_process_document(
                document_url
            )
            filename_info = None
        
        if not document_text.strip():
            return {"error": "No text could be extracted from the document"}
        
        logger.info(f"Processing document with {len(document_text)} characters")
        
        # Create semantic chunks with optimized parameters for accuracy
        chunks = text_processor.semantic_chunk_text(
            document_text, 
            target_chunk_size=800,   # Smaller chunks for better precision
            overlap_ratio=0.25       # Higher overlap for better context preservation
        )
        
        if not chunks:
            return {"error": "Document could not be processed into readable chunks"}
        
        logger.info(f"Created {len(chunks)} chunks for processing")
        
        # Generate answers for each question
        answers = []
        for question in questions:
            try:
                # Find relevant chunks using enhanced TF-IDF with better parameters
                relevant_chunks = text_processor.find_relevant_chunks(
                    question, 
                    chunks, 
                    top_k=8,        # More chunks for better accuracy
                    min_score=0.01  # Much lower threshold for comprehensive context
                )
                
                if not relevant_chunks:
                    logger.warning(f"No relevant chunks found for question: {question}")
                    # Use top chunks as fallback
                    relevant_chunks = chunks[:3]
                
                # Create optimized context with more space for comprehensive answers
                context = text_processor.create_context(
                    relevant_chunks, 
                    max_context_length=8000  # Much larger context for maximum accuracy
                )
                
                # Generate answer using LLM
                try:
                    answer_list = await llm_client.generate_answers([question], context)
                    answer = answer_list[0] if answer_list else "No answer generated"
                except Exception as llm_error:
                    logger.error(f"LLM error for question '{question}': {llm_error}")
                    # Use fallback client
                    answer_list = await fallback_client.generate_answers([question], context)
                    answer = answer_list[0] if answer_list else f"Error: {str(llm_error)}"
                
                answers.append(answer)
                
            except Exception as q_error:
                logger.error(f"Error processing question '{question}': {q_error}")
                answers.append(f"Error processing question: {str(q_error)}")
        
        return {
            "answers": answers,
            "doc_hash": doc_hash,
            "filename": filename_info,
            "document_length": len(document_text),
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Async processing error: {e}")
        return {"error": str(e)}


def save_query_to_db(document_url, filename, doc_hash, questions, answers, 
                    processing_time, document_length, chunks_created):
    """Save query request to database"""
    try:
        query_request = QueryRequest(
            document_url=document_url,
            document_filename=filename,
            document_hash=doc_hash,
            questions=json.dumps(questions),
            answers=json.dumps(answers),
            processing_time=processing_time,
            document_length=document_length,
            chunks_created=chunks_created
        )
        
        db.session.add(query_request)
        db.session.commit()
        
        logger.info(f"Saved query request to database with ID: {query_request.id}")
        
    except Exception as e:
        logger.error(f"Error saving query to database: {e}")
        db.session.rollback()


@app.route('/api/stats')
def get_stats():
    """API endpoint for performance statistics"""
    try:
        stats = PerformanceStats.get_stats()
        return jsonify({
            "total_requests": stats.total_requests,
            "total_questions": stats.total_questions,
            "avg_response_time": round(stats.avg_response_time, 2),
            "cache_hits": stats.cache_hits,
            "pdfs_processed": stats.pdfs_processed,
            "total_extracted_chars": stats.total_extracted_chars,
            "updated_at": stats.updated_at.isoformat() if stats.updated_at else None
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({"error": "Failed to load statistics"}), 500


@app.route('/api/query/<int:query_id>')
def get_query_details(query_id):
    """API endpoint for query details"""
    try:
        query = QueryRequest.query.get_or_404(query_id)
        return jsonify({
            "id": query.id,
            "document_url": query.document_url,
            "document_filename": query.document_filename,
            "questions": json.loads(query.questions),
            "answers": json.loads(query.answers),
            "processing_time": query.processing_time,
            "document_length": query.document_length,
            "chunks_created": query.chunks_created,
            "created_at": query.created_at.isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting query details: {e}")
        return jsonify({"error": "Query not found"}), 404


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear document cache"""
    try:
        deleted_count = db.session.query(DocumentCache).count()
        db.session.query(DocumentCache).delete()
        db.session.commit()
        
        flash(f"Cleared {deleted_count} cached documents", "success")
        logger.info(f"Cleared {deleted_count} cached documents")
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        flash("Error clearing cache", "error")
        db.session.rollback()
    
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html', error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    logger.error(f"Internal error: {error}")
    return render_template('index.html', error="Internal server error"), 500


@app.route('/api/query', methods=['POST'])
def api_query():
    """JSON API endpoint matching your original FastAPI format"""
    try:
        start_time = time.time()
        
        # Parse JSON request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        # Extract data matching your original format
        documents = data.get('documents', '')  # URL to document
        questions = data.get('questions', [])  # List of questions
        
        if not documents:
            return jsonify({"error": "Missing 'documents' URL"}), 400
            
        if not questions:
            return jsonify({"error": "Missing 'questions' list"}), 400
        
        if not isinstance(questions, list):
            return jsonify({"error": "'questions' must be a list"}), 400
        
        # Clean questions
        questions = [q.strip() for q in questions if q.strip()]
        if not questions:
            return jsonify({"error": "No valid questions provided"}), 400
        
        # Process document and generate answers
        result = asyncio.run(process_query_async(
            questions, documents, None  # No file upload for API
        ))
        
        if result.get('error'):
            return jsonify({"error": result['error']}), 400
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save to database
        save_query_to_db(
            document_url=documents,
            filename=result.get('filename'),
            doc_hash=result['doc_hash'],
            questions=questions,
            answers=result['answers'],
            processing_time=processing_time,
            document_length=result['document_length'],
            chunks_created=result['chunks_created']
        )
        
        # Update performance stats
        stats = PerformanceStats.get_stats()
        stats.update_stats(
            requests=1,
            questions=len(questions),
            response_time=processing_time
        )
        
        # Return JSON response matching your original format
        return jsonify({
            "answers": result['answers']
        })
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500
