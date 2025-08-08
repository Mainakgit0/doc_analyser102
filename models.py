from datetime import datetime
from app import db
from sqlalchemy import Text, DateTime, Integer, String, Float


class QueryRequest(db.Model):
    """Model for storing query requests and responses"""
    id = db.Column(Integer, primary_key=True)
    document_url = db.Column(String(500), nullable=True)
    document_filename = db.Column(String(255), nullable=True)
    document_hash = db.Column(String(64), nullable=False)
    questions = db.Column(Text, nullable=False)  # JSON string of questions
    answers = db.Column(Text, nullable=False)    # JSON string of answers
    processing_time = db.Column(Float, nullable=False)
    document_length = db.Column(Integer, nullable=False)
    chunks_created = db.Column(Integer, nullable=False)
    created_at = db.Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<QueryRequest {self.id}>'


class DocumentCache(db.Model):
    """Model for caching processed documents"""
    id = db.Column(Integer, primary_key=True)
    document_hash = db.Column(String(64), unique=True, nullable=False)
    document_url = db.Column(String(500), nullable=True)
    document_filename = db.Column(String(255), nullable=True)
    processed_text = db.Column(Text, nullable=False)
    text_length = db.Column(Integer, nullable=False)
    chunks_count = db.Column(Integer, nullable=False)
    created_at = db.Column(DateTime, default=datetime.utcnow)
    last_accessed = db.Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<DocumentCache {self.document_hash}>'


class PerformanceStats(db.Model):
    """Model for tracking system performance"""
    id = db.Column(Integer, primary_key=True)
    total_requests = db.Column(Integer, default=0)
    total_questions = db.Column(Integer, default=0)
    avg_response_time = db.Column(Float, default=0.0)
    cache_hits = db.Column(Integer, default=0)
    pdfs_processed = db.Column(Integer, default=0)
    total_extracted_chars = db.Column(Integer, default=0)
    updated_at = db.Column(DateTime, default=datetime.utcnow)
    
    @classmethod
    def get_stats(cls):
        """Get current performance stats"""
        stats = cls.query.first()
        if not stats:
            stats = cls()
            db.session.add(stats)
            db.session.commit()
        return stats
    
    def update_stats(self, requests=0, questions=0, response_time=0.0, 
                    cache_hit=False, pdf_processed=False, chars_extracted=0):
        """Update performance statistics"""
        self.total_requests += requests
        self.total_questions += questions
        
        # Update average response time
        if self.total_requests > 0 and response_time > 0:
            total_time = self.avg_response_time * (self.total_requests - requests)
            self.avg_response_time = (total_time + response_time) / self.total_requests
        
        if cache_hit:
            self.cache_hits += 1
        if pdf_processed:
            self.pdfs_processed += 1
        if chars_extracted > 0:
            self.total_extracted_chars += chars_extracted
            
        self.updated_at = datetime.utcnow()
        db.session.commit()
