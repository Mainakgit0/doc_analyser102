import os
import logging
import httpx
import asyncio
from typing import List

logger = logging.getLogger(__name__)

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")


class LLMClient:
    """Enhanced LLM client with improved prompt engineering"""
    
    def __init__(self):
        self.timeout = httpx.Timeout(30.0)  # Reduced timeout to prevent worker timeouts
        self.max_retries = 1  # Quick fail for faster fallback
    
    async def generate_answers(self, questions: List[str], context: str) -> List[str]:
        """Generate answers for multiple questions with enhanced prompts"""
        answers = []
        
        for question in questions:
            try:
                answer = await self._generate_single_answer(question, context)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error generating answer for question '{question}': {e}")
                answers.append(f"Error generating answer: {str(e)}")
        
        return answers
    
    async def _generate_single_answer(self, question: str, context: str) -> str:
        """Generate answer for a single question with enhanced prompt"""
        
        # Enhanced prompt engineering for better accuracy
        prompt = self._create_enhanced_prompt(question, context)
        
        for attempt in range(self.max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "llama3-70b-8192",  # Using more capable model
                    "messages": [
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 3000,
                    "temperature": 0.0,  # Maximum precision for accuracy
                    "top_p": 0.95,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                }
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["choices"][0]["message"]["content"].strip()
                        return self._post_process_answer(answer)
                    elif response.status_code == 429:
                        # Rate limit hit - fail fast to use fallback
                        logger.warning(f"Rate limit hit (429) - failing fast to use fallback")
                        raise Exception("Rate limit exceeded")
                    else:
                        logger.warning(f"API error on attempt {attempt + 1}: {response.status_code}")
                        if attempt == self.max_retries - 1:
                            raise Exception(f"API Error {response.status_code}: Unable to generate answer")
                        await asyncio.sleep(1)  # Shorter sleep
                        
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise e  # Re-raise to trigger fallback
                await asyncio.sleep(1)  # Shorter sleep
        
        return "Unable to generate answer after multiple attempts"
    
    def _get_system_prompt(self) -> str:
        """Ultra-precise system prompt for maximum accuracy"""
        return """You are a precision document analysis expert with exceptional accuracy in extracting and synthesizing information from provided context. Your primary objective is to deliver 100% accurate, contextually grounded answers.

CRITICAL ACCURACY REQUIREMENTS:
1. Base answers EXCLUSIVELY on the provided document context - never add external knowledge
2. If specific information is not in the context, state: "This information is not available in the provided document"
3. Quote exact phrases from the document when answering
4. Cross-reference multiple sections of the document to ensure completeness
5. Identify and resolve any contradictions within the document
6. Be extremely specific - avoid vague or general statements
7. Structure complex answers with clear organization (bullet points, numbered lists)
8. If the context is insufficient for a complete answer, specify what additional information would be needed

ANSWER FORMAT:
- Start with the direct answer
- Support with specific quotes/references from the document
- Organize information logically
- End with any relevant caveats or limitations from the context

Your expertise spans: insurance policies, legal contracts, HR documentation, compliance materials, technical specifications, and research papers. Prioritize precision over brevity."""
    
    def _create_enhanced_prompt(self, question: str, context: str) -> str:
        """Create ultra-precise prompt for maximum accuracy"""
        
        return f"""DOCUMENT ANALYSIS TASK:
Analyze the provided document context and answer the specific question with maximum precision and accuracy.

DOCUMENT CONTEXT:
{context}

QUESTION TO ANALYZE: {question}

REQUIRED RESPONSE FORMAT:
1. Provide a direct, specific answer based ONLY on the document content
2. Include exact quotes or specific references from the document
3. If multiple relevant sections exist, synthesize them comprehensively  
4. If any information is missing from the document, clearly state what is not available
5. Cross-check for consistency across different parts of the document
6. Organize your response clearly with bullet points or sections if complex

CRITICAL: Base your answer exclusively on the document context provided above. Do not add external knowledge or assumptions."""
    
    def _post_process_answer(self, answer: str) -> str:
        """Post-process the generated answer"""
        if not answer:
            return "No answer generated"
        
        # Clean up the answer
        answer = answer.strip()
        
        # Remove any prompt artifacts
        if answer.startswith("ANSWER:"):
            answer = answer[7:].strip()
        
        # Ensure proper formatting
        lines = answer.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n\n'.join(cleaned_lines) if cleaned_lines else answer


class FallbackLLMClient:
    """Fallback client for when primary LLM is unavailable"""
    
    def __init__(self):
        pass
    
    async def generate_answers(self, questions: List[str], context: str) -> List[str]:
        """Generate basic answers using keyword extraction"""
        answers = []
        
        for question in questions:
            # Simple keyword-based answer generation
            answer = self._extract_relevant_text(question, context)
            answers.append(answer)
        
        return answers
    
    def _extract_relevant_text(self, question: str, context: str) -> str:
        """Extract most relevant text segments"""
        if not context:
            return "No context available to answer the question."
        
        # Split context into sentences
        sentences = context.split('.')
        question_words = set(question.lower().split())
        
        # Score sentences based on keyword overlap
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(question_words & sentence_words)
            if overlap > 0:
                scored_sentences.append((overlap, sentence))
        
        # Sort by relevance and take top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_sentences:
            return "No relevant information found in the provided context."
        
        # Combine top relevant sentences
        top_sentences = [sentence for _, sentence in scored_sentences[:3]]
        return '. '.join(top_sentences) + '.'
