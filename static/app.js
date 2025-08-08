// Enhanced JavaScript for document query system

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('queryForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner-border');
    
    // Tab switching logic
    const urlTab = document.getElementById('url-tab');
    const uploadTab = document.getElementById('upload-tab');
    const urlInput = document.getElementById('documentUrl');
    const fileInput = document.getElementById('documentFile');
    
    // Clear inputs when switching tabs
    urlTab.addEventListener('click', function() {
        if (fileInput) {
            fileInput.value = '';
        }
    });
    
    uploadTab.addEventListener('click', function() {
        if (urlInput) {
            urlInput.value = '';
        }
    });
    
    // Form validation and submission
    if (form) {
        form.addEventListener('submit', function(e) {
            const questions = document.getElementById('questions').value.trim();
            const documentUrl = urlInput ? urlInput.value.trim() : '';
            const documentFile = fileInput ? fileInput.files[0] : null;
            
            // Validate questions
            if (!questions) {
                e.preventDefault();
                showAlert('Please enter at least one question.', 'danger');
                return;
            }
            
            // Validate document source
            if (!documentUrl && !documentFile) {
                e.preventDefault();
                showAlert('Please provide either a document URL or upload a file.', 'danger');
                return;
            }
            
            // Validate file size (10MB limit)
            if (documentFile && documentFile.size > 10 * 1024 * 1024) {
                e.preventDefault();
                showAlert('File size must be less than 10MB.', 'danger');
                return;
            }
            
            // Show loading state
            showLoadingState();
        });
    }
    
    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                // Check file type
                const allowedTypes = ['.pdf', '.docx', '.doc', '.txt'];
                const fileName = file.name.toLowerCase();
                const isValidType = allowedTypes.some(type => fileName.endsWith(type));
                
                if (!isValidType) {
                    showAlert('Please select a valid file type (PDF, DOCX, DOC, or TXT).', 'warning');
                    this.value = '';
                    return;
                }
                
                // Check file size
                if (file.size > 10 * 1024 * 1024) {
                    showAlert('File size must be less than 10MB.', 'warning');
                    this.value = '';
                    return;
                }
                
                // Show file info
                const fileSize = (file.size / 1024 / 1024).toFixed(2);
                showAlert(`Selected: ${file.name} (${fileSize} MB)`, 'info');
            }
        });
    }
    
    // URL input validation
    if (urlInput) {
        urlInput.addEventListener('blur', function() {
            const url = this.value.trim();
            if (url && !isValidUrl(url)) {
                showAlert('Please enter a valid URL.', 'warning');
                this.focus();
            }
        });
    }
    
    // Auto-resize textarea
    const textarea = document.getElementById('questions');
    if (textarea) {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
    
    // Show loading state
    function showLoadingState() {
        if (submitBtn && btnText && spinner) {
            submitBtn.disabled = true;
            btnText.textContent = 'Processing...';
            spinner.classList.remove('d-none');
        }
    }
    
    // Show alert message
    function showAlert(message, type = 'info') {
        // Remove existing alerts
        const existingAlerts = document.querySelectorAll('.alert.dynamic-alert');
        existingAlerts.forEach(alert => alert.remove());
        
        // Create new alert
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show dynamic-alert mt-3`;
        alertDiv.innerHTML = `
            <i data-feather="info" class="me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        // Insert after navigation
        const nav = document.querySelector('nav');
        if (nav) {
            nav.insertAdjacentElement('afterend', alertDiv);
            feather.replace();
            
            // Auto-dismiss after 5 seconds
            setTimeout(() => {
                if (alertDiv) {
                    alertDiv.remove();
                }
            }, 5000);
        }
    }
    
    // Validate URL format
    function isValidUrl(string) {
        try {
            const url = new URL(string);
            return url.protocol === "http:" || url.protocol === "https:";
        } catch (_) {
            return false;
        }
    }
    
    // Copy answer to clipboard
    window.copyAnswer = function(answerText) {
        navigator.clipboard.writeText(answerText).then(function() {
            showAlert('Answer copied to clipboard!', 'success');
        }, function() {
            showAlert('Failed to copy to clipboard.', 'danger');
        });
    };
    
    // Performance stats update (if on index page)
    if (document.querySelector('[data-stats]')) {
        updateStats();
        setInterval(updateStats, 30000); // Update every 30 seconds
    }
    
    function updateStats() {
        fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                if (!data.error) {
                    updateStatsDisplay(data);
                }
            })
            .catch(error => {
                console.log('Stats update failed:', error);
            });
    }
    
    function updateStatsDisplay(stats) {
        const statsElements = document.querySelectorAll('[data-stat]');
        statsElements.forEach(element => {
            const statType = element.getAttribute('data-stat');
            if (stats[statType] !== undefined) {
                element.textContent = formatStatValue(statType, stats[statType]);
            }
        });
    }
    
    function formatStatValue(type, value) {
        switch (type) {
            case 'total_requests':
            case 'total_questions':
            case 'cache_hits':
            case 'pdfs_processed':
                return value.toLocaleString();
            case 'avg_response_time':
                return value.toFixed(2) + 's';
            case 'total_extracted_chars':
                return (value / 1000000).toFixed(1) + 'M';
            default:
                return value;
        }
    }
});

// Question templates
const questionTemplates = {
    insurance: [
        "What types of coverage are included?",
        "What are the deductible amounts?",
        "What is excluded from coverage?",
        "What is the claims process?",
        "What are the policy limits?"
    ],
    legal: [
        "What are the key terms and conditions?",
        "What are the parties' obligations?",
        "What are the termination conditions?",
        "What is the governing law?",
        "What are the dispute resolution procedures?"
    ],
    hr: [
        "What are the eligibility requirements?",
        "What benefits are provided?",
        "What is the leave policy?",
        "What are the performance expectations?",
        "What is the disciplinary process?"
    ],
    technical: [
        "What are the system requirements?",
        "How do I install or configure this?",
        "What are the troubleshooting steps?",
        "What are the security considerations?",
        "How do I maintain or update this?"
    ]
};

// Add question template functionality
function addQuestionTemplate(category) {
    const textarea = document.getElementById('questions');
    if (!textarea) return;
    
    const templates = questionTemplates[category] || [];
    if (templates.length === 0) return;
    
    const currentText = textarea.value.trim();
    const newText = templates.join('\n');
    
    textarea.value = currentText ? currentText + '\n' + newText : newText;
    textarea.dispatchEvent(new Event('input')); // Trigger auto-resize
    textarea.focus();
}

// Export for global use
window.addQuestionTemplate = addQuestionTemplate;
