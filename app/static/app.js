const API_URL = 'http://127.0.0.1:8000';

// Global variables
let messages = [
    {role: 'assistant', content: "Hi! I'm your AI assistant. Upload files by clicking the 📎 button or drag & drop them here, then ask me anything! 🤖"}
];
let recentDocuments = [];
let selectedDocument = null;
let selectedFileForConversion = null;
let dragCounter = 0;

// Persistent session ID — generated once per page load so memory works across queries
let SESSION_ID = sessionStorage.getItem('rag_session_id');
if (!SESSION_ID) {
    SESSION_ID = 'sess_' + Math.random().toString(36).substr(2, 16);
    sessionStorage.setItem('rag_session_id', SESSION_ID);
}
console.log('Session ID:', SESSION_ID);

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showNotification(message, type = 'info', duration = 4000) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideInNotification 0.4s ease-out reverse';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 400);
    }, duration);
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    switch (ext) {
        case 'pdf': return 'fa-file-pdf';
        case 'doc': case 'docx': return 'fa-file-word';
        case 'jpg': case 'jpeg': case 'png': case 'gif': return 'fa-file-image';
        case 'mp3': case 'wav': case 'ogg': case 'm4a': return 'fa-file-audio';
        case 'xlsx': case 'xls': return 'fa-file-excel';
        default: return 'fa-file';
    }
}

function showQuickAction(action) {
    const chatInput = document.getElementById('chat-input');
    
    switch(action) {
        case 'convert':
            chatInput.value = 'I want to convert a file to a different format';
            chatInput.focus();
            showNotification('💡 Upload a file first, then I\'ll help you convert it!', 'info');
            break;
        case 'compare':
            chatInput.value = 'I want to compare two documents';
            chatInput.focus();
            showNotification('💡 Upload two files, then I\'ll help you compare them!', 'info');
            break;
        case 'docs':
            showDocumentLibrary();
            break;
    }
}

function showDocumentLibrary() {
    console.log('showDocumentLibrary called');
    console.log('recentDocuments:', recentDocuments);
    
    // Check if we can add messages to the chat
    const chatContainer = document.getElementById('chat-container');
    if (!chatContainer) {
        console.error('Chat container not found!');
        alert('Chat container not found! Please refresh the page.');
        return;
    }
    
    if (recentDocuments.length === 0) {
        addMessage('assistant', '📂 Your document library is empty. Upload some files first by clicking the 📎 button or dragging files here!');
        // Also try to fetch documents from backend immediately
        fetchDocumentsFromBackend();
        return;
    }
    
    let libraryMessage = '📂 **Your Document Library:**\n\n';
    recentDocuments.slice(-10).reverse().forEach((doc, index) => {
        const uploadDate = new Date(doc.uploadedAt).toLocaleDateString();
        libraryMessage += `${index + 1}. **${doc.name}** (${doc.size}) - uploaded ${uploadDate}\n`;
    });
    libraryMessage += '\nYou can ask me questions about any of these documents!';
    
    addMessage('assistant', libraryMessage);
    
    // Also try to fetch documents from backend
    fetchDocumentsFromBackend();
}

function fetchDocumentsFromBackend() {
    fetch(`${API_URL}/docs/list?session_id=${encodeURIComponent(SESSION_ID)}`)
        .then(response => response.json())
        .then(data => {
            const docIds = data.doc_ids || [];
            
            if (docIds.length > 0) {
                let backendDocsMessage = '\n\n🗄️ **Documents in System:**\n\n';
                docIds.slice(0, 20).forEach((docId, index) => {
                    backendDocsMessage += `${index + 1}. ${docId}\n`;
                });
                if (docIds.length > 20) {
                    backendDocsMessage += `\n... and ${docIds.length - 20} more documents`;
                }
                addMessage('assistant', backendDocsMessage);
            }
        })
        .catch(error => {
            console.log('Could not fetch backend documents:', error);
        });
}

function showTips() {
    const tips = [
        '💡 **Upload files:** Click the 📎 button or drag & drop files anywhere in the chat',
        '🔍 **Ask questions:** Once uploaded, ask me anything about your documents',
        '🔄 **Convert files:** Upload a file and say "convert this to PDF" or "convert to Word"',
        '⚖️ **Compare documents:** Upload two files and ask me to compare them',
        '🎤 **Audio files:** Upload audio files and I\'ll transcribe them automatically',
        '🔗 **Web content:** Paste URLs and I\'ll extract and analyze the content'
    ];
    const tipsMessage = '✨ **Here\'s what I can help you with:**\n\n' + tips.join('\n\n');
    addMessage('assistant', tipsMessage);
}

function addFileAttachmentMessage(files) {
    const fileList = Array.from(files);
    let attachmentHTML = '';
    
    fileList.forEach(file => {
        attachmentHTML += `
            <div class="file-attachment">
                <div class="file-icon">
                    <i class="fas ${getFileIcon(file.name)}"></i>
                </div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
                <div class="file-actions">
                    <button class="file-action-btn" onclick="convertFile('${file.name}')">
                        <i class="fas fa-exchange-alt"></i> Convert
                    </button>
                </div>
            </div>
        `;
    });
    
    const message = `📎 **Uploaded ${fileList.length} file(s):**${attachmentHTML}`;
    addMessage('user', message);
}

function convertFile(filename) {
    const chatInput = document.getElementById('chat-input');
    chatInput.value = `Convert ${filename} to PDF`;
    chatInput.focus();
}

function removeFileFromPreview(index) {
    const fileInput = document.getElementById('file-input');
    const dt = new DataTransfer();
    const files = Array.from(fileInput.files);
    
    files.splice(index, 1);
    files.forEach(file => dt.items.add(file));
    fileInput.files = dt.files;
    
    displayFilePreview(files);
}

function displaySelectedFile(file) {
    const container = document.getElementById('selected-file-display');
    container.innerHTML = `
        <div class="file-preview">
            <div class="file-icon">
                <i class="fas ${getFileIcon(file.name)}"></i>
            </div>
            <div class="file-details">
                <div class="file-name">📎 Selected: ${file.name}</div>
                <div class="file-size">${formatFileSize(file.size)}</div>
            </div>
            <button class="remove-file" onclick="clearSelectedFile()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
}

function clearSelectedFile() {
    selectedFileForConversion = null;
    document.getElementById('selected-file-display').innerHTML = '';
    document.getElementById('convert-file-input').value = '';
    updateConvertButton();
}

function updateConvertButton() {
    const button = document.getElementById('convert-button');
    button.disabled = !selectedFileForConversion && !selectedDocument;
}

function displayLibraryDocuments() {
    const container = document.getElementById('library-documents');
    
    if (recentDocuments.length === 0) {
        container.innerHTML = `
            <div style="text-align: center; padding: 40px; color: rgba(255,255,255,0.6);">
                <i class="fas fa-folder-open" style="font-size: 2rem; margin-bottom: 10px;"></i>
                <div>No documents uploaded yet</div>
                <div style="font-size: 0.9rem; margin-top: 5px;">Upload some files first to see them here</div>
            </div>
        `;
        return;
    }
    
    container.innerHTML = recentDocuments.map((doc, index) => `
        <div class="document-card ${selectedDocument === doc ? 'selected' : ''}" onclick="selectDocument(${index})">
            <div class="document-header">
                <div class="document-title">
                    <i class="fas ${getFileIcon(doc.name)}"></i>
                    ${doc.name}
                </div>
                <div class="document-type">${doc.type || 'Unknown'}</div>
            </div>
            <div class="document-info">
                <span><i class="fas fa-calendar"></i> ${new Date(doc.uploadedAt).toLocaleDateString()}</span>
                <span><i class="fas fa-weight"></i> ${doc.size}</span>
            </div>
        </div>
    `).join('');
}

function selectDocument(index) {
    selectedDocument = recentDocuments[index];
    selectedFileForConversion = null;
    document.getElementById('selected-file-display').innerHTML = `
        <div class="file-preview">
            <div class="file-icon">
                <i class="fas ${getFileIcon(selectedDocument.name)}"></i>
            </div>
            <div class="file-details">
                <div class="file-name">📎 Selected from Library: ${selectedDocument.name}</div>
                <div class="file-size">${selectedDocument.size}</div>
            </div>
            <button class="remove-file" onclick="clearSelectedDocument()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    displayLibraryDocuments();
    updateConvertButton();
}

function clearSelectedDocument() {
    selectedDocument = null;
    document.getElementById('selected-file-display').innerHTML = '';
    displayLibraryDocuments();
    updateConvertButton();
}

function updateRecentDocuments() {
    const container = document.getElementById('recent-uploads');
    
    if (recentDocuments.length === 0) {
        container.innerHTML = `
            <div style="text-align: center; padding: 30px; color: rgba(255,255,255,0.5);">
                <i class="fas fa-cloud-upload-alt" style="font-size: 1.5rem; margin-bottom: 8px;"></i>
                <div>No documents uploaded yet</div>
            </div>
        `;
        return;
    }
    
    const recent = recentDocuments.slice(-3).reverse();
    container.innerHTML = recent.map((doc, index) => `
        <div class="document-card">
            <div class="document-header">
                <div class="document-title">
                    <i class="fas ${getFileIcon(doc.name)}"></i>
                    ${doc.name}
                </div>
                <div class="document-type">${doc.type || 'Document'}</div>
            </div>
            <div class="document-info">
                <span><i class="fas fa-clock"></i> ${new Date(doc.uploadedAt).toLocaleString()}</span>
                <span><i class="fas fa-weight"></i> ${doc.size}</span>
            </div>
            <div class="document-actions">
                <button class="action-btn" onclick="queryDocument('${doc.name}')">
                    <i class="fas fa-search"></i> Query
                </button>
                <button class="action-btn" onclick="convertFromLibrary('${doc.name}')">
                    <i class="fas fa-exchange-alt"></i> Convert
                </button>
            </div>
        </div>
    `).join('');
}

function queryDocument(filename) {
    showTab('chat');
    document.getElementById('chat-input').value = `Tell me about ${filename}`;
    document.getElementById('chat-input').focus();
    showNotification(`Ready to query ${filename}`, 'info');
}

function convertFromLibrary(filename) {
    showTab('convert');
    const doc = recentDocuments.find(d => d.name === filename);
    if (doc) {
        selectedDocument = doc;
        selectedFileForConversion = null;
        document.getElementById('selected-file-display').innerHTML = `
            <div class="file-preview">
                <div class="file-icon">
                    <i class="fas ${getFileIcon(doc.name)}"></i>
                </div>
                <div class="file-details">
                    <div class="file-name">📎 Selected: ${doc.name}</div>
                    <div class="file-size">${doc.size}</div>
                </div>
                <button class="remove-file" onclick="clearSelectedDocument()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        updateConvertButton();
        showNotification(`${filename} selected for conversion`, 'success');
    }
}

// Navigation
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.nav-pill').forEach(pill => {
        pill.addEventListener('click', (e) => {
            const section = e.target.getAttribute('data-section');
            showTab(section);
        });
    });

// Chat functionality
const chatContainer = document.getElementById('chat-container');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const ttsEnabled = document.getElementById('tts-enabled');

function addMessage(role, content) {
    console.log('addMessage called with:', role, content);
    
    // Get chat container fresh each time
    const chatContainer = document.getElementById('chat-container');
    if (!chatContainer) {
        console.error('Chat container not found in addMessage!');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}-message`;
    
    // Convert simple markdown-like formatting to HTML
    let formattedContent = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // **bold**
        .replace(/\n/g, '<br>')  // line breaks
        .replace(/(\d+)\. /g, '<br>$1. ');  // numbered lists
    
    messageDiv.innerHTML = `<div class="message-bubble">${formattedContent}</div>`;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message
    addMessage('user', message);
    chatInput.value = '';

    // Show loading
    sendButton.disabled = true;
    sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Thinking...';

    // Send to API — always include session_id so memory works
    fetch(`${API_URL}/query/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: message,
            session_id: SESSION_ID,
            tts: ttsEnabled.checked,
            top_k: 5
        })
    })
    .then(response => response.json())
    .then(data => {
        const answer = data.answer || 'Sorry, I could not generate a response.';
        addMessage('assistant', answer);

        // Show source indicator
        if (data.fallback_used) {
            addMessage('assistant', '🌐 _Answer sourced from web search (no matching document found)_');
        }

        if (ttsEnabled.checked && data.tts_audio_path) {
            const audio = new Audio(`${API_URL}${data.tts_audio_path}`);
            audio.play().catch(e => console.log('Audio play failed:', e));
        }
    })
    .catch(error => {
        addMessage('assistant', `⚠️ Error: ${error.message}. Is the server running?`);
    })
    .finally(() => {
        sendButton.disabled = false;
        sendButton.innerHTML = '<i class="fas fa-paper-plane"></i> Send';
    });
}

sendButton.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

    // File upload functionality
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const urlInput = document.getElementById('url-input');
    const uploadUrlButton = document.getElementById('upload-url-button');

    // Full page drag and drop
    const dropZone = document.getElementById('upload-drop-zone');
    const body = document.body;
    
    body.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dragCounter++;
        if (dropZone) {
            dropZone.style.display = 'flex';
            console.log('Showing drop zone');
        }
    });
    
    body.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragCounter--;
        if (dragCounter === 0 && dropZone) {
            dropZone.style.display = 'none';
            console.log('Hiding drop zone');
        }
    });
    
    body.addEventListener('dragover', (e) => {
        e.preventDefault();
    });
    
    body.addEventListener('drop', (e) => {
        e.preventDefault();
        dragCounter = 0;
        dropZone.style.display = 'none';
        
        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            addFileAttachmentMessage(files);
            handleFileUpload(files);
        }
    });

    // Attachment button
    const attachmentButton = document.getElementById('attachment-button');
    if (attachmentButton) {
        attachmentButton.addEventListener('click', () => {
            fileInput.click();
            showNotification('📎 Select files to upload...', 'info', 2000);
        });
    } else {
        console.error('Attachment button not found!');
    }

    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            console.log('Files selected:', files.length);
            addFileAttachmentMessage(files);
            handleFileUpload(files);
            showNotification(`📎 Selected ${files.length} file(s) for upload`, 'info');
        }
        // Reset input so same file can be selected again
        e.target.value = '';
    });

    function handleFileUpload(files) {
        if (files.length === 0) return;

        // Upload each file individually using the 'file' key (matches FastAPI's UploadFile)
        const uploadPromises = files.map(file => {
            const formData = new FormData();
            formData.append('file', file);  // singular 'file' key — matches backend
            formData.append('session_id', SESSION_ID);
            return fetch(`${API_URL}/upload/`, {
                method: 'POST',
                body: formData
            }).then(r => r.json());
        });

        // Show uploading state
        addMessage('assistant', `⏳ Uploading and processing ${files.length} file(s)…`);

        Promise.all(uploadPromises)
        .then(results => {
            let totalChunks = 0;
            results.forEach((data, idx) => {
                const file = files[idx];
                totalChunks += data.chunk_count || 0;
                recentDocuments.push({
                    name: file.name,
                    size: formatFileSize(file.size),
                    type: file.type || 'Unknown',
                    doc_id: data.doc_id || file.name,
                    uploadedAt: new Date().toISOString(),
                });
            });

            const successMsg = `✅ Processed ${files.length} file(s) → **${totalChunks} chunks** indexed. You can now ask questions about your document(s)!`;
            addMessage('assistant', successMsg);
            showNotification(`${files.length} file(s) uploaded successfully`, 'success');
            sessionStorage.setItem('recentDocuments', JSON.stringify(recentDocuments));
        })
        .catch(error => {
            addMessage('assistant', `❌ Upload failed: ${error.message}`);
            showNotification(`Upload failed: ${error.message}`, 'error');
        });
    }

    // URL upload
    uploadUrlButton.addEventListener('click', () => {
        const url = urlInput.value.trim();
        if (!url) return;

        uploadStatus.innerHTML = '<div class="status-info"><i class="fas fa-spinner fa-spin"></i> Processing URL...</div>';

        fetch(`${API_URL}/upload/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `input_text=${encodeURIComponent(url)}&session_id=${encodeURIComponent(SESSION_ID)}`
        })
        .then(response => response.json())
        .then(data => {
            uploadStatus.innerHTML = `<div class="status-success"><i class="fas fa-check"></i> URL processed! Extracted ${data.chunks_created || 'some'} chunks.</div>`;
            urlInput.value = '';
            showNotification('URL processed successfully!', 'success');
        })
        .catch(error => {
            uploadStatus.innerHTML = `<div class="status-error"><i class="fas fa-times"></i> URL processing failed: ${error.message}</div>`;
            showNotification(`URL processing failed: ${error.message}`, 'error');
        });
    });

    // Convert functionality
    const convertFileInput = document.getElementById('convert-file-input');
    const convertButton = document.getElementById('convert-button');
    const outputFormat = document.getElementById('output-format');
    const convertStatus = document.getElementById('convert-status');

    // Convert section buttons
    document.getElementById('upload-new-convert').addEventListener('click', () => {
        document.getElementById('convert-upload-area').style.display = 'block';
        document.getElementById('document-library').style.display = 'none';
        convertFileInput.click();
    });

    document.getElementById('select-from-library').addEventListener('click', () => {
        document.getElementById('convert-upload-area').style.display = 'none';
        document.getElementById('document-library').style.display = 'block';
        displayLibraryDocuments();
    });

    convertFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            selectedFileForConversion = file;
            selectedDocument = null;
            displaySelectedFile(file);
            updateConvertButton();
        }
    });

    convertButton.addEventListener('click', () => {
        const outputFormatValue = outputFormat.value;
        
        let fileToConvert = null;
        let fileName = '';
        
        if (selectedFileForConversion) {
            fileToConvert = selectedFileForConversion;
            fileName = selectedFileForConversion.name;
        } else if (selectedDocument) {
            showNotification('Converting from library requires backend enhancement', 'info');
            return;
        } else {
            convertStatus.innerHTML = '<div class="status-error"><i class="fas fa-exclamation"></i> Please select a file to convert.</div>';
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileToConvert);
        formData.append('output_format', outputFormatValue);

        // Show progress
        convertStatus.innerHTML = `
            <div class="upload-progress">
                <div class="upload-progress-bar" style="width: 70%;"></div>
            </div>
            <div style="margin-top: 10px; color: white;">Converting ${fileName} to ${outputFormatValue.toUpperCase()}...</div>
        `;

        fetch(`${API_URL}/convert/`, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                return response.blob().then(blob => ({
                    blob,
                    filename: `${fileName.split('.')[0]}_converted.${outputFormatValue}`
                }));
            }
            throw new Error('Conversion failed');
        })
        .then(({blob, filename}) => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            window.URL.revokeObjectURL(url);
            
            convertStatus.innerHTML = `<div class="status-success"><i class="fas fa-check"></i> File converted successfully! 🎉 Downloaded as ${filename}</div>`;
            showNotification(`${fileName} converted to ${outputFormatValue.toUpperCase()} successfully!`, 'success');
            
            // Clear selection
            clearSelectedFile();
            clearSelectedDocument();
        })
        .catch(error => {
            convertStatus.innerHTML = `<div class="status-error"><i class="fas fa-times"></i> Conversion failed: ${error.message}</div>`;
            showNotification(`Conversion failed: ${error.message}`, 'error');
        });
    });

// Compare functionality
const compareFile1Input = document.getElementById('compare-file1');
const compareFile2Input = document.getElementById('compare-file2');
const compareButton = document.getElementById('compare-button');
const compareResults = document.getElementById('compare-results');

compareButton.addEventListener('click', () => {
    const file1 = compareFile1Input.files[0];
    const file2 = compareFile2Input.files[0];

    if (!file1 || !file2) {
        compareResults.innerHTML = '<div class="status-error"><i class="fas fa-exclamation"></i> Please select both documents to compare.</div>';
        return;
    }

    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);

    compareResults.innerHTML = '<div class="status-info"><i class="fas fa-spinner fa-spin"></i> Comparing documents...</div>';

    fetch(`${API_URL}/compare/`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        compareResults.innerHTML = `
            <div class="status-success">
                <i class="fas fa-check"></i> Comparison completed!
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 12px; margin-top: 15px; color: white; line-height: 1.6;">
                ${data.result || 'No comparison results available.'}
            </div>
        `;
    })
    .catch(error => {
        compareResults.innerHTML = `<div class="status-error"><i class="fas fa-times"></i> Comparison failed: ${error.message}</div>`;
    });
});

    // Audio functionality
    const audioUploadArea = document.getElementById('audio-upload-area');
    const audioFileInput = document.getElementById('audio-file-input');
    const transcribeButton = document.getElementById('transcribe-button');
    const transcribeStatus = document.getElementById('transcribe-status');

    audioUploadArea.addEventListener('click', () => {
        audioFileInput.click();
    });

    transcribeButton.addEventListener('click', () => {
        const file = audioFileInput.files[0];
        if (!file) {
            transcribeStatus.innerHTML = '<div class="status-error"><i class="fas fa-exclamation"></i> Please select an audio file.</div>';
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', SESSION_ID);

        transcribeStatus.innerHTML = '<div class="status-info"><i class="fas fa-spinner fa-spin"></i> Transcribing audio...</div>';

        fetch(`${API_URL}/upload/`, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            transcribeStatus.innerHTML = `<div class="status-success"><i class="fas fa-check"></i> Audio transcribed and added to knowledge base!</div>`;
            
            // Add audio file to recent documents
            recentDocuments.push({
                name: file.name,
                size: formatFileSize(file.size),
                type: 'Audio',
                uploadedAt: new Date().toISOString(),
                path: null
            });
            updateRecentDocuments();
            sessionStorage.setItem('recentDocuments', JSON.stringify(recentDocuments));
            
            showNotification('Audio transcribed successfully!', 'success');
        })
        .catch(error => {
            transcribeStatus.innerHTML = `<div class="status-error"><i class="fas fa-times"></i> Transcription failed: ${error.message}</div>`;
            showNotification(`Transcription failed: ${error.message}`, 'error');
        });
    });

// Load documents
function loadDocuments() {
    const documentList = document.getElementById('document-list');
    documentList.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading documents<span class="loading-dots"></span></div>';

    fetch(`${API_URL}/docs/list?session_id=${encodeURIComponent(SESSION_ID)}`)
    .then(response => response.json())
    .then(data => {
        const docIds = data.doc_ids || [];
        
        if (docIds.length === 0) {
            documentList.innerHTML = `
                <div class="status-info">
                    <i class="fas fa-inbox"></i>
                    No documents uploaded yet. Use the Upload tab to add some!
                </div>
            `;
        } else {
            documentList.innerHTML = `
                <div style="color: white; font-weight: 600; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
                    <i class="fas fa-folder-open"></i>
                    ${docIds.length} documents in your library:
                </div>
            `;
            
            docIds.forEach((docId, index) => {
                const item = document.createElement('div');
                item.className = 'document-item';
                item.innerHTML = `
                    <i class="fas fa-file-alt document-icon"></i>
                    <span class="document-name">${docId}</span>
                `;
                documentList.appendChild(item);
            });
        }
    })
    .catch(error => {
        documentList.innerHTML = `<div class="status-error"><i class="fas fa-times"></i> Failed to load documents: ${error.message}</div>`;
    });
}

    // Load documents function
    function loadDocuments() {
        const documentList = document.getElementById('document-list');
        documentList.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading documents<span class="loading-dots"></span></div>';

        fetch(`${API_URL}/docs/list?session_id=${encodeURIComponent(SESSION_ID)}`)
        .then(response => response.json())
        .then(data => {
            const docIds = data.doc_ids || [];
            
            if (docIds.length === 0) {
                documentList.innerHTML = `
                    <div class="status-info">
                        <i class="fas fa-inbox"></i>
                        No documents uploaded yet. Use the Upload tab to add some!
                    </div>
                `;
            } else {
                documentList.innerHTML = `
                    <div style="color: white; font-weight: 600; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
                        <i class="fas fa-folder-open"></i>
                        ${docIds.length} documents in your library:
                    </div>
                `;
                
                docIds.forEach((docId, index) => {
                    const item = document.createElement('div');
                    item.className = 'document-item';
                    item.innerHTML = `
                        <i class="fas fa-file-alt document-icon"></i>
                        <span class="document-name">${docId}</span>
                    `;
                    documentList.appendChild(item);
                });
            }
        })
        .catch(error => {
            documentList.innerHTML = `<div class="status-error"><i class="fas fa-times"></i> Failed to load documents: ${error.message}</div>`;
        });
    }

    // Initialize unified interface
    chatInput.focus();
    
    // Load documents from localStorage if available
    const savedDocs = localStorage.getItem('recentDocuments');
    if (savedDocs) {
        try {
            recentDocuments = JSON.parse(savedDocs);
            updateRecentDocuments();
        } catch (e) {
            console.log('No saved documents found');
        }
    }

    // Add welcome message with upload instructions
    if (document.getElementById('chat-container').children.length === 0) {
        addMessage('assistant', "👋 Hi! I'm your AI assistant. Here's how to get started:<br><br>📎 <strong>Upload files:</strong> Click the blue 📎 button below or drag & drop files anywhere<br>💬 <strong>Ask questions:</strong> Once uploaded, ask me anything about your documents<br>⚡ <strong>Quick actions:</strong> Use the buttons below for convert, compare, or view docs");
    }
});

// Make functions globally accessible via window object
window.showQuickAction = function(action) {
    console.log('Window.showQuickAction called with:', action);
    const chatInput = document.getElementById('chat-input');
    
    switch(action) {
        case 'convert':
            chatInput.value = 'I want to convert a file to a different format';
            chatInput.focus();
            showNotification('💡 Upload a file first, then I\'ll help you convert it!', 'info');
            break;
        case 'compare':
            chatInput.value = 'I want to compare two documents';
            chatInput.focus();
            showNotification('💡 Upload two files, then I\'ll help you compare them!', 'info');
            break;
        case 'docs':
            console.log('Calling showDocumentLibrary from window.showQuickAction');
            window.showDocumentLibrary();
            break;
    }
};

window.showDocumentLibrary = function() {
    console.log('Window.showDocumentLibrary called');
    console.log('recentDocuments:', recentDocuments);
    
    // Check if we can add messages to the chat
    const chatContainer = document.getElementById('chat-container');
    if (!chatContainer) {
        console.error('Chat container not found!');
        alert('Chat container not found! Please refresh the page.');
        return;
    }
    
    if (recentDocuments.length === 0) {
        window.addMessage('assistant', '📂 Your document library is empty. Upload some files first by clicking the 📎 button or dragging files here!');
        // Also try to fetch documents from backend immediately
        fetchDocumentsFromBackend();
        return;
    }
    
    let libraryMessage = '📂 **Your Document Library:**\n\n';
    recentDocuments.slice(-10).reverse().forEach((doc, index) => {
        const uploadDate = new Date(doc.uploadedAt).toLocaleDateString();
        libraryMessage += `${index + 1}. **${doc.name}** (${doc.size}) - uploaded ${uploadDate}\n`;
    });
    libraryMessage += '\nYou can ask me questions about any of these documents!';
    
    window.addMessage('assistant', libraryMessage);
};

window.addMessage = function(role, content) {
    console.log('Window.addMessage called with:', role, content);
    
    // Get chat container fresh each time
    const chatContainer = document.getElementById('chat-container');
    if (!chatContainer) {
        console.error('Chat container not found in addMessage!');
        return;
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}-message`;
    
    // Convert simple markdown-like formatting to HTML
    let formattedContent = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // **bold**
        .replace(/\n/g, '<br>')  // line breaks
        .replace(/(\d+)\. /g, '<br>$1. ');  // numbered lists
    
    messageDiv.innerHTML = `<div class="message-bubble">${formattedContent}</div>`;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
};

window.showTips = function() {
    const tips = [
        '💡 **Upload files:** Click the 📎 button or drag & drop files anywhere in the chat',
        '🔍 **Ask questions:** Once uploaded, ask me anything about your documents',
        '🔄 **Convert files:** Upload a file and say "convert this to PDF" or "convert to Word"',
        '⚖️ **Compare documents:** Upload two files and ask me to compare them',
        '🎤 **Audio files:** Upload audio files and I\'ll transcribe them automatically',
        '🔗 **Web content:** Paste URLs and I\'ll extract and analyze the content'
    ];
    
    const tipsMessage = '✨ **Here\'s what I can help you with:**\n\n' + tips.join('\n\n');
    window.addMessage('assistant', tipsMessage);
};

window.showNotification = showNotification;

// Debug logging
console.log('JavaScript loaded successfully!');
console.log('Functions available:', {
    showQuickAction: typeof showQuickAction,
    showTips: typeof showTips,
    showDocumentLibrary: typeof showDocumentLibrary
});