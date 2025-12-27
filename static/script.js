// API base URL - use relative paths for same-origin requests
const API_BASE = '';

// DOM elements
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const uploadContent = document.getElementById('uploadContent');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const removeImage = document.getElementById('removeImage');
const modelSelect = document.getElementById('modelSelect');
const predictBtn = document.getElementById('predictBtn');
const predictBtnText = document.getElementById('predictBtnText');
const predictBtnLoader = document.getElementById('predictBtnLoader');
const errorMessage = document.getElementById('errorMessage');
const results = document.getElementById('results');
const modelName = document.getElementById('modelName');
const top1Label = document.getElementById('top1Label');
const top1Confidence = document.getElementById('top1Confidence');
const top1Progress = document.getElementById('top1Progress');
const top5List = document.getElementById('top5List');

let selectedFile = null;

// Upload box click handler
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// Drag and drop handlers
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#f0f2ff';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Handle file selection
function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file (PNG, JPG, JPEG)');
        return;
    }
    
    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        imagePreview.style.display = 'block';
        uploadContent.style.display = 'none';
        predictBtn.disabled = false;
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

// Remove image handler
removeImage.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.style.display = 'none';
    uploadContent.style.display = 'block';
    predictBtn.disabled = true;
    hideError();
    hideResults();
});

// Predict button handler
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    const selectedModel = modelSelect.value;
    await predictImage(selectedFile, selectedModel);
});

// Predict image
async function predictImage(file, model) {
    // Show loading state
    setLoading(true);
    hideError();
    hideResults();
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', file);
        
        // Make API request - use relative path
        const url = `/predict?model=${model}`;
        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        showError(error.message || 'An error occurred during prediction');
    } finally {
        setLoading(false);
    }
}

// Display results
function displayResults(data) {
    // Set model name
    const modelDisplayName = data.model === 'resnet' ? 'ResNet18-SE' : 'Vision Transformer (ViT)';
    modelName.textContent = modelDisplayName;
    
    // Display top-1
    top1Label.textContent = data.top1.label;
    const top1Conf = (data.top1.confidence * 100).toFixed(2);
    top1Confidence.textContent = `${top1Conf}%`;
    top1Progress.style.width = `${data.top1.confidence * 100}%`;
    top1Progress.textContent = `${top1Conf}%`;
    
    // Display top-5
    top5List.innerHTML = '';
    data.top5.forEach((item, index) => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'top5-item';
        
        const labelSpan = document.createElement('span');
        labelSpan.className = 'label';
        labelSpan.textContent = `${index + 1}. ${item.label}`;
        
        const confSpan = document.createElement('span');
        confSpan.className = 'confidence';
        confSpan.textContent = `${(item.confidence * 100).toFixed(2)}%`;
        
        itemDiv.appendChild(labelSpan);
        itemDiv.appendChild(confSpan);
        top5List.appendChild(itemDiv);
    });
    
    // Show results
    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Set loading state
function setLoading(loading) {
    predictBtn.disabled = loading;
    if (loading) {
        predictBtnText.textContent = 'Running...';
        predictBtnLoader.style.display = 'block';
    } else {
        predictBtnText.textContent = 'Predict';
        predictBtnLoader.style.display = 'none';
    }
}

// Show error
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

// Hide error
function hideError() {
    errorMessage.style.display = 'none';
}

// Hide results
function hideResults() {
    results.style.display = 'none';
}

