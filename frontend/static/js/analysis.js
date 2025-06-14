// In static/js/analysis.js
function analyzeText() {
    const text = document.getElementById('textInput').value;
    const mode = document.getElementById('analysisMode').value;
    
    let endpoint = '/api/analyze';
    let options = {};
    
    if (mode === 'enhanced' || mode === 'cot_only') {
        endpoint = '/api/analyze/cot';
        options = {
            cot_mode: mode === 'cot_only' ? 'standalone' : 'enhanced'
        };
    }
    
    fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text, ...options})
    })
    .then(response => response.json())
    .then(data => {
        displayResults(data);
        if (data.analysis.chain_of_thought) {
            displayCoTResults(data.analysis.chain_of_thought);
        }
    });
}

function displayCoTResults(cotData) {
    const cotSection = document.getElementById('cotResults');
    const stepsContainer = document.getElementById('reasoningSteps');
    
    if (cotData.reasoning_chain && cotData.reasoning_chain.reasoning_steps) {
        let stepsHtml = '';
        cotData.reasoning_chain.reasoning_steps.forEach((step, index) => {
            stepsHtml += `
                <div class="card mb-2">
                    <div class="card-header">
                        <h6>Step ${index + 1}: ${step.step.replace('_', ' ').toUpperCase()}</h6>
                        <small>Confidence: ${(step.confidence * 100).toFixed(1)}%</small>
                    </div>
                    <div class="card-body">
                        <p>${step.reasoning}</p>
                        <strong>Key Findings:</strong>
                        <ul>
                            ${step.findings.slice(0, 3).map(f => `<li>${f}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            `;
        });
        stepsContainer.innerHTML = stepsHtml;
        cotSection.style.display = 'block';
    }
}