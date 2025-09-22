function sendMessage() {
    const userInput = document.getElementById('userInput').value;
    const lang = document.getElementById('language').value;
    const chatbox = document.getElementById('chatbox');
    
    if (userInput.trim() === '') return;
    
    chatbox.innerHTML += `<p><strong>You:</strong> ${userInput}</p>`;
    chatbox.scrollTop = chatbox.scrollHeight;
    
    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput, lang: lang })
    })
    .then(response => response.json())
    .then(data => {
        chatbox.innerHTML += `<p><strong>Bot:</strong> ${data.reply}</p>`;
        chatbox.scrollTop = chatbox.scrollHeight;
    });
    
    document.getElementById('userInput').value = '';
}

document.getElementById('reportForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const data = { 
        date: new Date().toISOString().split('T')[0], 
        symptom: formData.get('symptom'), 
        location: formData.get('location'), 
        count: parseInt(formData.get('count')) 
    };
    fetch('/report', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data) })
    .then(r => r.json()).then(d => alert(d.status));
});

document.getElementById('predictForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const features = [
        parseFloat(formData.get('diarrhea')),
        parseFloat(formData.get('typhoid')),
        parseFloat(formData.get('cholera')),
        parseFloat(formData.get('hepatitis'))
    ];
    fetch('/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({features: features}) })
    .then(r => r.json()).then(d => {
        document.getElementById('predResult').innerHTML = `<p>Risk: ${d.risk_level} (Prob: ${d.risk_prob.toFixed(2)})</p>`;
    });
});

function fetchHealth() {
    fetch('/api/health?district=Ferozepur').then(r => r.json()).then(d => {
        document.getElementById('healthData').innerHTML = `<pre>${JSON.stringify(d, null, 2)}\nSource: ${d[0].source}</pre>`;
    });
}

function fetchWater() {
    fetch('/api/water?district=Ferozepur').then(r => r.json()).then(d => {
        document.getElementById('waterData').innerHTML = `<pre>${JSON.stringify(d, null, 2)}\nSource: ${d.source}</pre>`;
    });
}