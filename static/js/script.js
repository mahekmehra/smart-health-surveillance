function appendMessage(role, text) {
    const chatbox = document.getElementById('chatbox');
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.textContent = text;
    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function setTyping(visible) {
    const t = document.getElementById('typing');
    if (!t) return;
    t.style.display = visible ? '' : 'none';
}

function sendMessage() {
    const inputEl = document.getElementById('userInput');
    const userInput = inputEl.value;
    const lang = document.getElementById('language').value;
    if (userInput.trim() === '') return;
    appendMessage('user', userInput);
    inputEl.value = '';
    setTyping(true);
    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userInput, lang: lang })
    })
    .then(response => response.json())
    .then(data => {
        setTyping(false);
        appendMessage('bot', data.reply || 'No response');
    })
    .catch(err => {
        setTyping(false);
        appendMessage('bot', `Error: ${err.message}`);
    });
}

const reportFormEl = document.getElementById('reportForm');
if (reportFormEl) {
    reportFormEl.addEventListener('submit', function(e) {
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
}

const predictFormEl = document.getElementById('predictForm');
if (predictFormEl) {
    predictFormEl.addEventListener('submit', function(e) {
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
}

// Chat: Enter to send
document.addEventListener('DOMContentLoaded', () => {
    const inputEl = document.getElementById('userInput');
    if (inputEl) {
        inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
    }
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

// Table interactions: sorting and filtering
(function() {
	function sortTable(table, columnIndex, type, asc) {
		const tbody = table.tBodies[0];
		const rows = Array.from(tbody.querySelectorAll('tr'));
		const dir = asc ? 1 : -1;
		rows.sort((a, b) => {
			const av = a.children[columnIndex].innerText.trim();
			const bv = b.children[columnIndex].innerText.trim();
			if (type === 'number') {
				const an = parseFloat(av.replace(/[,\s]/g, '')) || 0;
				const bn = parseFloat(bv.replace(/[,\s]/g, '')) || 0;
				return (an - bn) * dir;
			}
			return av.localeCompare(bv) * dir;
		});
		rows.forEach(r => tbody.appendChild(r));
	}

	function attachSorting(tableId) {
		const table = document.getElementById(tableId);
		if (!table) return;
		const headers = table.querySelectorAll('thead th');
		headers.forEach((th, idx) => {
			let asc = true;
			th.addEventListener('click', () => {
				sortTable(table, idx, th.getAttribute('data-sort') || 'text', asc);
				asc = !asc;
			});
		});
	}

	function attachFilter(inputId, tableId) {
		const input = document.getElementById(inputId);
		const table = document.getElementById(tableId);
		if (!input || !table) return;
		input.addEventListener('input', () => {
			const q = input.value.toLowerCase();
			const rows = table.tBodies[0].querySelectorAll('tr');
			rows.forEach(row => {
				const text = row.innerText.toLowerCase();
				row.style.display = text.includes(q) ? '' : 'none';
			});
		});
	}

	document.addEventListener('DOMContentLoaded', () => {
		attachSorting('table-flood');
		attachSorting('table-hotspots');
		attachFilter('filter-flood', 'table-flood');
		attachFilter('filter-hotspots', 'table-hotspots');

		// Theme toggle with persistence and OS preference sync
		const toggle = document.getElementById('theme-toggle');
		const root = document.documentElement;
		const saved = localStorage.getItem('theme');
		const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
		const apply = (mode) => {
			if (mode === 'dark') root.classList.add('dark'); else root.classList.remove('dark');
			localStorage.setItem('theme', mode);
			root.setAttribute('data-theme', mode);
			const isLight = mode === 'light';
			document.getElementById('sun-icon')?.classList.toggle('hidden', !isLight);
			document.getElementById('moon-icon')?.classList.toggle('hidden', isLight);
		};
		apply(saved || (prefersLight ? 'light' : 'dark'));
		if (toggle) {
				toggle.addEventListener('click', () => {
					apply(root.classList.contains('dark') ? 'light' : 'dark');
				});
		}
		if (window.matchMedia) {
			window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
				if (!localStorage.getItem('theme')) apply(e.matches ? 'light' : 'dark');
			});
		}
	});
})();