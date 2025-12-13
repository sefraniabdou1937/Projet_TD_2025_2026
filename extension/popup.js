document.addEventListener('DOMContentLoaded', function() {
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        let activeTab = tabs[0];
        
        // --- NETTOYAGE DU TITRE ---
        // On prend le titre et on coupe tout ce qui vient après un "-" ou un "|"
        let rawTitle = activeTab.title;
        let journalName = rawTitle.split('|')[0].split('-')[0].trim();

        // Appel à votre API
        fetch(`http://127.0.0.1:8000/predict?name=${encodeURIComponent(journalName)}`)
        .then(response => response.json())
        .then(data => {
            let div = document.getElementById('result');
            if (data.is_predatory) {
                div.innerHTML = `
                    <h2 style="color:red; margin: 0;">⚠️ DANGER</h2>
                    <p>Risque : <b>${data.risk_score}%</b></p>
                    <hr>
                    <small>Éditeur : ${data.details.Publisher}</small>
                `;
            } else {
                div.innerHTML = `
                    <h2 style="color:green; margin: 0;">✅ FIABLE</h2>
                    <p>Risque : <b>${data.risk_score}%</b></p>
                    <hr>
                    <small>Citations : ${data.details.oa_cited}</small>
                `;
            }
        })
        .catch(error => {
            document.getElementById('result').innerText = "Erreur : API non détectée. Vérifiez que api.py tourne.";
        });
    });
});