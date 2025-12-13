document.addEventListener('DOMContentLoaded', function() {
    // 1. On récupère le titre de la page active (Le nom de la revue)
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        let activeTab = tabs[0];
        let journalName = activeTab.title; 

        // 2. On interroge VOTRE API (api.py)
        // On utilise encodeURIComponent pour gérer les espaces et caractères spéciaux
        fetch(`http://127.0.0.1:8000/predict?name=${encodeURIComponent(journalName)}`)
        .then(response => response.json())
        .then(data => {
            let div = document.getElementById('result');
            
            // 3. Affichage du résultat selon la réponse de l'IA
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
            document.getElementById('result').innerText = "Erreur : L'API ne répond pas. Vérifiez que 'uvicorn' est lancé.";
        });
    });
});