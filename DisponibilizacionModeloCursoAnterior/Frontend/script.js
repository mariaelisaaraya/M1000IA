document.addEventListener('DOMContentLoaded', function () {
    const aboutUsLink = document.getElementById('about-us-link');
    const aboutUsSection = document.getElementById('about-us');
    const closeAboutUsButton = document.getElementById('close-about-us');
    const sentimentForm = document.getElementById('sentiment-form');
    const resultDiv = document.getElementById('result');

    aboutUsLink.addEventListener('click', function (event) {
        event.preventDefault();
        aboutUsSection.style.display = 'block';
    });

    closeAboutUsButton.addEventListener('click', function () {
        aboutUsSection.style.display = 'none';
    });

    sentimentForm.addEventListener('submit', async function (event) {
        event.preventDefault();
        const textInput = document.getElementById('text-input').value;

        const response = await fetch('http://127.0.0.1:8000/sentiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: textInput })
        });

        const result = await response.json();
        resultDiv.innerHTML = `<p>Sentiment: ${result.sentiment}</p>`;
    });
});
