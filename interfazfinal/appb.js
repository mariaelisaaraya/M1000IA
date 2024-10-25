document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const resultado = params.get('resultado');
    const info = params.get('info') || 'No hay información adicional.';

   
    const resultadoElement = document.getElementById('resultado');
    const infoElement = document.getElementById('info');

    if (resultadoElement) {
        resultadoElement.innerText = `Predicción: ${resultado}`;
    }

    if (infoElement) {
        infoElement.innerText = `Información adicional: ${info}`;
    }
});

//canvas download

// Descargar la imagen y la predicción
document.getElementById('download-btn').addEventListener('click', async function() {
    const predictionId = localStorage.getItem('prediction_id'); // Obtén el prediction_id de localStorage
    console.log('ID de predicción:', predictionId); // Verificar el ID
    if (!predictionId) {
        console.error('No se encontró el ID de la predicción.');
        return; // Salir temprano si no hay ID de predicción
    }
    
    const resultText = document.getElementById('resultado').innerText;

    // Define información adicional basada en la predicción
    let infoText = '';
    if (resultText.includes('Benigna')) {
        infoText = 'La predicción indica que la condición es benigna.';
    } else if (resultText.includes('Maligna')) {
        infoText = 'La predicción indica que la condición es maligna.';
    } else {
        infoText = 'Resultado no determinado.';
    }

    // Fetch the image from the API
    try {
        const apiUrl = `http://127.0.0.1:5000/download/${predictionId}`; // URL completa para la API
        console.log('URL de la API:', apiUrl); // Verificar la URL de la API
        const response = await fetch(apiUrl); // Cambia aquí
        if (!response.ok) {
            throw new Error('Error al obtener la imagen: ' + response.statusText);
        }

        const imageBlob = await response.blob(); // Obtén la imagen como un blob
        const imageUrl = URL.createObjectURL(imageBlob); // Crea una URL para el blob

        // Crear un canvas
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 600;
        canvas.height = 400;
        context.fillStyle = '#fff';
        context.fillRect(0, 0, canvas.width, canvas.height);

        // Dibujar texto del resultado
        context.font = '28px Roboto';
        context.fillStyle = '#333';
        context.fillText('Resultado del Análisis: ' + resultText, 20, 50);

        // Dibujar información adicional
        context.font = '20px Roboto';
        context.fillText(infoText, 20, 100);

        // Cargar la imagen y dibujarla en el canvas
        const img = new Image();
        img.src = imageUrl; // Usa la URL del blob

        img.onload = function() {
            context.drawImage(img, 20, 150, 200, 200);

            // Crear un enlace de descarga
            const downloadLink = document.createElement('a');
            downloadLink.href = canvas.toDataURL('image/png');
            downloadLink.download = 'resultado.png';
            document.body.appendChild(downloadLink);
            downloadLink.click();
            document.body.removeChild(downloadLink);
        };

        img.onerror = function() {
            console.error('Error al cargar la imagen para la descarga.');
        };

    } catch (error) {
        console.error('Error en la descarga de la imagen:', error);
    }
});
