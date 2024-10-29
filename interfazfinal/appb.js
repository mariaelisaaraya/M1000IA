document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const resultado = params.get('resultado');
    const info = params.get('info') || 'No hay información adicional.';

    const resultadoElement = document.getElementById('resultado');
    const infoElement = document.getElementById('info');

    if (resultadoElement) {
        resultadoElement.innerText = ` ${resultado}`;
    }

    if (infoElement) {
        infoElement.innerText = `Información adicional: ${info}`;
    }
});

// Descargar la imagen y la predicción
document.getElementById('download-btn').addEventListener('click', async function() {
    const predictionId = localStorage.getItem('prediction_id'); // Obtén el prediction_id de localStorage
    console.log('ID de predicción:', predictionId); // Verificar el ID
    if (!predictionId) {
        console.error('No se encontró el ID de la predicción.');
        return; // Salir temprano si no hay ID de predicción
    }
    
    const resultText = document.getElementById('resultado').innerText.trim().toLowerCase();

    // Define información adicional basada en la predicción
    let infoText = '';
    if (resultText.includes('benigna')) {
        infoText = 'La predicción indica que la condición es benigna.';
    } else if (resultText.includes('maligna')) {
        infoText = 'La predicción indica que la condición es maligna.';
    } else {
        infoText = 'Resultado no determinado.';
    }

    // Primero, obtenemos la información de la predicción (edad, sexo, probabilidades)
    try {
        const predictionUrl = `http://127.0.0.1:5000/predict/${predictionId}`; 
        const response = await fetch(predictionUrl);
        if (!response.ok) {
            throw new Error('Error al obtener la predicción: ' + response.statusText);
        }

        const data = await response.json();
        const age = data.edad;
        const sex = data.sexo;
        const predictedProbabilities = data.probabilities; // Cambia aquí a 'probabilities'

        // Asegúrate de que predictedProbabilities sea un array y tenga los índices que esperas
        if (!Array.isArray(predictedProbabilities) || predictedProbabilities.length < 2) {
            throw new Error('Probabilidades de predicción no válidas');
        }

        // Ahora, obtenemos la imagen de la API
        const apiUrl = `http://127.0.0.1:5000/download/${predictionId}`; // URL completa para la API
        console.log('URL de la API:', apiUrl); // Verificar la URL de la API
        const imageResponse = await fetch(apiUrl); // Cambia aquí
        if (!imageResponse.ok) {
            throw new Error('Error al obtener la imagen: ' + imageResponse.statusText);
        }

        const imageBlob = await imageResponse.blob(); // Obtén la imagen como un blob
        const imageUrl = URL.createObjectURL(imageBlob); // Crea una URL para el blob

        // Crear un canvas
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 800;
        canvas.height = 600;

        // Fondo blanco
        context.fillStyle = '#fff';
        context.fillRect(0, 0, canvas.width, canvas.height);

        // Título
        context.fillStyle = '#333';
        context.font = 'bold 40px Roboto';
        context.textAlign = 'center';
        context.fillText('Resultado del Análisis', canvas.width / 2, 50);

        // Información adicional
        context.font = '30px Roboto';
        context.textAlign = 'left';
        context.fillText(infoText, 20, 100);

        // Detalles de la predicción
        context.font = '24px Roboto';
        context.fillStyle = '#555';
        context.fillText(`Edad: ${age}`, 100, 160); // Ajustar x para centrar
        context.fillText(`Sexo: ${sex}`, 100, 200); // Ajustar x para centrar
        
        //context.fillText(`porcentaje de acierto: ${Math.round(predictedProbabilities[1] * 100)}%`, 20, 240);

        // Imagen
        const img = new Image();
        img.src = imageUrl;
        img.onload = function() {
            context.drawImage(img, canvas.width - 400, 120, 300, 250);

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
        console.error('Error en la descarga de la imagen o predicción:', error);
    }
});
