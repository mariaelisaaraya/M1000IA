// Obtener parámetros de la URL
const params = new URLSearchParams(window.location.search);
const resultado = params.get('resultado');
const info = params.get('info') || 'No hay información adicional.';

// Mostrar el resultado y la información adicional en la página
document.getElementById('resultado').innerText = `Predicción: ${resultado}. Información adicional: ${info}`;

// Función para redirigir a la página de resultados
function displayPrediction(prediction) {
    window.location.href = `resultado.html?resultado=${prediction}`;
}

// Manejo del evento de envío del formulario
document.getElementById('form-datos').addEventListener('submit', async function (event) {
    event.preventDefault(); // Prevenir el envío normal del formulario

    const formData = new FormData(this); // Captura todos los datos del formulario
    const edad = document.getElementById('edadInput').value; // Obtiene la edad usando el ID correcto
    const sexo = document.getElementById('sexoInput').value; // Obtiene el sexo usando el ID correcto
    const imagen = document.getElementById('imageInput').files[0]; // Obtiene la imagen usando el ID correcto

    // Verifica si la imagen fue seleccionada
    if (!imagen) {
        alert('Por favor, selecciona una imagen.');
        return;
    }

    // Convertir metadatos a JSON
    const metadata = JSON.stringify([parseInt(edad), sexo]);

    // Agrega metadatos al FormData
    formData.append('metadata', metadata);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (response.ok) {
            // Redirigir a la página de resultados
            window.location.href = `resultado.html?resultado=${result.prediction}&info=${result.details}`;
        } else {
            alert(result.error || 'Se produjo un error al procesar la solicitud.');
        }
    } catch (error) {
        console.error('Error al realizar la predicción:', error);
        alert('Error al enviar la solicitud. Por favor, inténtalo de nuevo más tarde.');
    }
});
