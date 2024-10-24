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
