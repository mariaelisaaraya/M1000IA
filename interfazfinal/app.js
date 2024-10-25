document.getElementById('form-datos').addEventListener('submit', async (event) => {
    event.preventDefault(); // Evitar el envío normal del formulario

    // Crear el FormData a partir del formulario
    const formData = new FormData(event.target);
    const edad = document.getElementById('edadInput').value;
    const sexo = document.getElementById('sexoInput').value;
    const imagen = document.getElementById('imageInput').files[0];

    // Verificar si la imagen fue seleccionada
    if (!imagen) {
        return alert('Por favor, selecciona una imagen.');
    }

    // Convertir metadatos a JSON y agregarlos al FormData
    formData.append('edad', edad);
    formData.append('sexo', sexo);

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
       

        if (response.ok) {
            // Proceso de interpretación: Si predicted_class es 0, es "Benigna", si es 1, es "Maligna"
            const prediction = result.predicted_class === 0 ? 'Benigna' : 'Maligna';
          // Guardar el ID de predicción correcto en localStorage
            localStorage.setItem('prediction_id', result.id); // Aquí es donde haces el cambio
            console.log('ID de predicción guardado:', result.id); // Esto debería mostrar el ID
        
            // Redirigir a la página de resultados con la predicción y los detalles
         window.location.href = `resultado.html?resultado=${prediction}&info=${result.details}`;
            } else {
             alert(result.error || 'Se produjo un error en la solicitud.');
             }
        } catch (error) {
        console.error('Error al realizar la predicción:', error);
        alert('Error al enviar la solicitud. Por favor, inténtalo de nuevo más tarde.');
    }
});


//const form = document.getElementById('form-datos');

        //form.addEventListener('submit', (e) => {
           // e.preventDefault();
           // window.location.href = "resultado.html";  // Simulando redirección al análisis
       // });
    