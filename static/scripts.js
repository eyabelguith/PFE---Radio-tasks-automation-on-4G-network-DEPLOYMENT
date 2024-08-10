document.addEventListener('DOMContentLoaded', function() {
    const chatButton = document.getElementById('chatButton');
    const chatModal = document.getElementById('chatModal');
    const closeButton = document.querySelector('.close');

    chatButton.addEventListener('click', function() {
        chatModal.style.display = 'block';
    });

    closeButton.addEventListener('click', function() {
        chatModal.style.display = 'none';
    });

    window.addEventListener('click', function(event) {
        if (event.target == chatModal) {
            chatModal.style.display = 'none';
        }
    });
});

function submitPrediction() {
    const temperature = document.getElementById('temperature').value;
    const humidity = document.getElementById('humidity').value;
    const windSpeed = document.getElementById('windSpeed').value;
    const cellName = document.getElementById('cellName').value;

    const inputData = {
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'Wind Speed (mph)': windSpeed,
        'Cell Name': cellName
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('resultText').textContent = `Prediction: ${data.prediction}`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function sendMessage() {
    const chatInput = document.getElementById('chatInput').value;
    const chatBox = document.getElementById('chatBox');

    if (chatInput.trim() !== "") {
        const message = document.createElement('p');
        message.textContent = chatInput;
        chatBox.appendChild(message);
        document.getElementById('chatInput').value = '';
    }
}
