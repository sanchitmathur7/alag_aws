// Toggle chat window visibility
function toggleChatWindow() {
    var chatWindow = document.getElementById("chat-window");
    if (chatWindow.style.display === "none" || chatWindow.style.display === "") {
        chatWindow.style.display = "flex";
    } else {
        chatWindow.style.display = "none";
    }
 }
 
 // Send message function
 function sendMessage() {
    var userInput = document.getElementById("user-input").value;
    if (userInput === "") return;
 
    var chatBox = document.getElementById("chat-box");
 
    var userMessage = document.createElement("div");
    userMessage.textContent = "You: " + userInput;
    chatBox.appendChild(userMessage);
 
    fetch("/api/get_response", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        var botMessage = document.createElement("div");
        botMessage.textContent = "Sam: " + data.response;
        chatBox.appendChild(botMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        var errorMessage = document.createElement("div");
        errorMessage.textContent = "Sam: Sorry, there was an error. Please try again later.";
        chatBox.appendChild(errorMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
    });
 
    document.getElementById("user-input").value = "";
 }
 
 // Listen for Enter key press to send message
 document.getElementById("user-input").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevent default form submission if inside a form
        sendMessage();
    }
 });