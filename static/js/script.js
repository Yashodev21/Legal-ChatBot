const input = document.getElementById("question");
const chatBox = document.getElementById("chat-box");
const messages = document.getElementById("messages");
const status = document.getElementById("status");
const button = document.getElementById("send-btn");

document.querySelectorAll(".suggestion").forEach((suggestion) => {
    suggestion.addEventListener("click", () => {
        input.value = suggestion.dataset.question;
        input.focus();
        sendMessage();
    });
});

input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

window.sendMessage = async function () {
    const question = input.value.trim();
    if (!question || button.disabled) return;

    const userMessage = document.createElement("div");
    userMessage.className = "message user";
    userMessage.textContent = question;
    messages.appendChild(userMessage);

    input.value = "";
    button.disabled = true;
    status.innerHTML = '<div class="message bot typing"><span></span><span></span><span></span></div>';
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Something went wrong.");

        const botMessage = document.createElement("div");
        botMessage.className = "message bot";
        botMessage.textContent = data.answer;
        status.innerHTML = "";
        messages.appendChild(botMessage);
    } catch (error) {
        const errorMessage = document.createElement("div");
        errorMessage.className = "message bot";
        errorMessage.textContent = "Error: " + error.message;
        status.innerHTML = "";
        messages.appendChild(errorMessage);
    } finally {
        button.disabled = false;
        status.innerHTML = "";
        chatBox.scrollTop = chatBox.scrollHeight;
        input.focus();
    }
};
