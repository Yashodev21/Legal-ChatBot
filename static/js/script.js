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
        const response = await fetch(`${window.location.origin}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question })
        });
        const contentType = response.headers.get("content-type") || "";
        const data = contentType.includes("application/json")
            ? await response.json()
            : {};

        if (!response.ok) {
            throw new Error(
                data.error || `Server error (${response.status}).`
            );
        }

        if (typeof data.answer !== "string" || !data.answer.trim()) {
            throw new Error("The server returned an empty answer.");
        }

        const botMessage = document.createElement("div");
        botMessage.className = "message bot";
        botMessage.textContent = data.answer;
        status.innerHTML = "";
        messages.appendChild(botMessage);
    } catch (error) {
        const errorMessage = document.createElement("div");
        errorMessage.className = "message bot";
        const message = error.name === "TypeError"
            ? "Unable to reach the Flask server. Confirm it is running and open this page from the Flask URL."
            : error.message;
        errorMessage.textContent = "Error: " + message;
        status.innerHTML = "";
        messages.appendChild(errorMessage);
    } finally {
        button.disabled = false;
        status.innerHTML = "";
        chatBox.scrollTop = chatBox.scrollHeight;
        input.focus();
    }
};
