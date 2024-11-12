document.addEventListener("DOMContentLoaded", () => {
    const audio = document.getElementById('welcomeAudio');
    const shouldPlay = Math.random() > 0.5;
    if (shouldPlay) {
        audio.play().catch(error => {
            console.log("Autoplay blocked. Waiting for user interaction.");
            document.addEventListener("click", () => {
                audio.play();
            }, { once: true });
        });
    }
});