document.addEventListener("mousemove", (e) => {
    document.body.style.backgroundPosition =
        `${e.clientX / 20}px ${e.clientY / 20}px`;
});
