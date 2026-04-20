// Copy-to-clipboard for the install block.
document.querySelectorAll(".copy-btn").forEach((button) => {
  button.addEventListener("click", async () => {
    const targetId = button.dataset.copy;
    const target = targetId ? document.getElementById(targetId) : null;
    if (!target) {
      return;
    }

    const text = target.innerText.trim();

    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection?.removeAllRanges();
      selection?.addRange(range);
      document.execCommand("copy");
      selection?.removeAllRanges();
    }

    const label = button.querySelector(".copy-label");
    const original = label?.textContent ?? "Copy";
    button.classList.add("copied");
    if (label) {
      label.textContent = "Copied";
    }

    setTimeout(() => {
      button.classList.remove("copied");
      if (label) {
        label.textContent = original;
      }
    }, 1400);
  });
});
