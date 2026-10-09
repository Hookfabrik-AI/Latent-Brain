(() => {
  document.querySelectorAll('[data-copy-target]').forEach(button => {
    button.addEventListener('click', async () => {
      const el = document.getElementById(button.dataset.copyTarget);
      if (!el) return;
      const previous = button.textContent;
      try {
        await navigator.clipboard.writeText(el.textContent.trim());
        button.textContent = 'Copied';
      } catch {
        button.textContent = 'Select code below';
      }
      setTimeout(() => { button.textContent = previous; }, 1800);
    });
  });
})();
