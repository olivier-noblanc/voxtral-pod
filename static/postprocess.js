document.addEventListener('DOMContentLoaded', () => {
    const contentEl = document.getElementById('content');
    if (!contentEl) return;

    const rawContent = contentEl.getAttribute('data-content');

    const copyBtn = document.getElementById('copyBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(rawContent).then(() => {
                alert("Texte copié !");
            }).catch(err => {
                console.error("Erreur de copie : ", err);
            });
        });
    }

    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            const title = downloadBtn.getAttribute('data-title') || 'document';
            const filename = downloadBtn.getAttribute('data-filename') || 'export.txt';
            const blob = new Blob([rawContent], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const safeTitle = title.replace(/\s+/g, '_');
            a.download = `${safeTitle}_${filename}`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        });
    }
});
