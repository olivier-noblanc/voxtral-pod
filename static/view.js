document.addEventListener('DOMContentLoaded', () => {
    // UI elements
    const elToggleSpeaker = document.getElementById('toggleSpeakerEditorBtn');
    const elHideTimestamps = document.getElementById('hideTimestamps');
    const elHideSpeakers = document.getElementById('hideSpeakers');
    const audioPlayer = document.getElementById('audioPlayer');

    // 1. Initialiser le lecteur audio si l'URL est fournie
    if (audioPlayer) {
        const audioUrl = document.body.dataset.audioUrl;
        if (audioUrl) {
            audioPlayer.src = audioUrl;
        }
    }

    // 2. Gestionnaire Speaker Editor
    if (elToggleSpeaker) {
        elToggleSpeaker.addEventListener('click', toggleSpeakerEditor);
    }

    // 3. Masquer les timestamps
    if (elHideTimestamps) {
        elHideTimestamps.addEventListener('change', (e) => {
            const display = e.target.checked ? 'none' : 'inline';
            document.querySelectorAll('.timestamp').forEach(span => {
                span.style.display = display;
            });
        });
    }

    // 4. Masquer les speakers
    if (elHideSpeakers) {
            elHideSpeakers.addEventListener('change', (e) => {
                const display = e.target.checked ? 'none' : 'inline';
                document.querySelectorAll('.speaker-label').forEach(span => {
                    span.style.display = display;
                });
            });
    }
});

function toggleSpeakerEditor() {
    const container = document.getElementById('speakerRenameContainer');
    if (!container) return;
    if (container.style.display === 'none' || container.style.display === '') {
        container.style.display = 'block';
        initSpeakerEditor();
    } else {
        container.style.display = 'none';
    }
}

function initSpeakerEditor() {
    const speakerSpans = document.querySelectorAll('.segment-speaker');
    const speakers = new Set();
    speakerSpans.forEach(span => {
        const speaker = span.dataset.speaker || span.textContent.replace(/\[|\]/g, "").trim();
        if (speaker) speakers.add(speaker);
    });
    const listDiv = document.getElementById('speakerRenameList');
    if (!listDiv) return;
    listDiv.innerHTML = '';
    speakers.forEach(speaker => {
        const colDiv = document.createElement('div');
        colDiv.className = 'fr-col-12 fr-col-md-6';
        const label = document.createElement('label');
        label.className = 'fr-label';
        label.textContent = 'Speaker "' + speaker + '" :';
        const input = document.createElement('input');
        input.type = 'text';
        input.value = speaker;
        input.className = 'fr-input';
        input.dataset.oldName = speaker;
        input.addEventListener('input', e => {
            const newName = e.target.value;
            const oldName = e.target.dataset.oldName;
            document.querySelectorAll('.segment-speaker').forEach(span => {
                const currentText = span.textContent.replace(/\[|\]/g, "").trim();
                if (currentText === oldName) {
                    span.textContent = '[' + newName + ']';
                }
            });
            e.target.dataset.oldName = newName;
        });
        colDiv.appendChild(label);
        colDiv.appendChild(input);
        listDiv.appendChild(colDiv);
    });
}
