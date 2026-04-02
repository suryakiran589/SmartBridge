// ── State ──────────────────────────────────────────────────────────────────────
let lastWord     = "";
let lastSentence = "";
let lastSigned   = "";
let history      = [];
let letterChips  = [];

// ── Poll API ───────────────────────────────────────────────────────────────────
async function poll() {
    try {
        const res  = await fetch('/api/current_state');
        const data = await res.json();
        updateUI(data);
    } catch (e) {
        document.getElementById('statusText').textContent = 'OFFLINE';
    }
    setTimeout(poll, 120);
}

// ── Update UI ──────────────────────────────────────────────────────────────────
function updateUI(data) {
    const gesture  = data.gesture    || "None";
    const action   = data.action     || "";
    const conf     = parseFloat(data.confidence) || 0;
    const word     = data.word       || "";
    const sentence = data.sentence   || "";
    const signed   = data.last_signed|| "";

    // ── Mode sync ──────────────────────────────────────────────────────────────
    if (data.mode) {
        const btn = document.getElementById('modeToggle');
        const currentClass = btn.classList.contains('control') ? 'control' : 'sign';
        if (data.mode !== currentClass) {
            updateModeUI(data.mode);
        }
    }

    // ── Gesture label ──────────────────────────────────────────────────────────
    const label = document.getElementById('gestureLabel');
    const isActive = gesture && gesture !== "None" && gesture !== "...";

    if (isActive) {
        if (label.textContent !== gesture) {
            label.textContent = gesture;
            label.classList.remove('dim', 'flash');
            void label.offsetWidth; // reflow to restart animation
            label.classList.add('flash');
        }
        label.classList.remove('dim');
    } else {
        label.textContent = gesture === "..." ? "Reading..." : "Waiting...";
        label.classList.add('dim');
        label.classList.remove('flash');
    }

    // ── Action text ────────────────────────────────────────────────────────────
    document.getElementById('gestureAction').textContent = action;

    // ── Confidence bar ─────────────────────────────────────────────────────────
    const pct = Math.round(conf * 100);
    const bar = document.getElementById('confBar');
    bar.style.width = pct + '%';
    bar.className   = 'conf-fill' + (conf < 0.5 ? ' low' : conf < 0.75 ? ' mid' : '');
    document.getElementById('confValue').textContent = pct + '%';

    // ── Word display ───────────────────────────────────────────────────────────
    if (word !== lastWord) {
        const wordEl = document.getElementById('wordDisplay');
        wordEl.innerHTML = word
            ? `${word}<span class="cursor"></span>`
            : `<span class="cursor"></span>`;
        lastWord = word;
    }

    // ── Sentence display ───────────────────────────────────────────────────────
    if (sentence !== lastSentence) {
        const sentEl = document.getElementById('sentenceDisplay');
        if (sentence) {
            sentEl.textContent = sentence;
            sentEl.classList.add('has-text');
        } else {
            sentEl.textContent = 'Sentence will appear here...';
            sentEl.classList.remove('has-text');
        }

        // Add to history when a word gets committed (word cleared, sentence grew)
        if (sentence && !word && sentence !== lastSentence) {
            addToHistory(sentence);
        }

        lastSentence = sentence;
    }

    // ── Letter chips on video ──────────────────────────────────────────────────
    if (signed && signed !== lastSigned) {
        addLetterChip(signed);
        lastSigned = signed;
    }
}

// ── Letter chips ───────────────────────────────────────────────────────────────
function addLetterChip(letter) {
    const row  = document.getElementById('lettersRow');
    const chip = document.createElement('div');
    chip.className   = 'letter-chip';
    chip.textContent = letter;
    row.appendChild(chip);
    letterChips.push(chip);

    // Max 8 chips visible
    if (letterChips.length > 8) {
        const old = letterChips.shift();
        if (old) old.remove();
    }

    // Fade out after 3s
    setTimeout(() => {
        chip.style.transition = 'opacity 0.5s ease';
        chip.style.opacity    = '0';
        setTimeout(() => {
            chip.remove();
            letterChips = letterChips.filter(c => c !== chip);
        }, 500);
    }, 3000);
}

// ── History ────────────────────────────────────────────────────────────────────
function addToHistory(sentence) {
    // Avoid duplicates
    if (history[0] === sentence) return;
    history.unshift(sentence);
    if (history.length > 5) history.pop();
    renderHistory();
}

function renderHistory() {
    const list  = document.getElementById('historyList');
    const empty = document.getElementById('emptyMsg');

    // Remove old items
    list.querySelectorAll('.history-item').forEach(el => el.remove());

    if (history.length === 0) {
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    history.forEach((text, i) => {
        const timeLabel = i === 0 ? 'just now' : `${i}m ago`;
        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `
            <div class="history-text">${text}</div>
            <div class="history-time">${timeLabel}</div>
            <button class="history-play" title="Speak again" onclick="speakText('${text.replace(/'/g, "\\'")}')">▶</button>
        `;
        list.insertBefore(item, empty);
    });
}

// ── Button actions ─────────────────────────────────────────────────────────────
async function speakNow() {
    try {
        await fetch('/api/speak', { method: 'POST' });
        const text = document.getElementById('sentenceDisplay').textContent;
        if (text && text !== 'Sentence will appear here...') {
            addToHistory(text);
        }
    } catch(e) { console.error(e); }
}

async function deleteLetter() {
    try {
        await fetch('/api/delete', { method: 'POST' });
    } catch(e) { console.error(e); }
}

async function clearAll() {
    try {
        await fetch('/api/clear', { method: 'POST' });
        document.getElementById('lettersRow').innerHTML = '';
        letterChips  = [];
        lastSentence = '';
        lastWord     = '';
        lastSigned   = '';
        const sentEl = document.getElementById('sentenceDisplay');
        sentEl.textContent = 'Sentence will appear here...';
        sentEl.classList.remove('has-text');
        document.getElementById('wordDisplay').innerHTML = '<span class="cursor"></span>';
    } catch(e) { console.error(e); }
}

async function quickPhrase(text) {
    try {
        await fetch('/api/quick_phrase', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ text })
        });
        addToHistory(text);
    } catch(e) { console.error(e); }
}

async function speakText(text) {
    try {
        await fetch('/api/speak_text', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ text })
        });
    } catch(e) { console.error(e); }
}

// ── Mode toggle ────────────────────────────────────────────────────────────────
async function toggleMode() {
    try {
        const res  = await fetch('/api/toggle_mode', { method: 'POST' });
        const data = await res.json();
        updateModeUI(data.mode);
    } catch(e) { console.error(e); }
}

function updateModeUI(mode) {
    const btn  = document.getElementById('modeToggle');
    const icon = document.getElementById('modeIcon');
    const text = document.getElementById('modeText');
    const rightPanel = document.querySelector('.right-panel');

    if (mode === 'control') {
        btn.classList.add('control');
        icon.textContent = '🖱️';
        text.textContent = 'Gesture Control';
        rightPanel.style.opacity = '0.4';
        rightPanel.style.pointerEvents = 'none';
    } else {
        btn.classList.remove('control');
        icon.textContent = '🤟';
        text.textContent = 'Sign Language';
        rightPanel.style.opacity = '1';
        rightPanel.style.pointerEvents = 'auto';
    }
}

// ── Start ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', poll);