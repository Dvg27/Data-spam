document.addEventListener('DOMContentLoaded', () => {
  const messageInput = document.getElementById('messageInput');
  const charCount = document.getElementById('charCount');
  const wordCount = document.getElementById('wordCount');
  const quickAlert = document.getElementById('quickAlert');
  const clearBtn = document.getElementById('clearBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const sampleBtns = document.querySelectorAll('.sample-btn');

  const emptyState = document.getElementById('emptyState');
  const resultsContent = document.getElementById('resultsContent');
  const verdictBanner = document.getElementById('verdictBanner');
  const verdictIcon = document.getElementById('verdictIcon');
  const verdictTag = document.getElementById('verdictTag');
  const confidenceVal = document.getElementById('confidenceVal');
  const hamProbText = document.getElementById('hamProbText');
  const hamProgressBar = document.getElementById('hamProgressBar');
  const spamProbText = document.getElementById('spamProbText');
  const spamProgressBar = document.getElementById('spamProgressBar');
  const indicatorsList = document.getElementById('indicatorsList');
  const featuresGrid = document.getElementById('featuresGrid');
  const cleanedTextDisplay = document.getElementById('cleanedTextDisplay');

  // Friendly labels for structural features
  const FEATURE_LABELS = {
    message_length: 'Char Length',
    word_count: 'Word Count',
    sentence_count: 'Sentences',
    punctuation_count: 'Punctuation',
    punctuation_ratio: 'Punct. Ratio',
    uppercase_count: 'Uppercase',
    capitalization_ratio: 'Caps Ratio',
    digit_count: 'Digits Count',
    digit_ratio: 'Digit Ratio',
    contains_url: 'Has URL',
    currency_count: 'Currency Sym',
    exclamation_count: 'Exclamations',
    question_count: 'Questions',
    repeated_char_count: 'Char Repeats'
  };

  function updateInputMetrics() {
    const text = messageInput.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;

    charCount.textContent = chars;
    wordCount.textContent = words;

    // Quick regex checks for live hints
    const hasUrl = /https?:\/\/|www\./i.test(text);
    const hasCurrency = /[\$£€₹]/.test(text);
    if (hasUrl && hasCurrency) {
      quickAlert.textContent = '⚡ URL & Currency detected';
    } else if (hasUrl) {
      quickAlert.textContent = '⚡ Link detected';
    } else if (hasCurrency) {
      quickAlert.textContent = '⚡ Currency symbol detected';
    } else {
      quickAlert.textContent = '';
    }
  }

  messageInput.addEventListener('input', updateInputMetrics);

  clearBtn.addEventListener('click', () => {
    messageInput.value = '';
    updateInputMetrics();
    messageInput.focus();
  });

  // Sample Buttons click
  sampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const sampleText = btn.getAttribute('data-text');
      messageInput.value = sampleText;
      updateInputMetrics();
      analyzeCurrentMessage();
    });
  });

  // Shortcut Ctrl+Enter or Cmd+Enter
  messageInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      analyzeCurrentMessage();
    }
  });

  analyzeBtn.addEventListener('click', analyzeCurrentMessage);

  async function analyzeCurrentMessage() {
    const text = messageInput.value.trim();
    if (!text) {
      alert('Please enter an SMS message to analyze.');
      messageInput.focus();
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: text })
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Server responded with status ${response.status}`);
      }

      const result = await response.json();
      renderResults(result);
    } catch (err) {
      console.error('Classification error:', err);
      alert(`Error during analysis: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  function setLoading(loading) {
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');

    if (loading) {
      analyzeBtn.disabled = true;
      btnText.textContent = 'Analyzing...';
      btnLoader.style.display = 'inline-block';
    } else {
      analyzeBtn.disabled = false;
      btnText.textContent = 'Analyze Message';
      btnLoader.style.display = 'none';
    }
  }

  function renderResults(res) {
    emptyState.style.display = 'none';
    resultsContent.style.display = 'block';

    const isSpam = res.label === 'SPAM';
    const hamPercent = (res.ham_probability * 100).toFixed(1);
    const spamPercent = (res.spam_probability * 100).toFixed(1);
    const confPercent = (res.confidence * 100).toFixed(1);

    // Verdict styling
    verdictBanner.className = `verdict-banner ${isSpam ? 'verdict-spam' : 'verdict-ham'}`;
    verdictIcon.textContent = isSpam ? '🚨' : '✅';
    verdictTag.textContent = isSpam ? 'SPAM / PHISHING DETECTED' : 'LEGITIMATE MESSAGE (HAM)';
    confidenceVal.textContent = `${confPercent}%`;

    // Probability bars
    hamProbText.textContent = `${hamPercent}%`;
    hamProgressBar.style.width = `${hamPercent}%`;

    spamProbText.textContent = `${spamPercent}%`;
    spamProgressBar.style.width = `${spamPercent}%`;

    // Indicators
    indicatorsList.innerHTML = '';
    if (res.indicators && res.indicators.length > 0) {
      res.indicators.forEach(indicator => {
        const chip = document.createElement('div');
        chip.className = 'indicator-chip';
        chip.innerHTML = `<span>⚠️</span> <span>${escapeHtml(indicator)}</span>`;
        indicatorsList.appendChild(chip);
      });
    } else {
      const safeChip = document.createElement('div');
      safeChip.className = 'indicator-none';
      safeChip.innerHTML = '<span>✅</span> <span>No high-risk structural anomalies detected</span>';
      indicatorsList.appendChild(safeChip);
    }

    // Features Grid
    featuresGrid.innerHTML = '';
    if (res.features) {
      Object.entries(res.features).forEach(([k, v]) => {
        const item = document.createElement('div');
        item.className = 'feat-item';
        
        const label = FEATURE_LABELS[k] || k;
        let displayVal = v;
        if (typeof v === 'number') {
          displayVal = Number.isInteger(v) ? v : v.toFixed(3);
        }

        item.innerHTML = `
          <span class="feat-label">${escapeHtml(label)}</span>
          <span class="feat-val">${displayVal}</span>
        `;
        featuresGrid.appendChild(item);
      });
    }

    // Cleaned NLP Text
    cleanedTextDisplay.textContent = res.cleaned_text || '(None / empty tokens)';
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
              .replace(/"/g, "&quot;")
              .replace(/'/g, "&#039;");
  }
});

