document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const rawTextInput = document.getElementById('rawTextInput');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const btnClear = document.getElementById('btnClear');
  const charCounter = document.getElementById('charCounter');
  const analyzeSpinner = document.getElementById('analyzeSpinner');
  const presetButtons = document.querySelectorAll('.preset-btn');

  // Gauge & Triage Elements
  const gaugeProgress = document.getElementById('gaugeProgress');
  const gaugeVal = document.getElementById('gaugeVal');
  const triageBadge = document.getElementById('triageBadge');
  const triageCategoryTitle = document.getElementById('triageCategoryTitle');
  const triageRecommendation = document.getElementById('triageRecommendation');
  const triageTimestamp = document.getElementById('triageTimestamp');

  // Annotated tokens & metrics
  const annotatedTextBox = document.getElementById('annotatedTextBox');
  const topicsContainer = document.getElementById('topicsContainer');
  const featSelfFocus = document.getElementById('featSelfFocus');
  const featAbsolutist = document.getElementById('featAbsolutist');
  const featDistress = document.getElementById('featDistress');
  const featHesitation = document.getElementById('featHesitation');
  const driversList = document.getElementById('driversList');

  // Presets mapping
  const presets = {
    despair: "I feel so completely exhausted and empty inside. Everything I try to do always ends in failure and I am just tired of waking up every single day feeling like a burden to everyone around me. Nobody understands how alone I feel.",
    panic: "I haven't slept in three days. My chest hurts constantly from panic and anxiety. Every time night comes, I feel like I am suffocating with zero way out.",
    control: "We spent Sunday morning painting the living room with friends. The natural sunlight really brightens up the space, and we celebrated with pizza afterwards.",
    work: "Work was pretty hectic today with tight deadlines and client meetings, but looking forward to unwinding with a book tonight."
  };

  // Update word counter
  function updateCounter() {
    const text = rawTextInput.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    charCounter.textContent = `${words} word${words === 1 ? '' : 's'}`;
  }

  rawTextInput.addEventListener('input', updateCounter);

  // Preset button clicks
  presetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const type = btn.getAttribute('data-preset');
      if (presets[type]) {
        rawTextInput.value = presets[type];
        updateCounter();
        runAnalysis();
      }
    });
  });

  // Clear button
  btnClear.addEventListener('click', () => {
    rawTextInput.value = '';
    updateCounter();
    resetDisplay();
  });

  // Analyze button
  btnAnalyze.addEventListener('click', runAnalysis);

  async function runAnalysis() {
    const text = rawTextInput.value.trim();
    if (!text) {
      alert("Please enter or paste a text snippet to analyze.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error(`Server returned error ${response.status}`);
      }

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      console.error(err);
      alert("Error analyzing text. Please check server logs.");
    } finally {
      setLoading(false);
    }
  }

  function setLoading(isLoading) {
    if (isLoading) {
      analyzeSpinner.style.display = 'inline-block';
      btnAnalyze.disabled = true;
    } else {
      analyzeSpinner.style.display = 'none';
      btnAnalyze.disabled = false;
    }
  }

  function renderResults(data) {
    const score = data.risk_score;
    const pct = data.risk_percentage;

    // 1. Update Gauge
    const circumference = 427.26; // 2 * PI * 68
    const offset = circumference * (1 - score);
    gaugeProgress.style.strokeDashoffset = offset;
    gaugeVal.textContent = `${pct}%`;

    // Dynamic stroke colors based on tier
    let gaugeColor = '#10B981'; // emerald
    if (score >= 0.65) {
      gaugeColor = '#F43F5E'; // rose
    } else if (score >= 0.35) {
      gaugeColor = '#F59E0B'; // amber
    }
    gaugeProgress.style.stroke = gaugeColor;

    // 2. Triage Category & Badge
    triageBadge.className = `badge badge-${data.badge_color}`;
    triageBadge.textContent = data.triage_category;
    triageCategoryTitle.textContent = `Triage: ${data.triage_category}`;
    triageRecommendation.textContent = data.recommendation;
    triageTimestamp.textContent = `Calibrated Score: ${(score).toFixed(3)}`;

    // 3. Highlighted phrase tokens
    if (data.annotated_tokens && data.annotated_tokens.length > 0) {
      let html = '';
      data.annotated_tokens.forEach(item => {
        if (item.tag) {
          let tagClass = 'tag-distress';
          if (item.tag === 'self_focus') tagClass = 'tag-self';
          if (item.tag === 'absolutist') tagClass = 'tag-absolutist';
          if (item.tag === 'somatic') tagClass = 'tag-somatic';

          html += `<span class="annotated-token ${tagClass}" title="Marker: ${item.tag}">${escapeHtml(item.word)}</span> `;
        } else {
          html += `${escapeHtml(item.word)} `;
        }
      });
      annotatedTextBox.innerHTML = html;
    }

    // 4. Psycholinguistic feature matrix
    const f = data.psycholinguistic_features;
    if (f) {
      featSelfFocus.textContent = `${(f.self_focus_index * 100).toFixed(1)}%`;
      featAbsolutist.textContent = `${(f.absolutist_ratio * 100).toFixed(2)}%`;
      featDistress.textContent = `${(f.distress_affect_ratio * 100).toFixed(2)}%`;
      featHesitation.textContent = `${f.question_density.toFixed(2)}`;
    }

    // 5. Unsupervised Topic Decomposition (LDA)
    if (data.topics && data.topics.length > 0) {
      let topicHtml = '';
      data.topics.forEach(t => {
        const pctWidth = Math.min(100, Math.max(2, t.weight * 100));
        topicHtml += `
          <div class="topic-row">
            <div class="topic-meta">
              <span class="topic-name">${t.name} <span class="topic-keywords">(${t.keywords.join(', ')})</span></span>
              <span class="topic-val">${(t.weight * 100).toFixed(1)}%</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width: ${pctWidth}%;"></div>
            </div>
          </div>
        `;
      });
      topicsContainer.innerHTML = topicHtml;
    }

    // 6. Primary Linguistic Drivers (Explainability)
    if (data.top_drivers && data.top_drivers.length > 0) {
      let driverHtml = '';
      data.top_drivers.forEach(d => {
        const isPos = d.impact > 0;
        const cls = isPos ? 'elevates' : 'protects';
        const impactSign = isPos ? `+${d.impact}` : `${d.impact}`;
        const impactCls = isPos ? 'pos' : 'neg';

        driverHtml += `
          <div class="driver-item ${cls}">
            <span class="driver-name">${d.feature}</span>
            <span class="driver-impact ${impactCls}">${impactSign} (${d.direction})</span>
          </div>
        `;
      });
      driversList.innerHTML = driverHtml;
    } else {
      driversList.innerHTML = '<p class="placeholder-text">Markers remain within baseline variance.</p>';
    }
  }

  function resetDisplay() {
    gaugeProgress.style.strokeDashoffset = 427.26;
    gaugeVal.textContent = '--%';
    gaugeProgress.style.stroke = 'var(--accent-primary)';
    triageBadge.className = 'badge';
    triageBadge.textContent = 'Awaiting Input';
    triageCategoryTitle.textContent = 'Screening Status';
    triageRecommendation.textContent = 'Submit a text sample to generate calibrated risk probability, clinical recommendations, and underlying linguistic drivers.';
    triageTimestamp.textContent = 'Ready';
    annotatedTextBox.innerHTML = '<p class="placeholder-text">Run an analysis to view token-level psycholinguistic annotations.</p>';
    topicsContainer.innerHTML = '<p class="placeholder-text">Latent theme decomposition will appear here after analysis.</p>';
    featSelfFocus.textContent = '--';
    featAbsolutist.textContent = '--';
    featDistress.textContent = '--';
    featHesitation.textContent = '--';
    driversList.innerHTML = '<p class="placeholder-text">No active drivers computed.</p>';
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Initial load: trigger preset despair for instant visual verification
  rawTextInput.value = presets.despair;
  updateCounter();
  runAnalysis();
});
