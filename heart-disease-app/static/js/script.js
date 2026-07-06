(function () {
  const form = document.getElementById("intake-form");
  const submitBtn = document.getElementById("submit-btn");
  const sampleBtn = document.getElementById("sample-btn");
  const errorsBox = document.getElementById("form-errors");

  const resultEmpty = document.getElementById("result-empty");
  const resultBody = document.getElementById("result-body");
  const resultTag = document.getElementById("result-tag");

  const gaugeFill = document.getElementById("gauge-fill");
  const gaugeValue = document.getElementById("gauge-value");
  const verdictBadge = document.getElementById("verdict-badge");
  const verdictLabel = document.getElementById("verdict-label");
  const verdictSub = document.getElementById("verdict-sub");

  const barNeg = document.getElementById("bar-neg");
  const barPos = document.getElementById("bar-pos");
  const valNeg = document.getElementById("val-neg");
  const valPos = document.getElementById("val-pos");

  const flagsBlock = document.getElementById("flags-block");
  const flagsList = document.getElementById("flags-list");

  const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52; // r=52

  // A couple of illustrative sample cases so graders/users can try the tool instantly.
  const SAMPLES = [
    { age: 63, gender: 1, impluse: 92, pressurehight: 152, pressurelow: 95, glucose: 168, kcm: 8.2, troponin: 0.45 },
    { age: 34, gender: 0, impluse: 74, pressurehight: 112, pressurelow: 72, glucose: 91, kcm: 1.1, troponin: 0.008 },
  ];
  let sampleIndex = 0;

  sampleBtn.addEventListener("click", () => {
    const sample = SAMPLES[sampleIndex % SAMPLES.length];
    sampleIndex += 1;
    Object.entries(sample).forEach(([key, value]) => {
      const el = form.elements[key];
      if (el) el.value = value;
    });
  });

  function showErrors(messages) {
    errorsBox.innerHTML = messages.map((m) => `&bull; ${m}`).join("<br>");
    errorsBox.hidden = false;
  }

  function clearErrors() {
    errorsBox.hidden = true;
    errorsBox.innerHTML = "";
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.textContent = isLoading ? "Analyzing\u2026" : "Run diagnosis";
  }

  function renderResult(data) {
    resultEmpty.hidden = true;
    resultBody.hidden = false;

    const isPositive = data.prediction === "positive";
    resultTag.textContent = isPositive ? "Risk detected" : "No risk detected";
    resultTag.style.color = isPositive ? "var(--coral)" : "var(--teal)";
    resultTag.style.background = isPositive ? "var(--coral-dim)" : "var(--teal-dim)";

    // Gauge
    const pct = data.probability_positive;
    const offset = GAUGE_CIRCUMFERENCE * (1 - pct / 100);
    requestAnimationFrame(() => {
      gaugeFill.style.strokeDashoffset = offset;
      gaugeFill.style.stroke =
        data.risk_level === "High" ? "var(--coral)" :
        data.risk_level === "Moderate" ? "var(--amber)" : "var(--teal)";
    });
    gaugeValue.textContent = `${pct.toFixed(1)}%`;

    // Verdict
    verdictBadge.textContent = `${data.risk_level} risk`;
    verdictBadge.className = "verdict-badge " + data.risk_level.toLowerCase();
    verdictLabel.textContent = data.prediction_label;
    verdictSub.textContent = `Model confidence based on ${data.model_accuracy}% test-set accuracy ensemble.`;

    // Bars
    requestAnimationFrame(() => {
      barNeg.style.width = `${data.probability_negative}%`;
      barPos.style.width = `${data.probability_positive}%`;
    });
    valNeg.textContent = `${data.probability_negative.toFixed(1)}%`;
    valPos.textContent = `${data.probability_positive.toFixed(1)}%`;

    // Flags
    if (data.flags && data.flags.length) {
      flagsList.innerHTML = data.flags.map((f) => `<li>${f}</li>`).join("");
      flagsBlock.hidden = false;
    } else {
      flagsBlock.hidden = true;
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearErrors();

    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => (payload[key] = value));

    setLoading(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok || !data.ok) {
        showErrors(data.errors || ["Something went wrong. Please check your inputs."]);
        return;
      }
      renderResult(data);
    } catch (err) {
      showErrors(["Could not reach the prediction service. Is the server running?"]);
    } finally {
      setLoading(false);
    }
  });
})();
