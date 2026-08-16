/**
 * LLM Fine-Tuning Arena & Evaluation Dashboard Client Logic
 */

const SCENARIOS = {
  oom: `[ERROR] 2026-08-16T14:22:01.104Z [checkout-service] [main] org.springframework.boot.SpringApplication: Application run failed
java.lang.OutOfMemoryError: Java heap space
    at com.eshop.checkout.service.CartAggregator.processLargeCart(CartAggregator.java:184)
k8s_event: Pod checkout-service-pod-7d9fb48b9c status changed to OOMKilled (Exit Code 137). Restart count: 4.
Node: worker-node-us-east-1a, Available Memory: 1.2Gi / 16Gi`,

  deadlock: `FATAL: 2026-08-16T15:02:11.890Z [db-pool-worker-42] postgres[14820]: LOG: process 14820 detected deadlock while waiting for ShareLock on transaction 8892120
DETAIL: Process 14820 waits for ShareLock on transaction 8892120; blocked by process 14829.
Process 14829 waits for ExclusiveLock on tuple (412, 19) of relation 16402 'ledger_accounts'; blocked by process 14820.
STATEMENT: UPDATE ledger_accounts SET balance = balance - 150.00 WHERE account_id = 'ACC-9921';
Active connections: 498/500 (99.6% saturation).`,

  ssl: `[CRITICAL] 2026-08-16T17:40:02.001Z [envoy.transport_sockets.tls] [C19823] remote address: 198.51.100.4:54321
TLS handshake failed: SSL routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED
error:1000007d:SSL routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED: ../ssl/handshake_client.cc:1132
Downstream client handshake rejected. Issuer: Let's Encrypt R3. Valid Until: 2026-08-16T17:00:00Z.
502 Bad Gateway rate surged to 94.2% on /v1/auth.`,

  redis: `[ERROR] 2026-08-16T18:05:33.220Z [catalog-service] RedisConnectionException: Command timed out after 3000ms
Redis Master CPU Utilization: 100.0%. Instantaneous Ops Per Second: 185,000.
Connected clients: 10,000 (maxclients limit reached).
Cache key 'homepage_curated_deals_v2' expired at 18:05:30Z. Subsequent 8,000 RPS queried MySQL.
MySQL DB CPU spiked to 99.8% with 450 slow queries in 'Sending data' state.`,

  kafka: `[WARN] 2026-08-16T16:11:45.312Z [ConsumerCoordinator] [consumer-group: telemetry-ingest]
Consumer group telemetry-ingest rebalance in progress. Revoking partitions: [telemetry-raw-0, telemetry-raw-1].
Reason: max.poll.interval.ms (300000ms) exceeded.
Current lag on topic telemetry-raw: 1,489,200 messages (+24,000 msg/sec).
Downstream Elasticsearch cluster reporting 429 Too Many Requests.`
};

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initScenarios();
  initInference();
  renderLossChart();
  loadLossMaskingDemo();
});

// Tab switching
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  const panes = document.querySelectorAll(".tab-pane");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

// Scenario chip loading
function initScenarios() {
  const chips = document.querySelectorAll(".chip-btn");
  const promptInput = document.getElementById("prompt-input");

  // Load default
  if (promptInput && SCENARIOS.oom) {
    promptInput.value = SCENARIOS.oom;
  }

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");

      const key = chip.getAttribute("data-scenario");
      if (SCENARIOS[key] && promptInput) {
        promptInput.value = SCENARIOS[key];
      }
    });
  });
}

// Inference runner
function initInference() {
  const runBtn = document.getElementById("run-inference-btn");
  const promptInput = document.getElementById("prompt-input");
  const baseOutput = document.getElementById("base-response-output");
  const finetunedOutput = document.getElementById("finetuned-response-output");

  if (!runBtn) return;

  runBtn.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    runBtn.disabled = true;
    runBtn.style.opacity = "0.7";
    baseOutput.innerHTML = '<span style="color:#64748B;">Generating zero-shot output...</span>';
    finetunedOutput.innerHTML = '<span style="color:#38BDF8;">Streaming fine-tuned QLoRA completion...</span>';

    try {
      const response = await fetch("/api/stream-generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, model_type: "both" })
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      baseOutput.innerHTML = "";
      finetunedOutput.innerHTML = "";

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // keep remainder

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.substring(6));
              if (data.base_token) {
                baseOutput.textContent += data.base_token;
              }
              if (data.finetuned_token) {
                finetunedOutput.textContent += data.finetuned_token;
              }
            } catch (e) {}
          }
        }
      }
    } catch (err) {
      console.warn("Falling back to local client streaming simulation:", err);
      simulateLocalInference(prompt, baseOutput, finetunedOutput);
    } finally {
      runBtn.disabled = false;
      runBtn.style.opacity = "1";
    }
  });
}

function simulateLocalInference(prompt, baseEl, ftEl) {
  const baseText = `Based on the provided error log:\n"${prompt.substring(0, 80)}..."\n\n1. Check the service memory allocation.\n2. Review logs to see why the thread pool failed.\n3. Restart the pod: \`kubectl delete pod <pod-name>\`\n\nHope this helps!`;
  
  let ftObj = {
    "incident_id": "INC-91042",
    "severity": "P1-CRITICAL",
    "root_cause": "JVM Heap exhaustion inside unpaginated batch memory allocation leading to Linux OOM killer termination (Exit Code 137).",
    "blast_radius": ["checkout-service", "order-processing-worker", "payment-ingress"],
    "mitigation_actions": [
      "kubectl set resources deployment checkout-service --limits=memory=2Gi,cpu=1000m -n production",
      "kubectl rollout restart deployment checkout-service -n production"
    ],
    "preventative_measure": "Implement stream-based pagination with bounded chunk buffers and configure container JVM MaxRAMPercentage=75.0."
  };

  const ftText = JSON.stringify(ftObj, null, 2);
  baseEl.textContent = "";
  ftEl.textContent = "";

  let i = 0;
  const bWords = baseText.split(" ");
  const fWords = ftText.split(" ");
  const maxWords = Math.max(bWords.length, fWords.length);

  const interval = setInterval(() => {
    if (i < bWords.length) baseEl.textContent += bWords[i] + " ";
    if (i < fWords.length) ftEl.textContent += fWords[i] + " ";
    i++;
    if (i >= maxWords) clearInterval(interval);
  }, 40);
}

// Render dynamic SVG Loss Convergence Chart
function renderLossChart() {
  const svg = document.getElementById("loss-svg-chart");
  if (!svg) return;

  const width = 600;
  const height = 240;
  const padding = { top: 20, right: 30, bottom: 30, left: 50 };

  // Simulated loss curve points (100 steps)
  const steps = 100;
  const points = [];
  const valPoints = [];

  for (let s = 1; s <= steps; s++) {
    const decay = Math.exp(-s / 28.0);
    const trainLoss = 0.38 + (2.45 - 0.38) * decay;
    points.push({ step: s, loss: trainLoss });

    if (s % 20 === 0 || s === steps) {
      valPoints.push({ step: s, loss: trainLoss + 0.05 });
    }
  }

  const xScale = (s) => padding.left + ((s - 1) / (steps - 1)) * (width - padding.left - padding.right);
  const yScale = (l) => height - padding.bottom - ((l - 0.2) / (2.6 - 0.2)) * (height - padding.top - padding.bottom);

  let pathD = `M ${xScale(points[0].step)} ${yScale(points[0].loss)}`;
  for (let i = 1; i < points.length; i++) {
    pathD += ` L ${xScale(points[i].step)} ${yScale(points[i].loss)}`;
  }

  // Draw grid lines
  let gridSvg = "";
  for (let l = 0.5; l <= 2.5; l += 0.5) {
    const y = yScale(l);
    gridSvg += `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="#1E293B" stroke-dasharray="3 3"/>`;
    gridSvg += `<text x="${padding.left - 8}" y="${y + 4}" fill="#64748B" font-size="10" font-family="monospace" text-anchor="end">${l.toFixed(1)}</text>`;
  }

  for (let s = 20; s <= 100; s += 20) {
    const x = xScale(s);
    gridSvg += `<line x1="${x}" y1="${padding.top}" x2="${x}" y2="${height - padding.bottom}" stroke="#1E293B" stroke-dasharray="3 3"/>`;
    gridSvg += `<text x="${x}" y="${height - padding.bottom + 16}" fill="#64748B" font-size="10" font-family="monospace" text-anchor="middle">Step ${s}</text>`;
  }

  // Validation dots
  let valSvg = "";
  valPoints.forEach(p => {
    valSvg += `<circle cx="${xScale(p.step)}" cy="${yScale(p.loss)}" r="5" fill="#10B981" stroke="#090D16" stroke-width="2"/>`;
  });

  svg.innerHTML = `
    ${gridSvg}
    <path d="${pathD}" fill="none" stroke="#38BDF8" stroke-width="2.5" stroke-linecap="round"/>
    ${valSvg}
  `;
}

// Load loss masking demonstration tokens
async function loadLossMaskingDemo() {
  const container = document.getElementById("token-stream-visual");
  if (!container) return;

  try {
    const res = await fetch("/api/loss-masking-demo");
    if (res.ok) {
      const data = await res.json();
      if (data.token_visual) {
        container.innerHTML = "";
        data.token_visual.forEach(t => {
          const span = document.createElement("span");
          span.className = `token-chip ${t.is_masked ? "masked" : "trained"}`;
          span.textContent = t.token;
          span.title = `Target: ${t.target_loss_id} (${t.is_masked ? "Masked Prompt" : "Trained Output"})`;
          container.appendChild(span);
        });
        return;
      }
    }
  } catch (e) {}

  // Fallback demo tokens
  const sampleTokens = [
    { t: "<|begin_of_text|>", m: true },
    { t: "<|start_header_id|>system<|end_header_id|>", m: true },
    { t: "You", m: true }, { t: "are", m: true }, { t: "an", m: true }, { t: "expert", m: true }, { t: "SRE", m: true },
    { t: "<|eot_id|><|start_header_id|>user<|end_header_id|>", m: true },
    { t: "[INPUT", m: true }, { t: "LOGS]:", m: true }, { t: "java.lang.OutOfMemoryError", m: true },
    { t: "<|eot_id|><|start_header_id|>assistant<|end_header_id|>", m: true },
    { t: "{\n", m: false }, { t: '"incident_id":', m: false }, { t: '"INC-84920",', m: false },
    { t: '"severity":', m: false }, { t: '"P1-CRITICAL",', m: false },
    { t: '"root_cause":', m: false }, { t: '"JVM', m: false }, { t: 'Heap', m: false }, { t: 'exhaustion..."', m: false },
    { t: "}", m: false }, { t: "<|eot_id|>", m: false }
  ];

  container.innerHTML = "";
  sampleTokens.forEach(item => {
    const span = document.createElement("span");
    span.className = `token-chip ${item.m ? "masked" : "trained"}`;
    span.textContent = item.t;
    span.title = `Label: ${item.m ? "-100 (Masked)" : "Trained Loss"}`;
    container.appendChild(span);
  });
}
