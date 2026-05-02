const dom = {
  heroMeta: document.getElementById("hero-meta"),
  envTag: document.getElementById("env-tag"),
  privacyTag: document.getElementById("privacy-tag"),
  messageInput: document.getElementById("message-input"),
  inputNote: document.getElementById("input-note"),
  runAnalysis: document.getElementById("run-analysis"),
  resetWorkspace: document.getElementById("reset-workspace"),

  riskPill: document.getElementById("risk-pill"),
  chatContext: document.getElementById("chat-context"),
  chatThread: document.getElementById("chat-thread"),
  chatInput: document.getElementById("chat-input"),
  sendChat: document.getElementById("send-chat"),
  logStream: document.getElementById("log-stream"),
};

const state = {
  bootstrap: null,
  currentAnalysis: null,
  analysisPending: false,
  chatPending: false,
  pendingChatNode: null,
  activeModels: null,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const fallback = `La peticion a ${path} fallo con estado ${response.status}.`;
    try {
      const payload = await response.json();
      throw new Error(payload.detail || fallback);
    } catch {
      throw new Error(fallback);
    }
  }

  return response.json();
}

function currentClock() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function appendLog(stage, text, tone = "info") {
  const node = document.createElement("div");
  const stageClass = (stage || "Sistema").toLowerCase().replace(/\s+/g, "-");
  node.className = `log-line ${tone} stage-${stageClass}`;
  node.innerHTML = `
    <span class="log-time">${currentClock()}</span>
    <span class="log-stage">${stage}</span>
    <span class="log-text">${text}</span>
  `;
  dom.logStream.prepend(node);
}

function renderResultLogs(logs = []) {
  dom.logStream.innerHTML = "";
  [...logs].reverse().forEach((entry) => {
    appendLog(entry.stage || "Sistema", entry.text || "", "info");
  });
}

function appendChatMessage(role, text, options = {}) {
  const node = document.createElement("div");
  node.className = `chat-message ${role}${options.pending ? " pending" : ""}`;
  node.innerHTML = `
    <div class="chat-message-head">
      <span class="chat-role">${role === "user" ? "Usuario" : "Sistema"}</span>
      <span class="chat-time">${options.time || currentClock()}</span>
    </div>
    <div class="chat-text"></div>
  `;

  const textNode = node.querySelector(".chat-text");
  if (options.pending) {
    textNode.innerHTML = `
      <span class="typing-row">
        <span class="spinner" aria-hidden="true"></span>
        <span>${text}</span>
      </span>
    `;
  } else {
    textNode.textContent = text;
  }

  dom.chatThread.appendChild(node);
  dom.chatThread.scrollTop = dom.chatThread.scrollHeight;
  return node;
}

function resolvePendingChatMessage(text) {
  if (!state.pendingChatNode) {
    appendChatMessage("assistant", text);
    return;
  }

  state.pendingChatNode.classList.remove("pending");
  const textNode = state.pendingChatNode.querySelector(".chat-text");
  const timeNode = state.pendingChatNode.querySelector(".chat-time");
  if (textNode) {
    textNode.textContent = text;
  }
  if (timeNode) {
    timeNode.textContent = currentClock();
  }
  state.pendingChatNode = null;
  dom.chatThread.scrollTop = dom.chatThread.scrollHeight;
}

function setAnalysisPending(isPending) {
  state.analysisPending = isPending;
  dom.runAnalysis.disabled = isPending;
  dom.sendChat.disabled = isPending || state.chatPending;
  dom.messageInput.disabled = isPending;

  dom.resetWorkspace.disabled = isPending;
  document.body.classList.toggle("analysis-busy", isPending);

  if (isPending) {
    dom.inputNote.textContent = "El mensaje esta bloqueado mientras corre el analisis local.";
    dom.runAnalysis.innerHTML = `<span class="btn-spinner" aria-hidden="true"></span><span>Analizando...</span>`;
  } else {
    dom.inputNote.textContent = "Pega el texto fuente y ejecuta el analisis antes de conversar con el sistema.";
    dom.runAnalysis.textContent = "Analizar mensaje";
  }
}

function setChatPending(isPending) {
  state.chatPending = isPending;
  dom.sendChat.disabled = isPending || state.analysisPending;
  dom.chatInput.disabled = isPending;
  dom.sendChat.textContent = isPending ? "Enviando..." : "Enviar";
}

function resetWorkspace() {
  state.currentAnalysis = null;
  state.pendingChatNode = null;
  dom.messageInput.value = "";
  dom.messageInput.disabled = false;

  dom.resetWorkspace.disabled = false;
  dom.chatThread.innerHTML = "";
  dom.chatInput.value = "";
  dom.chatContext.textContent =
    "Pega un mensaje y ejecuta el analisis. El sistema detecta idioma, estima senales de persuasion y genera una explicacion anclada al texto actual.";
  dom.riskPill.textContent = "";
  dom.riskPill.className = "status-pill";
  dom.inputNote.textContent = "Pega el texto fuente y ejecuta el analisis antes de conversar con el sistema.";

  dom.logStream.innerHTML = "";
  appendLog("Interfaz", "Estado reiniciado. Esperando texto fuente para iniciar el analisis contextual.", "info");
}

function renderAnalysis(result) {
  state.currentAnalysis = result;
  dom.messageInput.disabled = false;
  dom.riskPill.textContent =
    result.riskLevel === "high"
      ? "Presion persuasiva alta"
      : result.riskLevel === "medium"
      ? "Presion persuasiva media"
      : "Presion persuasiva baja";
  dom.riskPill.className = `status-pill ${result.riskLevel || ""}`.trim();
  const detectedPrinciples = result.detectedPrinciples || [];
  const visiblePrinciples = result.visiblePrinciples || [];
  const principleLabel = detectedPrinciples.length
    ? "Principios detectados"
    : visiblePrinciples.length
    ? "Senales visibles"
    : "Principios detectados";
  const principleText = detectedPrinciples.length
    ? detectedPrinciples.join(", ")
    : visiblePrinciples.length
    ? visiblePrinciples.join(", ")
    : "sin senales dominantes";
  dom.chatContext.innerHTML = `
    <div class="summary-container">
      <div class="summary-header">Mensaje analizado</div>
      <div class="summary-item">
        <span class="summary-key">Idioma</span>
        <span class="summary-value">${result.detectedLanguage || "N/D"}</span>
      </div>
      <div class="summary-item">
        <span class="summary-key">Sospecha Phishing</span>
        <span class="summary-value highlight verdict-${(result.phishingSuspicion?.verdict || "Bajo").toLowerCase().replace(" ", "-")}">${result.phishingSuspicion?.verdict || "Bajo"} (${Math.round((result.phishingSuspicion?.score || 0) * 100)}%)</span>
      </div>
      <div class="summary-item">
        <span class="summary-key">Intensidad</span>
        <span class="summary-value highlight risk-${result.riskLevel || "low"}">${result.overallRisk || "N/D"}</span>
      </div>
      <div class="summary-item">
        <span class="summary-key">${principleLabel}</span>
        <span class="summary-value">${principleText}</span>
      </div>
      <div class="summary-resumen">
        ${result.simpleSummary || "Sin resumen"}
      </div>
    </div>
  `;
  dom.chatThread.innerHTML = "";
  (result.chat || []).forEach((item) => appendChatMessage(item.role, item.text));
  
  const executionSummary = `El backend analizo el texto con el clasificador y luego uso el SLM para auditar y expandir esa hipotesis inicial. ` +
    `Idioma detectado: ${result.detectedLanguage || "N/D"}. ` +
    `Principios claramente detectados: ${detectedPrinciples.join(", ") || "ninguno"}. ` +
    `Senales visibles: ${visiblePrinciples.join(", ") || "ninguna"}.`;
  

  appendLog("Auditoria", executionSummary, "info");
  renderResultLogs(result.logs || []);
}

async function analyzeCurrentMessage() {
  const text = dom.messageInput.value.trim();
  if (!text) {
    appendChatMessage("assistant", "Primero necesito un mensaje inspeccionado para poder analizar y conversar.");
    appendLog("Validacion", "Se intento analizar sin pegar un mensaje.", "warn");
    return null;
  }

  setAnalysisPending(true);
  dom.chatContext.textContent =
    "El sistema esta detectando idioma, estimando presion persuasiva y preparando una explicacion anclada al texto actual.";

  dom.logStream.innerHTML = "";
  appendLog("Entrada", "Mensaje recibido desde la interfaz y enviado al backend para analisis contextual.", "info");
  appendLog("Clasificador", "Se ejecutara la inferencia multi-label para estimar senales de persuasion por principio.", "info");
  appendLog("SLM", "El SLM tomara el bundle del clasificador como punto de partida y lo auditara contra el texto real.", "info");

  try {
    const result = await api("/api/analyze", {
      method: "POST",
      body: JSON.stringify({
        text,
        language: "auto",
        mode: "detailed",
      }),
    });
    renderAnalysis(result);
    return result;
  } catch (error) {
    dom.chatContext.textContent = error.message;
    appendChatMessage("assistant", error.message);
    appendLog("Error", error.message, "error");
    return null;
  } finally {
    setAnalysisPending(false);
  }
}

async function sendChat() {
  const question = dom.chatInput.value.trim();
  if (!question || state.chatPending || state.analysisPending) {
    return;
  }

  if (!state.currentAnalysis) {
    const prepared = await analyzeCurrentMessage();
    if (!prepared) {
      return;
    }
  }

  appendChatMessage("user", question);
  dom.chatInput.value = "";
  setChatPending(true);
  state.pendingChatNode = appendChatMessage("assistant", "El SLM local esta preparando la respuesta...", { pending: true });
  appendLog("Chat", "Pregunta enviada al flujo conversacional contextual sobre el mensaje ya analizado.", "info");

  try {
    const response = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        text: dom.messageInput.value,
        question,
        language: "auto",
        mode: "detailed",
        analysis: state.currentAnalysis,
      }),
    });
    resolvePendingChatMessage(response.answer);
    appendLog(
      response.source === "slm" ? "SLM" : "Analisis",
      response.log?.text || "Respuesta generada y devuelta al chat.",
      "info"
    );
  } catch (error) {
    resolvePendingChatMessage(error.message);
    appendLog("Error", error.message, "error");
  } finally {
    setChatPending(false);
  }
}

function renderAuditItem(label, value) {
  if (!value || value === "N/A") return "";
  return `
    <div class="audit-item">
      <span class="audit-key">${label}:</span>
      <span class="audit-value" title="${value}">${value}</span>
    </div>
  `;
}

function renderModelAudit(title, meta) {
  if (meta.status === "missing") {
    return `
      <div class="audit-section missing">
        <h5>${title}</h5>
        <p>No detectado en ruta oficial</p>
      </div>
    `;
  }
  return `
    <div class="audit-section">
      <h5>${title}</h5>
      <div class="audit-grid">
        ${renderAuditItem("ID", meta.name)}
        ${renderAuditItem("Ruta", meta.path)}
        ${renderAuditItem("Base", meta.base_model)}
        ${renderAuditItem("Archi.", meta.architecture)}
        ${renderAuditItem("Fecha", meta.trained_at)}
        ${renderAuditItem("Tamano", meta.size)}
      </div>
    </div>
  `;
}

async function loadBootstrap() {
  try {
    state.bootstrap = await api("/api/bootstrap");
    const app = state.bootstrap.app || {};
    
    // Funcion para actualizar el header con info de HW
    const updateHeader = (hw) => {
        const gpuInfo = hw.active === "CUDA GPU" 
            ? `<b style="color:var(--teal)">${hw.device_name}</b> (<span style="color:var(--text)">${hw.vram_free_gb}/${hw.vram_total_gb} GB VRAM</span>)`
            : `<b style="color:var(--amber)">Standard CPU Mode</b>`;

        dom.envTag.innerHTML = `
            ${gpuInfo} | 
            CPU: <b style="color:var(--text)">${hw.cpu_model}</b> (${hw.cpu_count} Cores) |
            RAM: <span style="color:var(--text)">${hw.ram_free_gb}/${hw.ram_total_gb} GB</span>
        `;
        dom.privacyTag.innerHTML = `<span style="color:var(--ice)">*</span> Local-Only`;
    };

    // Primera carga
    updateHeader(app.hardware || {});
    
    // Polling dinamico cada 10 segundos
    setInterval(async () => {
        try {
            const hw = await api("/api/hardware");
            updateHeader(hw);
        } catch (e) { console.warn("Error polling hardware info", e); }
    }, 10000);

    state.activeModels = {
      slm: app.slm?.name || "No disponible",
      classifier: app.classifier?.name || "No disponible",
    };

    dom.heroMeta.innerHTML = `
      <div class="system-audit-panel">
        ${renderModelAudit("Analizador de Persuasion (SLM)", app.slm)}
        ${renderModelAudit("Clasificador Base (Encoder)", app.classifier)}
      </div>
    `;

    resetWorkspace();

    if (app.slm_status && app.slm_status !== "ready") {
      dom.chatContext.textContent = app.slm_error || "No se puede trabajar sin el SLM local entrenado.";
      dom.sendChat.disabled = true;
      dom.runAnalysis.disabled = true;
      appendLog("Sistema", app.slm_error || "No se pudo habilitar el SLM local.", "error");
    }
  } catch (error) {
    dom.heroMeta.innerHTML = `<div class="error-box">${error.message}</div>`;
    throw error;
  }
}

dom.runAnalysis.addEventListener("click", () => analyzeCurrentMessage());
dom.resetWorkspace.addEventListener("click", () => {
  resetWorkspace();
});
dom.sendChat.addEventListener("click", () => sendChat());
dom.chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendChat();
  }
});

loadBootstrap().catch((error) => {
  dom.heroMeta.textContent = error.message;
  dom.chatContext.textContent = error.message;
});
