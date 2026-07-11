"use strict";

const ACTIVE_STATUSES = new Set(["starting", "running", "cancelling"]);
const VALIDATION_STATUSES = new Set(["ready", "starting", "running", "cancelling", "cancelled", "succeeded", "failed"]);
const AGENT_ACTIVE_STATUSES = new Set([
  "starting",
  "loading",
  "generating",
  "executing",
  "cancelling",
]);
const AGENT_STATUSES = new Set(["offline", "starting", "loading", "ready", "generating", "executing", "cancelling", "succeeded", "cancelled", "failed"]);
const VIEW_IDS = new Set(["assistant", "mission", "inference", "architecture", "business", "evidence"]);
const MAX_AGENT_DOM_MESSAGES = 32;
const MAX_AGENT_RESPONSE_CHARS = 8192;
const API_ERROR_COPY = {
  AGENT_UNAVAILABLE: "로컬 AI 서비스가 준비되지 않았습니다. 잠시 후 다시 시도해 주세요.",
  AGENT_BUSY: "현재 AI 요청이 끝난 뒤 다시 시도해 주세요.",
  AUTH_REQUIRED: "로컬 세션 인증이 만료되었습니다. 데모를 다시 실행해 주세요.",
  COMPUTE_BUSY: "다른 로컬 GPU 작업이 실행 중입니다. 완료 후 다시 시도해 주세요.",
  EVOLUTION_UNAVAILABLE: "Self-Harness 제어부가 준비되지 않았습니다.",
  INVALID_BODY: "요청 형식이 올바르지 않습니다.",
  JOB_ALREADY_RUNNING: "이미 검증 작업이 실행 중입니다.",
  NO_ACTIVE_AGENT_TURN: "중단할 AI 요청이 없습니다.",
  NO_ACTIVE_JOB: "중단할 검증 작업이 없습니다.",
};
const AGENT_MODULE_DEFAULTS = {
  gemma: "LOCAL",
  router: "READY",
  swarm: "GATED",
  cts: "READY",
  fast: "GATED",
};
const PHASE_ORDER = [
  "verifying",
  "loading_model",
  "building_runtime",
  "running_inference",
  "postcheck",
];
const PHASE_TO_DOM = {
  verifying: "verify",
  loading_model: "model_load",
  building_runtime: "runtime_build",
  running_inference: "inference",
  postcheck: "postcheck",
};
const PHASE_LABELS = {
  ready: "대기",
  starting: "검증 준비",
  worker_started: "로컬 worker 시작",
  verifying: "무결성 검증",
  loading_model: "로컬 모델 적재",
  building_runtime: "런타임 구성",
  running_inference: "Depth 100 탐색",
  postcheck: "안전 조건 확인",
  complete: "검증 완료",
  succeeded: "검증 완료",
  failed: "검증 실패",
  offline: "연결 재시도",
  cancelling: "취소 중",
  cancelled: "취소됨",
};

const SCENARIOS = {
  defense: {
    prompt: "Air-gapped defense document integrity and bounded reasoning validation",
    copy: "외부망이 단절된 지휘 환경에서 검증 가능한 깊은 추론 경로를 구성합니다.",
  },
  bio: {
    prompt: "Confidential bio R&D evidence synthesis with bounded local reasoning",
    copy: "단백질·후보물질 IP를 외부로 보내지 않는 로컬 연구 추론 경로를 검증합니다.",
  },
  finance: {
    prompt: "Local financial risk analysis with bounded memory and deterministic telemetry",
    copy: "클라우드 왕복 없이 감사 가능한 금융 리스크 분석 경로를 검증합니다.",
  },
};

const STORY = {
  problem: {
    index: "01 / PROBLEM",
    copy: "국방·신약·금융의 핵심 데이터는 클라우드에 보낼 수 없고, 자체 구축은 보안·최적화·전문인력이라는 숨은 비용을 남깁니다.",
  },
  solution: {
    index: "02 / SOLUTION",
    copy: "Cogni-OS는 로컬 모델, 고정 용량 CTS 작업 텐서, 텐서 중심 내부 경로, 주·야간 안전 제어를 하나의 Appliance 경험으로 통합합니다.",
  },
  scale: {
    index: "03 / SCALE-UP",
    copy: "초기에는 유상 PoC와 Appliance 공급으로 신뢰를 만들고, 이후 서명된 오프라인 산업 모듈·연간 라이선스·현장 유지보수로 반복 매출을 확장합니다.",
  },
  development: {
    index: "04 / DEFENSIBILITY",
    copy: "성능 주장은 공인 시험으로, 제품 신뢰는 IP·ISO/IEC 42001·감사 로그로, 시장 진입은 설계 파트너와 조달 협력으로 방어합니다.",
  },
};

const NODE_COPY = {
  rhythm: ["Bio Rhythm", "추론과 진화의 GPU 점유를 시간으로 분리합니다. 활성 추론이 끝나기 전에는 야간 변경 작업이 시작되지 않습니다.", "IMPLEMENTED + TESTED"],
  router: ["BIO-HAMA Meta Router", "인지 부하와 불확실성을 고정 크기 텐서로 계산해 전략·전술·반응 모듈 예산을 조절합니다.", "IMPLEMENTED + TESTED"],
  aflow: ["AFlow", "실행 권한이 없는 bounded workflow 후보를 탐색합니다. 생성 payload는 검증 전까지 inert 상태입니다.", "IMPLEMENTED + TESTED"],
  harness: ["Self-Harness", "실패 수집 → 제안 → 정책 검사 → 격리 회귀 테스트 → 승격 또는 롤백으로 변경을 통제합니다.", "FAIL-CLOSED"],
  gemma: ["Local Gemma 4", "검증된 로컬 경로와 manifest만 허용합니다. Hub ID·URL·remote code·KV cache는 거부됩니다.", "MEASURED"],
  deq: ["DEQ Equilibrium", "제한 이력 Broyden solver가 고정점 잔차를 계산하고, 미수렴·비유한 상태를 성공으로 표시하지 않습니다.", "MEASURED"],
  cts: ["Cognitive Tree Search", "Depth가 커져도 사전 할당된 301-node arena와 bounded ancestor bank 안에서 탐색합니다.", "MEASURED"],
  fast: ["Fast Weight", "수렴된 상태를 bounded 저랭크 세션 overlay로 컴파일하며 원본 모델 가중치는 변경하지 않습니다.", "IMPLEMENTED + TESTED"],
  swarm: ["System 4", "내부 hot path에서 자연어 직렬화를 거치지 않고 연속 텐서를 결합합니다. 6.3ms는 아직 설계 목표입니다.", "TARGET PENDING"],
  experts: ["System 3", "사전 할당된 expert pool 안에서 novelty를 측정하고 무제한 동적 생성을 금지합니다.", "IMPLEMENTED + TESTED"],
  ewc: ["FP-EWC", "고정점 민감도를 matrix-free 방식으로 추정하고 C-FIRE 재투영으로 업데이트 후 안정성을 확인합니다.", "IMPLEMENTED + TESTED"],
};

const TOUR = [
  { view: "mission", kicker: "WHY NOW", title: "보안 때문에 AI를 포기하는 시장", copy: "국방·신약 고객은 데이터를 클라우드로 보낼 수 없고, 자체 구축은 숨은 비용을 만듭니다." },
  { view: "mission", kicker: "PROOF, NOT PROMISE", title: "한 대의 범용 GPU에서 실제로 측정", copy: "현재 장비의 Depth 100, VRAM, 잔차, 회귀 테스트를 공개하고 목표 RTX 4090과 구분합니다." },
  { view: "inference", kicker: "LIVE VALIDATION", title: "같은 명령으로 다시 검증", copy: "manifest 검증부터 로컬 Gemma 적재, DEQ·CTS, 사후 안전 점검까지 실제 이벤트만 시각화합니다." },
  { view: "architecture", kicker: "DEFENSIBLE ARCHITECTURE", title: "텐서 경계와 주·야간 안전 제어", copy: "GPU 소유자는 하나뿐이며, 추론과 진화가 동시에 실행되지 않도록 상태 기계가 차단합니다." },
  { view: "business", kicker: "BEACHHEAD TO SCALE", title: "Appliance에서 오프라인 모듈 경제로", copy: "비안전결정형 유상 PoC로 시작해 제품 공급, 연간 라이선스, 현장 유지보수로 반복 매출을 확장합니다." },
  { view: "evidence", kicker: "THE ASK", title: "다음 단계는 시각화가 아니라 객관화", copy: "RTX 4090 반복시험, 고객 인터뷰·LOI, 공인 성적서, TCO를 Fact-book으로 완성합니다." },
];

const ui = {
  currentView: "assistant",
  selectedScenario: "defense",
  lastSeq: -1,
  lastStatus: "ready",
  pollStopped: false,
  tourIndex: 0,
  toastTimer: null,
  eventHistory: [],
  eventRunStartSeq: -1,
  agentMode: "chat",
  agentSeq: -1,
  agentStatus: "offline",
  agentPollStopped: false,
  chatEmptyTemplate: null,
  agentRequestPending: false,
  agentCancelPending: false,
  validationRequestPending: false,
  validationCancelPending: false,
  evolutionRequestPending: false,
  evolutionRunning: false,
  evolutionPromotionEnabled: false,
  agentConnectionLost: false,
  validationConnectionLost: false,
  lastAgentErrorKey: "",
  lastEvolutionErrorKey: "",
};

function $(selector, root = document) {
  return root.querySelector(selector);
}

function $$(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

function setText(selector, value) {
  const element = $(selector);
  if (element && value !== undefined && value !== null) {
    element.textContent = String(value).slice(0, MAX_AGENT_RESPONSE_CHARS);
  }
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function showToast(message, tone = "info") {
  const toast = $("#toast");
  if (!toast) return;
  toast.dataset.tone = tone;
  toast.setAttribute("role", tone === "error" ? "alert" : "status");
  toast.setAttribute("aria-live", tone === "error" ? "assertive" : "polite");
  $("span", toast).textContent = String(message || "").slice(0, 512);
  toast.hidden = false;
  clearTimeout(ui.toastTimer);
  ui.toastTimer = setTimeout(() => {
    toast.hidden = true;
  }, 4200);
}

function switchView(view, options = {}) {
  if (!VIEW_IDS.has(view)) return false;
  const panel = $(`[data-view-panel="${view}"]`);
  if (!panel) return false;
  ui.currentView = view;
  $$('[data-view-panel]').forEach((candidate) => {
    const active = candidate === panel;
    candidate.hidden = !active;
    candidate.classList.toggle("is-active", active);
  });
  $$('[data-view]').forEach((button) => {
    const active = button.dataset.view === view;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  if (!options.skipHash) history.replaceState(null, "", `#${view}`);
  if (options.focus) $("#main-content")?.focus({ preventScroll: true });
  $("#main-content")?.scrollTo({ top: 0, behavior: options.instant ? "auto" : "smooth" });
  return true;
}

function setRuntimeStatus(status, stage) {
  const pill = $("#runtime-pill");
  const label = $("#runtime-label");
  if (!pill || !label) return;
  const display = {
    ready: "READY",
    starting: "STARTING",
    running: "LIVE",
    cancelling: "STOPPING",
    cancelled: "CANCELLED",
    succeeded: "VERIFIED",
    failed: "FAILED",
    offline: "RECONNECTING",
  }[status] || String(status || "READY").toUpperCase();
  label.textContent = display;
  pill.dataset.state = status;
  pill.title = String(PHASE_LABELS[stage] || display).slice(0, 128);
}

function updateMetrics(metrics = {}) {
  if (finiteNumber(metrics.peak_vram_gib)) {
    const vram = metrics.peak_vram_gib;
    const limit = finiteNumber(metrics.vram_limit_gib) ? metrics.vram_limit_gib : 16.7;
    setText("#metric-vram", vram.toFixed(4));
    setText("#telemetry-vram", vram.toFixed(4));
    setText("#ledger-vram", `Peak VRAM ${vram.toFixed(4)} GiB`);
    const meter = $("#vram-meter");
    const telemetry = $("#telemetry-vram-fill");
    const meterLabel = `VRAM 사용량 ${vram.toFixed(4)} GiB, 상한 ${limit.toFixed(1)} GiB`;
    if (meter) {
      meter.max = limit;
      meter.value = vram;
      meter.setAttribute("aria-label", meterLabel);
    }
    if (telemetry) {
      telemetry.max = limit;
      telemetry.value = vram;
      telemetry.setAttribute("aria-label", meterLabel);
    }
  }
  if (Number.isInteger(metrics.reached_depth)) {
    setText("#metric-depth", metrics.reached_depth);
    setText("#orbit-depth", metrics.reached_depth);
    setText("#reactor-depth", metrics.reached_depth);
  }
  if (Number.isInteger(metrics.nodes_used)) {
    const capacity = Number.isInteger(metrics.node_capacity) ? metrics.node_capacity : metrics.nodes_used;
    setText("#metric-nodes", `${metrics.nodes_used} / ${capacity} nodes`);
  }
  if (finiteNumber(metrics.transition_residual)) {
    setText("#metric-residual", metrics.transition_residual.toFixed(6));
    setText("#reactor-residual", metrics.transition_residual.toExponential(3));
  }
  if (typeof metrics.transition_used_fallback === "boolean") {
    const fallback = metrics.transition_used_fallback ? "사용" : "미사용";
    setText("#metric-fallback", fallback);
    setText("#telemetry-fallback", metrics.transition_used_fallback ? "USED" : "NOT USED");
  }
  if (typeof metrics.finite === "boolean") {
    setText("#telemetry-finite", metrics.finite ? "PASS" : "FAIL");
    $("#telemetry-finite")?.classList.toggle("positive", metrics.finite);
  }
  if (Number.isInteger(metrics.tests)) setText("#metric-tests", metrics.tests);
  if (Number.isInteger(metrics.verified_files)) setText("#rail-files", `${metrics.verified_files} files`);
  if (typeof metrics.device === "string" && metrics.device) {
    setText("#device-name", metrics.device);
    setText("#rail-device", metrics.device);
    const compactDevice = $("#device-name");
    const railDevice = $("#rail-device");
    if (compactDevice) compactDevice.title = metrics.device.slice(0, 256);
    if (railDevice) railDevice.title = metrics.device.slice(0, 256);
  }
  if (typeof metrics.measured_at === "string") {
    const date = new Date(metrics.measured_at);
    setText("#rail-time", Number.isNaN(date.valueOf()) ? "최근 검증 기록" : `${date.toLocaleDateString("ko-KR")} 내부 실측`);
  }
}

function updatePhases(state) {
  const stageIndex = PHASE_ORDER.indexOf(state.stage);
  const succeeded = state.status === "succeeded";
  const failed = state.status === "failed";
  $$("#phase-list li").forEach((item) => {
    const eventStage = Object.keys(PHASE_TO_DOM).find((key) => PHASE_TO_DOM[key] === item.dataset.phase);
    const index = PHASE_ORDER.indexOf(eventStage);
    item.classList.toggle("is-complete", succeeded || (stageIndex >= 0 && index < stageIndex));
    item.classList.toggle("is-active", ACTIVE_STATUSES.has(state.status) && index === stageIndex);
    item.classList.toggle("is-failed", failed && index === stageIndex);
  });
  setText("#phase-status", PHASE_LABELS[state.stage] || PHASE_LABELS[state.status] || "대기");
  const reactor = $("#reactor");
  reactor?.classList.toggle("is-running", ACTIVE_STATUSES.has(state.status));
  reactor?.classList.toggle("is-success", state.status === "succeeded");
  const indicator = $("#live-indicator");
  if (indicator) {
    indicator.classList.toggle("is-live", ACTIVE_STATUSES.has(state.status));
    indicator.classList.toggle("is-success", state.status === "succeeded");
    indicator.lastChild.textContent = ACTIVE_STATUSES.has(state.status) ? "LIVE" : state.status === "succeeded" ? "VERIFIED" : "STANDBY";
  }
}

function renderEvents(events = []) {
  if (!Array.isArray(events)) return;
  const incoming = events
    .filter((event) => event && typeof event === "object" && Number.isInteger(event.seq))
    .sort((left, right) => left.seq - right.seq);
  const newestStart = [...incoming]
    .reverse()
    .find((event) => event.status === "starting" && event.stage === "starting");
  if (newestStart && newestStart.seq > ui.eventRunStartSeq) {
    ui.eventRunStartSeq = newestStart.seq;
    ui.eventHistory = [];
  }
  const merged = new Map(ui.eventHistory.map((event) => [event.seq, event]));
  incoming.forEach((event) => {
    if (ui.eventRunStartSeq < 0 || event.seq >= ui.eventRunStartSeq) merged.set(event.seq, event);
  });
  ui.eventHistory = [...merged.values()]
    .sort((left, right) => left.seq - right.seq)
    .slice(-16);
  if (!ui.eventHistory.length) return;
  const list = $("#event-list");
  if (!list) return;
  const fragment = document.createDocumentFragment();
  ui.eventHistory.forEach((event) => {
    const item = document.createElement("li");
    const time = document.createElement("time");
    const message = document.createElement("span");
    const timestamp = typeof event.timestamp === "string" ? new Date(event.timestamp) : null;
    if (timestamp && !Number.isNaN(timestamp.valueOf())) {
      time.dateTime = event.timestamp.slice(0, 128);
      time.textContent = timestamp.toLocaleTimeString("ko-KR", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      });
    } else {
      time.textContent = `#${String(event.seq).padStart(3, "0")}`;
    }
    time.title = `event #${String(event.seq).padStart(3, "0")}`;
    const stage = event.stage || event.kind || "event";
    message.textContent = String(event.message || `${PHASE_LABELS[stage] || stage} · ${event.progress ?? "—"}%`).slice(0, 512);
    item.append(time, message);
    fragment.append(item);
  });
  list.replaceChildren(fragment);
}

function updateControlStates() {
  const validationActive = ACTIVE_STATUSES.has(ui.lastStatus);
  const agentActive = AGENT_ACTIVE_STATUSES.has(ui.agentStatus);
  const agentUnavailable = ui.agentStatus === "offline" || ui.agentConnectionLost;
  const evolutionActive = ui.evolutionRunning || ui.evolutionRequestPending;
  const validationTerminal = ["succeeded", "failed", "cancelled"].includes(ui.lastStatus);

  $$('[data-action="run"]').forEach((button) => {
    const busyElsewhere = agentActive || evolutionActive || ui.agentConnectionLost;
    const disabled = validationActive || busyElsewhere || ui.validationRequestPending || ui.validationConnectionLost;
    button.disabled = disabled;
    button.setAttribute("aria-busy", String(validationActive || ui.validationRequestPending));
    const label = $("[data-run-label]", button);
    if (label) {
      label.textContent = ui.validationConnectionLost || ui.agentConnectionLost
        ? "연결 확인 중"
        : ui.validationRequestPending
        ? "시작 중"
        : validationActive
          ? "검증 중"
          : busyElsewhere
            ? "GPU 사용 중"
            : validationTerminal
              ? "다시 검증"
              : button.dataset.readyLabel;
    }
  });
  $$('[data-action="cancel"]').forEach((button) => {
    button.disabled = !validationActive || ui.lastStatus === "cancelling" || ui.validationCancelPending;
    button.setAttribute("aria-busy", String(ui.validationCancelPending));
  });

  const agentBusy = agentActive || ui.agentRequestPending;
  const agentComputeBlocked = validationActive || evolutionActive;
  const send = $('[data-action="agent-send"]');
  if (send) {
    send.disabled = agentUnavailable || agentBusy || agentComputeBlocked;
    send.setAttribute("aria-busy", String(agentBusy));
  }
  const sendLabel = $("[data-agent-send-label]", send || document);
  if (sendLabel) {
    sendLabel.textContent = ui.agentRequestPending
      ? "전송 중"
      : ui.agentStatus === "cancelling"
        ? "중단 중"
        : ui.agentStatus === "executing"
          ? "작업 중"
          : ["starting", "loading", "generating"].includes(ui.agentStatus)
            ? "응답 중"
            : agentComputeBlocked
              ? "GPU 사용 중"
              : "보내기";
  }
  const cancel = $('[data-action="agent-cancel"]');
  if (cancel) {
    cancel.disabled = !agentActive || ui.agentStatus === "cancelling" || ui.agentCancelPending;
    cancel.setAttribute("aria-busy", String(ui.agentCancelPending));
  }
  const reset = $('[data-action="agent-reset"]');
  if (reset) reset.disabled = agentUnavailable || agentBusy;
  const input = $("#agent-input");
  if (input) input.disabled = agentUnavailable || agentBusy;
  $$('[data-agent-mode], [data-agent-prompt]').forEach((button) => {
    button.disabled = agentUnavailable || agentBusy;
  });
  const chatBusy = agentBusy || ui.agentConnectionLost;
  $(".chat-workspace")?.setAttribute("aria-busy", String(chatBusy));
  $("#chat-transcript")?.setAttribute("aria-busy", String(chatBusy));
  document.body.classList.toggle("agent-active", agentActive);

  const evolution = $('[data-action="evolution-run"]');
  if (evolution) {
    evolution.disabled = evolutionActive || validationActive || agentActive || agentUnavailable;
    evolution.setAttribute("aria-busy", String(evolutionActive));
  }
  const evolutionLabel = $("[data-evolution-label]", evolution || document);
  if (evolutionLabel) {
    const readyLabel = ui.evolutionPromotionEnabled
      ? "안전한 야간 주기 실행"
      : "제안 전용 야간 주기 실행";
    evolutionLabel.textContent = evolutionActive
      ? "야간 주기 실행 중"
      : validationActive || agentActive
        ? "다른 작업 완료 대기"
        : readyLabel;
  }
}

function updateState(state) {
  if (!state || typeof state !== "object") return;
  const prior = ui.lastStatus;
  const status = VALIDATION_STATUSES.has(state.status) ? state.status : "failed";
  ui.lastStatus = status;
  if (Number.isInteger(state.seq)) ui.lastSeq = Math.max(ui.lastSeq, state.seq);
  setRuntimeStatus(status, state.stage);
  updateMetrics(state.metrics || {});
  updatePhases({ ...state, status });
  renderEvents(state.events || []);

  const active = ACTIVE_STATUSES.has(status);
  document.body.classList.toggle("runtime-active", active);
  updateControlStates();

  const railSeal = $(".rail-seal");
  if (railSeal) railSeal.dataset.state = status;

  if (active) {
    setText("#rail-verdict", "LIVE RUNNING");
    setText("#rail-time", "이전 실측값 표시 · 완료 후 교체");
  } else if (status === "succeeded") {
    setText("#rail-verdict", "VERIFIED");
    if (prior !== "succeeded") showToast("실제 GPU 통합 검증을 통과했습니다.", "success");
  } else if (status === "failed") {
    setText("#rail-verdict", "NEW RUN FAILED");
    setText("#rail-time", "이전 실측값 유지 · 실패 로그 확인");
    const error = state.error?.message || state.error?.code || "검증 프로세스가 실패했습니다.";
    if (prior !== "failed") showToast(error, "error");
  } else if (status === "cancelled" && prior !== "cancelled") {
    setText("#rail-verdict", "RUN CANCELLED");
    setText("#rail-time", "이전 실측값 유지");
    showToast("검증을 취소하고 GPU worker를 정리했습니다.", "warning");
  } else if (status === "ready") {
    setText("#rail-verdict", "VERIFIED");
  }
}

function formatAgentTime(value) {
  if (typeof value !== "string") return "";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "";
  return date.toLocaleTimeString("ko-KR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function createAgentMessageElement() {
  const item = document.createElement("article");
  item.className = "chat-message";
  const avatar = document.createElement("span");
  avatar.className = "chat-avatar";
  avatar.setAttribute("aria-hidden", "true");
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  const header = document.createElement("header");
  const author = document.createElement("strong");
  const meta = document.createElement("span");
  meta.className = "chat-message-meta";
  const completion = document.createElement("span");
  completion.className = "chat-completion-status";
  completion.setAttribute("role", "status");
  const time = document.createElement("time");
  const content = document.createElement("p");
  const continueButton = document.createElement("button");
  continueButton.className = "chat-continue-button";
  continueButton.type = "button";
  continueButton.textContent = "이어서 작성";
  continueButton.hidden = true;
  continueButton.addEventListener("click", () => {
    ui.agentMode = "chat";
    $$('[data-agent-mode]').forEach((candidate) => {
      const selected = candidate.dataset.agentMode === "chat";
      candidate.classList.toggle("is-selected", selected);
      candidate.setAttribute("aria-pressed", String(selected));
    });
    const input = $("#agent-input");
    if (input) {
      input.value = "계속 이어서 답해주세요.";
      updateAgentCharacterCount();
      input.focus();
    }
  });
  meta.append(completion, time);
  header.append(author, meta);
  bubble.append(header, content, continueButton);
  item.append(avatar, bubble);
  return item;
}

function renderAgentConversation(messages = []) {
  if (!Array.isArray(messages)) return;
  const transcript = $("#chat-transcript");
  if (!transcript) return;
  if (!ui.chatEmptyTemplate) {
    const empty = $("#chat-empty", transcript);
    if (empty) ui.chatEmptyTemplate = empty.cloneNode(true);
  }
  const nearBottom = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 96;
  const priorScrollTop = transcript.scrollTop;
  const seen = new Set();
  const normalized = [];
  messages.slice(-MAX_AGENT_DOM_MESSAGES).forEach((message, index) => {
    if (!message || typeof message !== "object" || typeof message.content !== "string") return;
    const role = ["user", "assistant", "tool", "system"].includes(message.role) ? message.role : "assistant";
    const rawId = typeof message.id === "string" && message.id.length <= 128
      ? message.id
      : `${role}-${index}-${String(message.created_at || "").slice(0, 128)}`;
    let key = rawId;
    let duplicate = 0;
    while (seen.has(key)) {
      duplicate += 1;
      key = `${rawId}-${index}-${duplicate}`;
    }
    seen.add(key);
    normalized.push({
      key,
      role,
      content: message.content.slice(0, MAX_AGENT_RESPONSE_CHARS),
      createdAt: typeof message.created_at === "string" ? message.created_at.slice(0, 128) : "",
      streaming: message.streaming === true,
      finishReason: ["stop", "length", "cancelled", "error", "tool"].includes(message.finish_reason)
        ? message.finish_reason
        : null,
      continuations: Number.isInteger(message.continuations)
        ? Math.max(0, Math.min(2, message.continuations))
        : 0,
      truncated: message.truncated === true,
      generatedTokens: Number.isInteger(message.generated_tokens)
        ? Math.max(0, Math.min(1536, message.generated_tokens))
        : 0,
    });
  });

  if (!normalized.length) {
    if (ui.chatEmptyTemplate) transcript.replaceChildren(ui.chatEmptyTemplate.cloneNode(true));
    return;
  }

  $("#chat-empty", transcript)?.remove();
  const existing = new Map();
  $$(".chat-message", transcript).forEach((item) => {
    const key = item.dataset.messageId;
    if (!key || existing.has(key)) item.remove();
    else existing.set(key, item);
  });
  normalized.forEach((message) => {
    const item = existing.get(message.key) || createAgentMessageElement();
    existing.delete(message.key);
    item.dataset.messageId = message.key;
    item.classList.toggle("is-streaming", message.streaming);
    item.classList.toggle("is-truncated", message.truncated);
    item.setAttribute("aria-busy", String(message.streaming));
    const role = message.role;
    item.dataset.role = role;
    const avatar = $(".chat-avatar", item);
    avatar.textContent = { user: "YOU", assistant: "AI", tool: "TOOL", system: "SYS" }[role];
    const author = $(".chat-bubble header strong", item);
    author.textContent = { user: "사용자", assistant: "Cogni Agent", tool: "로컬 작업", system: "시스템" }[role];
    const time = $(".chat-bubble header time", item);
    time.textContent = formatAgentTime(message.createdAt);
    if (time.textContent) time.dateTime = message.createdAt;
    else time.removeAttribute("datetime");
    const completion = $(".chat-completion-status", item);
    if (completion) {
      let completionCopy = "";
      if (role === "assistant" && message.streaming) completionCopy = "작성 중";
      else if (role === "assistant" && message.truncated) completionCopy = "길이 한계 · 이어서 가능";
      else if (role === "assistant" && message.finishReason) {
        completionCopy = message.continuations
          ? `자동 이어쓰기 ${message.continuations}회 · 완료`
          : "완료";
      }
      completion.textContent = completionCopy;
      completion.hidden = !completionCopy;
      completion.title = message.generatedTokens
        ? `생성 ${message.generatedTokens.toLocaleString("ko-KR")} 토큰`
        : "";
    }
    const content = $(".chat-bubble p", item);
    content.textContent = message.content;
    const continueButton = $(".chat-continue-button", item);
    if (continueButton) continueButton.hidden = role !== "assistant" || !message.truncated || message.streaming;
    transcript.append(item);
  });
  existing.forEach((item) => item.remove());
  transcript.scrollTop = nearBottom ? transcript.scrollHeight : priorScrollTop;
}

function updateAgentCore(core = {}) {
  const active = new Set(Array.isArray(core.active_modules) ? core.active_modules : []);
  $$('[data-agent-module]').forEach((module) => {
    const enabled = active.has(module.dataset.agentModule);
    module.classList.toggle("is-active", enabled);
    const state = $("b", module);
    if (state && enabled) state.textContent = "ACTIVE";
    else if (state && core.modules && typeof core.modules[module.dataset.agentModule] === "string") {
      state.textContent = core.modules[module.dataset.agentModule].slice(0, 32).toUpperCase();
    } else if (state) {
      state.textContent = AGENT_MODULE_DEFAULTS[module.dataset.agentModule] || "READY";
    }
  });
  const badge = $("#agent-core-badge");
  if (badge) badge.textContent = typeof core.verdict === "string" ? core.verdict.slice(0, 64) : "대기";
}

function updateEvolutionState(evolution = {}) {
  if (!evolution || typeof evolution !== "object") return;
  if (Number.isInteger(evolution.failures)) setText("#evolution-failures", evolution.failures);
  if (typeof evolution.last_run === "string" && evolution.last_run) {
    setText("#evolution-last-run", formatAgentTime(evolution.last_run) || evolution.last_run);
  }
  if (typeof evolution.sandbox === "string") setText("#evolution-sandbox", evolution.sandbox);
  const badge = $("#evolution-badge");
  if (badge && typeof evolution.status === "string") badge.textContent = evolution.status.slice(0, 64);
  if (typeof evolution.running === "boolean") {
    ui.evolutionRunning = evolution.running;
  }
  if (typeof evolution.promotion_enabled === "boolean") {
    ui.evolutionPromotionEnabled = evolution.promotion_enabled;
  }
  const button = $('[data-action="evolution-run"]');
  if (button && typeof evolution.blocked_reason === "string") {
    button.title = evolution.blocked_reason.slice(0, 256);
  } else if (button) {
    button.removeAttribute("title");
  }
  if (evolution.error && typeof evolution.error.message === "string") {
    const errorKey = `${evolution.error.code || "ERROR"}:${evolution.error.message}`;
    if (errorKey !== ui.lastEvolutionErrorKey) {
      ui.lastEvolutionErrorKey = errorKey;
      showToast(evolution.error.message, "error");
    }
  } else if (!ui.evolutionRunning) {
    ui.lastEvolutionErrorKey = "";
  }
  updateControlStates();
}

function updateAgentState(state) {
  if (!state || typeof state !== "object") return;
  if (Number.isInteger(state.seq)) ui.agentSeq = Math.max(ui.agentSeq, state.seq);
  ui.agentStatus = AGENT_STATUSES.has(state.status) ? state.status : "offline";
  const labels = {
    offline: "OFFLINE",
    starting: "STARTING",
    loading: "MODEL LOADING",
    ready: "READY",
    generating: "THINKING",
    executing: "WORKING",
    cancelling: "STOPPING",
    succeeded: "READY",
    cancelled: "CANCELLED",
    failed: "FAILED",
  };
  setText("#agent-status-label", labels[ui.agentStatus] || ui.agentStatus.toUpperCase());
  const heading = $("#agent-heading-state");
  if (heading) heading.dataset.state = ui.agentStatus;
  if (Array.isArray(state.conversation)) renderAgentConversation(state.conversation);
  updateAgentCore(state.core || {});
  updateEvolutionState(state.evolution || {});
  updateControlStates();
  if (state.error && typeof state.error.message === "string") {
    const errorKey = `${state.error.code || "ERROR"}:${state.error.message}`;
    if (errorKey !== ui.lastAgentErrorKey) {
      ui.lastAgentErrorKey = errorKey;
      showToast(state.error.message, "error");
    }
  } else if (ui.agentStatus !== "failed") {
    ui.lastAgentErrorKey = "";
  }
}

function updateAgentCharacterCount() {
  const length = $("#agent-input")?.value.length || 0;
  setText("#agent-char-count", `${length.toLocaleString("ko-KR")} / 4,096자`);
}

function describeApiError(error, fallback) {
  const code = typeof error?.code === "string" ? error.code : "";
  if (API_ERROR_COPY[code]) return API_ERROR_COPY[code];
  if (code === "CONNECTION_LOST") return "로컬 제어 서비스와 연결할 수 없습니다. 데모가 실행 중인지 확인해 주세요.";
  if (code.startsWith("HTTP_5")) return "로컬 서비스가 일시적으로 응답하지 않습니다. 잠시 후 다시 시도해 주세요.";
  return fallback;
}

async function sendAgentMessage() {
  if (ui.agentRequestPending || AGENT_ACTIVE_STATUSES.has(ui.agentStatus)) return;
  const input = $("#agent-input");
  const message = input?.value.trim() || "";
  if (!message) {
    showToast("메시지를 입력해 주세요.", "warning");
    input?.focus();
    return;
  }
  ui.agentRequestPending = true;
  updateControlStates();
  try {
    const state = await api("/api/agent/chat", {
      method: "POST",
      body: { message, mode: ui.agentMode },
    });
    if (input) input.value = "";
    updateAgentCharacterCount();
    updateAgentState(state);
  } catch (error) {
    showToast(describeApiError(error, "요청을 시작하지 못했습니다."), "error");
  } finally {
    ui.agentRequestPending = false;
    updateControlStates();
  }
}

async function cancelAgentTurn() {
  if (ui.agentCancelPending || !AGENT_ACTIVE_STATUSES.has(ui.agentStatus)) return;
  ui.agentCancelPending = true;
  updateControlStates();
  try {
    updateAgentState(await api("/api/agent/cancel", { method: "POST", body: {} }));
  } catch (error) {
    showToast(describeApiError(error, "요청을 중단하지 못했습니다."), "error");
  } finally {
    ui.agentCancelPending = false;
    updateControlStates();
  }
}

async function resetAgentConversation() {
  if (ui.agentRequestPending || AGENT_ACTIVE_STATUSES.has(ui.agentStatus)) {
    showToast("현재 요청이 끝난 뒤 새 대화를 시작할 수 있습니다.", "warning");
    return;
  }
  ui.agentRequestPending = true;
  updateControlStates();
  try {
    ui.agentSeq = -1;
    const state = await api("/api/agent/reset", { method: "POST", body: {} });
    const input = $("#agent-input");
    if (input) input.value = "";
    updateAgentCharacterCount();
    updateAgentState(state);
  } catch (error) {
    showToast(describeApiError(error, "대화를 초기화하지 못했습니다."), "error");
  } finally {
    ui.agentRequestPending = false;
    updateControlStates();
  }
}

async function runEvolutionCycle() {
  if (ui.evolutionRequestPending || ui.evolutionRunning) return;
  ui.evolutionRequestPending = true;
  updateControlStates();
  try {
    const state = await api("/api/evolution/run", { method: "POST", body: {} });
    updateEvolutionState(state.evolution || state);
    showToast("Self-Harness 안전 주기를 시작했습니다.");
  } catch (error) {
    showToast(describeApiError(error, "Self-Harness 주기를 시작하지 못했습니다."), "error");
  } finally {
    ui.evolutionRequestPending = false;
    updateControlStates();
  }
}

async function pollAgentState() {
  if (ui.agentPollStopped) return;
  const path = ui.evolutionRunning || ui.agentSeq < 0
    ? "/api/agent/state"
    : `/api/agent/state?after=${ui.agentSeq}`;
  try {
    const state = await api(path);
    if (ui.agentConnectionLost && Number.isInteger(state.seq) && state.seq < ui.agentSeq) ui.agentSeq = -1;
    if (ui.agentConnectionLost) showToast("로컬 AI 연결이 복구되었습니다.", "success");
    ui.agentConnectionLost = false;
    updateAgentState(state);
  } catch (_) {
    if (!ui.agentConnectionLost) showToast("로컬 AI 연결을 다시 확인하고 있습니다.", "warning");
    ui.agentConnectionLost = true;
    updateAgentState({ status: "offline" });
  } finally {
    const active = AGENT_ACTIVE_STATUSES.has(ui.agentStatus);
    const delay = active ? 100 : ui.evolutionRunning ? 350 : 900;
    if (!ui.agentPollStopped) setTimeout(pollAgentState, delay);
  }
}

async function api(path, options = {}) {
  const request = {
    method: options.method || "GET",
    cache: "no-store",
    credentials: "same-origin",
    headers: {},
  };
  if (options.body !== undefined) {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(options.body);
  }
  let response;
  try {
    response = await fetch(path, request);
  } catch (cause) {
    const error = new Error("CONNECTION_LOST");
    error.code = "CONNECTION_LOST";
    error.cause = cause;
    throw error;
  }
  let payload = {};
  try { payload = await response.json(); } catch (_) { payload = {}; }
  if (!response.ok) {
    const code = payload?.error?.code || `HTTP_${response.status}`;
    const error = new Error(code);
    error.code = code;
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function runValidation() {
  if (ui.validationRequestPending || ACTIVE_STATUSES.has(ui.lastStatus)) return;
  const scenario = SCENARIOS[ui.selectedScenario];
  ui.validationRequestPending = true;
  updateControlStates();
  try {
    const state = await api("/api/run", { method: "POST", body: { prompt: scenario.prompt } });
    switchView("inference");
    updateState(state);
    showToast("로컬 Gemma 통합 검증을 시작했습니다.");
  } catch (error) {
    if (error.code === "JOB_ALREADY_RUNNING") {
      switchView("inference");
      showToast("이미 검증이 진행 중입니다.", "warning");
    } else {
      showToast(describeApiError(error, "검증을 시작하지 못했습니다."), "error");
    }
  } finally {
    ui.validationRequestPending = false;
    updateControlStates();
  }
}

async function cancelValidation() {
  if (ui.validationCancelPending || !ACTIVE_STATUSES.has(ui.lastStatus)) return;
  ui.validationCancelPending = true;
  updateControlStates();
  try {
    const state = await api("/api/cancel", { method: "POST", body: {} });
    updateState(state);
    showToast("안전한 worker 종료를 요청했습니다.", "warning");
  } catch (error) {
    showToast(describeApiError(error, "검증 취소를 요청하지 못했습니다."), "error");
  } finally {
    ui.validationCancelPending = false;
    updateControlStates();
  }
}

async function shutdownDemo() {
  if (ACTIVE_STATUSES.has(ui.lastStatus) || AGENT_ACTIVE_STATUSES.has(ui.agentStatus) || ui.evolutionRunning) {
    const confirmed = window.confirm("실행 중인 로컬 작업을 안전하게 중단하고 CogniBoard를 종료할까요?");
    if (!confirmed) return;
  }
  try {
    await api("/api/shutdown", { method: "POST", body: {} });
  } catch (_) {
    // The server can close the socket immediately after accepting shutdown.
  }
  ui.pollStopped = true;
  ui.agentPollStopped = true;
  $("#shutdown-screen").hidden = false;
}

function updateFullscreenState() {
  const button = $('[data-action="fullscreen"]');
  if (!button) return;
  const active = Boolean(document.fullscreenElement);
  button.setAttribute("aria-pressed", String(active));
  button.setAttribute("aria-label", active ? "발표 전체화면 종료" : "발표 전체화면 시작");
  button.title = active ? "전체화면 종료 (Esc)" : "발표 전체화면";
  document.body.classList.toggle("fullscreen-active", active);
}

async function toggleFullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch (_) {
    showToast("이 브라우저에서는 전체화면을 시작할 수 없습니다.", "warning");
  }
}

async function pollState() {
  if (ui.pollStopped) return;
  const path = ui.lastSeq >= 0 ? `/api/state?after=${ui.lastSeq}` : "/api/state";
  try {
    const state = await api(path);
    if (ui.validationConnectionLost && Number.isInteger(state.seq) && state.seq < ui.lastSeq) {
      ui.lastSeq = -1;
      ui.eventHistory = [];
      ui.eventRunStartSeq = -1;
    }
    if (ui.validationConnectionLost) showToast("검증 제어 연결이 복구되었습니다.", "success");
    ui.validationConnectionLost = false;
    updateState(state);
  } catch (_) {
    if (!ui.validationConnectionLost) showToast("검증 제어 연결을 다시 확인하고 있습니다.", "warning");
    ui.validationConnectionLost = true;
    if (!ui.pollStopped) setRuntimeStatus("offline", "연결 재시도 중");
    updateControlStates();
  } finally {
    if (!ui.pollStopped) setTimeout(pollState, ACTIVE_STATUSES.has(ui.lastStatus) ? 80 : 700);
  }
}

function updateTour() {
  const step = TOUR[ui.tourIndex];
  switchView(step.view, { skipHash: true, instant: true });
  setText("#tour-kicker", step.kicker);
  setText("#tour-title", step.title);
  setText("#tour-copy", step.copy);
  setText("#tour-count", `${String(ui.tourIndex + 1).padStart(2, "0")} / ${String(TOUR.length).padStart(2, "0")}`);
  const progress = $("#tour-progress");
  if (progress) { progress.max = TOUR.length; progress.value = ui.tourIndex + 1; }
  const previous = $('[data-action="tour-prev"]');
  if (previous) previous.disabled = ui.tourIndex === 0;
  const next = $('[data-action="tour-next"]');
  if (next) next.textContent = ui.tourIndex === TOUR.length - 1 ? "실제 검증 시작" : "다음 →";
}

function openTour() {
  ui.tourIndex = 0;
  $("#tour-panel").hidden = false;
  document.body.classList.add("tour-open");
  updateTour();
  $('[data-action="tour-next"]')?.focus();
}

function closeTour() {
  $("#tour-panel").hidden = true;
  document.body.classList.remove("tour-open");
}

function nextTour() {
  if (ui.tourIndex >= TOUR.length - 1) {
    closeTour();
    runValidation();
    return;
  }
  ui.tourIndex += 1;
  updateTour();
}

function previousTour() {
  if (ui.tourIndex <= 0) return;
  ui.tourIndex -= 1;
  updateTour();
}

function bindNavigation() {
  $$('[data-view]').forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  $$('[data-view-jump]').forEach((button) => button.addEventListener("click", () => switchView(button.dataset.viewJump)));
  $$('[data-view-link]').forEach((link) => link.addEventListener("click", (event) => {
    event.preventDefault();
    switchView(link.dataset.viewLink);
  }));
  const initial = location.hash.slice(1);
  if (VIEW_IDS.has(initial)) switchView(initial, { skipHash: true, instant: true });
}

function bindStories() {
  $$('[data-story]').forEach((button) => button.addEventListener("click", () => {
    $$('[data-story]').forEach((item) => item.classList.toggle("is-active", item === button));
    const story = STORY[button.dataset.story];
    const detail = $("#story-detail");
    $(".story-index", detail).textContent = story.index;
    $("p", detail).textContent = story.copy;
  }));
}

function bindScenarios() {
  $$('[data-scenario]').forEach((button) => button.addEventListener("click", () => {
    ui.selectedScenario = button.dataset.scenario;
    $$('[data-scenario]').forEach((item) => {
      const selected = item === button;
      item.classList.toggle("is-selected", selected);
      item.setAttribute("aria-pressed", String(selected));
    });
    setText("#scenario-copy", SCENARIOS[ui.selectedScenario].copy);
  }));
}

function bindArchitecture() {
  $$('[data-node]').forEach((button) => button.addEventListener("click", () => {
    $$('[data-node]').forEach((item) => item.classList.toggle("is-selected", item === button));
    const [title, copy, status] = NODE_COPY[button.dataset.node];
    const panel = $("#node-detail");
    $("h2", panel).textContent = title;
    $("p", panel).textContent = copy;
    $("div strong", panel).textContent = status;
  }));
}

function bindActions() {
  $$('[data-action="run"]').forEach((button) => button.addEventListener("click", runValidation));
  $$('[data-action="cancel"]').forEach((button) => button.addEventListener("click", cancelValidation));
  $$('[data-action="shutdown"]').forEach((button) => button.addEventListener("click", shutdownDemo));
  $$('[data-action="fullscreen"]').forEach((button) => button.addEventListener("click", toggleFullscreen));
  $('[data-action="agent-send"]')?.addEventListener("click", sendAgentMessage);
  $('[data-action="agent-cancel"]')?.addEventListener("click", cancelAgentTurn);
  $('[data-action="agent-reset"]')?.addEventListener("click", resetAgentConversation);
  $('[data-action="evolution-run"]')?.addEventListener("click", runEvolutionCycle);
  $$('[data-agent-mode]').forEach((button) => button.addEventListener("click", () => {
    ui.agentMode = button.dataset.agentMode;
    $$('[data-agent-mode]').forEach((candidate) => {
      const selected = candidate === button;
      candidate.classList.toggle("is-selected", selected);
      candidate.setAttribute("aria-pressed", String(selected));
    });
    $("#agent-input")?.focus();
  }));
  $$('[data-agent-prompt]').forEach((button) => button.addEventListener("click", () => {
    const mode = button.dataset.agentModeValue || "chat";
    ui.agentMode = mode;
    $$('[data-agent-mode]').forEach((candidate) => {
      const selected = candidate.dataset.agentMode === mode;
      candidate.classList.toggle("is-selected", selected);
      candidate.setAttribute("aria-pressed", String(selected));
    });
    const input = $("#agent-input");
    if (input) {
      input.value = button.dataset.agentPrompt || "";
      updateAgentCharacterCount();
      input.focus();
    }
  }));
  $$('[data-action="toggle-log"]').forEach((button) => button.addEventListener("click", () => {
    const log = $("#event-log");
    log.hidden = !log.hidden;
    if (!log.hidden) {
      switchView("inference");
      log.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }));
  $('[data-action="tour"]')?.addEventListener("click", openTour);
  $('[data-action="tour-close"]')?.addEventListener("click", closeTour);
  $('[data-action="tour-next"]')?.addEventListener("click", nextTour);
  $('[data-action="tour-prev"]')?.addEventListener("click", previousTour);
}

function bindKeyboard() {
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("#tour-panel").hidden) closeTour();
    if (event.altKey && /^[1-6]$/.test(event.key)) {
      const views = ["assistant", "mission", "inference", "architecture", "business", "evidence"];
      switchView(views[Number(event.key) - 1], { focus: true });
    }
  });
  $("#agent-input")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      sendAgentMessage();
    }
  });
  $("#agent-input")?.addEventListener("input", updateAgentCharacterCount);
}

function init() {
  bindNavigation();
  bindStories();
  bindScenarios();
  bindArchitecture();
  bindActions();
  bindKeyboard();
  document.addEventListener("fullscreenchange", updateFullscreenState);
  updateFullscreenState();
  updateAgentCharacterCount();
  updateControlStates();
  pollAgentState();
  pollState();
}

document.addEventListener("DOMContentLoaded", init);
