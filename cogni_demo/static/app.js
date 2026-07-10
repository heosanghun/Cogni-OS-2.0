"use strict";

const ACTIVE_STATUSES = new Set(["starting", "running", "cancelling"]);
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
  verifying: "무결성 검증",
  loading_model: "로컬 모델 적재",
  building_runtime: "런타임 구성",
  running_inference: "Depth 100 탐색",
  postcheck: "안전 조건 확인",
  succeeded: "검증 완료",
  failed: "검증 실패",
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
  currentView: "mission",
  selectedScenario: "defense",
  lastSeq: -1,
  lastStatus: "ready",
  pollStopped: false,
  tourIndex: 0,
  toastTimer: null,
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
    element.textContent = String(value);
  }
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function showToast(message, tone = "info") {
  const toast = $("#toast");
  if (!toast) return;
  toast.dataset.tone = tone;
  $("span", toast).textContent = message;
  toast.hidden = false;
  clearTimeout(ui.toastTimer);
  ui.toastTimer = setTimeout(() => {
    toast.hidden = true;
  }, 4200);
}

function switchView(view, options = {}) {
  const panel = $(`[data-view-panel="${view}"]`);
  if (!panel) return;
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
  }[status] || String(status || "READY").toUpperCase();
  label.textContent = display;
  pill.dataset.state = status;
  pill.title = PHASE_LABELS[stage] || display;
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
  if (!Array.isArray(events) || !events.length) return;
  const list = $("#event-list");
  if (!list) return;
  const fragment = document.createDocumentFragment();
  events.slice(-16).forEach((event) => {
    const item = document.createElement("li");
    const time = document.createElement("time");
    const message = document.createElement("span");
    time.textContent = String(event.seq ?? "—").padStart(3, "0");
    const stage = event.stage || event.kind || "event";
    message.textContent = event.message || `${PHASE_LABELS[stage] || stage} · ${event.progress ?? "—"}%`;
    item.append(time, message);
    fragment.append(item);
  });
  list.replaceChildren(fragment);
}

function updateState(state) {
  if (!state || typeof state !== "object") return;
  const prior = ui.lastStatus;
  ui.lastStatus = state.status || "ready";
  if (Number.isInteger(state.seq)) ui.lastSeq = Math.max(ui.lastSeq, state.seq);
  setRuntimeStatus(state.status, state.stage);
  updateMetrics(state.metrics || {});
  updatePhases(state);
  renderEvents(state.events || []);

  const active = ACTIVE_STATUSES.has(state.status);
  $$('[data-action="run"]').forEach((button) => { button.disabled = active; });
  $$('[data-action="cancel"]').forEach((button) => { button.disabled = !active || state.status === "cancelling"; });
  document.body.classList.toggle("runtime-active", active);

  const railSeal = $(".rail-seal");
  if (railSeal) railSeal.dataset.state = state.status || "ready";

  if (active) {
    setText("#rail-verdict", "LIVE RUNNING");
    setText("#rail-time", "이전 실측값 표시 · 완료 후 교체");
  } else if (state.status === "succeeded") {
    setText("#rail-verdict", "VERIFIED");
    if (prior !== "succeeded") showToast("실제 GPU 통합 검증을 통과했습니다.", "success");
  } else if (state.status === "failed") {
    setText("#rail-verdict", "NEW RUN FAILED");
    setText("#rail-time", "이전 실측값 유지 · 실패 로그 확인");
    const error = state.error?.message || state.error?.code || "검증 프로세스가 실패했습니다.";
    if (prior !== "failed") showToast(error, "error");
  } else if (state.status === "cancelled" && prior !== "cancelled") {
    setText("#rail-verdict", "RUN CANCELLED");
    setText("#rail-time", "이전 실측값 유지");
    showToast("검증을 취소하고 GPU worker를 정리했습니다.", "warning");
  } else if (state.status === "ready") {
    setText("#rail-verdict", "VERIFIED");
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
  const response = await fetch(path, request);
  let payload = {};
  try { payload = await response.json(); } catch (_) { payload = {}; }
  if (!response.ok) {
    const code = payload?.error?.code || `HTTP_${response.status}`;
    const error = new Error(code);
    error.code = code;
    throw error;
  }
  return payload;
}

async function runValidation() {
  const scenario = SCENARIOS[ui.selectedScenario];
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
      showToast(`검증을 시작하지 못했습니다: ${error.code || error.message}`, "error");
    }
  }
}

async function cancelValidation() {
  try {
    const state = await api("/api/cancel", { method: "POST", body: {} });
    updateState(state);
    showToast("안전한 worker 종료를 요청했습니다.", "warning");
  } catch (error) {
    showToast(`취소 요청 실패: ${error.code || error.message}`, "error");
  }
}

async function shutdownDemo() {
  if (ACTIVE_STATUSES.has(ui.lastStatus)) {
    const confirmed = window.confirm("실행 중인 검증을 중단하고 CogniBoard를 종료할까요?");
    if (!confirmed) return;
  }
  try {
    await api("/api/shutdown", { method: "POST", body: {} });
  } catch (_) {
    // The server can close the socket immediately after accepting shutdown.
  }
  ui.pollStopped = true;
  $("#shutdown-screen").hidden = false;
}

async function pollState() {
  if (ui.pollStopped) return;
  const path = ui.lastSeq >= 0 ? `/api/state?after=${ui.lastSeq}` : "/api/state";
  try {
    const state = await api(path);
    updateState(state);
  } catch (error) {
    if (!ui.pollStopped) setRuntimeStatus("failed", "failed");
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
  if (next) next.textContent = ui.tourIndex === TOUR.length - 1 ? "완료" : "다음 →";
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
    showToast("IR 가이드를 완료했습니다. 실제 검증을 실행해 보세요.", "success");
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
  if ($(`[data-view-panel="${initial}"]`)) switchView(initial, { skipHash: true, instant: true });
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
    $$('[data-scenario]').forEach((item) => item.classList.toggle("is-selected", item === button));
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
    if (event.altKey && /^[1-5]$/.test(event.key)) {
      const views = ["mission", "inference", "architecture", "business", "evidence"];
      switchView(views[Number(event.key) - 1], { focus: true });
    }
  });
}

function init() {
  bindNavigation();
  bindStories();
  bindScenarios();
  bindArchitecture();
  bindActions();
  bindKeyboard();
  pollState();
}

document.addEventListener("DOMContentLoaded", init);
