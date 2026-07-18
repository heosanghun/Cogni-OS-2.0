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
const MAX_AGENT_CHAT_INPUT_CHARS = 4096;
const MAX_AGENT_PROJECT_INPUT_CHARS = 1 * 1024 * 1024;
const MAX_PROPOSAL_REVIEW_ITEMS = 8;
const MAX_PROPOSAL_DIFF_CHARS = 40000;
const MAX_PROPOSAL_REVIEW_TEXT_CHARS = 4096;
const MAX_ATTACHMENT_UPLOAD_BYTES = 8 * 1024 * 1024;
const MAX_ATTACHMENT_UPLOAD_COUNT = 32;
const MAX_LIVE_VRAM_LIMIT_GIB = 16.7;
const ATTACHMENT_CONTENT_ENDPOINT = "/api/workspace/attachments/content";
const RAG_SOURCE_ENDPOINT = "/api/workspace/rag/source";
const MAX_RAG_SOURCE_TEXT_CHARS = 12000;
const RAG_SOURCE_OFFSET_BASES = new Set([
  "normalized_document_text_v1",
  "normalized_pdf_page_text_v1",
]);
const RAG_SOURCE_REPRESENTATION = "normalized_extracted_excerpt_v1";
const RAG_SOURCE_EXACT_KEYS = Object.freeze([
  "schema_version",
  "attachment_id",
  "chunk_index",
  "name",
  "media_type",
  "text",
  "representation",
  "page_number",
  "char_start",
  "char_end",
  "offset_basis",
  "excerpt_sha256",
].sort());
const VOICE_SAMPLE_RATE = 16000;
const VOICE_MAX_SECONDS = 30;
const VOICE_MAX_SOURCE_RATE = 192000;
const VOICE_MAX_RECORDED_BYTES = 4 * 1024 * 1024;
const VOICE_RECORDER_STOP_TIMEOUT_MS = 5000;
const VOICE_DECODE_TIMEOUT_MS = 5000;
const MAX_TTS_TEXT_CHARS = 2000;
const MAX_TTS_WAV_BYTES = 8 * 1024 * 1024;
const MAX_TTS_BASE64_CHARS = Math.ceil(MAX_TTS_WAV_BYTES / 3) * 4;
// Capability metadata is necessary but not sufficient: controls stay fail-closed
// until the corresponding browser-side execution path is actually implemented.
const WEB_SEARCH_UI_IMPLEMENTED = true;
const MICROPHONE_CAPTURE_UI_IMPLEMENTED = true;
const IMAGE_CHAT_MEDIA_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const ATTACHMENT_MEDIA_BY_SUFFIX = Object.freeze({
  ".txt": "text/plain",
  ".md": "text/markdown",
  ".csv": "text/csv",
  ".json": "application/json",
  ".pdf": "application/pdf",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
});
const API_ERROR_COPY = {
  AGENT_UNAVAILABLE: "로컬 AI 서비스가 준비되지 않았습니다. 잠시 후 다시 시도해 주세요.",
  AGENT_BUSY: "현재 AI 요청이 끝난 뒤 다시 시도해 주세요.",
  AUTH_REQUIRED: "로컬 세션 인증이 만료되었습니다. 데모를 다시 실행해 주세요.",
  COMPUTE_BUSY: "다른 로컬 GPU 작업이 실행 중입니다. 완료 후 다시 시도해 주세요.",
  EVOLUTION_UNAVAILABLE: "Self-Harness 제어부가 준비되지 않았습니다.",
  PROPOSAL_REVIEW_INTEGRITY_FAILED: "제안 증거와 현재 소스가 일치하지 않아 diff를 표시하지 않았습니다.",
  PROPOSAL_REVIEW_UNAVAILABLE: "Self-Harness 제안 검토 자료를 안전하게 불러올 수 없습니다.",
  INVALID_BODY: "요청 형식이 올바르지 않습니다.",
  BODY_TOO_LARGE: "요청이 안전한 로컬 처리 크기 한도를 초과했습니다.",
  JOB_ALREADY_RUNNING: "이미 검증 작업이 실행 중입니다.",
  NO_ACTIVE_AGENT_TURN: "중단할 AI 요청이 없습니다.",
  NO_ACTIVE_JOB: "중단할 검증 작업이 없습니다.",
  AKASICDB_NOT_CONFIGURED: "감사된 로컬 AkasicDB 클론이 구성되지 않았습니다.",
  AKASICDB_UNAVAILABLE: "로컬 AkasicDB 인덱스를 사용할 수 없습니다.",
  ATTACHMENT_LIMIT_REACHED: "로컬 첨부 파일 개수 한도에 도달했습니다.",
  ATTACHMENT_CONTENT_UNAVAILABLE: "이 첨부에는 안전한 이미지 콘텐츠 경로가 없습니다.",
  ATTACHMENT_NOT_FOUND: "로컬 첨부 파일을 찾을 수 없습니다. 목록을 새로 불러와 주세요.",
  ATTACHMENT_TOTAL_BYTES_REACHED: "로컬 첨부 저장소의 전체 용량 한도에 도달했습니다.",
  ATTACHMENT_NOT_INDEXABLE: "이 파일 형식은 현재 로컬 RAG에 인덱싱할 수 없습니다.",
  ATTACHMENT_PREVIEW_UNAVAILABLE: "이 파일은 현재 안전한 미리보기를 제공하지 않습니다.",
  ATTACHMENT_TOO_LARGE: "파일이 로컬 첨부 크기 한도를 초과했습니다.",
  INVALID_ATTACHMENT: "첨부 파일을 확인할 수 없습니다.",
  IMAGE_MODEL_UNAVAILABLE: "검증된 로컬 Gemma 이미지 입력 경로가 준비되지 않았습니다.",
  INVALID_JSON_ATTACHMENT: "JSON 파일 형식이 올바르지 않습니다.",
  JSON_ATTACHMENT_TOO_LARGE: "JSON 파일은 1MiB 이하만 안전하게 처리합니다.",
  JSON_NESTING_TOO_DEEP: "JSON 파일의 중첩 깊이가 안전 한도를 초과했습니다.",
  PDF_ENCRYPTED: "암호화된 PDF는 로컬 텍스트 추출을 지원하지 않습니다.",
  PDF_NO_EXTRACTABLE_TEXT: "이 PDF에는 추출 가능한 텍스트가 없습니다.",
  PDF_PAGE_LIMIT: "PDF 페이지 수가 안전 처리 한도를 초과했습니다.",
  PDF_TEXT_EXTRACTION_FAILED: "로컬 PDF 텍스트 추출에 실패했습니다.",
  PDF_TEXT_EXTRACTION_UNAVAILABLE: "로컬 PDF 추출기(pypdf)가 설치되지 않았습니다.",
  PDF_TEXT_LIMIT: "PDF에서 추출된 텍스트가 안전 처리 한도를 초과했습니다.",
  RAG_REINDEX_FAILED: "로컬 RAG 재색인을 완료하지 못했습니다.",
  RAG_SOURCE_INTEGRITY_FAILED: "정규화 추출 발췌의 SHA-256 무결성 검증에 실패해 표시하지 않았습니다.",
  RAG_SOURCE_NOT_FOUND: "선택한 로컬 RAG 정규화 추출 발췌를 찾을 수 없습니다.",
  RAG_SOURCE_UNAVAILABLE: "선택한 로컬 RAG 정규화 추출 발췌를 안전하게 불러올 수 없습니다.",
  MODEL_NOT_VERIFIED: "검증된 로컬 모델만 선택할 수 있습니다.",
  MODEL_SWITCH_UNAVAILABLE: "모델은 검증되었지만 안전한 전환 기능은 아직 활성화되지 않았습니다.",
  LOCAL_AUDIO_PREPROCESS_FAILED: "검증된 로컬 Gemma 오디오 전처리에 실패했습니다.",
  LOCAL_AUDIO_PROCESSOR_REQUIRED: "검증된 로컬 Gemma 오디오 프로세서가 필요합니다.",
  LOCAL_STT_ARTIFACT_REQUIRED: "녹음·로컬 전송·WAV 검증은 완료했지만, 전사를 위한 검증된 로컬 STT 모델이 필요합니다.",
  LOCAL_STT_INFERENCE_UNVERIFIED: "로컬 음성 전사는 구성되었지만 실제 실행 검증 전입니다.",
  LOCAL_STT_FAILED: "검증된 로컬 음성 인식기가 전사를 완료하지 못했습니다.",
  LOCAL_STT_OUTPUT_INVALID: "로컬 음성 인식 결과가 안전한 텍스트 형식이 아닙니다.",
  LOCAL_TTS_ARTIFACT_REQUIRED: "검증된 Windows 로컬 음성이 없어 읽어주기를 사용할 수 없습니다.",
  LOCAL_TTS_HOST_PROBE_REQUIRED: "로컬 음성 합성기는 구성되었지만 호스트 실행 검증 전입니다.",
  LOCAL_TTS_FAILED: "검증된 로컬 음성 합성을 완료하지 못했습니다.",
  LOCAL_TTS_OUTPUT_INVALID: "로컬 음성 합성 결과가 안전한 WAV 형식이 아닙니다.",
  TTS_TEXT_INVALID: "읽어줄 답변은 1~2,000자의 안전한 텍스트여야 합니다.",
  LENS_AIR_GAP_BLOCKED: "Lens 검색은 명시적 온라인 모드에서만 사용할 수 있습니다.",
  LENS_AUTH_FAILED: "Lens API가 구성된 인증 정보를 거부했습니다.",
  LENS_BUSY: "다른 Lens 검색이 끝난 뒤 다시 시도해 주세요.",
  LENS_HOST_NOT_ALLOWLISTED: "api.lens.org가 외부 허용 목록에 없습니다.",
  LENS_INVALID_KIND: "Lens 검색 유형은 특허 또는 논문이어야 합니다.",
  LENS_INVALID_LIMIT: "Lens 결과 수 한도를 확인해 주세요.",
  LENS_INVALID_QUERY: "Lens 검색어는 1~512자로 입력해 주세요.",
  LENS_NETWORK_ERROR: "Lens 공식 API에 연결하지 못했습니다.",
  LENS_RATE_LIMITED: "Lens API 요청 한도에 도달했습니다. 잠시 후 다시 시도해 주세요.",
  LENS_REQUEST_REJECTED: "Lens API가 검색 요청을 거부했습니다.",
  LENS_SERVICE_UNAVAILABLE: "Lens API가 현재 응답할 수 없습니다.",
  LENS_TOKEN_REQUIRED: "Lens API 토큰이 구성되지 않았습니다.",
  LENS_TERMS_REQUIRED: "Lens API 이용약관과 출처 표시 의무에 대한 명시적 동의가 필요합니다.",
  UNSUPPORTED_MEDIA_TYPE: "현재 지원하지 않는 로컬 파일 형식입니다.",
  VOICE_AUDIO_TOO_LARGE: "음성은 최대 30초까지 녹음할 수 있습니다.",
  VOICE_BASE64_INVALID: "로컬 음성 전송 형식이 올바르지 않습니다.",
  VOICE_LANGUAGE_INVALID: "지원하지 않는 음성 언어 설정입니다.",
  VOICE_WAV_FORMAT_INVALID: "음성은 16kHz 모노 16-bit PCM WAV 형식이어야 합니다.",
  VOICE_WAV_INVALID: "녹음된 WAV 파일을 확인할 수 없습니다.",
  VOICE_WAV_TRUNCATED: "녹음된 WAV 데이터가 완전하지 않습니다.",
  VOICE_SILENCE_DETECTED: "말소리가 감지되지 않았습니다. 마이크를 가까이 두고 다시 시도해 주세요.",
};
const RUNTIME_CAPABILITY_NAMES = Object.freeze([
  "aflow",
  "bio_hama",
  "cts_deq",
  "gemma4_e4b",
  "self_harness",
  "system_1_5",
  "system_2_5",
  "system_3",
  "system_4",
]);
const RUNTIME_CAPABILITY_STATES = new Set([
  "disabled", "research", "advisory", "canary", "authoritative",
  "gated", "night_only", "proposal_only",
]);
const RUNTIME_EVIDENCE_CLASSES = new Set(["measured", "verified", "target", "plan"]);
const EXECUTION_STATES = new Set(["active", "standby", "not_loaded", "off"]);
const PHASE_ORDER = [
  "verifying",
  "loading_model",
  "building_runtime",
  "running_inference",
  "validating_decode_bridge",
  "postcheck",
];
const PHASE_TO_DOM = {
  verifying: "verify",
  loading_model: "model_load",
  building_runtime: "runtime_build",
  running_inference: "inference",
  validating_decode_bridge: "inference",
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
  validating_decode_bridge: "CTS 인과 브릿지 검증",
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
  rhythm: ["Bio Rhythm", "상태 기계는 추론과 진화의 동시 실행을 금지합니다. 이 제어 계약은 답변 권한과 별개입니다."],
  router: ["BIO-HAMA Meta Router", "부하·불확실성 텔레메트리를 관측하는 advisory 경로입니다. 현재 답변 토큰을 결정하지 않습니다."],
  aflow: ["AFlow", "봉인된 held-in/held-out 평가로 bounded workflow 후보를 탐색합니다. 결과는 production에 설치되지 않고 연구 archive에만 남습니다."],
  harness: ["Self-Harness", "성공 invariant와 실패 trace를 영속 저장하고, 서로 다른 K≥3 후보를 정책 검사 후 검토 대기 상태로 보존합니다. 실행·소스 수정·자동 승격 API는 없습니다."],
  gemma: ["Local Gemma 4", "검증된 로컬 경로와 manifest만 허용합니다. Hub ID·URL·remote code·KV cache는 거부됩니다."],
  deq: ["DEQ Equilibrium", "제한 이력 Broyden solver가 고정점 잔차를 계산하고, 미수렴·비유한 상태를 성공으로 표시하지 않습니다."],
  cts: ["Cognitive Tree Search", "사전 할당된 301-node arena와 bounded ancestor bank 안에서 탐색하는 canary answer path입니다."],
  fast: ["Fast Weight", "수렴 상태의 bounded 저랭크 overlay 후보입니다. admitted checkpoint가 없으면 inference에서 꺼집니다."],
  swarm: ["System 4", "bounded tensor swarm 상태를 텔레메트리로 노출합니다. 현재 답변 토큰을 변경하지 않으며 지연 수치는 live benchmark 전 표시하지 않습니다."],
  experts: ["System 3", "bounded expert pool 상태를 텔레메트리로 노출합니다. 현재 답변 토큰을 변경하지 않으며 승격은 attestation 전 차단됩니다."],
  ewc: ["FP-EWC", "evolution lifecycle의 night-only 업데이트 후보입니다. C-FIRE는 해당 업데이트 투영 범위에만 적용되며 decoder 전체 인증이 아닙니다."],
};

const TOUR = [
  { view: "mission", kicker: "WHY NOW", title: "보안 때문에 AI를 포기하는 시장", copy: "국방·신약 고객은 데이터를 클라우드로 보낼 수 없고, 자체 구축은 숨은 비용을 만듭니다." },
  { view: "mission", kicker: "PROOF, NOT PROMISE", title: "현재 프로세스에서 직접 재검증", copy: "검증 실행이 성공한 뒤에만 현재 장비의 Depth, VRAM, 잔차를 공개하고 목표 RTX 4090과 구분합니다." },
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
  lastEvidenceVerified: false,
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
  evolutionProposalCount: 0,
  proposalReviewPending: false,
  proposalReviewReturnFocus: null,
  agentConnectionLost: false,
  validationConnectionLost: false,
  lastAgentErrorKey: "",
  lastEvolutionErrorKey: "",
  runtimeCapabilities: Object.create(null),
  runtimeExecutionModules: Object.create(null),
  workspaceCapabilities: null,
  workspaceCapabilitiesLoaded: false,
  workspaceCapabilityRequestId: 0,
  workspaceCapabilityAbortController: null,
  workspaceRequestPending: false,
  workspaceAttachments: [],
  selectedImageAttachmentId: "",
  imageModelIntegrationReady: false,
  imageModelFirstUseReady: false,
  imageAttestationPending: false,
  imageAttestationSettling: false,
  indexedAttachmentIds: new Set(),
  ragBackendReady: false,
  ragAnswerIntegrationReady: false,
  ragDocumentCount: 0,
  ragEnabled: false,
  modelSelectionPending: false,
  lensConnectorReady: false,
  lensSearchPending: false,
  lensSearchResults: [],
  voiceCaptureState: "idle",
  voiceSession: null,
  voiceTranscriptionConfigured: false,
  voiceTranscriptionReady: false,
  voiceTranscriptionAttemptReady: false,
  voiceTransportReady: false,
  voiceBrowserCaptureReady: false,
  voiceSynthesisReady: false,
  voiceSynthesisDisabledReason: "LOCAL_TTS_ARTIFACT_REQUIRED",
  voicePlaybackState: "idle",
  voicePlaybackAudio: null,
  voicePlaybackObjectUrl: "",
  voicePlaybackAbortController: null,
  voicePlaybackText: "",
  previewObjectUrl: "",
  previewReturnFocus: null,
  evidenceDrawerRequestId: 0,
  evidenceDrawerAbortController: null,
  evidenceDrawerReturnFocus: null,
  evidenceDrawerOpenerIdentity: null,
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

function normalizedLiveRuntimeMetrics(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const measuredAt = typeof raw.measured_at === "string" ? raw.measured_at : "";
  const source = typeof raw.source === "string" ? raw.source : "";
  const device = typeof raw.device === "string" ? raw.device.trim() : "";
  const modelClass = typeof raw.model_class === "string" ? raw.model_class.trim() : "";
  if (
    raw.evidence_kind !== "live_runtime_validation"
    || !measuredAt
    || measuredAt.length > 128
    || Number.isNaN(Date.parse(measuredAt))
    || source !== "scripts/validate_gemma4_runtime.py --event-stream"
    || raw.target !== "RTX 4090 24GB"
    || !device
    || device.length > 256
    || !modelClass
    || modelClass.length > 128
  ) return null;
  if (
    !Number.isInteger(raw.verified_files)
    || raw.verified_files <= 0
    || !Number.isInteger(raw.hidden_size)
    || raw.hidden_size <= 0
    || !finiteNumber(raw.load_seconds)
    || raw.load_seconds < 0
    || !finiteNumber(raw.inference_seconds)
    || raw.inference_seconds < 0
    || raw.requested_depth !== 100
    || raw.reached_depth !== raw.requested_depth
    || !Number.isInteger(raw.nodes_used)
    || raw.nodes_used <= 0
    || !Number.isInteger(raw.node_capacity)
    || raw.node_capacity <= 0
    || raw.nodes_used > raw.node_capacity
    || !Number.isInteger(raw.search_allocated_bytes)
    || raw.search_allocated_bytes <= 0
  ) return null;
  if (
    raw.transition_converged !== true
    || raw.finite !== true
    || !finiteNumber(raw.transition_residual)
    || raw.transition_residual < 0
    || raw.transition_residual > 0.005
    || raw.cts_protocol_version !== "SearchRequestV2"
    || raw.safe_for_decode !== true
    || raw.unsafe_silent_fallbacks !== 0
    || raw.linear_solve_fallbacks !== 0
    || raw.transition_used_fallback !== false
    || raw.solver_rank !== 16
    || !Number.isInteger(raw.solver_history_peak)
    || raw.solver_history_peak < 1
    || raw.solver_history_peak > 16
  ) return null;
  if (
    !Number.isInteger(raw.solver_failures)
    || raw.solver_failures < 0
    || raw.failed_edges !== raw.solver_failures
    || raw.q_zero_backups !== raw.failed_edges
    || !Number.isInteger(raw.mac_budget)
    || !Number.isInteger(raw.mac_reserved)
    || raw.mac_reserved <= 0
    || raw.mac_reserved > raw.mac_budget
    || raw.act_applied !== 301
    || typeof raw.trace_digest !== "string"
    || !/^[0-9a-f]{64}$/.test(raw.trace_digest)
    || raw.causal_bridge_answer_bearing !== true
    || raw.causal_bridge_bias_nonzero !== true
    || !finiteNumber(raw.causal_bridge_bias_max)
    || raw.causal_bridge_bias_max <= 0
    || raw.causal_bridge_bias_max > 0.1
    || raw.conditioned_generated_tokens !== 1
  ) return null;
  if (
    !finiteNumber(raw.vram_limit_gib)
    || raw.vram_limit_gib <= 0
    || raw.vram_limit_gib > MAX_LIVE_VRAM_LIMIT_GIB
    || !finiteNumber(raw.peak_allocated_vram_gib)
    || raw.peak_allocated_vram_gib < 0
    || !finiteNumber(raw.peak_reserved_vram_gib)
    || raw.peak_reserved_vram_gib < raw.peak_allocated_vram_gib
    || raw.peak_reserved_vram_gib > raw.vram_limit_gib
    || !finiteNumber(raw.peak_vram_gib)
    || raw.peak_vram_gib !== Math.max(
      raw.peak_allocated_vram_gib,
      raw.peak_reserved_vram_gib,
    )
  ) return null;
  return Object.freeze({ ...raw, device, model_class: modelClass });
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

function attachmentMediaType(file) {
  const name = typeof file?.name === "string" ? file.name : "";
  const dot = name.lastIndexOf(".");
  const suffix = dot >= 0 ? name.slice(dot).toLowerCase() : "";
  const expected = ATTACHMENT_MEDIA_BY_SUFFIX[suffix] || "";
  return file?.type === expected || !file?.type ? expected : "";
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const value = typeof reader.result === "string" ? reader.result : "";
      const marker = value.indexOf(",");
      if (marker <= 0 || !value.slice(0, marker).endsWith(";base64")) {
        reject(new Error("INVALID_ATTACHMENT"));
        return;
      }
      resolve(value.slice(marker + 1));
    }, { once: true });
    reader.addEventListener("error", () => reject(new Error("INVALID_ATTACHMENT")), { once: true });
    reader.addEventListener("abort", () => reject(new Error("INVALID_ATTACHMENT")), { once: true });
    reader.readAsDataURL(file);
  });
}

function normalizedAttachment(item) {
  if (!item || typeof item !== "object") return null;
  if (typeof item.attachment_id !== "string" || !item.attachment_id) return null;
  if (typeof item.name !== "string" || !item.name) return null;
  return {
    attachment_id: item.attachment_id.slice(0, 128),
    name: item.name.slice(0, 128),
    media_type: typeof item.media_type === "string" ? item.media_type.slice(0, 128) : "",
    size_bytes: Number.isInteger(item.size_bytes) ? Math.max(0, item.size_bytes) : 0,
    text_indexable: item.text_indexable === true,
    indexed: item.indexed === true,
    preview_kind: ["text", "image"].includes(item.preview_kind) ? item.preview_kind : "",
    preview_available: item.preview_available === true,
  };
}

function parseCanonicalRagSourceId(rawSourceId) {
  if (
    typeof rawSourceId !== "string"
    || !/^[0-9a-f]{24}\.(?:0|[1-9][0-9]{0,2})$/.test(rawSourceId)
  ) return null;
  const separator = rawSourceId.lastIndexOf(".");
  const chunkIndex = Number(rawSourceId.slice(separator + 1));
  if (!Number.isInteger(chunkIndex) || chunkIndex < 0 || chunkIndex > 127) return null;
  return Object.freeze({
    sourceId: rawSourceId,
    attachmentId: rawSourceId.slice(0, separator),
    chunkIndex,
  });
}

const RAG_PROVENANCE_EXACT_KEYS = Object.freeze([
  "answer_integration_schema",
  "embedding",
  "indexed_excerpt_chars",
  "indexed_excerpt_sha256",
  "prompt_excerpt_chars",
  "prompt_excerpt_representation",
  "prompt_excerpt_sha256",
  "repository",
  "retrieval_mode",
  "revision",
  "selected_excerpt_chars",
  "selected_excerpt_sha256",
  "selected_excerpt_truncated",
  "semantic_embedding",
  "source_sha256",
].sort());
// Keep static assets network-inert while still comparing the exact authority
// returned by the loopback product API.
const RAG_AKASICDB_REPOSITORY = [
  "https:",
  "",
  "github.com",
  "heosanghun",
  "AkasicDB.git",
].join("/");

function normalizedRetrievalProvenance(value, sourceIdentity) {
  if (!value || typeof value !== "object" || Array.isArray(value) || !sourceIdentity) return null;
  const keys = Object.keys(value).sort();
  if (
    keys.length !== RAG_PROVENANCE_EXACT_KEYS.length
    || keys.some((key, index) => key !== RAG_PROVENANCE_EXACT_KEYS[index])
  ) return null;
  if (value.repository !== RAG_AKASICDB_REPOSITORY) return null;
  if (value.revision !== "a6c8e8ebd487e7cb86079f9804a66aaf0914d1dc") return null;
  if (value.retrieval_mode !== "lexical_only") return null;
  if (value.embedding !== "stable_sha256_lexical_sketch_v1") return null;
  if (value.semantic_embedding !== false) return null;
  if (value.answer_integration_schema !== "cogni.agent.retrieval-evidence.v1") return null;
  if (!/^[0-9a-f]{64}$/.test(value.source_sha256)) return null;
  if (!value.source_sha256.startsWith(sourceIdentity.attachmentId)) return null;
  if (!/^[0-9a-f]{64}$/.test(value.indexed_excerpt_sha256)) return null;
  if (!/^[0-9a-f]{64}$/.test(value.selected_excerpt_sha256)) return null;
  if (!/^[0-9a-f]{64}$/.test(value.prompt_excerpt_sha256)) return null;
  if (
    !Number.isInteger(value.indexed_excerpt_chars)
    || value.indexed_excerpt_chars < 1
    || value.indexed_excerpt_chars > 1600
    || !Number.isInteger(value.selected_excerpt_chars)
    || value.selected_excerpt_chars < 1
    || value.selected_excerpt_chars > value.indexed_excerpt_chars
    || !Number.isInteger(value.prompt_excerpt_chars)
    || value.prompt_excerpt_chars < value.selected_excerpt_chars
    || value.prompt_excerpt_chars > 10000
    || value.prompt_excerpt_representation !== "xml_entity_escaped_v1"
  ) return null;
  const truncated = value.selected_excerpt_chars < value.indexed_excerpt_chars;
  if (value.selected_excerpt_truncated !== truncated) return null;
  if (!truncated && value.selected_excerpt_sha256 !== value.indexed_excerpt_sha256) return null;
  return Object.freeze({
    repository: value.repository,
    revision: value.revision,
    retrievalMode: value.retrieval_mode,
    embedding: value.embedding,
    semanticEmbedding: false,
    answerIntegrationSchema: value.answer_integration_schema,
    sourceSha256: value.source_sha256,
    indexedExcerptSha256: value.indexed_excerpt_sha256,
    indexedExcerptChars: value.indexed_excerpt_chars,
    selectedExcerptSha256: value.selected_excerpt_sha256,
    selectedExcerptChars: value.selected_excerpt_chars,
    selectedExcerptTruncated: truncated,
    promptExcerptSha256: value.prompt_excerpt_sha256,
    promptExcerptChars: value.prompt_excerpt_chars,
    promptExcerptRepresentation: value.prompt_excerpt_representation,
  });
}

function normalizedRetrievalSources(items) {
  if (!Array.isArray(items)) return [];
  const seen = new Set();
  const sources = [];
  items.slice(0, 5).forEach((item) => {
    if (!item || typeof item !== "object") return;
    const number = Number.isInteger(item.number) ? item.number : 0;
    if (number < 1 || number > 5 || seen.has(number)) return;
    const rawTitle = typeof item.title === "string" ? item.title : "";
    const title = rawTitle.replace(/[\u0000-\u001f\u007f-\u009f]/g, " ").trim().slice(0, 160);
    if (!title) return;
    const score = Number.isFinite(item.score)
      ? Math.max(0, Math.min(1, Number(item.score)))
      : null;
    const sourceIdentity = parseCanonicalRagSourceId(item.source_id);
    const provenance = normalizedRetrievalProvenance(item.provenance, sourceIdentity);
    if (!sourceIdentity || !provenance) return;
    seen.add(number);
    sources.push({
      number,
      title,
      score,
      sourceId: sourceIdentity?.sourceId || "",
      attachmentId: sourceIdentity?.attachmentId || "",
      chunkIndex: sourceIdentity?.chunkIndex ?? null,
      provenance,
    });
  });
  return sources;
}

function configureRagEvidenceButton(button, source) {
  if (!(button instanceof HTMLButtonElement)) return false;
  if (
    !source
    || !/^[0-9a-f]{24}$/.test(source.attachmentId)
    || !Number.isInteger(source.chunkIndex)
    || source.chunkIndex < 0
    || source.chunkIndex > 127
    || !/^[0-9a-f]{64}$/.test(source.provenance?.indexedExcerptSha256 || "")
  ) return false;
  button.type = "button";
  button.dataset.action = "rag-evidence-open";
  button.dataset.sourceNumber = String(source.number);
  button.dataset.attachmentId = source.attachmentId;
  button.dataset.chunkIndex = String(source.chunkIndex);
  button.dataset.sourceTitle = source.title;
  button.dataset.sourceScore = source.score === null ? "" : String(source.score);
  button.dataset.expectedExcerptSha256 = source.provenance.indexedExcerptSha256;
  button.setAttribute("aria-label", `근거 ${source.number} 정규화 추출 발췌 열기: ${source.title}`);
  button.setAttribute("aria-haspopup", "dialog");
  button.setAttribute("aria-controls", "evidence-drawer-layer");
  return true;
}

function ragSourceError(code) {
  const error = new Error(code);
  error.code = code;
  return error;
}

function evidenceSourceFromTrigger(trigger) {
  if (!(trigger instanceof HTMLButtonElement)) return null;
  const number = Number(trigger.dataset.sourceNumber);
  const chunkIndex = Number(trigger.dataset.chunkIndex);
  const attachmentId = trigger.dataset.attachmentId || "";
  const rawTitle = trigger.dataset.sourceTitle || "";
  const title = rawTitle.replace(/[\u0000-\u001f\u007f-\u009f]/g, " ").trim().slice(0, 160);
  const rawScore = trigger.dataset.sourceScore;
  const expectedExcerptSha256 = trigger.dataset.expectedExcerptSha256 || "";
  const parsedScore = rawScore === "" || rawScore === undefined ? null : Number(rawScore);
  if (
    !Number.isInteger(number)
    || number < 1
    || number > 5
    || !/^[0-9a-f]{24}$/.test(attachmentId)
    || !Number.isInteger(chunkIndex)
    || chunkIndex < 0
    || chunkIndex > 127
    || !/^[0-9a-f]{64}$/.test(expectedExcerptSha256)
    || !title
    || (parsedScore !== null && !Number.isFinite(parsedScore))
  ) return null;
  return {
    number,
    title,
    attachmentId,
    chunkIndex,
    expectedExcerptSha256,
    score: parsedScore === null ? null : Math.max(0, Math.min(1, parsedScore)),
  };
}

function renderMessageContentWithCitations(container, text, sources) {
  if (!(container instanceof HTMLElement) || typeof text !== "string") return;
  const sourceByNumber = new Map(
    sources
      .filter(
        (source) => (
          /^[0-9a-f]{24}$/.test(source.attachmentId)
          && Number.isInteger(source.chunkIndex)
          && source.chunkIndex >= 0
          && source.chunkIndex <= 127
        ),
      )
      .map((source) => [source.number, source]),
  );
  const fragment = document.createDocumentFragment();
  const citationPattern = /\[근거\s+([1-5])\]/g;
  let cursor = 0;
  let match;
  while ((match = citationPattern.exec(text)) !== null) {
    if (match.index > cursor) {
      fragment.append(document.createTextNode(text.slice(cursor, match.index)));
    }
    const source = sourceByNumber.get(Number(match[1]));
    if (source) {
      const button = document.createElement("button");
      button.className = "chat-citation-button";
      configureRagEvidenceButton(button, source);
      button.textContent = `[근거 ${source.number}]`;
      fragment.append(button);
    } else {
      fragment.append(document.createTextNode(match[0]));
    }
    cursor = citationPattern.lastIndex;
  }
  if (cursor < text.length) fragment.append(document.createTextNode(text.slice(cursor)));
  container.replaceChildren(fragment);
}

function normalizedExactRagSource(payload, requestedSource) {
  if (
    !payload
    || typeof payload !== "object"
    || Array.isArray(payload)
  ) throw ragSourceError("RAG_SOURCE_UNAVAILABLE");
  const payloadKeys = Object.keys(payload).sort();
  if (
    payloadKeys.length !== RAG_SOURCE_EXACT_KEYS.length
    || payloadKeys.some((key, index) => key !== RAG_SOURCE_EXACT_KEYS[index])
    || payload.schema_version !== 2
    || payload.attachment_id !== requestedSource.attachmentId
    || payload.chunk_index !== requestedSource.chunkIndex
    || typeof payload.name !== "string"
    || !payload.name
    || payload.name.length > 128
    || payload.name === "."
    || payload.name === ".."
    || /[\/\\\u0000-\u001f\u007f-\u009f]/.test(payload.name)
    || typeof payload.media_type !== "string"
    || !payload.media_type
    || payload.media_type.length > 128
    || !/^[a-z0-9.+-]+\/[a-z0-9.+-]+$/.test(payload.media_type)
    || typeof payload.text !== "string"
    || !payload.text
    || Array.from(payload.text).length > MAX_RAG_SOURCE_TEXT_CHARS
    || payload.representation !== RAG_SOURCE_REPRESENTATION
    || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/.test(payload.text)
    || !Number.isInteger(payload.char_start)
    || payload.char_start < 0
    || !Number.isInteger(payload.char_end)
    || payload.char_end <= payload.char_start
    || payload.char_end - payload.char_start !== Array.from(payload.text).length
    || !RAG_SOURCE_OFFSET_BASES.has(payload.offset_basis)
    || !/^[0-9a-f]{64}$/.test(payload.excerpt_sha256)
  ) throw ragSourceError("RAG_SOURCE_UNAVAILABLE");
  const pageNumber = payload.page_number;
  const documentLocationValid = (
    payload.offset_basis === "normalized_document_text_v1"
    && pageNumber === null
  );
  const pdfLocationValid = (
    payload.offset_basis === "normalized_pdf_page_text_v1"
    && Number.isInteger(pageNumber)
    && pageNumber >= 1
    && pageNumber <= 128
  );
  if (!documentLocationValid && !pdfLocationValid) {
    throw ragSourceError("RAG_SOURCE_UNAVAILABLE");
  }
  const name = payload.name
    .replace(/[\u0000-\u001f\u007f-\u009f]/g, " ")
    .trim();
  if (!name) throw ragSourceError("RAG_SOURCE_UNAVAILABLE");
  return {
    attachmentId: payload.attachment_id,
    chunkIndex: payload.chunk_index,
    name,
    mediaType: payload.media_type,
    text: payload.text,
    representation: payload.representation,
    pageNumber,
    charStart: payload.char_start,
    charEnd: payload.char_end,
    offsetBasis: payload.offset_basis,
    excerptSha256: payload.excerpt_sha256,
  };
}

async function sha256HexUtf8(text) {
  if (
    typeof TextEncoder !== "function"
    || !globalThis.crypto
    || !globalThis.crypto.subtle
  ) throw ragSourceError("RAG_SOURCE_INTEGRITY_FAILED");
  const bytes = new TextEncoder().encode(text);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(
    new Uint8Array(digest),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("");
}

function setEvidenceDrawerState(state, copy) {
  const drawer = $("#evidence-drawer");
  const status = $("#evidence-drawer-status");
  if (drawer) drawer.dataset.state = state;
  if (!status) return;
  status.textContent = String(copy || "").slice(0, 512);
  status.setAttribute("role", state === "error" ? "alert" : "status");
  status.setAttribute("aria-live", state === "error" ? "assertive" : "polite");
}

function evidenceTriggerMatchesIdentity(trigger, identity) {
  if (
    !identity
    || !/^[0-9a-f]{24}$/.test(identity.attachmentId)
    || !Number.isInteger(identity.chunkIndex)
    || identity.chunkIndex < 0
    || identity.chunkIndex > 127
    || !/^[0-9a-f]{64}$/.test(identity.expectedExcerptSha256 || "")
  ) return false;
  const candidate = evidenceSourceFromTrigger(trigger);
  return Boolean(
    candidate
    && candidate.attachmentId === identity.attachmentId
    && candidate.chunkIndex === identity.chunkIndex
    && candidate.expectedExcerptSha256 === identity.expectedExcerptSha256
  );
}

function evidenceOpenerIdentity(source, returnFocus) {
  if (
    !source
    || !/^[0-9a-f]{24}$/.test(source.attachmentId)
    || !Number.isInteger(source.chunkIndex)
    || source.chunkIndex < 0
    || source.chunkIndex > 127
    || !/^[0-9a-f]{64}$/.test(source.expectedExcerptSha256 || "")
  ) return null;
  const message = (
    returnFocus instanceof HTMLElement
    && evidenceTriggerMatchesIdentity(returnFocus, source)
  )
    ? returnFocus.closest(".chat-message")
    : null;
  const rawMessageId = message?.dataset.messageId;
  const messageId = typeof rawMessageId === "string" && rawMessageId.length <= 128
    ? rawMessageId
    : "";
  return Object.freeze({
    attachmentId: source.attachmentId,
    chunkIndex: source.chunkIndex,
    expectedExcerptSha256: source.expectedExcerptSha256,
    messageId,
  });
}

function latestEvidenceOpener(identity) {
  const transcript = $("#chat-transcript");
  if (!(transcript instanceof HTMLElement) || !identity) return null;
  const matching = $$(
    'button[data-action="rag-evidence-open"]',
    transcript,
  ).filter(
    (candidate) => (
      !candidate.disabled
      && candidate.isConnected
      && evidenceTriggerMatchesIdentity(candidate, identity)
    ),
  );
  if (!matching.length) return null;
  if (identity.messageId) {
    const sameMessage = matching.filter((candidate) => {
      const candidateMessage = candidate.closest(".chat-message");
      return candidateMessage?.dataset.messageId === identity.messageId;
    });
    if (sameMessage.length) return sameMessage[sameMessage.length - 1];
  }
  return matching[matching.length - 1];
}

function evidenceDrawerFallbackFocus() {
  const transcript = $("#chat-transcript");
  if (transcript instanceof HTMLElement && transcript.isConnected) return transcript;
  const input = $("#agent-input");
  return input instanceof HTMLElement && input.isConnected ? input : null;
}

function closeRagEvidenceDrawer() {
  ui.evidenceDrawerRequestId += 1;
  if (ui.evidenceDrawerAbortController) {
    ui.evidenceDrawerAbortController.abort();
    ui.evidenceDrawerAbortController = null;
  }
  const layer = $("#evidence-drawer-layer");
  const excerpt = $("#evidence-drawer-excerpt");
  if (excerpt) {
    excerpt.textContent = "";
    excerpt.hidden = true;
  }
  if (layer) layer.hidden = true;
  syncModalBackgroundBlock();
  const returnFocus = ui.evidenceDrawerReturnFocus;
  const openerIdentity = ui.evidenceDrawerOpenerIdentity;
  ui.evidenceDrawerReturnFocus = null;
  ui.evidenceDrawerOpenerIdentity = null;
  const connectedExactOpener = (
    returnFocus instanceof HTMLElement
    && returnFocus.isConnected
    && evidenceTriggerMatchesIdentity(returnFocus, openerIdentity)
  )
    ? returnFocus
    : null;
  const focusTarget = connectedExactOpener
    || latestEvidenceOpener(openerIdentity)
    || evidenceDrawerFallbackFocus();
  focusTarget?.focus({ preventScroll: true });
}

async function openRagEvidenceSource(source, returnFocus) {
  if (
    !source
    || !/^[0-9a-f]{24}$/.test(source.attachmentId)
    || !Number.isInteger(source.chunkIndex)
    || source.chunkIndex < 0
    || source.chunkIndex > 127
    || !/^[0-9a-f]{64}$/.test(source.expectedExcerptSha256 || "")
  ) return;
  if (ui.evidenceDrawerAbortController) ui.evidenceDrawerAbortController.abort();
  const requestId = ui.evidenceDrawerRequestId + 1;
  ui.evidenceDrawerRequestId = requestId;
  const layer = $("#evidence-drawer-layer");
  const drawer = $("#evidence-drawer");
  const excerpt = $("#evidence-drawer-excerpt");
  const empty = $("#evidence-drawer-empty");
  if (!layer || !drawer || !excerpt || !empty) return;
  ui.evidenceDrawerReturnFocus = returnFocus instanceof HTMLElement ? returnFocus : null;
  ui.evidenceDrawerOpenerIdentity = evidenceOpenerIdentity(source, returnFocus);
  excerpt.textContent = "";
  excerpt.hidden = true;
  empty.hidden = false;
  empty.textContent = "정규화 추출 위치와 SHA-256 무결성을 확인하고 있습니다.";
  setText("#evidence-drawer-title", `근거 ${source.number} 추출 발췌 확인 중`);
  setText(
    "#evidence-drawer-location",
    `attachment_id ${source.attachmentId} · chunk ${source.chunkIndex}`,
  );
  setText(
    "#evidence-drawer-score",
    source.score === null ? "검색 점수 미제공" : source.score.toFixed(4),
  );
  setText("#evidence-drawer-digest", "검증 중");
  setText("#evidence-drawer-representation", "정규화 추출 발췌 · 원본 첨부 바이트 아님");
  setEvidenceDrawerState("loading", "정확한 로컬 정규화 추출 발췌를 불러오고 있습니다.");
  layer.hidden = false;
  drawer.focus();
  syncModalBackgroundBlock();
  const controller = new AbortController();
  ui.evidenceDrawerAbortController = controller;
  try {
    const payload = await api(
      `${RAG_SOURCE_ENDPOINT}?attachment_id=${encodeURIComponent(source.attachmentId)}&chunk_index=${encodeURIComponent(source.chunkIndex)}`,
      { signal: controller.signal },
    );
    if (requestId !== ui.evidenceDrawerRequestId) return;
    const exactSource = normalizedExactRagSource(payload, source);
    if (exactSource.excerptSha256 !== source.expectedExcerptSha256) {
      throw ragSourceError("RAG_SOURCE_INTEGRITY_FAILED");
    }
    setText("#evidence-drawer-title", exactSource.name);
    const pageLocation = exactSource.pageNumber === null
      ? "정규화 문서 텍스트"
      : `PDF 물리 ${exactSource.pageNumber}쪽의 정규화 추출 텍스트`;
    setText(
      "#evidence-drawer-location",
      `${pageLocation} · 문자 ${exactSource.charStart}–${exactSource.charEnd} · ${exactSource.offsetBasis}`,
    );
    const actualDigest = await sha256HexUtf8(exactSource.text);
    if (requestId !== ui.evidenceDrawerRequestId) return;
    if (actualDigest !== exactSource.excerptSha256) {
      throw ragSourceError("RAG_SOURCE_INTEGRITY_FAILED");
    }
    setText("#evidence-drawer-digest", exactSource.excerptSha256);
    setText("#evidence-drawer-representation", "정규화 추출 발췌 · 원본 첨부 바이트 아님");
    excerpt.textContent = exactSource.text;
    excerpt.hidden = false;
    empty.hidden = true;
    setEvidenceDrawerState(
      "ready",
      `정규화 추출 발췌 SHA-256 확인 완료 · ${exactSource.mediaType}`,
    );
  } catch (error) {
    if (requestId !== ui.evidenceDrawerRequestId) return;
    excerpt.textContent = "";
    excerpt.hidden = true;
    empty.hidden = false;
    empty.textContent = "무결성이 확인되지 않아 정규화 추출 발췌를 표시하지 않았습니다.";
    setText("#evidence-drawer-digest", "검증 실패");
    setText("#evidence-drawer-representation", "검증 실패");
    setEvidenceDrawerState(
      "error",
      describeApiError(error, "선택한 정규화 추출 근거를 안전하게 불러올 수 없습니다."),
    );
  } finally {
    if (requestId === ui.evidenceDrawerRequestId) {
      ui.evidenceDrawerAbortController = null;
    }
  }
}

function trapFocusWithin(event, root) {
  if (event.key !== "Tab" || !(root instanceof HTMLElement)) return;
  const candidates = $$(
    'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    root,
  ).filter((candidate) => !candidate.hidden && candidate.getAttribute("aria-hidden") !== "true");
  if (!candidates.length) {
    event.preventDefault();
    root.focus();
    return;
  }
  const first = candidates[0];
  const last = candidates[candidates.length - 1];
  if (event.shiftKey && (document.activeElement === first || !root.contains(document.activeElement))) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (document.activeElement === last || !root.contains(document.activeElement))) {
    event.preventDefault();
    first.focus();
  }
}

function syncModalBackgroundBlock() {
  const modalOpen = [
    "#attachment-preview-layer",
    "#evidence-drawer-layer",
    "#proposal-review-layer",
  ].some((selector) => {
    const layer = $(selector);
    return layer instanceof HTMLElement && !layer.hidden;
  });
  const shell = $(".app-shell");
  if (!(shell instanceof HTMLElement)) return;
  shell.inert = modalOpen;
  if (modalOpen) shell.setAttribute("aria-hidden", "true");
  else shell.removeAttribute("aria-hidden");
}

function renderWorkspaceAttachments() {
  const tray = $("#agent-attachment-tray");
  const list = $("#agent-attachment-list");
  if (!tray || !list) return;
  const fragment = document.createDocumentFragment();
  ui.workspaceAttachments.slice(0, MAX_ATTACHMENT_UPLOAD_COUNT).forEach((attachment) => {
    const chip = document.createElement("span");
    const name = document.createElement("button");
    const state = document.createElement("small");
    const imageSelect = document.createElement("button");
    const remove = document.createElement("button");
    const indexed = ui.indexedAttachmentIds.has(attachment.attachment_id);
    const imageSelectable = (
      ui.imageModelIntegrationReady || ui.imageModelFirstUseReady
    )
      && !ui.imageAttestationPending
      && IMAGE_CHAT_MEDIA_TYPES.has(attachment.media_type);
    const imageSelected = imageSelectable
      && ui.selectedImageAttachmentId === attachment.attachment_id;
    chip.className = "attachment-chip";
    chip.dataset.indexed = String(indexed);
    chip.dataset.imageSelected = String(imageSelected);
    chip.setAttribute("role", "listitem");
    chip.title = `${attachment.name} · ${attachment.media_type || "로컬 파일"}`.slice(0, 256);
    name.type = "button";
    name.className = "attachment-preview-trigger";
    name.dataset.action = "workspace-attachment-preview";
    name.dataset.attachmentId = attachment.attachment_id;
    name.dataset.previewAvailable = String(attachment.preview_available);
    name.setAttribute("aria-label", `${attachment.name} 미리보기`);
    name.title = attachment.preview_available
      ? `${attachment.name} 로컬 미리보기`
      : `${attachment.name} 미리보기 사용 불가`;
    name.textContent = attachment.name;
    state.textContent = indexed
      ? "RAG"
      : imageSelected
        ? "다음 대화 이미지"
        : imageSelectable
          ? ui.imageModelIntegrationReady
            ? "이미지 선택 가능"
            : "첫 사용 검증"
      : attachment.text_indexable
        ? "인덱스 대기"
        : attachment.media_type === "application/pdf"
          ? "PDF 추출기 없음"
          : "로컬 저장·모델 미전달";
    imageSelect.type = "button";
    imageSelect.className = "attachment-image-select";
    imageSelect.dataset.action = "workspace-image-select";
    imageSelect.dataset.attachmentId = attachment.attachment_id;
    imageSelect.setAttribute("aria-pressed", String(imageSelected));
    imageSelect.setAttribute(
      "aria-label",
      imageSelected
        ? `${attachment.name} 다음 대화 이미지 선택 해제`
        : `${attachment.name} 다음 대화 이미지로 선택`,
    );
    imageSelect.title = imageSelected
      ? "다음 대화 이미지 선택 해제"
      : imageSelectable
        ? ui.imageModelIntegrationReady
          ? "이 이미지 한 장을 다음 대화에만 사용"
          : "이 이미지 한 장으로 로컬 모델 경로를 처음 검증한 뒤 답변"
        : "검증된 로컬 이미지 모델 경로가 필요합니다.";
    imageSelect.textContent = imageSelected ? "선택됨" : "이미지 사용";
    imageSelect.hidden = !IMAGE_CHAT_MEDIA_TYPES.has(attachment.media_type);
    imageSelect.disabled = !imageSelectable;
    remove.type = "button";
    remove.className = "attachment-remove";
    remove.dataset.action = "workspace-attachment-delete";
    remove.dataset.attachmentId = attachment.attachment_id;
    remove.setAttribute("aria-label", `${attachment.name} 첨부 삭제`);
    remove.title = "첨부와 로컬 RAG 인덱스에서 삭제";
    remove.textContent = "×";
    chip.append(name, state, imageSelect, remove);
    fragment.append(chip);
  });
  list.replaceChildren(fragment);
  tray.hidden = ui.workspaceAttachments.length === 0;
  const indexedCount = Math.max(ui.indexedAttachmentIds.size, ui.ragDocumentCount);
  setText(
    "#agent-attachment-live-status",
    ui.workspaceAttachments.length
      ? `${ui.workspaceAttachments.length}개 첨부 · ${indexedCount}개 로컬 인덱스${ui.selectedImageAttachmentId ? " · 이미지 1장 선택" : ""}`
      : "첨부 파일이 없습니다.",
  );
  updateWorkspaceControlStates();
}

function renderWorkspaceModels(models) {
  const selector = $("#agent-model-selector");
  if (!selector) return;
  const items = Array.isArray(models?.items)
    ? models.items.filter((item) => (
      item
      && typeof item === "object"
      && typeof item.model_id === "string"
      && typeof item.label === "string"
      && item.verification
    )).slice(0, 16)
    : [];
  const fragment = document.createDocumentFragment();
  if (!items.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "검증 모델 없음";
    fragment.append(option);
  } else {
    items.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.model_id.slice(0, 256);
      option.textContent = item.label.slice(0, 128);
      option.selected = item.selected === true;
      option.disabled = item.selectable !== true && item.selected !== true;
      fragment.append(option);
    });
  }
  selector.replaceChildren(fragment);
  selector.dataset.verifiedCount = String(items.length);
  selector.dataset.selectableCount = String(
    items.filter((item) => item.selectable === true).length,
  );
  setText(
    "#agent-model-status",
    items.length > 1
      ? items.filter((item) => item.selectable === true).length > 1
        ? `검증 모델 ${items.length}개`
        : `검증 모델 ${items.length}개 · 전환 잠금`
      : items.length === 1 ? "검증 모델 고정" : "사용 불가",
  );
}

function workspaceRagStatusLabel() {
  const indexedDocuments = Math.max(ui.indexedAttachmentIds.size, ui.ragDocumentCount);
  if (!ui.ragBackendReady) return "미연결";
  if (!ui.ragAnswerIntegrationReady) return "인덱스 전용";
  if (ui.ragEnabled) return "사용 중";
  return indexedDocuments > 0 ? "사용 가능" : "문서 필요";
}

function updateExternalCallDisclosure(mode, externalCalls) {
  const validMode = mode === "air_gapped" || mode === "online_opt_in";
  const validCalls = Number.isInteger(externalCalls) && externalCalls >= 0;
  const verified = validMode && validCalls;
  const networkLabel = !verified
    ? "NETWORK UNVERIFIED"
    : mode === "online_opt_in"
      ? "ONLINE OPT-IN"
      : "LOCAL ONLY";
  const disclosure = !verified
    ? "UNVERIFIED"
    : mode === "online_opt_in"
      ? `ONLINE OPT-IN · ${externalCalls} CALLS`
      : `DISABLED · ${externalCalls} CALLS`;

  setText("#network-mode-label", networkLabel);
  setText("#external-call-count", verified ? String(externalCalls) : "—");
  setText("#telemetry-external-calls", disclosure);
  setText("#rail-external-calls", disclosure);
  setText(
    "#mission-external-calls",
    !verified
      ? "UNVERIFIED · 앱 외부 호출 상태 미검증"
      : mode === "online_opt_in"
        ? `ONLINE OPT-IN · 앱 외부 호출 ${externalCalls}회`
        : `LOCAL ONLY · 앱 외부 호출 ${externalCalls}회`,
  );
  const pill = $("#network-mode-pill");
  pill?.classList.toggle("status-local", verified && mode === "air_gapped");
  pill?.classList.toggle("is-online-opt-in", verified && mode === "online_opt_in");
  ["#telemetry-external-calls", "#rail-external-calls"].forEach((selector) => {
    const element = $(selector);
    element?.classList.toggle("positive", verified && mode === "air_gapped");
  });
}

function revokeWorkspaceCapabilities() {
  ui.workspaceCapabilities = null;
  ui.workspaceCapabilitiesLoaded = false;
  ui.ragBackendReady = false;
  ui.ragAnswerIntegrationReady = false;
  ui.ragEnabled = false;
  ui.imageModelIntegrationReady = false;
  ui.imageModelFirstUseReady = false;
  ui.imageAttestationPending = false;
  ui.imageAttestationSettling = false;
  ui.selectedImageAttachmentId = "";
  ui.lensConnectorReady = false;
  ui.voiceTransportReady = false;
  ui.voiceBrowserCaptureReady = false;
  ui.voiceTranscriptionConfigured = false;
  ui.voiceTranscriptionReady = false;
  ui.voiceTranscriptionAttemptReady = false;
  ui.voiceSynthesisReady = false;
}

function browserMicrophoneSupport() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (window.isSecureContext !== true) {
    return { ready: false, reason: "보안 컨텍스트가 아니어서 마이크를 사용할 수 없습니다." };
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return { ready: false, reason: "이 브라우저는 getUserMedia를 지원하지 않습니다." };
  }
  if (typeof window.MediaRecorder !== "function") {
    return { ready: false, reason: "이 브라우저는 MediaRecorder를 지원하지 않습니다." };
  }
  if (!AudioContextClass) {
    return { ready: false, reason: "이 브라우저는 AudioContext를 지원하지 않습니다." };
  }
  return { ready: true, reason: "" };
}

function applyWorkspaceCapabilities(capabilities) {
  if (!capabilities || typeof capabilities !== "object" || Array.isArray(capabilities)) {
    revokeWorkspaceCapabilities();
    updateExternalCallDisclosure("", null);
    setText("#agent-web-status", "미검증");
    updateWorkspaceControlStates();
    return false;
  }
  ui.workspaceCapabilities = capabilities;
  ui.workspaceCapabilitiesLoaded = true;
  const attachments = capabilities.attachments || {};
  const rag = capabilities.rag || {};
  const microphone = capabilities.microphone || {};
  const captureTransport = microphone.capture_transport || {};
  const processor = microphone.processor || {};
  const transcriber = microphone.transcriber || {};
  const stt = microphone.stt || {};
  const tts = microphone.tts || {};
  const web = capabilities.web_search || {};
  const lens = web.official_lens_connector || {};
  ui.ragBackendReady = rag.state === "local_index_ready";
  ui.ragAnswerIntegrationReady = ui.ragBackendReady
    && rag.answer_integration === true
    && rag.answer_integration_schema === "cogni.agent.retrieval-evidence.v1";
  const imageCapability = attachments.image_capability || {};
  const serverImageAttestationPending = imageCapability.state === "first_image_attestation_in_progress";
  ui.imageModelIntegrationReady = attachments.image_to_model_integration === true
    && imageCapability.runtime_ready === true
    && imageCapability.model_inference_attested === true;
  if (serverImageAttestationPending) {
    ui.imageAttestationPending = true;
  } else if (ui.imageModelIntegrationReady || ui.imageAttestationSettling) {
    // A settling refresh is the only configured-unverified response allowed to
    // clear a locally admitted probe. This prevents an older in-flight GET
    // from erasing the POST result before the server publishes pending state.
    ui.imageAttestationPending = false;
  }
  ui.imageModelFirstUseReady = attachments.state === "enabled"
    && imageCapability.state === "configured_unverified"
    && imageCapability.configured === true
    && imageCapability.first_use_attestation_allowed === true;
  if (!ui.imageModelIntegrationReady && !ui.imageModelFirstUseReady) {
    ui.selectedImageAttachmentId = "";
  }
  ui.ragDocumentCount = Number.isInteger(rag.documents) ? Math.max(0, rag.documents) : 0;
  if (
    !ui.ragAnswerIntegrationReady
    || (!ui.ragDocumentCount && !ui.indexedAttachmentIds.size)
  ) {
    ui.ragEnabled = false;
  }
  const browserCapture = browserMicrophoneSupport();
  ui.voiceBrowserCaptureReady = browserCapture.ready;
  ui.voiceTransportReady = captureTransport.state === "configured"
    && microphone.capture_state === "browser_get_user_media"
    && microphone.transport_state === "authenticated_loopback_ready";
  ui.voiceTranscriptionReady = microphone.transcription_state === "ready"
    && microphone.runtime_audio_input === true
    && processor.probe_passed === true
    && transcriber.configured === true
    && microphone.model_inference_attested === true;
  ui.voiceTranscriptionConfigured = microphone.transcription_state === "configured_unverified"
    && microphone.runtime_audio_input === false
    && microphone.model_inference_attested === false
    && processor.configured === true
    && transcriber.configured === true
    && transcriber.artifact_verified === true
    && stt.mode === "local_only"
    && stt.artifact_verified === true
    && stt.runtime_ready === false
    && stt.disabled_reason === "LOCAL_STT_INFERENCE_UNVERIFIED";
  ui.voiceTranscriptionAttemptReady = ui.voiceTranscriptionReady
    || ui.voiceTranscriptionConfigured;
  ui.voiceSynthesisReady = tts.state === "ready"
    && tts.mode === "local_only"
    && tts.source === "verified_windows_system_speech"
    && tts.host_probe_passed === true;
  ui.voiceSynthesisDisabledReason = typeof tts.disabled_reason === "string"
    ? tts.disabled_reason.slice(0, 64)
    : "LOCAL_TTS_ARTIFACT_REQUIRED";
  const attachmentButton = $('[data-action="workspace-attach"]');
  if (attachmentButton) {
    const ready = attachments.state === "enabled";
    attachmentButton.setAttribute("aria-label", ready ? "로컬 파일 또는 이미지 첨부" : "파일 또는 이미지 첨부, 사용 불가");
    attachmentButton.title = ready
      ? "파일은 이 PC의 content-addressed 저장소에만 보관됩니다."
      : "백엔드가 로컬 첨부 기능을 활성화하지 않았습니다.";
  }
  setText(
    "#agent-attachment-status",
    ui.imageAttestationPending
      ? "이미지 검증 중"
      : ui.imageModelFirstUseReady
        ? "구성됨 · 첫 이미지 검증 필요"
        : attachments.state === "enabled"
          ? "로컬"
          : "사용 불가",
  );
  setText("#agent-rag-status", workspaceRagStatusLabel());
  const networkMode = web.mode === "air_gapped" || web.mode === "online_opt_in" ? web.mode : "";
  const externalCalls = Number.isInteger(lens.external_calls) && lens.external_calls >= 0
    ? lens.external_calls
    : null;
  updateExternalCallDisclosure(networkMode, externalCalls);
  ui.lensConnectorReady = WEB_SEARCH_UI_IMPLEMENTED
    && networkMode === "online_opt_in"
    && lens.executor_implemented === true
    && lens.state === "ready";
  const webButton = $('[data-action="workspace-web-search"]');
  if (webButton) {
    webButton.setAttribute(
      "aria-label",
      ui.lensConnectorReady
        ? "공식 Lens API에서 특허와 논문 검색"
        : "Lens 특허·논문 검색, 정책 또는 자격 증명 필요",
    );
    webButton.title = ui.lensConnectorReady
      ? "api.lens.org 공식 HTTPS POST 검색만 실행합니다."
      : lens.state === "credentials_required"
        ? "COGNI_OS_LENS_API_TOKEN이 필요합니다."
        : lens.state === "terms_acceptance_required"
          ? "Lens API 약관 확인 후 COGNI_OS_LENS_TERMS_ACCEPTED=1 설정이 필요합니다."
        : lens.state === "allowlist_required"
          ? "COGNI_OS_WEB_ALLOWLIST에 api.lens.org가 필요합니다."
          : "명시적 온라인 모드가 꺼져 있습니다.";
  }
  setText(
    "#agent-web-status",
    ui.lensConnectorReady
      ? "공식 API"
      : lens.state === "credentials_required"
        ? "토큰 필요"
        : lens.state === "terms_acceptance_required"
          ? "약관 동의 필요"
        : lens.state === "allowlist_required"
          ? "허용 필요"
          : "오프라인",
  );
  const lensIndex = $("#agent-lens-index");
  if (lensIndex) {
    lensIndex.disabled = !ui.ragBackendReady || lens.lens_to_akasicdb !== true;
    if (lensIndex.disabled) lensIndex.checked = false;
  }
  const sttConfigured = ui.voiceTranscriptionConfigured;
  setText(
    "#agent-microphone-status",
    MICROPHONE_CAPTURE_UI_IMPLEMENTED && ui.voiceTransportReady && ui.voiceBrowserCaptureReady
      ? ui.voiceTranscriptionReady
        ? "로컬 STT"
        : sttConfigured
          ? "구성됨·실행 미검증"
          : "STT 필요"
      : !ui.voiceBrowserCaptureReady ? "브라우저 미지원"
      : "미연결",
  );
  const microphoneButton = $('[data-action="workspace-microphone"]');
  if (microphoneButton) {
    const captureReady = MICROPHONE_CAPTURE_UI_IMPLEMENTED
      && ui.voiceTransportReady
      && ui.voiceBrowserCaptureReady;
    microphoneButton.setAttribute(
      "aria-label",
      captureReady
        ? ui.voiceTranscriptionReady
          ? "로컬 음성 입력 시작"
          : sttConfigured
            ? "로컬 음성 입력 시작 및 첫 실행 검증"
            : "음성 입력 사용 불가: 검증된 로컬 STT 모델 필요"
        : "음성 입력, 로컬 캡처 파이프라인 미연결",
    );
    microphoneButton.title = captureReady
      ? ui.voiceTranscriptionReady
        ? "마이크 권한은 이 버튼을 누를 때만 요청되며 외부로 전송하지 않습니다."
        : sttConfigured
          ? "로컬 STT 구성은 확인됐습니다. 첫 사용자 주도 전사가 성공하면 실행 검증 상태로 전환합니다."
          : "텍스트 전사에는 검증된 로컬 STT와 오디오 프로세서가 필요합니다."
      : browserCapture.reason || "브라우저 캡처와 인증된 로컬 전송 경로가 준비되지 않았습니다.";
  }
  setText(
    "#agent-tts-status",
    ui.voiceSynthesisReady
      ? tts.browser_playback_verified === true
        ? "로컬 TTS"
        : "호스트 검증·재생 미검증"
      : tts.state === "configured_unverified"
        ? "구성됨·실행 미검증"
        : "음성 필요",
  );
  renderWorkspaceModels(capabilities.models || {});
  renderWorkspaceAttachments();
  updateWorkspaceControlStates();
  return true;
}

async function requestLatestWorkspaceCapabilities() {
  ui.workspaceCapabilityAbortController?.abort();
  const requestId = ui.workspaceCapabilityRequestId + 1;
  const controller = new AbortController();
  ui.workspaceCapabilityRequestId = requestId;
  ui.workspaceCapabilityAbortController = controller;
  try {
    const capabilities = await api("/api/workspace/capabilities", {
      signal: controller.signal,
    });
    if (requestId !== ui.workspaceCapabilityRequestId) {
      return { latest: false, applied: false, requestId };
    }
    return {
      latest: true,
      applied: applyWorkspaceCapabilities(capabilities) === true,
      requestId,
    };
  } catch (error) {
    const latest = requestId === ui.workspaceCapabilityRequestId
      && !controller.signal.aborted;
    return { latest, applied: false, requestId, error };
  } finally {
    if (requestId === ui.workspaceCapabilityRequestId) {
      ui.workspaceCapabilityAbortController = null;
    }
  }
}

async function refreshWorkspaceCapabilityDisclosure() {
  const result = await requestLatestWorkspaceCapabilities();
  if (!result.latest) return false;
  if (!result.applied) {
    revokeWorkspaceCapabilities();
    updateExternalCallDisclosure("", null);
    setText("#agent-web-status", "미검증");
    updateWorkspaceControlStates();
    return false;
  }
  return true;
}

function updateWorkspaceControlStates() {
  const capabilities = ui.workspaceCapabilities || {};
  const attachments = capabilities.attachments || {};
  const microphone = capabilities.microphone || {};
  const tts = microphone.tts || {};
  const web = capabilities.web_search || {};
  const lens = web.official_lens_connector || {};
  const agentBusy = AGENT_ACTIVE_STATUSES.has(ui.agentStatus) || ui.agentRequestPending;
  const unavailable = !ui.workspaceCapabilitiesLoaded || ui.workspaceRequestPending || agentBusy;
  const voiceComputeBusy = ACTIVE_STATUSES.has(ui.lastStatus)
    || ui.evolutionRunning
    || ui.evolutionRequestPending;
  const attachmentReady = attachments.state === "enabled";
  const attachmentButton = $('[data-action="workspace-attach"]');
  const attachmentInput = $("#agent-attachment-input");
  if (attachmentButton) attachmentButton.disabled = unavailable || !attachmentReady;
  if (attachmentInput) attachmentInput.disabled = unavailable || !attachmentReady;
  const indexedDocuments = Math.max(ui.indexedAttachmentIds.size, ui.ragDocumentCount);
  setText("#agent-rag-status", workspaceRagStatusLabel());
  const ragButton = $('[data-action="workspace-rag-toggle"]');
  if (ragButton) {
    ragButton.disabled = unavailable
      || ui.agentMode !== "chat"
      || !ui.ragBackendReady
      || !ui.ragAnswerIntegrationReady
      || indexedDocuments < 1;
    ragButton.setAttribute("aria-pressed", String(ui.ragEnabled));
    ragButton.setAttribute(
      "aria-label",
      !ui.ragBackendReady
        ? "로컬 RAG, AkasicDB 미연결"
        : !ui.ragAnswerIntegrationReady
          ? "로컬 RAG, 인덱스 준비·답변 연결 미검증"
        : indexedDocuments < 1
          ? "로컬 RAG, 인덱싱된 문서 필요"
          : ui.ragEnabled
            ? "로컬 RAG 사용 중, 선택하여 해제"
            : "로컬 RAG 사용 가능, 선택하여 활성화",
    );
    ragButton.title = !ui.ragBackendReady
      ? "감사된 로컬 AkasicDB가 준비되지 않았습니다."
      : !ui.ragAnswerIntegrationReady
        ? "로컬 검색 인덱스는 준비됐지만 Agent 답변 연결은 검증되지 않았습니다."
      : indexedDocuments < 1
        ? "UTF-8 텍스트 파일을 첨부하면 로컬 인덱싱을 시도합니다."
        : "현재 대화에 로컬 AkasicDB 검색을 적용합니다.";
  }
  const webButton = $('[data-action="workspace-web-search"]');
  if (webButton) {
    webButton.disabled = unavailable
      || !WEB_SEARCH_UI_IMPLEMENTED
      || !ui.lensConnectorReady
      || lens.state !== "ready";
  }
  const lensSubmit = $("#agent-lens-submit");
  if (lensSubmit) lensSubmit.disabled = unavailable || !ui.lensConnectorReady || ui.lensSearchPending;
  const lensIndex = $("#agent-lens-index");
  if (lensIndex) {
    lensIndex.disabled = unavailable
      || !ui.ragBackendReady
      || lens.lens_to_akasicdb !== true
      || ui.lensSearchPending;
    if (lensIndex.disabled) lensIndex.checked = false;
  }
  const microphoneButton = $('[data-action="workspace-microphone"]');
  if (microphoneButton) {
    const recording = ui.voiceCaptureState === "recording";
    microphoneButton.disabled = recording
      ? false
      : unavailable
        || voiceComputeBusy
        || ui.voicePlaybackState !== "idle"
        || !MICROPHONE_CAPTURE_UI_IMPLEMENTED
        || microphone.capture_state !== "browser_get_user_media"
        || microphone.transport_state !== "authenticated_loopback_ready"
        || !ui.voiceBrowserCaptureReady
        || !ui.voiceTranscriptionAttemptReady
        || ["requesting", "encoding", "uploading"].includes(ui.voiceCaptureState);
  }
  const voiceStop = $('[data-action="workspace-voice-stop"]');
  if (voiceStop) voiceStop.disabled = ui.voiceCaptureState !== "recording";
  const voiceCancel = $('[data-action="workspace-voice-cancel"]');
  if (voiceCancel) {
    voiceCancel.disabled = !["requesting", "recording", "encoding", "uploading"].includes(
      ui.voiceCaptureState,
    );
  }
  const ttsText = latestRenderedAssistantText();
  const ttsPlay = $('[data-action="workspace-tts-play"]');
  if (ttsPlay) {
    const playing = ui.voicePlaybackState === "playing";
    const loading = ui.voicePlaybackState === "loading";
    const capabilityReady = ui.voiceSynthesisReady
      && tts.state === "ready"
      && tts.source === "verified_windows_system_speech"
      && tts.host_probe_passed === true;
    ttsPlay.disabled = unavailable
      || voiceComputeBusy
      || ui.voiceCaptureState !== "idle"
      || loading
      || playing
      || !capabilityReady
      || !ttsText;
    ttsPlay.classList.toggle("is-playing", playing);
    ttsPlay.setAttribute("aria-pressed", String(playing));
    const disabledCopy = API_ERROR_COPY[ui.voiceSynthesisDisabledReason]
      || "검증된 로컬 음성 합성기가 필요합니다.";
    ttsPlay.setAttribute(
      "aria-label",
      !capabilityReady
        ? `마지막 Cogni Agent 답변 읽어주기, 사용 불가: ${disabledCopy}`
        : !ttsText
          ? "마지막 Cogni Agent 답변 읽어주기, 완료된 답변 필요"
          : playing
            ? "마지막 Cogni Agent 답변 재생 중"
            : loading
              ? "마지막 Cogni Agent 답변 음성 준비 중"
              : "마지막 Cogni Agent 답변 읽어주기",
    );
    ttsPlay.title = !capabilityReady
      ? disabledCopy
      : !ttsText
        ? "화면에 표시된 완료 답변이 있어야 읽어줄 수 있습니다."
        : playing
          ? "재생 중지 버튼으로 멈출 수 있습니다."
          : tts.browser_playback_verified === true
            ? "현재 화면에 표시된 마지막 Cogni Agent 답변을 로컬 Windows 음성으로 읽습니다."
            : "호스트 WAV 합성은 검증됐지만 이 브라우저의 실제 오디오 재생은 아직 검증 전입니다.";
  }
  const ttsStop = $('[data-action="workspace-tts-stop"]');
  if (ttsStop) {
    ttsStop.disabled = !["loading", "playing"].includes(ui.voicePlaybackState);
  }
  const modelSelector = $("#agent-model-selector");
  if (modelSelector) {
    const selectableCount = Number(modelSelector.dataset.selectableCount || "0");
    modelSelector.disabled = unavailable || ui.modelSelectionPending || selectableCount < 2;
  }
  $$('.attachment-remove').forEach((button) => {
    button.disabled = unavailable || ui.workspaceRequestPending;
  });
  $$('.attachment-preview-trigger').forEach((button) => {
    button.disabled = unavailable || button.dataset.previewAvailable !== "true";
  });
  $$('.attachment-image-select').forEach((button) => {
    button.disabled = unavailable
      || ui.agentMode !== "chat"
      || ui.imageAttestationPending
      || (!ui.imageModelIntegrationReady && !ui.imageModelFirstUseReady);
  });
  const reindexButton = $('[data-action="workspace-rag-reindex"]');
  if (reindexButton) {
    reindexButton.disabled = unavailable
      || !ui.ragBackendReady
      || !ui.workspaceAttachments.some((item) => item.text_indexable);
  }
}

async function loadWorkspaceCapabilities() {
  const result = await requestLatestWorkspaceCapabilities();
  if (!result.latest) return;
  if (result.applied) {
    if (
      ui.imageAttestationPending
      && ["succeeded", "cancelled", "failed"].includes(ui.agentStatus)
      && !ui.imageAttestationSettling
    ) {
      void settleFirstImageAttestation();
    }
    try {
      const inventory = await api("/api/workspace/attachments");
      ui.workspaceAttachments = Array.isArray(inventory?.items)
        ? inventory.items.map(normalizedAttachment).filter(Boolean).slice(0, MAX_ATTACHMENT_UPLOAD_COUNT)
        : [];
      ui.indexedAttachmentIds = new Set(
        ui.workspaceAttachments
          .filter((item) => item.indexed === true)
          .map((item) => item.attachment_id),
      );
      if (!ui.workspaceAttachments.some(
        (item) => item.attachment_id === ui.selectedImageAttachmentId,
      )) {
        ui.selectedImageAttachmentId = "";
      }
      renderWorkspaceAttachments();
    } catch (_) {
      showToast("로컬 첨부 목록을 불러오지 못했습니다.", "warning");
    }
  } else {
    revokeWorkspaceCapabilities();
    setText("#agent-attachment-status", "사용 불가");
    setText("#agent-rag-status", "사용 불가");
    setText("#agent-web-status", "오프라인");
    setText("#agent-microphone-status", "미연결");
    setText("#agent-model-status", "확인 실패");
    updateExternalCallDisclosure("", null);
    updateWorkspaceControlStates();
  }
}

async function settleFirstImageAttestation() {
  if (ui.imageAttestationSettling) return;
  ui.imageAttestationSettling = true;
  try {
    for (let attempt = 0; attempt < 20; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 100));
      await refreshWorkspaceCapabilityDisclosure();
      const imageCapability = ui.workspaceCapabilities?.attachments?.image_capability || {};
      if (ui.imageModelIntegrationReady) {
        showToast("현재 로컬 모델의 이미지 전처리·추론 경로가 검증되었습니다.", "success");
        return;
      }
      if (imageCapability.state !== "first_image_attestation_in_progress") {
        showToast("첫 이미지 검증을 통과하지 못했습니다. 이미지 기능은 미검증 상태로 유지됩니다.", "warning");
        return;
      }
    }
    showToast("이미지 검증 결과를 확인하지 못해 기능을 활성화하지 않았습니다.", "warning");
  } finally {
    ui.imageAttestationSettling = false;
  }
}

async function indexWorkspaceAttachment(attachment) {
  if (!ui.ragBackendReady || !attachment.text_indexable) return false;
  const result = await api("/api/workspace/rag/index", {
    method: "POST",
    body: { attachment_ids: [attachment.attachment_id] },
  });
  const indexed = Array.isArray(result?.results)
    && result.results.some((item) => item?.indexed === true || item?.already_indexed === true);
  if (indexed) {
    ui.indexedAttachmentIds.add(attachment.attachment_id);
    ui.ragDocumentCount = Number.isInteger(result.documents)
      ? Math.max(ui.ragDocumentCount, result.documents)
      : Math.max(ui.ragDocumentCount, ui.indexedAttachmentIds.size);
  }
  return indexed;
}

async function uploadWorkspaceFiles(fileList) {
  if (ui.workspaceRequestPending) return;
  const files = Array.from(fileList || []).slice(0, MAX_ATTACHMENT_UPLOAD_COUNT);
  if (!files.length) return;
  const capabilities = ui.workspaceCapabilities?.attachments || {};
  const accepted = new Set(Array.isArray(capabilities.accepted_media_types) ? capabilities.accepted_media_types : []);
  const serverLimit = Number.isInteger(capabilities.max_bytes_each) ? capabilities.max_bytes_each : MAX_ATTACHMENT_UPLOAD_BYTES;
  const byteLimit = Math.min(MAX_ATTACHMENT_UPLOAD_BYTES, Math.max(1, serverLimit));
  ui.workspaceRequestPending = true;
  updateWorkspaceControlStates();
  let uploaded = 0;
  let indexed = 0;
  try {
    for (const file of files) {
      const mediaType = attachmentMediaType(file);
      if (!mediaType || (accepted.size && !accepted.has(mediaType))) {
        showToast(`${file.name.slice(0, 80)}: 지원하지 않는 형식입니다.`, "warning");
        continue;
      }
      if (!Number.isInteger(file.size) || file.size < 1 || file.size > byteLimit) {
        showToast(`${file.name.slice(0, 80)}: 8MiB 이하 파일만 첨부할 수 있습니다.`, "warning");
        continue;
      }
      const contentBase64 = await readFileAsBase64(file);
      const response = await api("/api/workspace/attachments/add", {
        method: "POST",
        body: {
          name: file.name.slice(0, 128),
          media_type: mediaType,
          content_base64: contentBase64,
        },
      });
      const attachment = normalizedAttachment(response);
      if (!attachment) throw new Error("INVALID_ATTACHMENT");
      const existing = ui.workspaceAttachments.findIndex((item) => item.attachment_id === attachment.attachment_id);
      if (existing >= 0) ui.workspaceAttachments[existing] = attachment;
      else ui.workspaceAttachments.push(attachment);
      uploaded += 1;
      try {
        if (await indexWorkspaceAttachment(attachment)) indexed += 1;
      } catch (_) {
        showToast(`${attachment.name}: 로컬 첨부는 완료했지만 RAG 인덱싱은 보류되었습니다.`, "warning");
      }
      renderWorkspaceAttachments();
    }
    if (uploaded) {
      showToast(
        `${uploaded}개 파일을 로컬에 첨부했습니다${indexed ? ` · ${indexed}개 AkasicDB 인덱싱` : ""}.`,
        "success",
      );
    }
  } catch (error) {
    showToast(describeApiError(error, "로컬 파일을 첨부하지 못했습니다."), "error");
  } finally {
    ui.workspaceRequestPending = false;
    const input = $("#agent-attachment-input");
    if (input) input.value = "";
    renderWorkspaceAttachments();
    updateWorkspaceControlStates();
  }
}

async function deleteWorkspaceAttachment(attachmentId) {
  if (ui.workspaceRequestPending || typeof attachmentId !== "string") return;
  const attachment = ui.workspaceAttachments.find((item) => item.attachment_id === attachmentId);
  if (!attachment) return;
  ui.workspaceRequestPending = true;
  updateWorkspaceControlStates();
  try {
    const result = await api("/api/workspace/attachments/delete", {
      method: "POST",
      body: { attachment_id: attachmentId },
    });
    if (result?.deleted !== true || result?.blob_deleted !== true) {
      throw new Error("ATTACHMENT_DELETE_FAILED");
    }
    ui.workspaceAttachments = ui.workspaceAttachments.filter(
      (item) => item.attachment_id !== attachmentId,
    );
    ui.indexedAttachmentIds.delete(attachmentId);
    if (ui.selectedImageAttachmentId === attachmentId) {
      ui.selectedImageAttachmentId = "";
    }
    ui.ragDocumentCount = Number.isInteger(result.indexed_documents)
      ? Math.max(0, result.indexed_documents)
      : ui.indexedAttachmentIds.size;
    if (ui.ragDocumentCount < 1) ui.ragEnabled = false;
    renderWorkspaceAttachments();
    showToast(`${attachment.name}: 로컬 첨부와 검색 인덱스에서 삭제했습니다.`, "success");
  } catch (error) {
    showToast(describeApiError(error, "로컬 첨부를 삭제하지 못했습니다."), "error");
  } finally {
    ui.workspaceRequestPending = false;
    renderWorkspaceAttachments();
    updateWorkspaceControlStates();
  }
}

function selectWorkspaceImageAttachment(attachmentId) {
  if (
    ui.workspaceRequestPending
    || ui.agentMode !== "chat"
    || ui.imageAttestationPending
    || (!ui.imageModelIntegrationReady && !ui.imageModelFirstUseReady)
    || typeof attachmentId !== "string"
  ) return;
  const attachment = ui.workspaceAttachments.find(
    (item) => item.attachment_id === attachmentId,
  );
  if (!attachment || !IMAGE_CHAT_MEDIA_TYPES.has(attachment.media_type)) return;
  const selecting = ui.selectedImageAttachmentId !== attachmentId;
  ui.selectedImageAttachmentId = selecting ? attachmentId : "";
  if (selecting) ui.ragEnabled = false;
  renderWorkspaceAttachments();
  setText("#agent-rag-status", ui.ragEnabled ? "사용 중" : "사용 가능");
  showToast(
    selecting
      ? `${attachment.name}: 다음 대화에 사용할 이미지 한 장을 선택했습니다.`
      : `${attachment.name}: 다음 대화 이미지 선택을 해제했습니다.`,
    selecting ? "success" : "info",
  );
}

function closeWorkspaceAttachmentPreview() {
  const layer = $("#attachment-preview-layer");
  const image = $("#attachment-preview-image");
  const text = $("#attachment-preview-text");
  if (ui.previewObjectUrl) {
    URL.revokeObjectURL(ui.previewObjectUrl);
    ui.previewObjectUrl = "";
  }
  if (image) {
    image.removeAttribute("src");
    image.alt = "";
    image.hidden = true;
  }
  if (text) {
    text.textContent = "";
    text.hidden = true;
  }
  if (layer) layer.hidden = true;
  syncModalBackgroundBlock();
  const returnFocus = ui.previewReturnFocus;
  ui.previewReturnFocus = null;
  if (returnFocus instanceof HTMLElement && returnFocus.isConnected) returnFocus.focus();
}

function boundedProposalReviewText(value, maximum = MAX_PROPOSAL_REVIEW_TEXT_CHARS) {
  if (typeof value !== "string") return "";
  return value
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]/g, " ")
    .slice(0, maximum);
}

function normalizedProposalReviewItem(item) {
  if (!item || typeof item !== "object") return null;
  const proposalId = typeof item.proposal_id === "string" ? item.proposal_id : "";
  const baseDigest = typeof item.base_sha256 === "string" ? item.base_sha256 : "";
  const replacementDigest = typeof item.replacement_sha256 === "string"
    ? item.replacement_sha256
    : "";
  const relativePath = boundedProposalReviewText(item.relative_path, 512);
  if (
    !/^[0-9a-f]{64}$/.test(proposalId)
    || !/^[0-9a-f]{64}$/.test(baseDigest)
    || !/^[0-9a-f]{64}$/.test(replacementDigest)
    || !relativePath
    || relativePath.startsWith("/")
    || relativePath.startsWith("\\")
    || /^[a-zA-Z]:/.test(relativePath)
    || relativePath.split(/[\\/]/).includes("..")
  ) return null;
  const stale = item.status === "stale_base";
  return {
    proposalId,
    relativePath,
    baseDigest,
    replacementDigest,
    rationale: boundedProposalReviewText(item.rationale),
    expectedBehavior: boundedProposalReviewText(item.expected_behavior),
    risk: boundedProposalReviewText(item.risk),
    reproductionTest: boundedProposalReviewText(item.reproduction_test),
    rollbackTrigger: boundedProposalReviewText(item.rollback_trigger),
    status: stale ? "stale_base" : "pending_review",
    unifiedDiff: boundedProposalReviewText(item.unified_diff, MAX_PROPOSAL_DIFF_CHARS),
    diffTruncated: item.diff_truncated === true,
    readOnly: item.execution_allowed === false && item.source_mutation_allowed === false,
  };
}

function appendProposalFact(list, label, value) {
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = value || "미제공";
  list.append(term, description);
}

function renderProposalReview(payload) {
  if (
    !payload
    || payload.mode !== "proposal_only_read_only"
    || payload.mutation_endpoint !== false
    || payload.execution_endpoint !== false
    || !Array.isArray(payload.items)
  ) throw new Error("PROPOSAL_REVIEW_UNAVAILABLE");
  const items = payload.items
    .slice(0, MAX_PROPOSAL_REVIEW_ITEMS)
    .map(normalizedProposalReviewItem)
    .filter((item) => item?.readOnly === true);
  const list = $("#proposal-review-list");
  if (!list) return;
  const fragment = document.createDocumentFragment();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "proposal-review-empty";
    empty.textContent = "현재 무결성 검사를 통과한 읽기 전용 제안 diff가 없습니다.";
    fragment.append(empty);
  } else {
    items.forEach((item) => {
      const article = document.createElement("article");
      const header = document.createElement("header");
      const path = document.createElement("strong");
      const status = document.createElement("span");
      const rationale = document.createElement("p");
      const facts = document.createElement("dl");
      const diff = document.createElement("pre");
      article.className = "proposal-review-item";
      article.dataset.stale = String(item.status === "stale_base");
      path.textContent = item.relativePath;
      status.textContent = item.status === "stale_base" ? "STALE · 표시 차단" : "PENDING REVIEW";
      rationale.textContent = item.rationale || "제안 근거가 제공되지 않았습니다.";
      appendProposalFact(facts, "제안 ID", item.proposalId.slice(0, 16));
      appendProposalFact(facts, "기대 동작", item.expectedBehavior);
      appendProposalFact(facts, "위험", item.risk);
      appendProposalFact(facts, "재현 검사", item.reproductionTest);
      appendProposalFact(facts, "롤백 조건", item.rollbackTrigger);
      appendProposalFact(
        facts,
        "Digest",
        `${item.baseDigest.slice(0, 12)} → ${item.replacementDigest.slice(0, 12)}`,
      );
      diff.textContent = item.status === "stale_base"
        ? "기준 소스 digest가 변경되어 오해를 부를 수 있는 diff를 표시하지 않습니다."
        : item.unifiedDiff || "변경 diff가 비어 있습니다.";
      if (item.diffTruncated) {
        appendProposalFact(facts, "표시 한계", "40,000자에서 안전하게 잘림");
      }
      header.append(path, status);
      article.append(header, rationale, facts, diff);
      fragment.append(article);
    });
  }
  list.replaceChildren(fragment);
  setText("#proposal-review-count", `${items.length}개 제안`);
}

function closeProposalReview() {
  const layer = $("#proposal-review-layer");
  if (layer) layer.hidden = true;
  syncModalBackgroundBlock();
  const returnFocus = ui.proposalReviewReturnFocus;
  ui.proposalReviewReturnFocus = null;
  if (returnFocus instanceof HTMLElement && returnFocus.isConnected) returnFocus.focus();
}

async function openProposalReview(returnFocus) {
  if (ui.proposalReviewPending || ui.evolutionProposalCount < 1) return;
  ui.proposalReviewPending = true;
  ui.proposalReviewReturnFocus = returnFocus instanceof HTMLElement ? returnFocus : null;
  updateControlStates();
  try {
    const payload = await api("/api/evolution/proposals");
    renderProposalReview(payload);
    const layer = $("#proposal-review-layer");
    const dialog = $(".proposal-review-dialog", layer || document);
    if (!layer || !dialog) throw new Error("PROPOSAL_REVIEW_UNAVAILABLE");
    layer.hidden = false;
    dialog.focus();
    syncModalBackgroundBlock();
  } catch (error) {
    closeProposalReview();
    showToast(describeApiError(error, "Self-Harness 제안 diff를 열지 못했습니다."), "error");
  } finally {
    ui.proposalReviewPending = false;
    updateControlStates();
  }
}

async function openWorkspaceAttachmentPreview(attachmentId, returnFocus) {
  if (ui.workspaceRequestPending || typeof attachmentId !== "string") return;
  const attachment = ui.workspaceAttachments.find((item) => item.attachment_id === attachmentId);
  if (!attachment || !attachment.preview_available) return;
  ui.workspaceRequestPending = true;
  ui.previewReturnFocus = returnFocus instanceof HTMLElement ? returnFocus : null;
  updateWorkspaceControlStates();
  try {
    const preview = await api(
      `/api/workspace/attachments/preview?attachment_id=${encodeURIComponent(attachmentId)}`,
    );
    if (!preview || preview.attachment_id !== attachmentId) throw new Error("ATTACHMENT_PREVIEW_UNAVAILABLE");
    const layer = $("#attachment-preview-layer");
    const dialog = $(".attachment-preview-dialog", layer || document);
    const text = $("#attachment-preview-text");
    const image = $("#attachment-preview-image");
    const title = $("#attachment-preview-title");
    const meta = $("#attachment-preview-meta");
    if (!layer || !dialog || !text || !image || !title || !meta) return;
    closeWorkspaceAttachmentPreview();
    ui.previewReturnFocus = returnFocus instanceof HTMLElement ? returnFocus : null;
    title.textContent = attachment.name;
    const size = Number.isInteger(preview.size_bytes) ? preview.size_bytes.toLocaleString("ko-KR") : "—";
    if (preview.kind === "text" && typeof preview.text === "string") {
      text.textContent = preview.text.slice(0, 12000);
      text.hidden = false;
      image.hidden = true;
      meta.textContent = `${preview.media_type || "text/plain"} · ${size} bytes · ${preview.extraction || "utf8"}${preview.truncated === true ? " · 12,000자에서 미리보기 생략" : ""}`;
    } else if (preview.kind === "image" && typeof preview.content_url === "string") {
      const blob = await apiBlob(preview.content_url);
      ui.previewObjectUrl = URL.createObjectURL(blob);
      image.src = ui.previewObjectUrl;
      image.alt = `${attachment.name} 로컬 미리보기`;
      image.hidden = false;
      text.hidden = true;
      meta.textContent = `${preview.media_type || blob.type || "image"} · ${size} bytes · 인증된 로컬 세션`;
    } else {
      throw new Error("ATTACHMENT_PREVIEW_UNAVAILABLE");
    }
    layer.hidden = false;
    dialog.focus();
    syncModalBackgroundBlock();
  } catch (error) {
    closeWorkspaceAttachmentPreview();
    showToast(describeApiError(error, "첨부 미리보기를 열지 못했습니다."), "error");
  } finally {
    ui.workspaceRequestPending = false;
    updateWorkspaceControlStates();
  }
}

async function reindexWorkspaceAttachments() {
  if (ui.workspaceRequestPending || !ui.ragBackendReady) return;
  const attachmentIds = ui.workspaceAttachments
    .filter((item) => item.text_indexable)
    .map((item) => item.attachment_id)
    .slice(0, MAX_ATTACHMENT_UPLOAD_COUNT);
  if (!attachmentIds.length) return;
  ui.workspaceRequestPending = true;
  updateWorkspaceControlStates();
  try {
    const result = await api("/api/workspace/rag/reindex", {
      method: "POST",
      body: { attachment_ids: attachmentIds },
    });
    const completed = Array.isArray(result?.reindexed_attachment_ids)
      ? result.reindexed_attachment_ids.filter((item) => attachmentIds.includes(item))
      : [];
    completed.forEach((item) => ui.indexedAttachmentIds.add(item));
    ui.ragDocumentCount = Number.isInteger(result?.documents)
      ? Math.max(0, result.documents)
      : ui.indexedAttachmentIds.size;
    renderWorkspaceAttachments();
    showToast(`${completed.length}개 로컬 문서를 AkasicDB에 재색인했습니다.`, "success");
  } catch (error) {
    showToast(describeApiError(error, "로컬 문서를 재색인하지 못했습니다."), "error");
  } finally {
    ui.workspaceRequestPending = false;
    renderWorkspaceAttachments();
    updateWorkspaceControlStates();
  }
}

function toggleWorkspaceRag() {
  const indexedDocuments = Math.max(ui.indexedAttachmentIds.size, ui.ragDocumentCount);
  if (
    ui.agentMode !== "chat"
    || !ui.ragBackendReady
    || !ui.ragAnswerIntegrationReady
    || indexedDocuments < 1
  ) return;
  ui.ragEnabled = !ui.ragEnabled;
  if (ui.ragEnabled) ui.selectedImageAttachmentId = "";
  setText("#agent-rag-status", ui.ragEnabled ? "사용 중" : "사용 가능");
  renderWorkspaceAttachments();
  updateWorkspaceControlStates();
  showToast(ui.ragEnabled ? "이 대화에 로컬 AkasicDB 검색을 사용합니다." : "이 대화의 로컬 RAG를 해제했습니다.");
}

function toggleLensSearchDrawer(forceOpen) {
  const drawer = $("#agent-lens-search-drawer");
  const button = $('[data-action="workspace-web-search"]');
  if (!drawer || !button) return;
  const open = typeof forceOpen === "boolean" ? forceOpen : drawer.hidden;
  if (open && (button.disabled || !ui.lensConnectorReady)) return;
  drawer.hidden = !open;
  button.setAttribute("aria-expanded", String(open));
  if (open) $("#agent-lens-query")?.focus();
}

function verifiedLensUrl(value, lensId) {
  if (typeof value !== "string" || typeof lensId !== "string") return "";
  if (!/^[0-9A-Z]{3}(?:-[0-9A-Z]{3}){4}$/.test(lensId)) return "";
  try {
    const parsed = new URL(value);
    return parsed.protocol === "https:"
      && parsed.hostname === "lens.org"
      && parsed.pathname === `/${lensId}`
      && !parsed.search
      && !parsed.hash
      ? parsed.href
      : "";
  } catch (_) {
    return "";
  }
}

function renderLensSearchResults(results) {
  const list = $("#agent-lens-search-results");
  if (!list) return;
  const fragment = document.createDocumentFragment();
  const bounded = Array.isArray(results) ? results.slice(0, 20) : [];
  bounded.forEach((result) => {
    if (!result || typeof result !== "object") return;
    const lensId = typeof result.lens_id === "string" ? result.lens_id.slice(0, 32) : "";
    const provenance = result.provenance && typeof result.provenance === "object" ? result.provenance : {};
    const canonicalUrl = verifiedLensUrl(provenance.canonical_url, lensId);
    if (!canonicalUrl) return;
    const item = document.createElement("li");
    item.className = "lens-result-card";
    const top = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = typeof result.title === "string" && result.title.trim()
      ? result.title.slice(0, 1000)
      : "제목 미제공";
    const link = document.createElement("a");
    link.href = canonicalUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `Lens ID ${lensId} · 외부 사이트`;
    link.title = "선택하면 앱 밖의 lens.org 페이지가 새 탭에서 열립니다.";
    link.setAttribute("aria-label", `Lens ID ${lensId}, 외부 lens.org 페이지 새 탭에서 열기`);
    top.append(title, link);
    const metadata = document.createElement("small");
    const kind = result.kind === "patent" ? "특허" : "논문";
    const year = Number.isInteger(result.publication_year) ? String(result.publication_year) : "연도 미제공";
    const type = typeof result.publication_type === "string" ? result.publication_type.slice(0, 128) : "유형 미제공";
    metadata.textContent = `${kind} · ${year} · ${type} · ${provenance.provider === "Lens.org official API" ? "공식 API 근거" : "근거 확인 필요"}`;
    const abstract = document.createElement("p");
    abstract.textContent = typeof result.abstract === "string" && result.abstract.trim()
      ? result.abstract.slice(0, 700)
      : "초록이 제공되지 않았습니다.";
    item.append(top, metadata, abstract);
    fragment.append(item);
  });
  list.replaceChildren(fragment);
}

async function searchLensOfficialApi(event) {
  event?.preventDefault();
  if (ui.lensSearchPending || !ui.lensConnectorReady) return;
  const queryInput = $("#agent-lens-query");
  const kindInput = $("#agent-lens-kind");
  const limitInput = $("#agent-lens-limit");
  const indexInput = $("#agent-lens-index");
  const query = typeof queryInput?.value === "string" ? queryInput.value.trim() : "";
  const kind = kindInput?.value === "scholarly" ? "scholarly" : "patent";
  const parsedLimit = Number.parseInt(limitInput?.value || "5", 10);
  const limit = [5, 10, 20].includes(parsedLimit) ? parsedLimit : 5;
  if (!query || query.length > 512) {
    setText("#agent-lens-search-status", "검색어를 1~512자로 입력해 주세요.");
    queryInput?.focus();
    return;
  }
  const shouldIndex = indexInput?.checked === true && indexInput.disabled === false;
  ui.lensSearchPending = true;
  setText("#agent-lens-search-status", "Lens 공식 API 검색 중…");
  updateWorkspaceControlStates();
  try {
    const response = await api(
      shouldIndex ? "/api/workspace/lens/search-and-index" : "/api/workspace/lens/search",
      { method: "POST", body: { kind, query, limit } },
    );
    const search = shouldIndex ? response?.search : response;
    const results = Array.isArray(search?.results) ? search.results.slice(0, limit) : [];
    ui.lensSearchResults = results;
    renderLensSearchResults(results);
    const total = Number.isInteger(search?.total) ? Math.max(0, search.total) : results.length;
    const calls = Number.isInteger(search?.external_calls) ? Math.max(0, search.external_calls) : 0;
    updateExternalCallDisclosure("online_opt_in", calls);
    setText(
      "#agent-lens-search-status",
      `${total.toLocaleString("ko-KR")}건 중 ${results.length}건 표시${shouldIndex ? " · 출처 포함 AkasicDB 인덱싱 요청 완료" : ""}`,
    );
    if (shouldIndex && Array.isArray(response?.indexed) && response.indexed.length) {
      ui.ragDocumentCount = Math.max(ui.ragDocumentCount, response.indexed.length);
      setText("#agent-rag-status", ui.ragEnabled ? "사용 중" : "사용 가능");
    }
  } catch (error) {
    ui.lensSearchResults = [];
    renderLensSearchResults([]);
    const message = describeApiError(error, "Lens 공식 API 검색을 완료하지 못했습니다.");
    setText("#agent-lens-search-status", message);
    showToast(message, "error");
  } finally {
    await refreshWorkspaceCapabilityDisclosure();
    ui.lensSearchPending = false;
    updateWorkspaceControlStates();
  }
}

async function selectWorkspaceModel() {
  const selector = $("#agent-model-selector");
  if (!selector || selector.disabled || !selector.value) return;
  ui.modelSelectionPending = true;
  updateWorkspaceControlStates();
  try {
    const selected = await api("/api/workspace/models/select", {
      method: "POST",
      body: { model_id: selector.value },
    });
    setText("#agent-model-status", selected?.label ? "검증 모델 선택됨" : "검증 모델 고정");
  } catch (error) {
    showToast(describeApiError(error, "검증 모델을 선택하지 못했습니다."), "error");
    await loadWorkspaceCapabilities();
  } finally {
    ui.modelSelectionPending = false;
    updateWorkspaceControlStates();
  }
}

function latestRenderedAssistantText() {
  const messages = $$(
    '#chat-transcript .chat-message[data-role="assistant"]:not(.is-streaming)',
  );
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const content = $(".chat-bubble > p", messages[index])?.textContent?.trim() || "";
    if (content) return content.slice(0, MAX_TTS_TEXT_CHARS);
  }
  return "";
}

function setVoicePlaybackState(state, copy = "") {
  ui.voicePlaybackState = state;
  const panel = $("#agent-voice-playback");
  if (panel) {
    panel.hidden = state === "idle" && !copy;
    panel.dataset.state = state;
  }
  const labels = {
    idle: "대기",
    loading: "로컬 음성 준비 중",
    playing: "로컬 재생 중",
  };
  setText("#agent-voice-playback-state", labels[state] || "대기");
  setText("#agent-voice-playback-copy", copy);
  setText(
    "#agent-tts-status",
    state === "playing" ? "재생 중" : state === "loading" ? "준비 중" : ui.voiceSynthesisReady ? "로컬 TTS" : "음성 필요",
  );
  updateWorkspaceControlStates();
}

function releaseVoicePlayback(copy = "") {
  ui.voicePlaybackAbortController?.abort();
  ui.voicePlaybackAbortController = null;
  const audio = ui.voicePlaybackAudio;
  ui.voicePlaybackAudio = null;
  if (audio) {
    audio.onended = null;
    audio.onerror = null;
    try {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    } catch (_) { /* playback already released */ }
  }
  if (ui.voicePlaybackObjectUrl) {
    URL.revokeObjectURL(ui.voicePlaybackObjectUrl);
    ui.voicePlaybackObjectUrl = "";
  }
  ui.voicePlaybackText = "";
  setVoicePlaybackState("idle", copy);
}

function ttsWavBlobFromBase64(encoded) {
  if (
    typeof encoded !== "string"
    || encoded.length < 60
    || encoded.length > MAX_TTS_BASE64_CHARS
  ) throw new Error("LOCAL_TTS_OUTPUT_INVALID");
  let binary;
  try {
    binary = atob(encoded);
  } catch (_) {
    throw new Error("LOCAL_TTS_OUTPUT_INVALID");
  }
  if (binary.length < 44 || binary.length > MAX_TTS_WAV_BYTES) {
    throw new Error("LOCAL_TTS_OUTPUT_INVALID");
  }
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  const signature = String.fromCharCode(...bytes.subarray(0, 12));
  if (signature.slice(0, 4) !== "RIFF" || signature.slice(8, 12) !== "WAVE") {
    throw new Error("LOCAL_TTS_OUTPUT_INVALID");
  }
  return new Blob([bytes], { type: "audio/wav" });
}

async function speakLatestAssistantReply() {
  if (ui.voicePlaybackState !== "idle" || !ui.voiceSynthesisReady) return;
  const text = latestRenderedAssistantText();
  if (!text) {
    showToast("읽어줄 완료 답변이 없습니다.", "warning");
    return;
  }
  releaseVoicePlayback();
  const controller = new AbortController();
  ui.voicePlaybackAbortController = controller;
  ui.voicePlaybackText = text;
  setVoicePlaybackState(
    "loading",
    `현재 표시된 마지막 답변 ${text.length.toLocaleString("ko-KR")}자를 로컬 음성으로 변환합니다.`,
  );
  try {
    const result = await api("/api/workspace/voice/synthesize", {
      method: "POST",
      body: { text, language: "auto" },
      signal: controller.signal,
    });
    if (controller.signal.aborted || ui.voicePlaybackAbortController !== controller) return;
    if (
      result?.external_calls !== 0
      || result?.source !== "verified_windows_system_speech"
    ) throw new Error("LOCAL_TTS_OUTPUT_INVALID");
    const blob = ttsWavBlobFromBase64(result.audio_wav_base64);
    const objectUrl = URL.createObjectURL(blob);
    const audio = new Audio();
    ui.voicePlaybackObjectUrl = objectUrl;
    ui.voicePlaybackAudio = audio;
    audio.preload = "auto";
    audio.src = objectUrl;
    audio.onended = () => {
      if (ui.voicePlaybackAudio === audio) releaseVoicePlayback("마지막 답변 재생을 완료했습니다.");
    };
    audio.onerror = () => {
      if (ui.voicePlaybackAudio !== audio) return;
      releaseVoicePlayback("로컬 WAV를 재생하지 못했습니다.");
      showToast("로컬 WAV를 재생하지 못했습니다.", "error");
    };
    await audio.play();
    if (ui.voicePlaybackAudio !== audio) return;
    ui.voicePlaybackAbortController = null;
    setVoicePlaybackState(
      "playing",
      `${String(result.voice || "Windows 로컬 음성").slice(0, 128)} · 외부 전송 0`,
    );
  } catch (error) {
    if (controller.signal.aborted) return;
    const code = error?.code || error?.message;
    const copy = API_ERROR_COPY[code] || "마지막 답변 읽어주기를 시작하지 못했습니다.";
    releaseVoicePlayback(copy);
    showToast(copy, code === "LOCAL_TTS_ARTIFACT_REQUIRED" ? "warning" : "error");
  }
}

function stopVoicePlayback() {
  if (!["loading", "playing"].includes(ui.voicePlaybackState)) return;
  releaseVoicePlayback("재생을 중지하고 임시 오디오 URL을 폐기했습니다.");
}

function setVoiceCaptureState(state, copy = "") {
  ui.voiceCaptureState = state;
  const panel = $("#agent-voice-capture");
  if (panel) {
    panel.hidden = state === "idle" && !copy;
    panel.dataset.state = state;
  }
  const labels = {
    idle: "대기",
    requesting: "마이크 권한 확인",
    recording: "로컬 녹음 중",
    encoding: "16kHz WAV 변환",
    uploading: "loopback 전송·검증",
  };
  setText("#agent-voice-capture-state", labels[state] || "대기");
  setText("#agent-voice-capture-copy", copy);
  const button = $('[data-action="workspace-microphone"]');
  if (button) {
    const recording = state === "recording";
    button.classList.toggle("is-recording", recording);
    button.setAttribute("aria-pressed", String(recording));
    const label = $("span", button);
    if (label) label.textContent = recording ? "중지" : "음성";
  }
  updateWorkspaceControlStates();
}

function releaseVoiceSession(session) {
  if (!session) return;
  clearTimeout(session.limitTimer);
  clearInterval(session.elapsedTimer);
  session.stopCleanup?.();
  session.stopCleanup = null;
  if (session.recorder) {
    session.recorder.ondataavailable = null;
    session.recorder.onerror = null;
    if (session.recorder.state !== "inactive") {
      try { session.recorder.stop(); } catch (_) { /* already stopped */ }
    }
  }
  if (session.stream) {
    session.stream.getTracks().forEach((track) => track.stop());
  }
  if (session.context && session.context.state !== "closed") {
    session.context.close().catch(() => {});
  }
}

function resampleVoiceSamples(source, sourceRate) {
  if (!(source instanceof Float32Array) || !source.length) throw new Error("VOICE_WAV_INVALID");
  if (!Number.isFinite(sourceRate) || sourceRate < VOICE_SAMPLE_RATE || sourceRate > VOICE_MAX_SOURCE_RATE) {
    throw new Error("VOICE_WAV_FORMAT_INVALID");
  }
  const targetLength = Math.min(
    VOICE_SAMPLE_RATE * VOICE_MAX_SECONDS,
    Math.max(1, Math.floor(source.length * VOICE_SAMPLE_RATE / sourceRate)),
  );
  if (sourceRate === VOICE_SAMPLE_RATE && source.length === targetLength) return source;
  const target = new Float32Array(targetLength);
  const ratio = sourceRate / VOICE_SAMPLE_RATE;
  for (let index = 0; index < targetLength; index += 1) {
    const position = index * ratio;
    const left = Math.min(source.length - 1, Math.floor(position));
    const right = Math.min(source.length - 1, left + 1);
    const weight = position - left;
    target[index] = source[left] + (source[right] - source[left]) * weight;
  }
  return target;
}

function encodeVoiceWav(samples) {
  if (!(samples instanceof Float32Array) || !samples.length || samples.length > VOICE_SAMPLE_RATE * VOICE_MAX_SECONDS) {
    throw new Error("VOICE_AUDIO_TOO_LARGE");
  }
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeAscii = (offset, value) => {
    for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index));
  };
  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, VOICE_SAMPLE_RATE, true);
  view.setUint32(28, VOICE_SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => {
    const bounded = Math.max(-1, Math.min(1, Number.isFinite(sample) ? sample : 0));
    view.setInt16(44 + index * 2, bounded < 0 ? bounded * 0x8000 : bounded * 0x7fff, true);
  });
  return new Uint8Array(buffer);
}

function voiceBytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

function insertVoiceTranscript(transcript) {
  const input = $("#agent-input");
  if (!input || typeof transcript !== "string" || !transcript.trim()) return false;
  const before = input.value.trimEnd();
  const joined = before ? `${before}\n${transcript.trim()}` : transcript.trim();
  if (joined.length > input.maxLength) return false;
  input.value = joined;
  updateAgentCharacterCount();
  input.focus();
  return true;
}

function preferredVoiceRecorderMimeType() {
  const candidates = [
    "audio/ogg;codecs=opus",
    "audio/webm;codecs=opus",
    "audio/ogg",
    "audio/webm",
  ];
  if (typeof window.MediaRecorder?.isTypeSupported !== "function") return "";
  return candidates.find((candidate) => window.MediaRecorder.isTypeSupported(candidate)) || "";
}

function stopVoiceRecorder(session) {
  const recorder = session?.recorder;
  if (!recorder || recorder.state === "inactive") {
    return Promise.reject(new Error("VOICE_RECORDER_NOT_ACTIVE"));
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    let timeoutId = 0;
    const cleanup = () => {
      if (timeoutId) window.clearTimeout(timeoutId);
      recorder.removeEventListener("stop", onStop);
      recorder.removeEventListener("error", onError);
      if (session.stopCleanup === cleanup) session.stopCleanup = null;
    };
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback(value);
    };
    const onStop = () => finish(resolve);
    const onError = (event) => finish(
      reject,
      event?.error instanceof Error ? event.error : new Error("VOICE_RECORDER_FAILED"),
    );
    session.stopCleanup = cleanup;
    recorder.addEventListener("stop", onStop, { once: true });
    recorder.addEventListener("error", onError, { once: true });
    timeoutId = window.setTimeout(
      () => finish(reject, new Error("VOICE_RECORDER_STOP_TIMEOUT")),
      VOICE_RECORDER_STOP_TIMEOUT_MS,
    );
    try {
      recorder.stop();
    } catch (error) {
      finish(reject, error);
    }
  });
}

async function decodeVoiceRecording(blob) {
  if (!(blob instanceof Blob) || blob.size < 1) throw new Error("VOICE_WAV_INVALID");
  if (blob.size > VOICE_MAX_RECORDED_BYTES) throw new Error("VOICE_AUDIO_TOO_LARGE");
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error("VOICE_WAV_FORMAT_INVALID");
  const context = new AudioContextClass({ sampleRate: VOICE_SAMPLE_RATE });
  let timeoutId = 0;
  try {
    const encoded = await blob.arrayBuffer();
    const decoded = await Promise.race([
      context.decodeAudioData(encoded.slice(0)),
      new Promise((_, reject) => {
        timeoutId = window.setTimeout(
          () => reject(new Error("VOICE_AUDIO_DECODE_TIMEOUT")),
          VOICE_DECODE_TIMEOUT_MS,
        );
      }),
    ]);
    const sourceRate = decoded?.sampleRate;
    if (
      !decoded
      || decoded.numberOfChannels < 1
      || !Number.isFinite(sourceRate)
      || sourceRate < VOICE_SAMPLE_RATE
      || sourceRate > VOICE_MAX_SOURCE_RATE
    ) {
      throw new Error("VOICE_WAV_FORMAT_INVALID");
    }
    const maxFrames = Math.floor(sourceRate * VOICE_MAX_SECONDS);
    const channel = decoded.getChannelData(0);
    if (!(channel instanceof Float32Array) || !channel.length) throw new Error("VOICE_WAV_INVALID");
    return {
      samples: new Float32Array(channel.subarray(0, Math.min(channel.length, maxFrames))),
      sourceRate,
    };
  } finally {
    if (timeoutId) window.clearTimeout(timeoutId);
    if (context.state !== "closed") context.close().catch(() => {});
  }
}

async function startVoiceCapture() {
  if (
    ui.voiceCaptureState !== "idle"
    || !ui.voiceTransportReady
    || !ui.voiceBrowserCaptureReady
    || !ui.voiceTranscriptionAttemptReady
  ) return;
  if (ui.voicePlaybackState !== "idle") stopVoicePlayback();
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const browserCapture = browserMicrophoneSupport();
  if (!browserCapture.ready || !navigator.mediaDevices?.getUserMedia || !AudioContextClass) {
    const copy = browserCapture.reason || "이 브라우저는 로컬 마이크 캡처를 지원하지 않습니다.";
    setVoiceCaptureState("idle", copy);
    showToast(copy, "warning");
    return;
  }
  const session = {
    cancelled: false,
    encodedChunks: [],
    encodedBytes: 0,
    recorderError: null,
  };
  ui.voiceSession = session;
  setVoiceCaptureState("requesting", "버튼을 누른 이 요청에서만 Windows 마이크 권한을 확인합니다.");
  try {
    session.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });
    if (session.cancelled || ui.voiceSession !== session) {
      releaseVoiceSession(session);
      return;
    }
    const mimeType = preferredVoiceRecorderMimeType();
    const recorderOptions = mimeType ? { mimeType, audioBitsPerSecond: 64000 } : undefined;
    session.recorder = recorderOptions
      ? new window.MediaRecorder(session.stream, recorderOptions)
      : new window.MediaRecorder(session.stream);
    session.recorder.ondataavailable = (event) => {
      if (session.cancelled || ui.voiceSession !== session || !(event.data instanceof Blob)) return;
      if (event.data.size < 1) return;
      session.encodedBytes += event.data.size;
      if (session.encodedBytes > VOICE_MAX_RECORDED_BYTES) {
        session.recorderError = new Error("VOICE_AUDIO_TOO_LARGE");
        if (ui.voiceCaptureState === "recording") setTimeout(stopVoiceCapture, 0);
        return;
      }
      session.encodedChunks.push(event.data);
    };
    session.recorder.onerror = (event) => {
      session.recorderError = event?.error instanceof Error
        ? event.error
        : new Error("VOICE_RECORDER_FAILED");
      if (ui.voiceCaptureState === "recording") setTimeout(stopVoiceCapture, 0);
    };
    session.recorder.start(100);
    session.startedAt = performance.now();
    session.limitTimer = setTimeout(stopVoiceCapture, VOICE_MAX_SECONDS * 1000);
    session.elapsedTimer = setInterval(() => {
      const elapsed = Math.min(VOICE_MAX_SECONDS, (performance.now() - session.startedAt) / 1000);
      setText("#agent-voice-capture-copy", `${elapsed.toFixed(1)}초 / ${VOICE_MAX_SECONDS}초 · 외부 전송 0`);
    }, 250);
    setVoiceCaptureState("recording", `0.0초 / ${VOICE_MAX_SECONDS}초 · 외부 전송 0`);
  } catch (error) {
    releaseVoiceSession(session);
    if (ui.voiceSession === session) ui.voiceSession = null;
    if (session.cancelled) return;
    const denied = error?.name === "NotAllowedError" || error?.name === "SecurityError";
    const copy = denied
      ? "마이크 권한이 허용되지 않았습니다. Windows 개인정보 보호 설정을 확인해 주세요."
      : "로컬 마이크를 시작하지 못했습니다.";
    setVoiceCaptureState("idle", copy);
    showToast(copy, "warning");
  }
}

async function stopVoiceCapture() {
  const session = ui.voiceSession;
  if (!session || ui.voiceCaptureState !== "recording") return;
  setVoiceCaptureState("encoding", "메모리 안에서 16kHz 모노 PCM WAV로 변환합니다.");
  try {
    await stopVoiceRecorder(session);
    if (session.recorderError) throw session.recorderError;
    const mimeType = session.recorder?.mimeType || session.encodedChunks[0]?.type || "";
    const recorded = new Blob(session.encodedChunks, mimeType ? { type: mimeType } : undefined);
    session.encodedChunks = [];
    releaseVoiceSession(session);
    const decoded = await decodeVoiceRecording(recorded);
    const resampled = resampleVoiceSamples(decoded.samples, decoded.sourceRate);
    const wav = encodeVoiceWav(resampled);
    const audioWavBase64 = voiceBytesToBase64(wav);
    session.abortController = new AbortController();
    setVoiceCaptureState("uploading", "인증된 127.0.0.1 API에서 형식과 로컬 STT 상태를 검증합니다.");
    const result = await api("/api/workspace/voice/transcribe", {
      method: "POST",
      body: { audio_wav_base64: audioWavBase64, language: "ko" },
      signal: session.abortController.signal,
    });
    if (session.cancelled || ui.voiceSession !== session) return;
    const refreshed = await refreshWorkspaceCapabilityDisclosure();
    if (!refreshed || !ui.voiceTranscriptionReady) {
      throw new Error("LOCAL_STT_INFERENCE_UNVERIFIED");
    }
    if (!insertVoiceTranscript(result?.transcript)) throw new Error("LOCAL_STT_OUTPUT_INVALID");
    setVoiceCaptureState("idle", "검증된 로컬 STT 결과를 입력창에 추가했습니다. 자동 전송하지 않습니다.");
    showToast("로컬 음성 전사를 입력창에 추가했습니다.", "success");
  } catch (error) {
    if (session.cancelled || ui.voiceSession !== session) return;
    const code = error?.code || error?.message;
    const copy = API_ERROR_COPY[code] || "로컬 음성 입력을 완료하지 못했습니다.";
    setVoiceCaptureState("idle", copy);
    showToast(copy, code === "LOCAL_STT_ARTIFACT_REQUIRED" ? "warning" : "error");
  } finally {
    releaseVoiceSession(session);
    session.encodedChunks = [];
    if (ui.voiceSession === session) ui.voiceSession = null;
    updateWorkspaceControlStates();
  }
}

function cancelVoiceCapture() {
  const session = ui.voiceSession;
  if (!session) return;
  session.cancelled = true;
  session.abortController?.abort();
  releaseVoiceSession(session);
  ui.voiceSession = null;
  setVoiceCaptureState("idle", "음성 입력을 취소했습니다. 녹음 데이터는 저장하지 않았습니다.");
}

function toggleVoiceCapture() {
  if (ui.voiceCaptureState === "recording") stopVoiceCapture();
  else if (ui.voiceCaptureState === "idle") startVoiceCapture();
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

function setRuntimeStatus(status, stage, evidenceVerified = true) {
  const pill = $("#runtime-pill");
  const label = $("#runtime-label");
  if (!pill || !label) return;
  const invalidEvidence = status === "succeeded" && !evidenceVerified;
  const display = invalidEvidence
    ? "검증 INVALID EVIDENCE"
    : {
      ready: "검증 READY",
      starting: "검증 STARTING",
      running: "검증 LIVE",
      cancelling: "검증 STOPPING",
      cancelled: "검증 CANCELLED",
      succeeded: "검증 VERIFIED",
      failed: "검증 FAILED",
      offline: "검증 RECONNECTING",
    }[status] || `검증 ${String(status || "READY").toUpperCase()}`;
  label.textContent = display;
  pill.dataset.state = invalidEvidence ? "failed" : status;
  pill.title = String(invalidEvidence ? "불완전한 실측 payload" : PHASE_LABELS[stage] || display).slice(0, 128);
}

function resetLiveRuntimePresentation() {
  $$('[data-live-evidence-badge]').forEach((badge) => {
    badge.textContent = "미검증";
    badge.dataset.scope = "unverified";
    badge.classList.remove("measured");
    badge.classList.add("planned");
  });
  $$('[data-live-evidence-status]').forEach((status) => {
    status.textContent = "검증 전";
    status.classList.remove("measured");
    status.classList.add("planned");
  });
  $$('[data-live-evidence-result]').forEach((result) => {
    result.textContent = "PENDING";
  });
  setText("#metric-vram", "—");
  setText("#telemetry-vram", "—");
  setText("#ledger-vram", "Peak VRAM — GiB");
  setText("#metric-depth", "—");
  setText("#orbit-depth", "—");
  setText("#reactor-depth", "—");
  setText("#metric-nodes", "검증 전");
  setText("#metric-residual", "—");
  setText("#reactor-residual", "—");
  setText("#metric-fallback", "미측정");
  setText("#telemetry-fallback", "미측정");
  setText("#telemetry-finite", "미측정");
  setText("#metric-tests", "—");
  $("#telemetry-finite")?.classList.remove("positive");
  setText("#rail-files", "—");
  setText("#device-name", "실행 전 미측정");
  setText("#rail-device", "실행 전 미측정");
  setText("#rail-time", "현재 프로세스 검증 전");
  setText("[data-live-cts-copy]", "현재 프로세스 라이브 검증 대기");
  setText("[data-live-vram-copy]", "현재 프로세스 라이브 검증 대기 · 상한 16.7 GiB");
  for (const selector of ["#vram-meter", "#telemetry-vram-fill"]) {
    const meter = $(selector);
    if (meter) {
      meter.max = MAX_LIVE_VRAM_LIMIT_GIB;
      meter.value = 0;
      meter.setAttribute("aria-label", "현재 프로세스 VRAM 검증 전");
    }
  }
  for (const selector of ["#device-name", "#rail-device"]) {
    $(selector)?.removeAttribute("title");
  }
}

function updateMetrics(metrics = null, evidenceScope = "unverified") {
  resetLiveRuntimePresentation();
  if (metrics === null) return;
  const currentEvidence = evidenceScope === "current";
  $$('[data-live-evidence-badge]').forEach((badge) => {
    badge.textContent = currentEvidence ? "현재 실측" : "직전 검증 실측";
    badge.dataset.scope = currentEvidence ? "current" : "prior";
    badge.classList.add("measured");
    badge.classList.remove("planned");
  });
  $$('[data-live-evidence-status]').forEach((status) => {
    status.textContent = currentEvidence ? "실측" : "직전 실측";
    status.classList.add("measured");
    status.classList.remove("planned");
  });
  $$('[data-live-evidence-result]').forEach((result) => {
    result.textContent = currentEvidence ? "PASS" : "PRIOR PASS";
  });

  const vram = metrics.peak_vram_gib;
  const allocatedVram = metrics.peak_allocated_vram_gib;
  const reservedVram = metrics.peak_reserved_vram_gib;
  const limit = metrics.vram_limit_gib;
  setText("#metric-vram", vram.toFixed(4));
  setText("#telemetry-vram", vram.toFixed(4));
  setText("#ledger-vram", `Peak VRAM ${vram.toFixed(4)} GiB`);
  $("#ledger-vram")?.setAttribute(
    "title",
    `Peak allocated ${allocatedVram.toFixed(4)} GiB; peak reserved ${reservedVram.toFixed(4)} GiB`,
  );
  const meterLabel = `VRAM 사용량 ${vram.toFixed(4)} GiB, 상한 ${limit.toFixed(1)} GiB`;
  for (const selector of ["#vram-meter", "#telemetry-vram-fill"]) {
    const meter = $(selector);
    if (meter) {
      meter.max = limit;
      meter.value = vram;
      meter.setAttribute("aria-label", meterLabel);
    }
  }
  setText("#metric-depth", metrics.reached_depth);
  setText("#orbit-depth", metrics.reached_depth);
  setText("#reactor-depth", metrics.reached_depth);
  setText("#metric-nodes", `${metrics.nodes_used} / ${metrics.node_capacity} nodes`);
  setText(
    "[data-live-cts-copy]",
    `${currentEvidence ? "현재 프로세스" : "직전 검증"} 실측 · ${metrics.nodes_used}/${metrics.node_capacity} node · finite`,
  );
  setText("#metric-residual", metrics.transition_residual.toFixed(6));
  setText("#reactor-residual", metrics.transition_residual.toExponential(3));
  setText("#metric-fallback", "미사용");
  setText("#telemetry-fallback", "NOT USED");
  setText("#telemetry-finite", "PASS");
  $("#telemetry-finite")?.classList.add("positive");
  setText("#rail-files", `${metrics.verified_files} files`);
  setText("#device-name", metrics.device);
  setText("#rail-device", metrics.device);
  const compactDevice = $("#device-name");
  const railDevice = $("#rail-device");
  if (compactDevice) compactDevice.title = metrics.device.slice(0, 256);
  if (railDevice) railDevice.title = metrics.device.slice(0, 256);
  setText(
    "[data-live-vram-copy]",
    `${currentEvidence ? "현재" : "직전 검증"} 실측 · ${metrics.device.slice(0, 96)} · 상한 ${limit.toFixed(1)} GiB`,
  );
  const date = new Date(metrics.measured_at);
  setText(
    "#rail-time",
    `${date.toLocaleDateString("ko-KR")} ${currentEvidence ? "현재" : "직전 검증"} 실측`,
  );
}

function updatePhases(state, evidenceVerified = false) {
  const stageIndex = PHASE_ORDER.indexOf(state.stage);
  const invalidEvidence = state.status === "succeeded" && !evidenceVerified;
  const succeeded = state.status === "succeeded" && evidenceVerified;
  const failed = state.status === "failed" || invalidEvidence;
  $$("#phase-list li").forEach((item) => {
    const indices = Object.keys(PHASE_TO_DOM)
      .filter((key) => PHASE_TO_DOM[key] === item.dataset.phase)
      .map((key) => PHASE_ORDER.indexOf(key))
      .filter((index) => index >= 0);
    const first = indices.length ? Math.min(...indices) : -1;
    const last = indices.length ? Math.max(...indices) : -1;
    const isCurrent = stageIndex >= first && stageIndex <= last && first >= 0;
    const isInvalidTerminal = invalidEvidence && item.dataset.phase === "postcheck";
    item.classList.toggle(
      "is-complete",
      succeeded || (!invalidEvidence && stageIndex >= 0 && last < stageIndex),
    );
    item.classList.toggle("is-active", ACTIVE_STATUSES.has(state.status) && isCurrent);
    item.classList.toggle("is-failed", failed && (isCurrent || isInvalidTerminal));
  });
  setText(
    "#phase-status",
    invalidEvidence ? "INVALID EVIDENCE" : PHASE_LABELS[state.stage] || PHASE_LABELS[state.status] || "대기",
  );
  const reactor = $("#reactor");
  reactor?.classList.toggle("is-running", ACTIVE_STATUSES.has(state.status));
  reactor?.classList.toggle("is-success", succeeded);
  const indicator = $("#live-indicator");
  if (indicator) {
    indicator.classList.toggle("is-live", ACTIVE_STATUSES.has(state.status));
    indicator.classList.toggle("is-success", succeeded);
    indicator.lastChild.textContent = ACTIVE_STATUSES.has(state.status)
      ? "LIVE"
      : succeeded
        ? "VERIFIED"
        : invalidEvidence
          ? "INVALID"
          : "STANDBY";
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
  $$('[data-agent-mode], [data-agent-prompt], [data-action="agent-focus"]').forEach((button) => {
    button.disabled = agentUnavailable || agentBusy;
  });
  const chatBusy = agentBusy || ui.agentConnectionLost;
  $(".chat-workspace")?.setAttribute("aria-busy", String(chatBusy));
  $("#chat-transcript")?.setAttribute("aria-busy", String(chatBusy));
  document.body.classList.toggle("agent-active", agentActive);
  updateWorkspaceControlStates();

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
  const proposalReview = $('[data-action="evolution-review"]');
  if (proposalReview) {
    proposalReview.disabled = ui.proposalReviewPending
      || evolutionActive
      || ui.evolutionProposalCount < 1
      || ui.agentConnectionLost;
    proposalReview.setAttribute("aria-busy", String(ui.proposalReviewPending));
    proposalReview.title = ui.evolutionProposalCount < 1
      ? "검토 대기 중인 무결성 확인 제안이 없습니다."
      : "현재 소스와 digest를 확인한 읽기 전용 unified diff를 엽니다.";
  }
  const proposalReviewLabel = $("[data-evolution-review-label]", proposalReview || document);
  if (proposalReviewLabel) {
    proposalReviewLabel.textContent = ui.proposalReviewPending
      ? "제안 확인 중"
      : `제안 diff 검토${ui.evolutionProposalCount ? ` · ${ui.evolutionProposalCount}` : ""}`;
  }
}

function updateState(state) {
  if (!state || typeof state !== "object") return;
  const prior = ui.lastStatus;
  const priorEvidenceVerified = ui.lastEvidenceVerified === true;
  const status = VALIDATION_STATUSES.has(state.status) ? state.status : "failed";
  const liveMetrics = normalizedLiveRuntimeMetrics(state.metrics);
  const hasLiveEvidence = liveMetrics !== null;
  const evidenceVerified = status === "succeeded" && hasLiveEvidence;
  ui.lastStatus = status;
  ui.lastEvidenceVerified = evidenceVerified;
  if (Number.isInteger(state.seq)) ui.lastSeq = Math.max(ui.lastSeq, state.seq);
  setRuntimeStatus(status, state.stage, evidenceVerified);
  updateMetrics(liveMetrics, evidenceVerified ? "current" : "prior");
  updatePhases({ ...state, status }, evidenceVerified);
  renderEvents(state.events || []);

  const active = ACTIVE_STATUSES.has(status);
  document.body.classList.toggle("runtime-active", active);
  updateControlStates();

  const railSeal = $(".rail-seal");
  if (railSeal) {
    railSeal.dataset.state = status === "succeeded" && !evidenceVerified ? "failed" : status;
  }

  if (active) {
    setText("#rail-verdict", "LIVE RUNNING");
    setText("#rail-time", hasLiveEvidence ? "직전 현재 프로세스 실측 유지 · 완료 후 교체" : "현재 실측 생성 중");
  } else if (status === "succeeded" && evidenceVerified) {
    setText("#rail-verdict", "VERIFIED");
    if (prior !== "succeeded" || !priorEvidenceVerified) showToast("완전한 현재 프로세스 실측이 CTS 안전 계약을 통과했습니다.", "success");
  } else if (status === "succeeded") {
    setText("#rail-verdict", "INVALID EVIDENCE");
    setText("#rail-time", "완전한 현재 프로세스 실측 없음");
    if (prior !== "succeeded" || priorEvidenceVerified) {
      showToast("검증 종료 payload가 완전한 CTS 안전 계약을 충족하지 못했습니다.", "error");
    }
  } else if (status === "failed") {
    setText("#rail-verdict", "NEW RUN FAILED");
    setText("#rail-time", hasLiveEvidence ? "직전 현재 프로세스 실측 유지 · 실패 로그 확인" : "유효한 현재 프로세스 실측 없음");
    const error = state.error?.message || state.error?.code || "검증 프로세스가 실패했습니다.";
    if (prior !== "failed") showToast(error, "error");
  } else if (status === "cancelled") {
    setText("#rail-verdict", "RUN CANCELLED");
    setText("#rail-time", hasLiveEvidence ? "직전 현재 프로세스 실측 유지" : "유효한 현재 프로세스 실측 없음");
    if (prior !== "cancelled") showToast("검증을 취소하고 GPU worker를 정리했습니다.", "warning");
  } else if (status === "ready") {
    setText("#rail-verdict", hasLiveEvidence ? "PRIOR EVIDENCE" : "NOT VERIFIED");
    setText(
      "#rail-time",
      hasLiveEvidence ? "직전 검증 실측 · 현재 실행 미검증" : "현재 프로세스 검증 전",
    );
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
  const sources = document.createElement("details");
  sources.className = "chat-sources";
  sources.hidden = true;
  const sourcesSummary = document.createElement("summary");
  sourcesSummary.textContent = "로컬 RAG 근거";
  const sourcesList = document.createElement("ol");
  sourcesList.setAttribute("aria-label", "답변에 사용된 로컬 RAG 근거");
  sources.append(sourcesSummary, sourcesList);
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
  bubble.append(header, content, sources, continueButton);
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
      generationMode: ["cogni_core", "cogni_core_rag", "rag_no_evidence", "conversation_fastpath", "factbook", "quality_fallback"].includes(message.generation_mode)
        ? message.generation_mode
        : null,
      sources: normalizedRetrievalSources(message.sources),
    });
  });

  if (!normalized.length) {
    if (ui.chatEmptyTemplate) transcript.replaceChildren(ui.chatEmptyTemplate.cloneNode(true));
    if (ui.voicePlaybackState !== "idle") releaseVoicePlayback();
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
    if (message.generationMode) item.dataset.generationMode = message.generationMode;
    else delete item.dataset.generationMode;
    item.setAttribute("aria-busy", String(message.streaming));
    const role = message.role;
    item.dataset.role = role;
    const avatar = $(".chat-avatar", item);
    const fastPathAnswer = role === "assistant" && message.generationMode === "conversation_fastpath";
    const factbookAnswer = role === "assistant" && message.generationMode === "factbook";
    avatar.textContent = factbookAnswer
      ? "FACT"
      : { user: "YOU", assistant: "AI", tool: "TOOL", system: "SYS" }[role];
    const author = $(".chat-bubble header strong", item);
    author.textContent = factbookAnswer
      ? "Runtime Fact-book"
      : { user: "사용자", assistant: "Cogni Agent", tool: "로컬 작업", system: "시스템" }[role];
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
        if (fastPathAnswer) {
          completionCopy = "대화 FAST PATH · 완료";
        } else if (message.generationMode === "factbook") {
          completionCopy = "FACT-BOOK · 검증된 사실";
        } else if (message.generationMode === "cogni_core_rag") {
          completionCopy = "로컬 RAG · 근거 연결 완료";
        } else if (message.generationMode === "rag_no_evidence") {
          completionCopy = "로컬 RAG · 근거 없음";
        } else if (message.generationMode === "quality_fallback") {
          completionCopy = "품질 검증 실패 · 복구 필요";
        } else {
          completionCopy = message.continuations
            ? `자동 이어쓰기 ${message.continuations}회 · 완료`
            : "완료";
        }
      }
      completion.textContent = completionCopy;
      completion.hidden = !completionCopy;
      completion.title = fastPathAnswer
        ? "짧은 소셜 대화를 제한된 로컬 규칙으로 한 번만 응답"
        : message.generationMode === "factbook"
          ? "모델 생성 없이 검증된 Runtime Fact-book에서 구성한 응답"
          : message.generationMode === "quality_fallback"
          ? "모델 후보가 품질 기준을 통과하지 못해 실제 답변을 제공하지 못했습니다."
          : message.generatedTokens
            ? `생성 ${message.generatedTokens.toLocaleString("ko-KR")} 토큰`
            : "";
    }
    const content = $(".chat-bubble p", item);
    renderMessageContentWithCitations(content, message.content, message.sources);
    const sources = $(".chat-sources", item);
    const sourcesList = $(".chat-sources ol", item);
    if (sources && sourcesList) {
      const fragment = document.createDocumentFragment();
      message.sources.forEach((source) => {
        const row = document.createElement("li");
        const title = document.createElement("button");
        const score = document.createElement("small");
        title.type = "button";
        title.className = "chat-source-button";
        if (!configureRagEvidenceButton(title, source)) {
          title.disabled = true;
          title.setAttribute("aria-label", `근거 ${source.number} 출처 위치 미제공`);
        }
        title.textContent = `[근거 ${source.number}] ${source.title}`;
        const provenance = [];
        if (source.attachmentId) provenance.push(`attachment_id ${source.attachmentId}`);
        if (source.chunkIndex !== null) provenance.push(`chunk_index ${source.chunkIndex}`);
        provenance.push(source.score === null ? "score 미제공" : `score ${source.score.toFixed(4)}`);
        provenance.push(source.provenance.retrievalMode);
        provenance.push(`revision ${source.provenance.revision.slice(0, 12)}`);
        provenance.push(`source_sha256 ${source.provenance.sourceSha256.slice(0, 12)}…`);
        provenance.push(
          `selected ${source.provenance.selectedExcerptChars}/${source.provenance.indexedExcerptChars} chars`,
        );
        if (source.provenance.selectedExcerptTruncated) provenance.push("selected excerpt truncated");
        score.textContent = provenance.join(" · ");
        row.append(title, score);
        fragment.append(row);
      });
      sourcesList.replaceChildren(fragment);
      sources.hidden = role !== "assistant" || message.sources.length === 0;
    }
    const continueButton = $(".chat-continue-button", item);
    if (continueButton) continueButton.hidden = role !== "assistant" || !message.truncated || message.streaming;
    transcript.append(item);
  });
  existing.forEach((item) => item.remove());
  transcript.scrollTop = nearBottom ? transcript.scrollHeight : priorScrollTop;
  if (
    ui.voicePlaybackState !== "idle"
    && ui.voicePlaybackText
    && latestRenderedAssistantText() !== ui.voicePlaybackText
  ) {
    releaseVoicePlayback("새 답변이 표시되어 이전 음성 재생을 종료했습니다.");
  } else {
    updateWorkspaceControlStates();
  }
}

function normalizedRuntimeCapability(raw, expectedName) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  if (raw.name !== expectedName) return null;
  if (!RUNTIME_CAPABILITY_STATES.has(raw.state)) return null;
  if (!RUNTIME_EVIDENCE_CLASSES.has(raw.evidence)) return null;
  if (typeof raw.answer_bearing !== "boolean") return null;
  if (typeof raw.runtime_mutation_allowed !== "boolean") return null;
  if (typeof raw.detail !== "string" || raw.detail.length < 1 || raw.detail.length > 256) return null;
  const nonAnswerStates = new Set(["disabled", "research", "advisory", "night_only", "proposal_only"]);
  if (nonAnswerStates.has(raw.state) && raw.answer_bearing) return null;
  if (raw.state === "proposal_only" && raw.runtime_mutation_allowed) return null;
  return Object.freeze({
    name: expectedName,
    state: raw.state,
    evidence: raw.evidence,
    answerBearing: raw.answer_bearing,
    runtimeMutationAllowed: raw.runtime_mutation_allowed,
    detail: raw.detail,
  });
}

function normalizedRuntimeCapabilities(raw) {
  const result = Object.create(null);
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return result;
  RUNTIME_CAPABILITY_NAMES.forEach((name) => {
    const record = normalizedRuntimeCapability(raw[name], name);
    if (record) result[name] = record;
  });
  return result;
}

function capabilityAuthorityLabel(record) {
  if (!record) return "UNVERIFIED · OFF";
  if (record.state === "authoritative" && record.answerBearing) {
    return "AUTHORITATIVE · ANSWER-BEARING";
  }
  if (record.state === "canary" && record.answerBearing) return "CANARY · ANSWER-BEARING";
  if (record.state === "advisory" && !record.answerBearing) return "ADVISORY · TELEMETRY ONLY";
  if (record.state === "gated" && !record.answerBearing) return "GATED · OFF";
  if (record.state === "night_only" && !record.answerBearing) {
    return "NIGHT ONLY · INFERENCE OFF";
  }
  if (record.state === "research" && !record.answerBearing) return "RESEARCH ARCHIVE ONLY";
  if (
    record.state === "proposal_only"
    && !record.answerBearing
    && !record.runtimeMutationAllowed
  ) return "PROPOSAL ONLY · MUTATION BLOCKED";
  if (record.state === "disabled" && !record.answerBearing) return "DISABLED · OFF";
  return "UNVERIFIED · OFF";
}

function capabilityEvidenceLabel(record) {
  return record ? `EVIDENCE ${record.evidence.toUpperCase()}` : "EVIDENCE UNVERIFIED";
}

function executionLabel(status) {
  if (status === "active") return "ACTIVE";
  if (status === "standby") return "STANDBY";
  if (status === "not_loaded") return "NOT LOADED";
  if (status === "off") return "OFF";
  return "UNVERIFIED";
}

function renderCapabilityDisclosures() {
  $$("[data-capability-name]").forEach((container) => {
    const record = ui.runtimeCapabilities[container.dataset.capabilityName] || null;
    const authority = capabilityAuthorityLabel(record);
    const evidence = capabilityEvidenceLabel(record);
    $$("[data-capability-authority]", container).forEach((element) => {
      element.textContent = authority;
    });
    $$("[data-capability-evidence]", container).forEach((element) => {
      element.textContent = evidence;
    });
    if (container.matches("button")) {
      container.title = `${authority} · ${evidence}`;
    }
  });
  const complete = RUNTIME_CAPABILITY_NAMES.every((name) => ui.runtimeCapabilities[name]);
  setText("#agent-core-badge", complete ? "FACT-BOOK DISCLOSED" : "AUTHORITY UNVERIFIED");
  setText(
    "#architecture-authority-badge",
    complete ? "FACT-BOOK DISCLOSED" : "AUTHORITY UNVERIFIED",
  );
}

function updateExecutionIndicator(container, status) {
  const label = executionLabel(status);
  const active = status === "active";
  const unavailable = status === "not_loaded" || status === "off" || !EXECUTION_STATES.has(status);
  container.classList.toggle("is-active", active);
  container.classList.toggle("is-unavailable", unavailable);
  const element = $("[data-execution-status]", container);
  if (element) element.textContent = label;
}

function updateNodeDetail(button) {
  if (!button) return;
  const nodeCopy = NODE_COPY[button.dataset.node];
  if (!nodeCopy) return;
  const panel = $("#node-detail");
  if (!panel) return;
  const [title, copy] = nodeCopy;
  const capabilityName = button.dataset.capabilityName || "";
  const executionModule = button.dataset.executionModule || "";
  const record = capabilityName ? ui.runtimeCapabilities[capabilityName] || null : null;
  const execution = executionModule
    ? executionLabel(ui.runtimeExecutionModules[executionModule])
    : button.dataset.node === "rhythm"
      ? "STATE MACHINE"
      : "UNVERIFIED";
  $("h2", panel).textContent = title;
  $("p", panel).textContent = copy;
  setText("#node-detail-execution", execution);
  setText("#node-detail-authority", capabilityAuthorityLabel(record));
  setText("#node-detail-evidence", capabilityEvidenceLabel(record));
}

function updateAgentCore(core = {}) {
  ui.runtimeCapabilities = normalizedRuntimeCapabilities(core.capabilities);
  ui.runtimeExecutionModules = Object.create(null);
  const allowedExecutionModules = new Set([
    "gemma", "router", "swarm", "experts", "cts",
    "fast", "stability", "aflow", "harness",
  ]);
  if (core.modules && typeof core.modules === "object" && !Array.isArray(core.modules)) {
    Object.entries(core.modules).forEach(([name, status]) => {
      if (allowedExecutionModules.has(name) && EXECUTION_STATES.has(status)) {
        ui.runtimeExecutionModules[name] = status;
      }
    });
  }

  renderCapabilityDisclosures();
  $$("[data-agent-module]").forEach((module) => {
    updateExecutionIndicator(module, ui.runtimeExecutionModules[module.dataset.agentModule]);
  });
  $$("[data-execution-module]").forEach((module) => {
    updateExecutionIndicator(module, ui.runtimeExecutionModules[module.dataset.executionModule]);
  });
  updateNodeDetail($("[data-node].is-selected"));
}

function updateEvolutionState(evolution = {}) {
  if (!evolution || typeof evolution !== "object") return;
  const evidenceFailures = Number.isInteger(evolution.evidence_failures)
    ? evolution.evidence_failures
    : evolution.failures;
  if (Number.isInteger(evidenceFailures)) setText("#evolution-failures", evidenceFailures);
  if (Number.isFinite(evolution.evidence_capture_ratio)) {
    const ratio = Math.max(0, Math.min(1, evolution.evidence_capture_ratio));
    setText("#evolution-coverage", `${(ratio * 100).toFixed(ratio === 1 ? 0 : 1)}%`);
  }
  const pending = Number.isInteger(evolution.pending_proposals)
    ? evolution.pending_proposals
    : evolution.rich_pending_proposals;
  if (Number.isInteger(pending)) {
    ui.evolutionProposalCount = Math.max(0, pending);
    setText("#evolution-proposals", ui.evolutionProposalCount);
  }
  const unreviewable = Number.isInteger(evolution.unreviewable_proposals)
    ? Math.max(0, evolution.unreviewable_proposals)
    : 0;
  if (typeof evolution.last_run === "string" && evolution.last_run) {
    setText("#evolution-last-run", formatAgentTime(evolution.last_run) || evolution.last_run);
  }
  if (typeof evolution.promotion_enabled === "boolean") {
    setText("#evolution-sandbox", evolution.promotion_enabled ? "독립 검증 필요" : "차단 · 제안 전용");
  }
  const badge = $("#evolution-badge");
  if (badge) {
    const degraded = evolution.integrity_degraded === true || unreviewable > 0;
    badge.dataset.integrity = degraded ? "degraded" : "healthy";
    if (degraded) badge.textContent = `무결성 제외 ${unreviewable}`;
    else if (typeof evolution.status === "string") badge.textContent = evolution.status.slice(0, 64);
  }
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
  let agentLabel = labels[ui.agentStatus] || ui.agentStatus.toUpperCase();
  const modelLoaded = state.core?.model_loaded === true;
  if (ui.agentStatus === "succeeded" && state.completion?.generation_mode === "quality_fallback") {
    agentLabel = "RESPONSE FAILED";
  }
  if ((ui.agentStatus === "ready" || ui.agentStatus === "succeeded") && !modelLoaded) {
    if (state.completion?.generation_mode === "conversation_fastpath") {
      agentLabel = "CONVERSATION READY";
    } else {
      agentLabel = state.completion?.generation_mode === "factbook"
        ? "FACT-BOOK ONLY"
        : "MODEL STANDBY";
    }
  }
  setText("#agent-status-label", agentLabel);
  const heading = $("#agent-heading-state");
  if (heading) heading.dataset.state = ui.agentStatus;
  if (Array.isArray(state.conversation)) renderAgentConversation(state.conversation);
  updateAgentCore(state.core || {});
  updateEvolutionState(state.evolution || {});
  if (
    ui.imageAttestationPending
    && ["succeeded", "cancelled", "failed"].includes(ui.agentStatus)
    && !ui.imageAttestationSettling
  ) {
    ui.imageModelFirstUseReady = false;
    void settleFirstImageAttestation();
  }
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

function syncAgentInputLimit() {
  const input = $("#agent-input");
  if (!input) return;
  input.maxLength = ui.agentMode === "task"
    ? MAX_AGENT_PROJECT_INPUT_CHARS
    : MAX_AGENT_CHAT_INPUT_CHARS;
}

function updateAgentCharacterCount() {
  syncAgentInputLimit();
  const input = $("#agent-input");
  const length = input?.value.length || 0;
  const limit = input?.maxLength || MAX_AGENT_CHAT_INPUT_CHARS;
  setText(
    "#agent-char-count",
    `${length.toLocaleString("ko-KR")} / ${limit.toLocaleString("ko-KR")}자`,
  );
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
    const requestBody = {
      message,
      mode: ui.agentMode,
      rag: ui.agentMode === "chat" && ui.ragEnabled
        && ui.ragAnswerIntegrationReady,
    };
    if (
      ui.agentMode === "chat"
      && (ui.imageModelIntegrationReady || ui.imageModelFirstUseReady)
      && ui.selectedImageAttachmentId
    ) {
      requestBody.image_attachment_id = ui.selectedImageAttachmentId;
    }
    const state = await api("/api/agent/chat", {
      method: "POST",
      body: requestBody,
    });
    if (input) input.value = "";
    if (state?.image_input_admitted === true) {
      ui.selectedImageAttachmentId = "";
      renderWorkspaceAttachments();
    }
    if (state?.image_attestation_probe === true) {
      ui.imageAttestationPending = true;
      showToast("첫 이미지로 로컬 전처리·모델 추론 경로를 검증하고 있습니다.", "info");
    }
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
    showToast("Self-Harness 제안 전용 증거 주기를 시작했습니다.");
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
  if (options.signal) request.signal = options.signal;
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

async function apiBlob(path) {
  const contentPrefix = `${ATTACHMENT_CONTENT_ENDPOINT}?attachment_id=`;
  if (!path.startsWith(contentPrefix) || !/^[0-9a-f]{24}$/.test(path.slice(contentPrefix.length))) {
    const invalid = new Error("ATTACHMENT_PREVIEW_UNAVAILABLE");
    invalid.code = "ATTACHMENT_PREVIEW_UNAVAILABLE";
    throw invalid;
  }
  let response;
  try {
    response = await fetch(path, {
      method: "GET",
      cache: "no-store",
      credentials: "same-origin",
    });
  } catch (cause) {
    const error = new Error("CONNECTION_LOST");
    error.code = "CONNECTION_LOST";
    error.cause = cause;
    throw error;
  }
  if (!response.ok) {
    let payload = {};
    try { payload = await response.json(); } catch (_) { payload = {}; }
    const code = payload?.error?.code || `HTTP_${response.status}`;
    const error = new Error(code);
    error.code = code;
    error.status = response.status;
    throw error;
  }
  const contentType = (response.headers.get("Content-Type") || "").split(";", 1)[0];
  if (!new Set(["image/png", "image/jpeg", "image/webp"]).has(contentType)) {
    const error = new Error("ATTACHMENT_PREVIEW_UNAVAILABLE");
    error.code = "ATTACHMENT_PREVIEW_UNAVAILABLE";
    throw error;
  }
  const blob = await response.blob();
  if (blob.size < 1 || blob.size > MAX_ATTACHMENT_UPLOAD_BYTES) {
    const error = new Error("ATTACHMENT_TOO_LARGE");
    error.code = "ATTACHMENT_TOO_LARGE";
    throw error;
  }
  return blob;
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
  releaseVoicePlayback();
  if (ui.voiceSession) cancelVoiceCapture();
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
  $$("[data-node]").forEach((button) => button.addEventListener("click", () => {
    $$("[data-node]").forEach((item) => item.classList.toggle("is-selected", item === button));
    updateNodeDetail(button);
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
  $('[data-action="evolution-review"]')?.addEventListener("click", (event) => {
    openProposalReview(event.currentTarget);
  });
  $('[data-action="evolution-review-close"]')?.addEventListener("click", closeProposalReview);
  $("#proposal-review-layer")?.addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeProposalReview();
  });
  $('[data-action="workspace-attach"]')?.addEventListener("click", () => {
    const input = $("#agent-attachment-input");
    if (input && !input.disabled) input.click();
  });
  $("#agent-attachment-input")?.addEventListener("change", (event) => {
    uploadWorkspaceFiles(event.target.files);
  });
  $("#agent-attachment-list")?.addEventListener("click", (event) => {
    const trigger = event.target.closest(
      '[data-action="workspace-attachment-delete"], [data-action="workspace-attachment-preview"], [data-action="workspace-image-select"]',
    );
    if (!trigger || trigger.disabled) return;
    if (trigger.dataset.action === "workspace-attachment-preview") {
      openWorkspaceAttachmentPreview(trigger.dataset.attachmentId, trigger);
    } else if (trigger.dataset.action === "workspace-image-select") {
      selectWorkspaceImageAttachment(trigger.dataset.attachmentId);
    } else {
      deleteWorkspaceAttachment(trigger.dataset.attachmentId);
    }
  });
  $$('[data-action="rag-evidence-close"]').forEach((button) => {
    button.addEventListener("click", closeRagEvidenceDrawer);
  });
  $("#evidence-drawer-layer")?.addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeRagEvidenceDrawer();
  });
  $('[data-action="workspace-rag-toggle"]')?.addEventListener("click", toggleWorkspaceRag);
  $('[data-action="workspace-rag-reindex"]')?.addEventListener("click", reindexWorkspaceAttachments);
  $('[data-action="workspace-preview-close"]')?.addEventListener("click", closeWorkspaceAttachmentPreview);
  $("#attachment-preview-layer")?.addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeWorkspaceAttachmentPreview();
  });
  $('[data-action="workspace-microphone"]')?.addEventListener("click", toggleVoiceCapture);
  $('[data-action="workspace-voice-stop"]')?.addEventListener("click", stopVoiceCapture);
  $('[data-action="workspace-voice-cancel"]')?.addEventListener("click", cancelVoiceCapture);
  $('[data-action="workspace-tts-play"]')?.addEventListener("click", speakLatestAssistantReply);
  $('[data-action="workspace-tts-stop"]')?.addEventListener("click", stopVoicePlayback);
  $('[data-action="workspace-web-search"]')?.addEventListener("click", () => toggleLensSearchDrawer());
  $('[data-action="workspace-web-close"]')?.addEventListener("click", () => toggleLensSearchDrawer(false));
  $('[data-action="workspace-web-submit"]')?.addEventListener("click", searchLensOfficialApi);
  $("#agent-lens-search-form")?.addEventListener("submit", searchLensOfficialApi);
  $("#agent-model-selector")?.addEventListener("change", selectWorkspaceModel);
  $("#chat-transcript")?.addEventListener("click", (event) => {
    const evidenceTrigger = event.target.closest('[data-action="rag-evidence-open"]');
    if (evidenceTrigger && !evidenceTrigger.disabled) {
      const source = evidenceSourceFromTrigger(evidenceTrigger);
      if (source) openRagEvidenceSource(source, evidenceTrigger);
      return;
    }
    const trigger = event.target.closest('[data-action="agent-focus"]');
    if (!trigger || trigger.disabled) return;
    const input = $("#agent-input");
    if (!input || input.disabled) return;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    input.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "nearest" });
    input.focus({ preventScroll: true });
  });
  $$('[data-agent-mode]').forEach((button) => button.addEventListener("click", () => {
    ui.agentMode = button.dataset.agentMode;
    if (ui.agentMode !== "chat") {
      ui.ragEnabled = false;
      ui.selectedImageAttachmentId = "";
      renderWorkspaceAttachments();
    }
    $$('[data-agent-mode]').forEach((candidate) => {
      const selected = candidate === button;
      candidate.classList.toggle("is-selected", selected);
      candidate.setAttribute("aria-pressed", String(selected));
    });
    updateWorkspaceControlStates();
    updateAgentCharacterCount();
    $("#agent-input")?.focus();
  }));
  $$('[data-agent-prompt]').forEach((button) => button.addEventListener("click", () => {
    const mode = button.dataset.agentModeValue || "chat";
    ui.agentMode = mode;
    if (ui.agentMode !== "chat") {
      ui.ragEnabled = false;
      ui.selectedImageAttachmentId = "";
      renderWorkspaceAttachments();
    }
    $$('[data-agent-mode]').forEach((candidate) => {
      const selected = candidate.dataset.agentMode === mode;
      candidate.classList.toggle("is-selected", selected);
      candidate.setAttribute("aria-pressed", String(selected));
    });
    updateWorkspaceControlStates();
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
    const evidenceLayer = $("#evidence-drawer-layer");
    if (evidenceLayer && !evidenceLayer.hidden) {
      if (event.key === "Escape") {
        closeRagEvidenceDrawer();
        return;
      }
      trapFocusWithin(event, $("#evidence-drawer"));
      return;
    }
    const attachmentLayer = $("#attachment-preview-layer");
    if (attachmentLayer && !attachmentLayer.hidden) {
      if (event.key === "Escape") {
        closeWorkspaceAttachmentPreview();
        return;
      }
      trapFocusWithin(event, $(".attachment-preview-dialog", attachmentLayer));
      return;
    }
    const proposalLayer = $("#proposal-review-layer");
    if (proposalLayer && !proposalLayer.hidden) {
      if (event.key === "Escape") {
        closeProposalReview();
        return;
      }
      trapFocusWithin(event, $(".proposal-review-dialog", proposalLayer));
      return;
    }
    if (event.key === "Escape" && !$("#tour-panel").hidden) closeTour();
    if (event.key === "Escape" && ui.voiceSession) cancelVoiceCapture();
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
  window.addEventListener("pagehide", () => releaseVoicePlayback(), { once: true });
  updateFullscreenState();
  updateAgentCharacterCount();
  updateControlStates();
  loadWorkspaceCapabilities();
  pollAgentState();
  pollState();
}

document.addEventListener("DOMContentLoaded", init);
