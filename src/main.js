import { pipeline } from '@huggingface/transformers';
import { KokoroTTS } from 'kokoro-js';
import 'aframe';
import './style.css';

const MODEL_ID = 'Xenova/whisper-base';
const TTS_MODEL_ID = 'onnx-community/Kokoro-82M-v1.0-ONNX';
const TARGET_SAMPLE_RATE = 16_000;

const recordButton = document.querySelector('#record-button');
const buttonLabel = document.querySelector('#button-label');
const status = document.querySelector('#status');
const transcript = document.querySelector('#transcript');
const audioPlayer = document.querySelector('#audio-player');
const hermesStatus = document.querySelector('#hermes-status');
const hermesPlatform = document.querySelector('#hermes-platform');
const hermesVersion = document.querySelector('#hermes-version');
const hermesModel = document.querySelector('#hermes-model');
const hermesOwner = document.querySelector('#hermes-owner');
const hermesModelCreated = document.querySelector('#hermes-model-created');
const hermesCurrentModel = document.querySelector('#hermes-current-model');
const hermesGatewayState = document.querySelector('#hermes-gateway-state');
const hermesRuntime = document.querySelector('#hermes-runtime');
const hermesReadiness = document.querySelector('#hermes-readiness');
const hermesPlatforms = document.querySelector('#hermes-platforms');
const hermesActivity = document.querySelector('#hermes-activity');
const hermesQueues = document.querySelector('#hermes-queues');
const hermesDisk = document.querySelector('#hermes-disk');
const hermesRequests = document.querySelector('#hermes-requests');
const hermesTokens = document.querySelector('#hermes-tokens');
const hermesHeartbeat = document.querySelector('#hermes-heartbeat');
const hermesToolsets = document.querySelector('#hermes-toolsets');
const hermesCapabilities = document.querySelector('#hermes-capabilities');
const hermesSessionWindow = document.querySelector('#hermes-session-window');
const hermesSessionCount = document.querySelector('#hermes-session-count');
const hermesSessionActive = document.querySelector('#hermes-session-active');
const hermesSessionMessages = document.querySelector('#hermes-session-messages');
const hermesSessionTools = document.querySelector('#hermes-session-tools');
const hermesSessionInputTokens = document.querySelector('#hermes-session-input-tokens');
const hermesSessionOutputTokens = document.querySelector('#hermes-session-output-tokens');
const hermesSessionSources = document.querySelector('#hermes-session-sources');
const hermesNote = document.querySelector('#hermes-note');
const hermesIndicator = document.querySelector('#hermes-indicator');
const hermesRefresh = document.querySelector('#hermes-refresh');
const timelineRefresh = document.querySelector('#timeline-refresh');
const timelineStatus = document.querySelector('#timeline-status');
const timelineScroll = document.querySelector('#timeline-scroll');
const timelineChart = document.querySelector('#timeline-chart');
const timelineDetail = document.querySelector('#timeline-detail');
const timelineDetailTitle = document.querySelector('#timeline-detail-title');
const timelineDetailMeta = document.querySelector('#timeline-detail-meta');
const timelineMessages = document.querySelector('#timeline-messages');
const timelineDetailClose = document.querySelector('#timeline-detail-close');
const activityDialog = document.querySelector('#activity-dialog');
const activityDialogTitle = document.querySelector('#activity-dialog-title');
const activityDialogMeta = document.querySelector('#activity-dialog-meta');
const activityDialogSummary = document.querySelector('#activity-dialog-summary');
const activityDialogEvents = document.querySelector('#activity-dialog-events');
const activityDialogClose = document.querySelector('#activity-dialog-close');
const activityTrackToggle = document.querySelector('#activity-track-toggle');
const activityTrack = document.querySelector('#activity-track');
const activityTrackLegend = document.querySelector('#activity-track-legend');
const xrOpenButton = document.querySelector('#xr-open');
const xrCloseButton = document.querySelector('#xr-close');
const xrViewer = document.querySelector('#xr-viewer');
const xrSceneHost = document.querySelector('#xr-scene-host');
const xrStatus = document.querySelector('#xr-status');
const hermesKeyForm = document.querySelector('#hermes-key-form');
const hermesKeyInput = document.querySelector('#hermes-key');
const hermesForgetButton = document.querySelector('#hermes-forget');
const HERMES_API_ORIGIN = 'http://executi.tail5b9fc7.ts.net:8642';
const HERMES_API_KEY_STORAGE_KEY = 'local-voice.hermes-api-key';
const HERMES_ENDPOINTS = {
  health: '/health/detailed',
  models: '/v1/models',
  modelOptions: '/api/model/options',
  capabilities: '/v1/capabilities',
  toolsets: '/v1/toolsets',
  sessions: '/api/sessions?limit=100&offset=0',
};
const TIMELINE_SESSION_LIMIT = 50;
const MESSAGE_PAGE_SIZE = 500;
const MESSAGE_DETAIL_PAGE_SIZE = 25;
const ACTIVITY_GAP_MS = 15 * 60 * 1000;
const ACTIVITY_COLORS = {
  'Text input': '#4d8a68',
  'Text output': '#7272ba',
  Reasoning: '#b077b5',
  'Tool call': '#d38a3d',
  'Tool result': '#4e9bab',
  Other: '#90978d',
};

let recorder;
let recordedChunks = [];
let microphoneStream;
let transcriberPromise;
let ttsPromise;
let busy = false;
let refreshingHermes = false;
let hermesPollingTimer;
let timelineLoadId = 0;
let activitySelectionId = 0;
let xrSessions = [];
let xrTimeRange;
let selectedActivityMessages = [];
let xrSelectedPeriod;

function setStatus(message) {
  status.textContent = message;
}

function waitForPaint() {
  return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

function readHermesKey() {
  try {
    return window.localStorage.getItem(HERMES_API_KEY_STORAGE_KEY) || '';
  } catch {
    return '';
  }
}

function setHermesKey(key) {
  try {
    if (key) window.localStorage.setItem(HERMES_API_KEY_STORAGE_KEY, key);
    else window.localStorage.removeItem(HERMES_API_KEY_STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}

async function refreshHermesStatus() {
  if (refreshingHermes) return;
  refreshingHermes = true;
  hermesRefresh.disabled = true;
  hermesRefresh.textContent = 'Checking…';
  hermesStatus.textContent = 'Checking…';
  hermesNote.textContent = 'Fetching health, model, runtime, and tool information…';
  hermesIndicator.className = 'service-indicator is-loading';

  try {
    const apiKey = readHermesKey();
    if (!apiKey) {
      throw new Error('Enter your gateway API key above to load Hermes information.');
    }

    if (window.location.protocol === 'https:') {
      throw new Error('The Hermes gateway uses HTTP. Browsers block requests to it from an HTTPS page; the gateway needs HTTPS for direct browser access.');
    }

    const headers = { Accept: 'application/json', Authorization: `Bearer ${apiKey}` };
    const responses = await Promise.all(Object.entries(HERMES_ENDPOINTS).map(async ([name, path]) => {
      const response = await fetch(`${HERMES_API_ORIGIN}${path}`, {
        headers,
        cache: 'no-store',
        signal: AbortSignal.timeout(10_000),
      });
      if (!response.ok) throw new Error(`${name} endpoint returned HTTP ${response.status}.`);
      return [name, await response.json()];
    }));
    const api = Object.fromEntries(responses);
    const info = api.health;
    const model = api.models.data?.[0];
    const runtime = api.capabilities.runtime || {};
    const metrics = info.metrics_today || {};
    const readinessChecks = Object.values(info.readiness?.checks || {});
    const readyCount = readinessChecks.filter((check) => check.status === 'ok').length;
    const connectedPlatforms = Object.entries(info.platforms || {})
      .filter(([, platform]) => platform.state === 'connected')
      .map(([name]) => name);
    const enabledToolsets = (api.toolsets.data || []).filter((toolset) => toolset.enabled);
    const features = api.capabilities.features || {};
    const enabledCapabilities = Object.entries(features)
      .filter(([, value]) => value === true)
      .map(([name]) => name.replaceAll('_', ' '));
    const sessionSummary = summarizeSessions(api.sessions);
    const queues = info.readiness?.checks?.background_queues || {};
    const disk = info.readiness?.checks?.disk || {};
    const currentModel = api.modelOptions.model || model?.id || '—';
    const currentProvider = api.modelOptions.provider || 'provider unknown';

    hermesStatus.textContent = info.status || 'Available';
    hermesPlatform.textContent = info.platform || '—';
    hermesVersion.textContent = info.version || '—';
    hermesModel.textContent = model?.id || 'None reported';
    hermesOwner.textContent = model?.owned_by || '—';
    hermesModelCreated.textContent = model?.created
      ? new Date(model.created * 1000).toLocaleString()
      : '—';
    hermesCurrentModel.textContent = `${currentProvider} / ${currentModel}`;
    hermesGatewayState.textContent = info.gateway_state || '—';
    hermesRuntime.textContent = runtime.mode?.replaceAll('_', ' ') || '—';
    hermesReadiness.textContent = readinessChecks.length ? `${readyCount} / ${readinessChecks.length} OK` : '—';
    hermesPlatforms.textContent = connectedPlatforms.length ? connectedPlatforms.join(', ') : 'None';
    hermesActivity.textContent = `${info.active_agents ?? 0} agents / ${info.api_server?.active_runs ?? 0} runs`;
    hermesQueues.textContent = `${queues.active_api_runs ?? 0} API / ${queues.process_completions ?? 0} completions / ${queues.active_delegations ?? 0} delegations`;
    hermesDisk.textContent = Number.isFinite(disk.used_percent) && Number.isFinite(disk.free_bytes)
      ? `${disk.used_percent}% / ${formatBytes(disk.free_bytes)}`
      : '—';
    hermesRequests.textContent = `${formatCount(metrics.requests)} / ${formatCount(metrics.messages)}`;
    hermesTokens.textContent = `${formatCount(metrics.tokens)} / ${formatLatency(metrics.latency_p95_ms)}`;
    hermesHeartbeat.textContent = info.last_heartbeat ? new Date(info.last_heartbeat).toLocaleString() : '—';
    renderChips(hermesToolsets, enabledToolsets.map((toolset) => toolset.label || toolset.name));
    renderChips(hermesCapabilities, enabledCapabilities);
    renderSessionSummary(sessionSummary);
    hermesIndicator.className = info.status === 'ok'
      ? 'service-indicator is-online'
      : 'service-indicator is-offline';
    hermesNote.textContent = `Runtime: ${runtime.description || runtime.mode || 'not reported'}. Capability flags describe API support; they do not guarantee every provider or tool is configured.`;
  } catch (error) {
    hermesStatus.textContent = 'Unavailable';
    hermesPlatform.textContent = '—';
    hermesVersion.textContent = '—';
    hermesModel.textContent = '—';
    hermesOwner.textContent = '—';
    hermesModelCreated.textContent = '—';
    hermesCurrentModel.textContent = '—';
    hermesGatewayState.textContent = '—';
    hermesRuntime.textContent = '—';
    hermesReadiness.textContent = '—';
    hermesPlatforms.textContent = '—';
    hermesActivity.textContent = '—';
    hermesQueues.textContent = '—';
    hermesDisk.textContent = '—';
    hermesRequests.textContent = '—';
    hermesTokens.textContent = '—';
    hermesHeartbeat.textContent = '—';
    renderChips(hermesToolsets, []);
    renderChips(hermesCapabilities, []);
    clearSessionSummary();
    hermesIndicator.className = 'service-indicator is-offline';
    hermesNote.textContent = error instanceof TypeError
      ? 'The browser could not read the gateway response. Enable CORS on the gateway for this page origin, including the Authorization request header and GET method.'
      : error.message || 'The Hermes API could not be reached.';
  } finally {
    refreshingHermes = false;
    hermesRefresh.disabled = false;
    hermesRefresh.textContent = 'Refresh status';
  }
}

function summarizeSessions(response) {
  const sessions = Array.isArray(response.data) ? response.data : [];
  const sources = new Map();
  const totals = sessions.reduce((summary, session) => {
    const source = session.source || 'unknown';
    sources.set(source, (sources.get(source) || 0) + 1);
    summary.active += session.ended_at == null ? 1 : 0;
    summary.messages += session.message_count || 0;
    summary.tools += session.tool_call_count || 0;
    summary.inputTokens += session.input_tokens || 0;
    summary.outputTokens += session.output_tokens || 0;
    return summary;
  }, { active: 0, messages: 0, tools: 0, inputTokens: 0, outputTokens: 0 });

  return {
    ...totals,
    count: sessions.length,
    limit: response.limit || sessions.length,
    hasMore: Boolean(response.has_more),
    sources: [...sources.entries()]
      .sort((left, right) => right[1] - left[1])
      .map(([name, count]) => `${name}: ${formatCount(count)}`),
  };
}

function renderSessionSummary(summary) {
  hermesSessionWindow.textContent = summary.hasMore
    ? `Latest ${formatCount(summary.count)} (more available)`
    : `${formatCount(summary.count)} sessions`;
  hermesSessionCount.textContent = formatCount(summary.count);
  hermesSessionActive.textContent = formatCount(summary.active);
  hermesSessionMessages.textContent = formatCount(summary.messages);
  hermesSessionTools.textContent = formatCount(summary.tools);
  hermesSessionInputTokens.textContent = formatCount(summary.inputTokens);
  hermesSessionOutputTokens.textContent = formatCount(summary.outputTokens);
  renderChips(hermesSessionSources, summary.sources);
}

function clearSessionSummary() {
  hermesSessionWindow.textContent = 'Unavailable';
  hermesSessionCount.textContent = '—';
  hermesSessionActive.textContent = '—';
  hermesSessionMessages.textContent = '—';
  hermesSessionTools.textContent = '—';
  hermesSessionInputTokens.textContent = '—';
  hermesSessionOutputTokens.textContent = '—';
  renderChips(hermesSessionSources, []);
  xrSessions = [];
  xrSelectedPeriod = undefined;
  selectedActivityMessages = [];
  closeXrViewer();
  timelineLoadId += 1;
  activitySelectionId += 1;
  timelineChart.replaceChildren();
  timelineScroll.hidden = true;
  timelineStatus.textContent = 'Enter your gateway key to load session activity.';
  timelineDetail.hidden = true;
  timelineRefresh.disabled = false;
  timelineRefresh.textContent = 'Load timeline';
}

async function loadTimeline() {
  if (timelineRefresh.disabled) return;
  timelineRefresh.disabled = true;
  timelineRefresh.textContent = 'Loading…';
  timelineStatus.textContent = 'Loading recent sessions and message timestamps…';
  timelineDetail.hidden = true;
  const requestId = ++timelineLoadId;

  try {
    const apiKey = readHermesKey();
    if (!apiKey) throw new Error('Enter and save your gateway API key first.');
    if (window.location.protocol === 'https:') {
      throw new Error('The Hermes gateway uses HTTP. HTTPS pages cannot load its timeline until the API also supports HTTPS.');
    }

    const sessionsResponse = await fetchHermesJson(`/api/sessions?limit=${TIMELINE_SESSION_LIMIT}&offset=0`, apiKey);
    const sessions = sessionsResponse.data || [];
    let nextSession = 0;
    let completedSessions = 0;
    let completedMessages = 0;
    const totalMessages = sessions.reduce((sum, session) => sum + (Number(session.message_count) || 0), 0);
    const workers = Array.from({ length: Math.min(4, sessions.length) }, async () => {
      while (nextSession < sessions.length && requestId === timelineLoadId) {
        const session = sessions[nextSession++];
        try {
          session.timelineEvents = await loadSessionEventMetadata(session, apiKey, (count) => {
            completedMessages += count;
            timelineStatus.textContent = `Loading message timestamps… ${formatCount(completedMessages)} of ${formatCount(totalMessages)} messages`;
          });
        } catch (error) {
          session.timelineError = error.message;
          session.timelineEvents = [];
        }
        completedSessions += 1;
        if (requestId === timelineLoadId) {
          timelineStatus.textContent = `Loading message timestamps… ${formatCount(completedMessages)} of ${formatCount(totalMessages)} messages · ${completedSessions} of ${sessions.length} sessions`;
        }
      }
    });
    await Promise.all(workers);
    if (requestId !== timelineLoadId) return;

    drawSessionTimeline(sessions, Boolean(sessionsResponse.has_more));
    const failed = sessions.filter((session) => session.timelineError).length;
    timelineStatus.textContent = `${sessions.length} recent sessions · ${formatCount(sessions.reduce((sum, session) => sum + session.timelineEvents.length, 0))} timestamped messages${failed ? ` · ${failed} session${failed === 1 ? '' : 's'} could not be loaded` : ''}. Select an activity bar to inspect it.`;
  } catch (error) {
    if (requestId === timelineLoadId) {
      timelineStatus.textContent = error.message || 'Could not load session activity.';
      timelineScroll.hidden = true;
    }
  } finally {
    if (requestId === timelineLoadId) {
      timelineRefresh.disabled = false;
      timelineRefresh.textContent = 'Refresh timeline';
    }
  }
}

async function fetchHermesJson(path, apiKey, { signal } = {}) {
  const response = await fetch(`${HERMES_API_ORIGIN}${path}`, {
    headers: { Accept: 'application/json', Authorization: `Bearer ${apiKey}` },
    cache: 'no-store',
    signal: signal || AbortSignal.timeout(15_000),
  });
  if (!response.ok) throw new Error(`Hermes returned HTTP ${response.status} for ${path.split('?')[0]}.`);
  return response.json();
}

async function loadSessionEventMetadata(session, apiKey, onPageLoaded) {
  const events = [];
  let offset = 0;
  const expectedMessages = Number(session.message_count) || 0;

  while (offset < expectedMessages) {
    const page = await fetchHermesJson(
      `/api/sessions/${encodeURIComponent(session.id)}/messages?limit=${MESSAGE_PAGE_SIZE}&offset=${offset}&order=oldest`,
      apiKey,
    );
    const messages = page.data || [];
    if (!messages.length) break;
    onPageLoaded(messages.length);
    for (let index = 0; index < messages.length; index += 1) {
      const message = messages[index];
      if (!Number.isFinite(message.timestamp)) continue;
      events.push({
        timestamp: message.timestamp,
        role: message.role || 'unknown',
        toolName: message.tool_name || '',
        displayKind: message.display_kind || '',
        hasToolCalls: Array.isArray(message.tool_calls) && message.tool_calls.length > 0,
        hasReasoning: Boolean(message.reasoning || message.reasoning_content),
        messageIndex: offset + index,
        messageId: message.id,
      });
    }
    offset += messages.length;
    if (messages.length < MESSAGE_PAGE_SIZE) break;
  }

  events.sort((left, right) => left.timestamp - right.timestamp || left.messageIndex - right.messageIndex);
  return events;
}

function makeActivityPeriods(events) {
  const periods = [];
  for (const event of events) {
    let period = periods.at(-1);
    if (!period || (event.timestamp - period.end) * 1000 > ACTIVITY_GAP_MS) {
      period = { start: event.timestamp, end: event.timestamp, events: [] };
      periods.push(period);
    }
    period.end = event.timestamp;
    period.events.push(event);
  }
  return periods;
}

function drawSessionTimeline(sessions, hasMoreSessions) {
  const drawableSessions = sessions.map((session) => ({
    ...session,
    periods: makeActivityPeriods(session.timelineEvents || []),
  })).filter((session) => session.periods.length > 0);
  if (!drawableSessions.length) {
    timelineChart.replaceChildren();
    timelineScroll.hidden = true;
    timelineStatus.textContent = 'No timestamped messages were found in the recent sessions.';
    return;
  }
  const timestamps = drawableSessions.flatMap((session) => (session.timelineEvents || []).map((event) => event.timestamp));
  if (!timestamps.length) {
    timelineChart.replaceChildren();
    timelineScroll.hidden = true;
    timelineStatus.textContent = 'No timestamped messages were found in these sessions.';
    return;
  }

  const chartWidth = 1120;
  const labelWidth = 250;
  const rightPadding = 24;
  const plotWidth = chartWidth - labelWidth - rightPadding;
  const topPadding = 45;
  const rowHeight = 29;
  const chartHeight = topPadding + drawableSessions.length * rowHeight + 30;
  let minTimestamp = Math.min(...timestamps);
  let maxTimestamp = Math.max(...timestamps);
  if (minTimestamp === maxTimestamp) {
    minTimestamp -= 60;
    maxTimestamp += 60;
  }
  const range = maxTimestamp - minTimestamp;
  const xFor = (timestamp) => labelWidth + ((timestamp - minTimestamp) / range) * plotWidth;
  timelineChart.setAttribute('viewBox', `0 0 ${chartWidth} ${chartHeight}`);
  timelineChart.setAttribute('width', String(chartWidth));
  timelineChart.setAttribute('height', String(chartHeight));
  timelineChart.replaceChildren();

  for (let tick = 0; tick <= 4; tick += 1) {
    const timestamp = minTimestamp + (range * tick) / 4;
    const x = xFor(timestamp);
    timelineChart.append(svgElement('line', { x1: x, y1: 26, x2: x, y2: chartHeight - 17, class: 'timeline-gridline' }));
    const label = svgElement('text', { x, y: 17, class: 'timeline-axis-label', 'text-anchor': tick === 0 ? 'start' : tick === 4 ? 'end' : 'middle' });
    label.textContent = formatDateTime(timestamp);
    timelineChart.append(label);
  }

  drawableSessions.forEach((session, index) => {
    const y = topPadding + index * rowHeight;
    if (index % 2 === 0) {
      timelineChart.append(svgElement('rect', { x: 0, y: y - 16, width: chartWidth, height: rowHeight, class: 'timeline-row-shade' }));
    }
    const label = svgElement('text', { x: 9, y: y + 1, class: 'timeline-session-label' });
    label.textContent = `${session.source || 'session'} · ${session.id.slice(-10)}`;
    label.setAttribute('aria-label', `${session.source || 'session'} session ${session.id}`);
    timelineChart.append(label);

    for (const period of session.periods) {
      const rawWidth = xFor(period.end) - xFor(period.start);
      const width = Math.max(8, rawWidth);
      const x = Math.max(labelWidth, Math.min(xFor(period.start) - (width - rawWidth) / 2, chartWidth - rightPadding - width));
      const bar = svgElement('rect', {
        x,
        y: y - 9,
        width,
        height: 15,
        rx: 5,
        class: session.ended_at == null ? 'timeline-period is-active' : 'timeline-period',
        role: 'button',
        tabindex: 0,
        'aria-label': `${session.source || 'Session'} ${session.id.slice(-10)}, activity ${formatDateTime(period.start)} to ${formatDateTime(period.end)}, ${period.events.length} messages. Activate to show activity.`,
      });
      const selectPeriod = () => showActivityAtPeriod(session, period);
      bar.addEventListener('click', selectPeriod);
      bar.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          selectPeriod();
        }
      });
      const title = svgElement('title');
      title.textContent = `${session.id} · ${formatDateTime(period.start)} · ${period.events.length} messages`;
      bar.append(title);
      timelineChart.append(bar);
    }
  });

  timelineScroll.hidden = false;
  xrSessions = drawableSessions;
  xrTimeRange = { min: minTimestamp, max: maxTimestamp };
  if (!xrViewer.hidden) buildXrScene();
  if (hasMoreSessions) {
    timelineStatus.textContent += ` Chart covers the latest ${sessions.length} sessions; older sessions are available from the API.`;
  }
}

function svgElement(tag, attributes = {}) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  return element;
}

async function showActivityAtPeriod(session, period) {
  const selectionId = ++activitySelectionId;
  timelineDetail.hidden = false;
  timelineDetailTitle.textContent = `${session.source || 'Session'} · ${session.id}`;
  const periodDescription = `${formatDateTime(period.start)} – ${formatDateTime(period.end)} · ${formatCount(period.events.length)} messages · ${formatDuration(period.end - period.start)}${period.events.length > 80 ? ' · showing first 80' : ''}`;
  timelineDetailMeta.textContent = periodDescription;
  activityDialogTitle.textContent = `${session.source || 'Session'} · ${session.id}`;
  activityDialogMeta.textContent = periodDescription;
  timelineMessages.replaceChildren();
  activityDialogEvents.replaceChildren();
  activityDialogSummary.replaceChildren();
  activityTrack.replaceChildren();
  activityTrackLegend.replaceChildren();
  setActivityDetailsExpanded(false);
  const loading = document.createElement('li');
  loading.className = 'timeline-message-loading';
  loading.textContent = 'Loading activity details…';
  timelineMessages.append(loading);
  activityDialogEvents.append(loading.cloneNode(true));
  if (xrViewer.hidden) {
    if (!activityDialog.open) activityDialog.showModal();
    timelineDetail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  try {
    const apiKey = readHermesKey();
  const detailEvents = period.events.slice(0, 80);
    const messageIndices = detailEvents.map((event) => event.messageIndex);
    const pageOffsets = [...new Set(messageIndices.map((index) => Math.floor(index / MESSAGE_DETAIL_PAGE_SIZE) * MESSAGE_DETAIL_PAGE_SIZE))];
    const pageResults = await Promise.all(pageOffsets.map((offset) => fetchHermesJson(
      `/api/sessions/${encodeURIComponent(session.id)}/messages?limit=${MESSAGE_DETAIL_PAGE_SIZE}&offset=${offset}&order=oldest`,
      apiKey,
    )));
    if (selectionId !== activitySelectionId) return;
    const selectedIds = new Set(detailEvents.map((event) => event.messageId).filter(Boolean));
    const messages = [];
    const byId = new Map();
    const indexedMessages = new Map();
    pageResults.forEach((page, pageIndex) => {
      const pageOffset = pageOffsets[pageIndex];
      (page.data || []).forEach((message, index) => {
        if (message.id) byId.set(message.id, message);
        indexedMessages.set(pageOffset + index, message);
      });
    });
    detailEvents.forEach((event) => {
      const message = event.messageId ? byId.get(event.messageId) : indexedMessages.get(event.messageIndex);
      if (message) messages.push(message);
    });
    messages.sort((left, right) => left.timestamp - right.timestamp);
    const activities = classifyActivity(messages);
    selectedActivityMessages = messages;
    xrSelectedPeriod = { session, period };
    renderActivitySummary(activities);
    renderActivityTrack(messages);
    if (!xrViewer.hidden) buildXrScene();
    if (!xrViewer.hidden) updateXrActivityLayer();
    renderActivityMessages(messages, timelineMessages);
    renderActivityMessages(messages, activityDialogEvents);
    if (!xrViewer.hidden && xrSceneHost.querySelector('#xr-event-banner') && messages[0]) {
      showXrEventBanner(messages[0], getActivityKinds(messages[0])[0]);
    }
  } catch (error) {
    if (selectionId !== activitySelectionId) return;
    timelineMessages.replaceChildren();
    const item = document.createElement('li');
    item.className = 'timeline-message-loading';
    item.textContent = error.message || 'Could not load activity details.';
    timelineMessages.append(item);
    activityDialogEvents.replaceChildren(item.cloneNode(true));
  }
}

function classifyActivity(messages) {
  const counts = { 'Text input': 0, 'Text output': 0, Reasoning: 0, 'Tool calling': 0, 'Tool result': 0, Other: 0 };
  for (const message of messages) {
    if (message.role === 'user') counts['Text input'] += 1;
    else if (message.role === 'tool') counts['Tool result'] += 1;
    else if (message.role === 'assistant') {
      if (message.reasoning || message.reasoning_content) counts.Reasoning += 1;
      if (message.tool_calls?.length) counts['Tool calling'] += message.tool_calls.length;
      if (extractText(message.content)) counts['Text output'] += 1;
      if (!message.tool_calls?.length && !extractText(message.content) && !message.reasoning && !message.reasoning_content) counts.Other += 1;
    } else counts.Other += 1;
  }
  return Object.entries(counts).filter(([, count]) => count > 0);
}

function getActivityKinds(message) {
  const kinds = [];
  if (message.role === 'user') return ['Text input'];
  if (message.role === 'tool') return ['Tool result'];
  if (message.role === 'assistant') {
    if (extractText(message.content)) kinds.push('Text output');
    if (message.reasoning || message.reasoning_content) kinds.push('Reasoning');
    if (message.tool_calls?.length) kinds.push('Tool call');
  }
  return kinds.length ? kinds : ['Other'];
}

function renderActivityTrack(messages) {
  const events = messages.map((message) => ({
    message,
    kinds: getActivityKinds(message),
  })).filter((event) => Number.isFinite(event.message.timestamp));
  activityTrack.replaceChildren();
  activityTrackLegend.replaceChildren();
  if (!events.length) return;

  const width = 1000;
  const centerY = 39;
  const minTime = Math.min(...events.map(({ message }) => message.timestamp));
  const maxTime = Math.max(...events.map(({ message }) => message.timestamp));
  const range = maxTime - minTime || 1;
  activityTrack.setAttribute('viewBox', `0 0 ${width} 78`);
  activityTrack.append(svgElement('line', { x1: 12, y1: centerY, x2: width - 12, y2: centerY, class: 'activity-track-axis' }));

  const visibleKinds = new Set();
  events.forEach(({ message, kinds }, index) => {
    const x = 12 + ((message.timestamp - minTime) / range) * (width - 24);
    kinds.forEach((kind, kindIndex) => {
      visibleKinds.add(kind);
      const lane = ((index + kindIndex) % 5) - 2;
      const circle = svgElement('circle', {
        cx: x,
        cy: centerY + lane * 7,
        r: kinds.length > 1 ? 5 : 6,
        fill: ACTIVITY_COLORS[kind],
        class: 'activity-track-point',
        tabindex: 0,
        role: 'img',
        'aria-label': `${kind} at ${formatDateTime(message.timestamp)}`,
      });
      const title = svgElement('title');
      title.textContent = `${kind} · ${formatDateTime(message.timestamp)}${message.tool_name ? ` · ${message.tool_name}` : ''}`;
      circle.append(title);
      activityTrack.append(circle);
    });
    if (index === 0 || index === events.length - 1) {
      const label = svgElement('text', {
        x,
        y: index === 0 ? 13 : 72,
        class: 'activity-track-time',
        'text-anchor': index === 0 ? 'start' : 'end',
      });
      label.textContent = formatDateTime(message.timestamp);
      activityTrack.append(label);
    }
  });

  for (const kind of visibleKinds) {
    const legendItem = document.createElement('span');
    legendItem.className = 'activity-track-legend-item';
    const swatch = document.createElement('span');
    swatch.className = 'activity-track-swatch';
    swatch.style.backgroundColor = ACTIVITY_COLORS[kind];
    const label = document.createElement('span');
    label.textContent = kind;
    legendItem.append(swatch, label);
    activityTrackLegend.append(legendItem);
  }
}

function setActivityDetailsExpanded(expanded) {
  activityTrackToggle.setAttribute('aria-expanded', String(expanded));
  activityDialogEvents.querySelectorAll('details.activity-detail').forEach((details) => {
    details.open = expanded;
  });
  activityTrackToggle.setAttribute('aria-label', expanded
    ? 'Activity timeline. Click to collapse all event details.'
    : 'Activity timeline. Click to expand all event details.');
}

function renderActivitySummary(activities) {
  activityDialogSummary.replaceChildren();
  for (const [label, count] of activities) {
    const chip = document.createElement('span');
    chip.className = 'hermes-chip activity-summary-chip';
    chip.textContent = `${label}: ${formatCount(count)}`;
    activityDialogSummary.append(chip);
  }
  if (!activities.length) renderChips(activityDialogSummary, []);
}

function extractText(content) {
  if (typeof content === 'string') return content.trim();
  if (!Array.isArray(content)) return '';
  return content.filter((part) => part?.type === 'text').map((part) => part.text || '').join('\n').trim();
}

function addExpandableDetail(item, label, content, { preformatted = false } = {}) {
  if (!content) return;
  const details = document.createElement('details');
  details.className = 'activity-detail';
  const summary = document.createElement('summary');
  summary.textContent = label;
  const body = document.createElement('div');
  body.className = 'activity-detail-body';
  if (preformatted) body.classList.add('is-preformatted');
  body.textContent = content;
  details.append(summary, body);
  item.append(details);
}

function toolCallDetails(toolCall) {
  const fn = toolCall.function || {};
  let argumentsText = fn.arguments || '';
  try {
    argumentsText = JSON.stringify(JSON.parse(argumentsText), null, 2);
  } catch {
    // Preserve raw arguments when the server returns non-JSON text.
  }
  return { name: fn.name || 'tool', argumentsText };
}

function renderActivityMessages(messages, container) {
  container.replaceChildren();
  if (!messages.length) {
    const item = document.createElement('li');
    item.className = 'timeline-message-loading';
    item.textContent = 'No message details found for this activity period.';
    container.append(item);
    return;
  }

  const displayLimit = 80;
  messages.slice(0, displayLimit).forEach((message) => {
    const item = document.createElement('li');
    item.className = `timeline-message role-${message.role || 'unknown'}`;
    const header = document.createElement('div');
    header.className = 'timeline-message-header';
    const role = document.createElement('strong');
    role.textContent = activityLabel(message);
    const time = document.createElement('time');
    time.dateTime = new Date(message.timestamp * 1000).toISOString();
    time.textContent = formatDateTime(message.timestamp);
    header.append(role, time);
    item.append(header);

    if (message.role === 'user') {
      addExpandableDetail(item, 'Text input', message.display_kind === 'hidden' ? '[Hidden message]' : extractText(message.content));
    } else if (message.role === 'assistant') {
      const text = extractText(message.content);
      addExpandableDetail(item, 'Text output', text);
      const reasoning = message.reasoning || message.reasoning_content;
      if (reasoning) addExpandableDetail(item, 'Reasoning', reasoning);
      for (const toolCall of message.tool_calls || []) {
        const tool = toolCallDetails(toolCall);
        addExpandableDetail(item, `Tool call · ${tool.name}`, tool.argumentsText, { preformatted: true });
      }
      if (!text && !reasoning && !message.tool_calls?.length) {
      const body = document.createElement('p');
      body.className = 'timeline-tool-summary';
      body.textContent = 'Assistant activity';
        item.append(body);
      }
    } else if (message.role === 'tool') {
      addExpandableDetail(item, `Tool result · ${message.tool_name || 'activity'}`, extractText(message.content));
    } else {
      addExpandableDetail(item, message.role || 'Activity', extractText(message.content));
    }
    container.append(item);
  });

  if (messages.length > displayLimit) {
    const item = document.createElement('li');
    item.className = 'timeline-message-loading';
    item.textContent = `${formatCount(messages.length - displayLimit)} additional messages omitted from this view.`;
    container.append(item);
  }
}

function activityLabel(message) {
  if (message.role === 'user') return 'Text input';
  if (message.role === 'tool') return `Tool result · ${message.tool_name || 'activity'}`;
  if (message.role === 'assistant') {
    const labels = [];
    if (extractText(message.content)) labels.push('Text output');
    if (message.reasoning || message.reasoning_content) labels.push('Reasoning');
    if (message.tool_calls?.length) labels.push('Tool calling');
    return labels.length ? labels.join(' · ') : 'Assistant activity';
  }
  return message.role || 'Activity';
}

function formatDateTime(timestamp) {
  return new Date(timestamp * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' });
}

function formatDuration(seconds) {
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return `${Math.max(0, Math.round(seconds))} sec`;
  if (minutes < 60) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const remainder = minutes % 60;
  return remainder ? `${hours} hr ${remainder} min` : `${hours} hr`;
}

function openXrViewer() {
  if (!xrSessions.length) {
    xrStatus.textContent = 'Load the 2D timeline first, then open the 3D view.';
    return;
  }
  xrViewer.hidden = false;
  document.body.classList.add('xr-viewer-open');
  buildXrScene();
}

function closeXrViewer() {
  const scene = xrSceneHost.querySelector('a-scene');
  if (scene?.renderer?.xr?.isPresenting) scene.renderer.xr.getSession()?.end();
  xrViewer.hidden = true;
  document.body.classList.remove('xr-viewer-open');
  xrSceneHost.replaceChildren();
}

function buildXrScene() {
  if (xrViewer.hidden || !xrSessions.length) return;
  const oldScene = xrSceneHost.querySelector('a-scene');
  if (oldScene?.renderer?.xr?.isPresenting) {
    xrStatus.textContent = 'Exit XR before refreshing the 3D scene.';
    return;
  }

  const sessionRange = xrTimeRange || (() => {
    const all = xrSessions.flatMap((session) => session.timelineEvents.map((event) => event.timestamp));
    return { min: Math.min(...all), max: Math.max(...all) };
  })();
  const rowSpacing = .52;
  const rowSpan = Math.max(rowSpacing, (xrSessions.length - 1) * rowSpacing);
  const sceneWidth = Math.max(14, rowSpan + 4);
  const timeDepth = 14;
  const cameraDistance = Math.max(22, sceneWidth * 1.15);
  const timeFar = -timeDepth / 2;
  const timeNear = timeDepth / 2;
  const timestampRange = Math.max(1, sessionRange.max - sessionRange.min);
  const zFor = (timestamp) => timeFar + ((timestamp - sessionRange.min) / timestampRange) * timeDepth;
  const sessionX = (index) => ((xrSessions.length - 1) / 2 - index) * rowSpacing;
  const scene = document.createElement('a-scene');
  scene.setAttribute('embedded', '');
  scene.setAttribute('width', '100%');
  scene.setAttribute('height', '100%');
  scene.setAttribute('vr-mode-ui', 'enabled: true; enterVRButton: true; enterARButton: false');
  scene.setAttribute('renderer', 'antialias: true; colorManagement: true; physicallyCorrectLights: true');
  scene.setAttribute('xr-mode-ui', 'enabled: true');
  scene.setAttribute('background', 'color: #e9efe8');
  scene.setAttribute('style', 'width: 100%; height: 100%;');
  scene.setAttribute('loading-screen', 'enabled: false');
  const asset = document.createElement('a-assets');
  scene.append(asset);

  const rig = document.createElement('a-entity');
  rig.setAttribute('position', `0 11 ${cameraDistance}`);
  const camera = document.createElement('a-entity');
  camera.setAttribute('camera', 'active: true; fov: 68');
  camera.setAttribute('id', 'xr-camera');
  camera.setAttribute('look-controls', 'pointerLockEnabled: false');
  camera.setAttribute('wasd-controls', 'acceleration: 18');
  camera.setAttribute('position', '0 0 0');
  camera.setAttribute('rotation', '-25 0 0');
  camera.setAttribute('cursor', 'rayOrigin: mouse; fuse: false');
  camera.setAttribute('raycaster', 'objects: .xr-clickable; far: 40');
  rig.append(camera);
  const leftController = document.createElement('a-entity');
  leftController.setAttribute('laser-controls', 'hand: left');
  leftController.setAttribute('raycaster', 'objects: .xr-clickable; far: 40');
  const rightController = document.createElement('a-entity');
  rightController.setAttribute('laser-controls', 'hand: right');
  rightController.setAttribute('raycaster', 'objects: .xr-clickable; far: 40');
  rig.append(leftController, rightController);
  scene.append(rig);

  scene.append(makeXrEntity('a-plane', { position: `0 -0.14 0`, rotation: '-90 0 0', width: sceneWidth, height: timeDepth + 2, color: '#fbfcfa', material: 'shader: flat' }));
  addXrText(scene, 'HERMES AGENT · SPATIAL SESSION TIMELINE', -sceneWidth / 2 + .5, 2.5, timeFar, 14, '#304132', 'left');
  addXrText(scene, 'SESSIONS RUN LEFT ↔ RIGHT · TIME FLOWS FRONT ↔ BACK · SELECT A CYLINDER', -sceneWidth / 2 + .5, 2.05, timeFar, 14, '#667365', 'left');
  addXrText(scene, 'TIME · NEWER →', sceneWidth / 2 - 1.6, .14, timeNear + .75, 2.2, '#62705f', 'right');

  const ticks = 6;
  for (let tick = 0; tick <= ticks; tick += 1) {
    const timestamp = sessionRange.min + (timestampRange * tick) / ticks;
    const z = zFor(timestamp);
    scene.append(makeXrEntity('a-box', { position: `0 0 ${z}`, width: sceneWidth - 1, height: .012, depth: .012, color: '#dce3da', material: 'shader: flat' }));
    addXrText(scene, formatDateTime(timestamp), -sceneWidth / 2 + .5, .11, z, 2.5, '#6c7669', 'left');
  }

  xrSessions.forEach((session, index) => {
    const x = sessionX(index);
    scene.append(makeXrEntity('a-box', { position: `${x} 0 0`, width: .012, height: .012, depth: timeDepth, color: '#e8ede7', material: 'shader: flat' }));
    const banner = makeXrEntity('a-entity', { position: `${x} .48 ${timeFar - .15}` });
    banner.append(makeXrEntity('a-plane', { width: rowSpacing * .94, height: .66, color: session.ended_at == null ? '#d39b3c' : '#405a45', opacity: .97, material: 'shader: flat; side: double' }));
    addXrText(banner, `${session.source || 'session'} · ${session.id.slice(-8)}`, 0, .02, .025, rowSpacing * .86, '#fffdf7', 'center');
    addXrText(banner, `${session.periods.length} periods`, 0, -.18, .03, rowSpacing * .8, '#f0e7d1', 'center');
    scene.append(banner);
    for (const period of session.periods) {
      const startZ = zFor(period.start);
      const endZ = zFor(period.end);
      const length = Math.max(.12, endZ - startZ);
      const centerZ = (startZ + endZ) / 2;
      const bar = makeXrEntity('a-cylinder', {
        position: `${x} .14 ${centerZ}`,
        rotation: '90 0 0',
        radius: '.11',
        height: length,
        segmentsRadial: '14',
        color: session.ended_at == null ? '#d39b3c' : '#699371',
        material: 'metalness: .12; roughness: .34',
        class: 'xr-clickable',
        'data-session-id': session.id,
        'data-period-start': period.start,
      });
      attachXrClick(bar, () => selectXrPeriod(session, period));
      scene.append(bar);
      addXrText(scene, `${period.events.length} msgs`, x + .16, .31, centerZ, 1.2, '#384738', 'left');
    }
  });

  if (xrSelectedPeriod && selectedActivityMessages.length) {
    const selectedIndex = xrSessions.findIndex((session) => session.id === xrSelectedPeriod.session.id);
    appendXrActivityLine(scene, zFor, sessionX(selectedIndex), timeFar, sceneWidth, timeDepth);
  } else {
    addXrText(scene, 'SELECT A RAISED CYLINDER TO REVEAL THE SESSION’S COLOR-CODED EVENT TRAIL', -sceneWidth / 2 + .5, .32, timeNear + .5, 14, '#667365', 'left');
  }
  scene.append(makeXrEntity('a-sky', { color: '#e9efe8' }));
  xrSceneHost.replaceChildren(scene);
  xrStatus.textContent = 'Spatial timeline: sessions run left-to-right, time front-to-back, periods are raised cylinders.';
}

function appendXrActivityLine(scene, zFor, selectedX, timeFar, sceneWidth, timeDepth) {
  const messages = selectedActivityMessages.slice(0, 80);
  const zAt = zFor;
  const lineY = .8;
  const layer = makeXrEntity('a-entity', { id: 'xr-activity-layer' });
  layer.append(makeXrEntity('a-cylinder', { position: `${selectedX} ${lineY} 0`, rotation: '90 0 0', radius: '.025', height: timeDepth, color: '#647361', material: 'shader: flat' }));
  const kindsByMessage = selectedActivityMessages.map((message) => ({ message, kinds: getActivityKinds(message) }));
  const kindSeen = new Set();
  kindsByMessage.slice(0, 80).forEach(({ message, kinds }, index) => {
    const z = zAt(message.timestamp);
    kinds.forEach((kind, kindIndex) => {
      kindSeen.add(kind);
      const marker = makeXrEntity('a-sphere', {
        position: `${selectedX + (kindIndex % 2 ? .08 : 0)} ${lineY} ${z}`,
        radius: '.075',
        color: ACTIVITY_COLORS[kind],
        class: 'xr-clickable',
        'data-event-index': index,
      });
      attachXrClick(marker, () => showXrEventBanner(message, kind));
      layer.append(marker);
    });
  });
  const selected = xrSelectedPeriod;
  const summaryText = classifyActivity(selectedActivityMessages)
    .map(([kind, count]) => `${kind.toUpperCase()} ${count}`)
    .join('  ·  ');
  const banner = makeXrEntity('a-entity', { position: `${selectedX} ${lineY + .8} ${timeFar - .6}` });
  banner.append(makeXrEntity('a-plane', { width: 4.8, height: .72, color: '#243429', opacity: .95, material: 'shader: flat; side: double' }));
  addXrText(banner, `${selected.session.source || 'SESSION'} · ${selected.session.id.slice(-8)} · ${formatDateTime(selected.period.start)}`, 0, .16, .035, 4.4, '#f4f5ef');
  addXrText(banner, summaryText || 'ACTIVITY DETAILS', 0, -.14, .04, 4.4, '#e8c779');
  layer.append(banner);

  let legendX = -Math.min(5, kindSeen.size * .7);
  const legendY = .25;
  for (const kind of kindSeen) {
    layer.append(makeXrEntity('a-sphere', { position: `${legendX} ${legendY} ${timeFar - .35}`, radius: '.045', color: ACTIVITY_COLORS[kind] }));
    addXrText(layer, kind, legendX + .09, legendY, timeFar - .35, 1.3, '#53604f', 'left');
    legendX += Math.max(1.4, kind.length * .085 + .7);
  }
  scene.append(layer);
}

function updateXrActivityLayer() {
  const scene = xrSceneHost.querySelector('a-scene');
  if (!scene || !xrSelectedPeriod || !selectedActivityMessages.length) return;
  scene.querySelector('#xr-activity-layer')?.remove();
  const range = xrTimeRange || (() => {
    const all = xrSessions.flatMap((session) => session.timelineEvents.map((event) => event.timestamp));
    return { min: Math.min(...all), max: Math.max(...all) };
  })();
  const selectedIndex = xrSessions.findIndex((session) => session.id === xrSelectedPeriod.session.id);
  const selectedX = ((xrSessions.length - 1) / 2 - selectedIndex) * .52;
  const timeDepth = 14;
  const timeFar = -timeDepth / 2;
  const timeRange = Math.max(1, range.max - range.min);
  const zFor = (timestamp) => timeFar + ((timestamp - range.min) / timeRange) * timeDepth;
  appendXrActivityLine(scene, zFor, selectedX, timeFar, 18, timeDepth);
}

function showXrEventBanner(message, kind) {
  xrSceneHost.querySelector('#xr-event-banner')?.remove();
  const scene = xrSceneHost.querySelector('a-scene');
  if (!scene) return;
  const camera = scene.querySelector('#xr-camera');
  const banner = makeXrEntity('a-entity', { id: 'xr-event-banner', position: '0 0 -3' });
  banner.append(makeXrEntity('a-plane', { width: 11, height: 1.15, color: '#243429', opacity: .96, material: 'shader: flat; side: double' }));
  const title = `${kind.toUpperCase()} · ${formatDateTime(message.timestamp)}${message.tool_name ? ` · ${message.tool_name}` : ''}`;
  addXrText(banner, title, 0, .31, .035, 10.2, '#e8c779', 'center');
  const toolCall = message.role === 'assistant' && kind === 'Tool call' ? message.tool_calls?.[0] : undefined;
  const content = kind === 'Text input' || kind === 'Text output'
    ? extractText(message.content)
    : kind === 'Tool result'
      ? extractText(message.content)
      : kind === 'Reasoning'
        ? (message.reasoning || message.reasoning_content || 'No reasoning text was returned.')
        : kind === 'Tool call'
          ? (toolCall ? `${toolCall.function?.name || 'tool'}(${toolCall.function?.arguments || ''})` : 'Tool call')
          : 'Assistant activity';
  addXrText(banner, truncateText(content || '[No text content]', 150), 0, -.02, .04, 10.2, '#ffffff', 'center');
  camera?.append(banner);
  xrStatus.textContent = `${kind} · ${formatDateTime(message.timestamp)}. Select another marker to inspect it.`;
}

function selectXrPeriod(session, period) {
  xrSelectedPeriod = { session, period };
  selectedActivityMessages = [];
  buildXrScene();
  showActivityAtPeriod(session, period);
  xrStatus.textContent = `Loading details for ${session.id.slice(-10)}…`;
}

function truncateText(text, limit) {
  const normalized = String(text).replace(/\s+/g, ' ').trim();
  return normalized.length > limit ? `${normalized.slice(0, limit - 1)}…` : normalized;
}

function makeXrEntity(tag, attributes) {
  const entity = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) entity.setAttribute(name, String(value));
  return entity;
}

function addXrText(parent, value, x, y, z, width, color, align = 'center') {
  const text = makeXrEntity('a-text', {
    value,
    position: `${x} ${y} ${z}`,
    width,
    color,
    align,
  });
  parent.append(text);
  return text;
}

function attachXrClick(entity, handler) {
  entity.classList.add('xr-clickable');
  entity.addEventListener('click', handler);
  entity.addEventListener('mouseenter', () => entity.setAttribute('scale', '1.12 1.12 1.12'));
  entity.addEventListener('mouseleave', () => entity.setAttribute('scale', '1 1 1'));
}

function updateHermesPolling() {
  window.clearInterval(hermesPollingTimer);
  hermesPollingTimer = undefined;
  if (readHermesKey()) {
    hermesPollingTimer = window.setInterval(() => {
      if (!document.hidden) {
        refreshHermesStatus();
        if (!timelineRefresh.disabled && !timelineScroll.hidden) loadTimeline();
      }
    }, 30_000);
  }
}

function formatCount(value) {
  return Number.isFinite(value) ? new Intl.NumberFormat().format(value) : '—';
}

function formatLatency(value) {
  return Number.isFinite(value) ? `${formatCount(value)} ms` : '—';
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

function renderChips(container, values) {
  container.replaceChildren();
  if (!values.length) {
    const empty = document.createElement('span');
    empty.className = 'hermes-empty';
    empty.textContent = 'None reported';
    container.append(empty);
    return;
  }

  for (const value of values) {
    const chip = document.createElement('span');
    chip.className = 'hermes-chip';
    chip.textContent = value;
    container.append(chip);
  }
}

hermesKeyForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const key = hermesKeyInput.value.trim();
  if (!key) {
    hermesNote.textContent = 'Paste a gateway API key first.';
    hermesKeyInput.focus();
    return;
  }

  if (!setHermesKey(key)) {
    hermesNote.textContent = 'Could not save the key in browser storage. Check the browser’s storage settings and try again.';
    return;
  }

  hermesKeyInput.value = '';
  hermesForgetButton.hidden = false;
  updateHermesPolling();
  refreshHermesStatus();
  loadTimeline();
});

hermesForgetButton.addEventListener('click', () => {
  setHermesKey('');
  hermesKeyInput.value = '';
  hermesForgetButton.hidden = true;
  timelineLoadId += 1;
  activitySelectionId += 1;
  timelineRefresh.disabled = false;
  timelineRefresh.textContent = 'Load timeline';
  hermesStatus.textContent = 'Key removed';
  hermesPlatform.textContent = '—';
  hermesVersion.textContent = '—';
  hermesModel.textContent = '—';
  hermesOwner.textContent = '—';
  hermesModelCreated.textContent = '—';
  hermesIndicator.className = 'service-indicator';
  hermesNote.textContent = 'Saved gateway key removed from this browser.';
  clearSessionSummary();
  updateHermesPolling();
});

function setBusy(value) {
  busy = value;
  recordButton.disabled = value;
  recordButton.classList.toggle('is-busy', value);
}

function modelProgress(label) {
  return ({ status: progressStatus, file, progress }) => {
    if (progressStatus === 'progress' && Number.isFinite(progress)) {
      setStatus(`${label}: downloading ${Math.round(progress)}%`);
    } else if (progressStatus === 'initiate') {
      setStatus(`${label}: downloading model files…`);
    } else if (progressStatus === 'done' && file) {
      setStatus(`${label}: loaded model files…`);
    }
  };
}

function loadTranscriber() {
  transcriberPromise ??= pipeline('automatic-speech-recognition', MODEL_ID, {
    dtype: 'q8',
    device: 'wasm',
    progress_callback: modelProgress('Whisper Base'),
  });
  return transcriberPromise;
}

function loadTTS() {
  ttsPromise ??= KokoroTTS.from_pretrained(TTS_MODEL_ID, {
    dtype: 'q8',
    device: 'wasm',
    progress_callback: modelProgress('Kokoro-82M'),
  });
  return ttsPromise;
}

function stopMicrophone() {
  microphoneStream?.getTracks().forEach((track) => track.stop());
  microphoneStream = undefined;
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    throw new Error('Audio recording is not supported in this browser.');
  }

  microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  recorder = new MediaRecorder(microphoneStream);
  recorder.addEventListener('dataavailable', (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  });
  recorder.addEventListener('stop', () => {
    stopMicrophone();
    transcribeAndSpeak().catch(showError);
  }, { once: true });
  recorder.start();

  recordButton.classList.add('is-recording');
  recordButton.setAttribute('aria-pressed', 'true');
  buttonLabel.textContent = 'Stop and transcribe';
  setStatus('Listening… select the button when you are finished.');
}

async function toMono16k(blob) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error('Audio decoding is not supported in this browser.');

  const decodeContext = new AudioContextClass();
  try {
    const decoded = await decodeContext.decodeAudioData(await blob.arrayBuffer());
    const frameCount = Math.ceil(decoded.duration * TARGET_SAMPLE_RATE);
    const offlineContext = new OfflineAudioContext(1, frameCount, TARGET_SAMPLE_RATE);
    const source = offlineContext.createBufferSource();
    source.buffer = decoded;
    source.connect(offlineContext.destination);
    source.start();
    const resampled = await offlineContext.startRendering();
    return resampled.getChannelData(0);
  } finally {
    await decodeContext.close();
  }
}

async function transcribeAndSpeak() {
  recordButton.classList.remove('is-recording');
  recordButton.setAttribute('aria-pressed', 'false');
  buttonLabel.textContent = 'Start listening';
  setBusy(true);
  audioPlayer.hidden = true;
  audioPlayer.removeAttribute('src');

  try {
    const recording = new Blob(recordedChunks, { type: recorder.mimeType || 'audio/webm' });
    if (!recording.size) throw new Error('No audio was captured. Try recording again.');

    setStatus('Preparing your recording…');
    const audio = await toMono16k(recording);
    setStatus('Loading Whisper Base… the first download may take a little while.');
    const asr = await loadTranscriber();
    setStatus('Transcribing locally…');
    // Transformers.js accepts a typed audio array as pipeline input. Its
    // pipeline handles resampling only for URL/string input; this array is
    // already mono float32 at the model's 16 kHz sample rate.
    const result = await asr(audio, {
      chunk_length_s: 30,
      stride_length_s: 5,
    });
    const text = result.text?.trim();
    if (!text) {
      transcript.value = '';
      setStatus('No speech was detected. Try again.');
      return;
    }

    transcript.value = text;
    setStatus('Transcript ready. Loading Kokoro-82M…');
    await waitForPaint();
    const tts = await loadTTS();
    setStatus('Speaking your transcript…');
    const speech = await tts.generate(text, { voice: 'af_heart' });
    const audioUrl = URL.createObjectURL(speech.toBlob());
    audioPlayer.src = audioUrl;
    audioPlayer.onended = () => URL.revokeObjectURL(audioUrl);
    audioPlayer.hidden = false;
    try {
      await audioPlayer.play();
      setStatus('Done. Your audio and transcript stay in this browser.');
    } catch {
      setStatus('Speech is ready. Press play to hear your transcript.');
    }
  } finally {
    setBusy(false);
  }
}

function showError(error) {
  console.error(error);
  setBusy(false);
  recordButton.classList.remove('is-recording');
  recordButton.setAttribute('aria-pressed', 'false');
  buttonLabel.textContent = 'Start listening';
  stopMicrophone();
  setStatus(error?.message || 'Something went wrong. Please try again.');
}

recordButton.addEventListener('click', async () => {
  if (busy) return;
  if (recorder?.state === 'recording') {
    setStatus('Finishing your recording…');
    recorder.stop();
    return;
  }

  transcript.value = '';
  try {
    await startRecording();
  } catch (error) {
    showError(error);
  }
});

hermesRefresh.addEventListener('click', refreshHermesStatus);
timelineRefresh.addEventListener('click', loadTimeline);
timelineDetailClose.addEventListener('click', () => {
  timelineDetail.hidden = true;
});
activityDialogClose.addEventListener('click', () => activityDialog.close());
activityDialog.addEventListener('click', (event) => {
  if (event.target === activityDialog) activityDialog.close();
});
activityTrackToggle.addEventListener('click', () => {
  const detailItems = [...activityDialogEvents.querySelectorAll('details.activity-detail')];
  const expanded = detailItems.some((details) => !details.open);
  setActivityDetailsExpanded(expanded);
});
xrOpenButton.addEventListener('click', openXrViewer);
xrCloseButton.addEventListener('click', closeXrViewer);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !xrViewer.hidden) closeXrViewer();
});
  activityDialogEvents.addEventListener('toggle', () => {
  const detailItems = [...activityDialogEvents.querySelectorAll('details.activity-detail')];
  const allExpanded = detailItems.length > 0 && detailItems.every((details) => details.open);
  activityTrackToggle.setAttribute('aria-expanded', String(allExpanded));
  activityTrackToggle.setAttribute('aria-label', allExpanded
    ? 'Activity timeline. Click to collapse all event details.'
    : 'Activity timeline. Click to expand all event details.');
}, true);
hermesForgetButton.hidden = !readHermesKey();
if (readHermesKey()) {
  refreshHermesStatus();
  loadTimeline();
  updateHermesPolling();
}
else hermesNote.textContent = 'Enter and save your gateway API key to load Hermes information.';

window.localVoiceReady = true;
