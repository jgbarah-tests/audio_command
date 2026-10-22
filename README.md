# Local Voice

A minimal browser-only voice loop: record speech, transcribe it with multilingual Whisper Base, then speak the transcript with Kokoro-82M.

## Run locally

```sh
npm install
npm run dev
```

Open the URL printed by Vite (usually `http://localhost:5173`) in a modern browser and allow microphone access. Don’t open `index.html` directly or serve the source folder using a static-only server: Vite is needed to bundle and resolve the JavaScript packages. `getUserMedia` requires localhost or HTTPS.

The first run downloads the quantized model files from Hugging Face in the browser. Speech transcription and synthesis happen on-device with WebAssembly; subsequent model use can benefit from the browser cache. No speech inference API is used.

## Hermes Agent details

Enter your gateway key in the Hermes panel. The page stores it in this browser's `localStorage` until you select **Forget key**, then sends it as a Bearer token directly to the gateway. The key is not compiled into the app or sent to this app's host. Browser storage is accessible to scripts running on this page's origin, so only use this on a trusted device and site.

The page reads `/health/detailed`, `/v1/models`, `/api/model/options`, `/v1/capabilities`, `/v1/toolsets`, and the first 100 metadata records from `/api/sessions`. It displays health/readiness checks, gateway/runtime state, connected platforms, active work, daily API metrics, disk use, current model/provider, enabled toolsets, supported API capabilities, and a session activity summary (session/activity/token counts by source). It does not display session titles, previews, or message contents. The summary refreshes every 30 seconds while the page is visible; use **Refresh status** to update it immediately.

The bottom-of-page **Session timeline** loads message records for the latest 50 sessions, extracts timestamp/role/tool-name metadata, and groups messages less than 15 minutes apart into estimated activity bars. Select a bar to open an activity popup and the bottom detail panel. Both show categorized user input, assistant output, reasoning, tool calls (including arguments), and tool results as expandable per-event details. The popup also includes a color-coded event track; click it to expand all details and click again to collapse them. **Open 3D / XR timeline** presents sessions as parallel rails, with time moving front-to-back and activity periods as raised cylinders. Session IDs appear on floating banners; selecting a cylinder adds a colored message-event trail and a summary banner, and selecting an event marker shows its text in a camera-facing banner. Desktop supports mouse and WASD navigation; compatible XR devices can enter VR with the in-scene button. XR mode requires a browser/device that supports WebXR and a secure context (HTTPS or localhost). The timeline refreshes with the session summary every 30 seconds while visible after loading, or manually with **Refresh timeline**. Reasoning, arguments, and tool results can contain sensitive data.

The gateway now responds successfully to CORS preflight for `GET` with the `Authorization` header. For a static site hosted on HTTPS, the API still needs HTTPS; browsers block requests from an HTTPS page to the current HTTP API. Once HTTPS is enabled on the API, the built `dist/` folder can be hosted as a static site with no Node proxy or app server. For development, use `npm run dev` as usual.
