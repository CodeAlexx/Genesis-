# Genesis Roadmap — full MojoMedia feature parity + Shotcut look

> **STATUS (2026-06): the original P1–P10 build-out below is COMPLETE.** Genesis has the persistent
> worker + frame/audio protocol (P1), dock shell (P2), full timeline editing + arbitrary V/A tracks
> (P3), media pool + drag-to-timeline (P4), compositing wired to the model (P5), audio incl. the
> per-track mixer (P6), render/export (P7), serde-JSON save/load (P8), scopes + keyframe editor (P9),
> and the Shotcut visual pass (P10). Work since then has been a **source-grounded Shotcut-parity
> push** (waves P1–P42: filter catalog, transitions, masks, blend modes, keyframe interp, audio
> effects, scopes, mixer, export depth, auto-save, …) — each build→measurement→fold-regression gated
> + committed. The current feature set is summarized in the repo `README.md`; remaining gaps and the
> cleanly-gateable stop point are tracked in `PARITY_GAPS.md`. The phase descriptions below are kept
> as the historical build-out record.

Goal: bring every MojoMedia feature into Genesis (Rust/egui UI over the isolated C engine
worker) and make it look like Shotcut. The C engine is already built and verified — most work
is UI + orchestration in Rust, which is where egui pays off.

## What's already reusable (C engine — done)
- `fpx_gpu.c` (OpenCL): composite (V2-over-V1 alpha), **PiP** rect, **grade** (bright/contrast/sat),
  **looks** (VHS, LUT3D), **8 transitions** (crossfade/wipes/slide/zoom/dissolve), **scopes**
  (RGB histogram, luma waveform, vectorscope). All parity-gated in MojoMedia.
- `fpx_decode.c` (vendored), `fpx_encode.c`, `fpx_audio.c`/`fpx_aread.c`/`fpx_aplay.c` (to vendor).
- `assets/icons_dark_32.rgba` — 39 Shotcut dark icons (32×32 RGBA8).

## Architecture rule (load-bearing)
OpenCL lives ONLY in the `gcompose` worker process; the egui UI never links/calls it (the two
GPU drivers race in one process — measured). The UI owns the project model (Rust structs); it
sends the worker a resolved frame spec and gets back RGBA.

## Phases (each: build → framebuffer/numeric gate → commit)

**P1 — Persistent worker + frame protocol (foundation).** Turn `gcompose` from spawn-per-frame
into a long-lived process: UI writes a `CompositeSpec` (base/over/trans media + source frames,
PiP rect, grade, look, transition, opacity) over a pipe; worker keeps decoder handles + the
OpenCL context warm and returns the RGBA frame via shared memory. *Gate: request 30 fps of
distinct frames, verify correctness + no per-frame process spawn.* Everything else builds on this.

**P2 — Dock layout shell (Shotcut composition).** `egui_dock` 3-column layout: left (Project /
Media pool · Filters), center (Player + Timeline), right (Properties · Scopes). Resizable docks,
dark theme, dock headers. *Gate: framebuffer shows the panel skeleton.*

**P3 — Project model + timeline editing.** Rust structs for clips (media, src_in, len, t0, track,
look, fades, PiP rect), transitions, keyframes. egui timeline: drag-move, edge-trim, split (razor),
select, undo/redo, snap, markers, ruler + timecode, **playhead scrub → worker preview**. *Gate:
edits mutate the model; preview reflects the playhead frame.*

**P4 — Media pool + import + drag-to-timeline.** File-picker import; thumbnail grid (thumbs via
worker); drag a pool item onto a lane. *Gate: import → drag → clip composites.*

**P5 — Compositing features wired to the model.** V2-over-V1 + per-clip **PiP** (+ keyframes),
**grade** + keyframes, **looks** (LUT/VHS), per-clip **fades**, **transitions** per boundary.
All resolved from the Rust model into the worker spec. *Gate: each feature measured in the
rendered frame, parity vs MojoMedia outputs.*

**P6 — Audio.** Vendor `fpx_audio`/`aread`/`aplay`; in-clip waveforms; PulseAudio playback synced
to the playhead; the mixer view (faders, VU, pan, EQ, gate/limiter/compressor). *Gate: waveform
matches, playback synced, mix RMS.*

**P7 — Render / export.** Vendor `fpx_encode`; render the program (worker composites every frame
+ encodes) → mp4 with audio. *Gate: rendered file matches the preview composite (PSNR/sample).*

**P8 — Project save/load.** Serialize the Rust model (serde JSON — cleaner than the `.mmp` text).
*Gate: round-trip exact.*

**P9 — Scopes + keyframe curve editor + node graph.** Scopes panel (histogram/waveform/vectorscope
from the worker); keyframe curve editor (egui custom widget); optional node-graph transition editor.
*Gate: scope values + keyframe interp.*

**P10 — Shotcut visual fidelity pass.** Now that egui gives rounded rects + gradients natively:
rounded clips with real gradients, the **real Shotcut PNG icons** (toolbar + track heads), Master
+ multi V/A track heads with speaker/eye/lock, **thumbnails + in-clip waveforms**, fade/trim
handles, ruler/timecode/playhead styling. *Gate: framebuffer comparisons vs the Shotcut reference.*

Theme is already Shotcut-dark from P0; visual fidelity is woven through P2/P3/P10.

## Notes
- P1 is the critical path; P5/P7 reuse the same worker compose loop.
- Real-time playback needs the persistent worker (P1) + a frame cache.
- MojoMedia (`/home/alex/MojoMedia`) stays the reference for exact numeric targets.
