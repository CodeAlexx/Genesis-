# Genesis

A Rust + [egui](https://github.com/emilk/egui) NLE front-end built on the **verified C
engine** from the Mojo `MojoMedia` project (FFmpeg decode/encode/audio + an OpenCL compute
shim). Genesis rewrites only the app/UI layer; the engine stays C, called over FFI.

**Why:** the Mojo editor proved the engine (compositing, grade, scopes, keyframes — all
numerically gated), but hand-rolling a GUI toolkit in Mojo over a bare GL backend hit
ergonomic limits (no retained widgets/layout, no rounded-rects/gradients, manual hit-testing).
egui supplies a mature immediate-mode toolkit (custom painting, interaction, theming, docking)
with no C++ and no Qt. `MojoMedia` is preserved as the reference; we revisit Mojo as it matures.

## Architecture — two processes (this is load-bearing)

```
genesis (ui/)              egui front-end. Links NO engine/OpenCL.
  │  spawn + read RGBA      chrome · timeline · interaction · theming · textures
  ▼
gcompose (gcompose/)       engine WORKER. Links the C engine, NO GUI libs.
  csrc/fpx_decode.c          FFmpeg decode
  csrc/fpx_gpu.c             OpenCL composite / grade / PiP / scopes
  → writes a GVW×GVH RGBA8 frame the UI uploads as an egui texture
```

**Why two processes (measured):** initializing the NVIDIA **OpenCL** driver in the *same
process* as eframe's **GL/GLX** driver intermittently segfaults at startup — ~20–40% across
every in-process ordering we tried. The engine-only worker is ~stable, and the egui UI — which
never calls OpenCL — is **0 crashes / 20**. Isolating OpenCL in its own process removes the
race by construction. (The worker still has a small residual init flake; the UI retries the
spawn a few times, which makes the composite reliable — 15/15 with retry.)

Composited RGBA8 frames cross the process boundary as a raw file the UI reads (Phase 0). A
persistent worker + shared-memory/pipe protocol is the planned upgrade for real-time playback.

## Phase 0 (this slice) — verified
- eframe window + Shotcut dark theme, labeled toolbar
- a **real OpenCL composite** (decode base + over → PiP inset + grade) from `gcompose`, shown
  as a texture in the preview. Measured: inset magenta `[246,0,246]`, base navy `[6,6,180]`.
- a custom-painted timeline with **draggable clips** (egui interaction)
- deterministic screenshot gate: `GENESIS_SHOT=<ppm>` captures egui's framebuffer then exits

## Build / run
```
cargo run -p genesis --release -- [/path/to/media.mp4]   # default: /tmp/editor_clip.mp4
```
Deps: a C compiler + FFmpeg dev libs (`libavformat/avcodec/swscale/avutil`), OpenCL
(`libOpenCL` + `CL/cl.h`, here from CUDA), and the egui/winit system libs (GL, xkbcommon).
The `genesis` binary locates `gcompose` next to itself in the target dir.

## Status
Phase 0 complete + measured. Engine C vendored from `MojoMedia/ffi`. Icon blob
`assets/icons_dark_32.rgba` (39 Shotcut dark icons, 32×32 RGBA8) carried over for the
toolbar/track-head pass. Open: root-cause the worker's residual OpenCL-init flake; persistent
worker + shm for playback; port timeline interactions, project save/load, scopes.
