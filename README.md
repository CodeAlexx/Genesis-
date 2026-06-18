# Genesis

A Rust + [egui](https://github.com/emilk/egui) NLE front-end built on the **verified C
engine** from the Mojo `MojoMedia` project (FFmpeg decode/encode/audio + an OpenCL compute
shim). Genesis rewrites only the app/UI layer; the engine stays C, called over FFI.

**Why:** the Mojo editor proved the engine (compositing, grade, scopes, keyframes — all
numerically gated), but hand-rolling a GUI toolkit in Mojo over a bare GL backend hit
ergonomic limits (no retained widgets/layout, no rounded-rects/gradients, manual hit-testing).
egui supplies a mature immediate-mode toolkit (custom painting, interaction, theming, docking)
with no C++ and no Qt. `MojoMedia` is preserved as the reference; we revisit Mojo as it matures.

## Architecture

```
Rust + egui (UI: chrome, timeline, interaction, theming)
        │  FFI (extern "C")
        ▼
C engine  (vendored in csrc/, built by build.rs)
  fpx_decode.c   FFmpeg decode            ← Phase 0
  fpx_gpu.c      OpenCL composite/scopes  ← later
  fpx_encode.c   FFmpeg encode            ← later
  fpx_audio*.c   audio                    ← later
```
Composited RGBA8 frames cross the boundary as egui textures.

## Phase 0 (this slice)
- eframe window + Shotcut dark theme
- labeled toolbar
- a real frame decoded by `fpx_decode.c` shown as a texture in the preview
- a custom-painted timeline with **draggable clips** (egui interaction)

## Build / run
```
cargo run --release -- [/path/to/media.mp4]   # default: /tmp/editor_clip.mp4
```
Deps: a C compiler + FFmpeg dev libs (`libavformat/avcodec/swscale/avutil`), and the usual
egui/winit system libs (GL, xkbcommon).

## Status
Phase 0. Engine C vendored from `MojoMedia/ffi`. Icon blob `assets/icons_dark_32.rgba`
(39 Shotcut dark icons, 32×32 RGBA8) carried over for the toolbar/track-head pass.
