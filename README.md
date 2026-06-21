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
  csrc/fpx_decode.c          FFmpeg decode (seek + EOF flush-drain)
  csrc/fpx_encode.c          FFmpeg encode (video + audio mux)
  csrc/fpx_audio.c/_aread.c  audio decode / filter-graph / program mix
  csrc/fpx_gpu.c             OpenCL composite / grade / PiP / filters / transitions / scopes
  → writes a GVW×GVH RGBA8 frame the UI uploads as an egui texture, and (persistent
    `--serve` mode) mixes program audio + answers scope queries over a line protocol
```

**Why two processes (measured):** initializing the NVIDIA **OpenCL** driver in the *same
process* as eframe's **GL/GLX** driver intermittently segfaults at startup — ~20–40% across
every in-process ordering we tried. The engine-only worker is ~stable, and the egui UI — which
never calls OpenCL — is **0 crashes / 20**. Isolating OpenCL in its own process removes the
race by construction. (The worker still has a small residual init flake; the UI retries the
spawn a few times, which makes the composite reliable — 15/15 with retry.)

Composited RGBA8 frames cross the process boundary as a raw file the UI reads. A persistent
worker (`gcompose --serve`, a fixed-arity line protocol: PREVIEW/ENC/AUDIO/OPEN + scope queries)
keeps decoder handles and the OpenCL context warm for preview, render, and audio assembly.

## Capabilities (shipped + gated)

Genesis grew well past Phase 0 into a feature-rich NLE. Every item below was integrated through a
build → behavioral-measurement → fold-regression gate before commit (see `PARITY_GAPS.md` for the
source-grounded Shotcut coverage audit and per-wave gate numbers).

- **Timeline / editing** — arbitrary V/A tracks; drag-move, edge-trim, split, ripple/slip/roll/slide,
  razor-all-tracks, frame-nudge, snap, markers, clip grouping, replace-clip, detach-audio,
  paste-filters, freeze-frame, export in/out region, multi-level undo/redo; interactive track headers
  (rename / hide / mute / lock / add / remove); on-clip fade + transition-length **drag handles**;
  **audio align** (waveform cross-correlation auto-sync); **nested sequences** (compound clips). A
  **timeline toolbar** surfaces these edits as icon buttons (split / razor-all / lift / ripple / cut /
  copy / paste / append / overwrite / insert / marker / snap toggle / zoom in-out-fit), each firing the
  same code path as its keyboard shortcut. New clips (Add-as-clip / drag-drop) are **length-clamped to
  their source frame count** (engine `NFRAMES` query) so a clip never references frames past the media
  end.
- **Compositing** — N-layer video fold, V2-over-V1, per-clip PiP transform (+ keyframes), 8 blend
  modes, per-clip fades (audio + **video fade-to-black**), chroma key + spill suppression, shape mask
  (rect/ellipse + feather/invert).
- **Transitions** — 11 per-boundary (crossfade / wipes / slide / zoom / dissolve / iris / clock /
  barn-door).
- **Colour & grade** — bright/contrast/sat, lift-gamma-gain, curves, HSL, levels, white balance,
  selective colour (per-hue band), gradient map, solarize, colour temperature.
- **Video filters (~36)** — stylize (vignette/sharpen/flip/invert/sepia/grayscale/posterize/mosaic/
  halftone/emboss/edge), FX (denoise/glow/RGB-shift), old-film (grain/scratches/diffusion), distort
  (wave/swirl/threshold/lens/crop/glitch/mirror/kaleidoscope/dither), 360 reframe, per-clip
  speed/time-remap + reverse.
- **Audio** — ~18 effects (3-band + 10-band graphic EQ, pan, compress/gate/normalize, reverb/delay/
  pitch, lowpass/highpass/tremolo, bass/treble/notch/chorus, flanger/phaser/limiter); a per-track
  **mixer** (fader / pan / mute / solo, every track); master gain automation (volume envelope).
- **Keyframes** — 36 MLT interpolation types (discrete/linear/smooth + Catmull-Rom variants + easings);
  per-clip filter-parameter keyframing.
- **Scopes** — RGB histogram, luma waveform, vectorscope, RGB parade, audio peak+RMS meter, audio
  spectrum (FFT), audio waveform oscilloscope — in a **resizable tabbed right dock** (Properties /
  Scopes / Audio) so each group gets the full panel instead of one crammed column.
- **Monitors** — Program + Source (3-point) preview panes, each with a **transport bar** (skip-to-start
  / rewind / play-pause / fast-forward / skip-to-end + SMPTE `HH:MM:SS:FF` timecode) and a **scrub
  slider** beneath the image.
- **Subtitles** — timeline-wide timed captions (import SRT / edit) rendered over the program.
- **Media management** — media-pool **thumbnails** (decoded first frame per item), 3D-LUT library
  (browse a `.cube` folder), recent-files menu, media bins (organize the pool), media relink (fix
  missing files).
- **Export** — codec / CRF / GOP / preset, audio codec (aac/mp3/ac3/pcm) + bitrate, in/out region.
- **Project** — serde-JSON save/load (round-trip exact); periodic auto-save + crash recovery.
- **UI chrome** — File/Edit/View/Help menu bar (alternate entry to the same actions), top toolbar,
  timeline toolbar, tabbed resizable right dock (Properties / Filters / Scopes / Audio), maximized 4K
  window.
- **Filters dock** — Shotcut-style per-clip filter **stack** over the existing effect fields: an
  "Applied" list (each removable with ✕ = reset) + a searchable "Add filter" catalog (BCS, color
  wheels, white balance, transform, blur, curves, HSL, levels, vignette, sharpen, flip, simple FX,
  mosaic, gradient map, denoise, glow, RGB-shift, halftone, chroma key, look). Parameters are tuned in
  the Properties tab; one undo step per add/remove. UI-only — a clip at defaults renders unchanged.

## Build / run
```
cargo run -p genesis --release -- [/path/to/media.mp4]   # default: /tmp/editor_clip.mp4
```
Deps: a C compiler + FFmpeg dev libs (`libavformat/avcodec/swscale/avutil`), OpenCL
(`libOpenCL` + `CL/cl.h`, here from CUDA), and the egui/winit system libs (GL, xkbcommon).
The `genesis` binary locates `gcompose` next to itself in the target dir.

### Headless gate hooks (env-driven, for measurement)
`GENESIS_OPEN=<project.json>` opens a project at launch; `GENESIS_RENDER=<out.mp4>` renders it then
exits; `GENESIS_SHOT=<ppm>` captures egui's framebuffer; `GENESIS_SOURCE=<idx>` opens the source
monitor; `GENESIS_SPECTRUM`/`GENESIS_SAMPLES=<out>` dump audio-scope buffers; `GENESIS_ALIGN=<out>`
runs the audio-align cross-correlation. The engine worker (`gcompose --serve`) speaks a line protocol
(PREVIEW/ENC/AUDIO/OPEN + CLIPAUD/scope queries).

## Status
Active. Engine C vendored from `MojoMedia/ffi`; icon blob `assets/icons_dark_32.rgba` (39 Shotcut
dark icons). The window opens **maximized**. The two-process OpenCL-isolation design (above) is stable
with worker-spawn retry — a live worker's `ERR` reply (a legitimately-failed command, e.g. a thumbnail
for a source frame past a clip's media end) is now a **soft miss**, so it no longer triggers the
worker+OpenCL restart cycle that machinery reserves for a genuinely dead worker. The video decoder
(`fpx_decode_frame_letterbox`) tracks its position and **decodes forward without seeking** for
sequential/in-GOP access (playback + render), via a frame-accurate drain-first loop — a ~4× per-frame
speedup over re-seeking to a keyframe every frame, byte-identical to the seek path (random access /
backward scrubbing still seeks).
**Editing, compositing, audio, scopes, export, project/media management, subtitles, and nested
sequences are complete** (each shipped through a build → behavioral-measurement → fold-regression
gate; see `PARITY_GAPS.md`). The remaining Shotcut gaps need a new subsystem or hardware this box
lacks — proxy editing + speech-to-text (transcode pipeline / a model), HW encoders / RNN denoise
(env-blocked), and low-value export knobs (two-pass / B-frames / 10-bit / rich text).
`docs/ROADMAP.md` records the original P1–P10 build-out (all complete).
