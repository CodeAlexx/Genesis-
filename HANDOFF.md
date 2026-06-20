# Genesis — Engineering Handoff

> A practical guide for the next person (or future-you) to pick up Genesis and keep it correct.
> Everything here is grounded in the code at HEAD `01e5311` and the gates run while building it.
> Companion docs: `README.md` (what it is + capability list), `PARITY_GAPS.md` (Shotcut coverage
> audit + per-wave gate numbers), `docs/ROADMAP.md` (the original P1–P10 build-out).

---

## 1. What Genesis is

A non-linear video editor: a **Rust + [egui](https://github.com/emilk/egui)** front-end over a
**verified C engine** (FFmpeg decode/encode/audio + an OpenCL compute shim) called via FFI. The C
engine was proven in the Mojo `MojoMedia` project; Genesis rewrites only the app/UI layer in Rust.

Cargo workspace, two crates:
- **`ui/`** — the `genesis` binary. egui front-end. Links **NO** engine / OpenCL.
- **`gcompose/`** — the `gcompose` binary. The engine **worker**: links the C engine, **NO** GUI libs.

~24k lines total. The `genesis` binary locates `gcompose` next to itself in the target dir and
spawns it.

---

## 2. Architecture — two processes (load-bearing, do not "simplify" away)

```
genesis (ui/)                       egui front-end: chrome · timeline · panels · interaction ·
  │   spawn + line protocol         theming · textures · the PROJECT MODEL (Rust structs)
  │   (stdin/stdout, fixed-arity)   builds per-frame wire lines; uploads returned RGBA as egui textures
  ▼
gcompose (gcompose/ --serve)        engine WORKER (persistent): decoder handles + OpenCL context warm
  csrc/fpx_decode.c                   FFmpeg decode (seek + EOF flush-drain)
  csrc/fpx_encode.c                   FFmpeg encode (video + audio mux)
  csrc/fpx_audio.c / fpx_aread.c      audio decode / libavfilter graph / program mix / .cube + envelope
  csrc/fpx_gpu.c                      OpenCL: composite / grade / PiP / filters / transitions / scopes
  → writes GVW×GVH RGBA8 frames + numeric scope buffers the UI reads back
```

**Why two processes (measured, not stylistic):** initializing the NVIDIA **OpenCL** driver in the
*same* process as eframe's **GL/GLX** driver intermittently segfaults at startup (~20–40% across
every in-process ordering tried). Isolating OpenCL in its own process removes the race by
construction. The worker has a small residual init flake; the UI **retries the spawn** a few times,
which makes it reliable. **Never link the engine/OpenCL into the `ui` crate.**

Consequence you WILL hit: the UI has **no decoder**. Anything that needs decoded media or OpenCL must
go through a **wire query** to `gcompose` (see §5). Example: audio-align added the `CLIPAUD` query
precisely because the UI can't call `decode_audio_range` directly.

---

## 3. Build / run / gate commands

```bash
# build (both crates)
cd /home/alex/Genesis && cargo build            # workspace: builds gcompose
(cd ui && cargo build)                          # the genesis binary (separate build invocation)

# run the editor (needs an X display)
DISPLAY=:1 [GENESIS_OPEN=proj.json] target/debug/genesis

# tests (model unit tests live in ui/src/model.rs; 57 at HEAD)
cargo test

# INIT-GATE the engine (compiles every OpenCL kernel + runs the NULL-guards):
printf 'CLOSE\n' | ./target/debug/gcompose --serve 2>&1 | grep -i "init rc"
#   -> "[gpu] init rc=0"   (nonzero == a kernel failed to build / a guard tripped)
```

- **0-warn is the bar.** `cargo build 2>&1 | grep -iE "warning|error"` must be empty for BOTH crates.
- To kill a running editor: `pkill -x genesis` — **NOT** `pkill -f genesis` (the `-f` pattern
  self-matches the launch command line and kills the shell, exit 144).

---

## 4. The project model (`ui/src/model.rs`, ~4.5k lines — the most-edited file)

`Project` is the document; it serializes to JSON via `project_io` (serde). Every field added over the
parity waves is `#[serde(default ...)]` so **old `.json` projects always load**.

Key types:
- **`Project`** — `media: Vec<String>` (paths) + `names` (parallel display names) + `bin_names` /
  `media_bin` (P47 bins, parallel to media) + `clips: Vec<Clip>` + `tracks: Vec<Track>` +
  `transitions: Vec<Transition>` + keyframe tracks (`bright_kf/contrast_kf/sat_kf/opacity_kf/gain_kf`,
  `pip_kf`) + `markers` + `subtitles: Vec<Subtitle>` (P48) + `subseqs: Vec<SubSeq>` (P49) + `export:
  ExportSettings` + `export_in/out` (P43 region) + program grade `bright/contrast/sat`.
  **Trap:** `Project` derives `Default`, whose `contrast`/`sat` are `0.0` (f32 default) — NOT the
  `1.0` identity. Building a transient `Project { ..Project::default() }` to reuse the compose path
  crushes the frame to grey (this bit P49; `subseq_view` sets the identity grade explicitly).
- **`Clip`** — `media`(usize), `src_in`/`len`/`t0`(i64 frames), `track`(u8), `gain`(per-clip AUDIO
  gain), `speed`/`reverse` (P24; `speed==0` = freeze still), `seq:i32` (P49: -1 normal / >=0 = a
  compound clip sourcing `subseqs[seq]`), `group`(u32), `fade_in`/`fade_out`(frames), `title`,
  `chroma`, `audio_fx`, and the long tail of per-clip filter params (grade / look / lut / mask /
  distort / selective-color / solarize / temp / fade-factor inputs …). `Clip::fade_factor(t)`,
  `Clip::end()`, `Clip::video(...)` ctor.
- **`Track`** — `kind`(Video/Audio), `name`, `hidden`/`muted`/`locked`, `gain`/`pan`/`solo` (P42
  mixer). `Clip.track` indexes `Project.tracks`. Lane order = video tracks reversed (top) ++ audio
  (bottom). **Every track carries audio** (`track_is_audible` never checks kind).
- **`Transition`** — `{track, center, dur(>=2), kind(0..10)}` per same-track boundary.
- **`SubSeq`** (P49) — `{name, len, clips, tracks}`; a self-contained sub-timeline over the parent's
  shared media pool. `Project.subseq_view(idx)` returns a transient parent-less `Project` to compose
  it. One level deep (a subseq carries no subseqs → no infinite recursion).
- **`Subtitle`** (P48) — `{start, end, text}` timeline frames. `parse_srt(s, fps)` +
  `active_subtitle_at(t)`.
- **`Kf`/`PipKey`** — keyframes (36 MLT interp types incl Catmull-Rom variants; `eval_track`/`eval_pip`).

Notable model ops (all unit-tested): `split_clip`/`split_all_at`, `trim_*`/`ripple_*`, `slip`, `roll`,
`slide`, `nudge_clip`, `replace_clip`, `group_clips`/`ungroup`/`move_group`, `detach_audio`,
`copy_filters_from`/`paste_filters`, `freeze_frame`, `relink_media`, `set_fade_in/out`,
`set_transition_dur`, `set_media_bin`/`add_bin`, `cross_correlation_offset` (audio align),
`export_range`, `fade_factor`.

---

## 5. The wire protocol (`gcompose --serve`, fixed-arity, whitespace-delimited)

The UI sends one line per request on the worker's stdin; the worker replies `DONE <out>` / `ERR`.
Paths are percent-encoded (`enc_path`/`dec_path`) so spaces are wire-safe. Fixed arity means **a
media path can never be mistaken for a command** — but it also means **every new field shifts the
count, and the UI emit + the engine parse must agree exactly** or every frame becomes "bad ENC".

Commands + current arities (field counts after the keyword) at HEAD:

| Command | Arity | Purpose |
|---|---|---|
| `OPEN` | 14 | open render session + export config; allocates the program-audio accumulator |
| `PREVIEW` | 103 | compose one frame → RGBA file (out path LAST); editor preview / fold steps |
| `ENC` | 103 | compose one frame → feed the encoder (no out path; last field is `fade`) |
| `AUDIO` | 11 | mix one clip's decoded+filtered+gained range into the accumulator |
| `GAINENV` | 2 | master gain envelope (packed) |
| `MEAS`/`LEVELS`/`SPECTRUM`/`SAMPLES` | 2/2/3/3 | audio meter / FFT / time-domain scope (write f32 then clear) |
| `WAVE`/`WAVECLOSE` | 3/2 | playback accumulator session |
| `ENV` | 4 | whole-file peak envelope (waveform thumbnail) |
| `CLIPAUD` | 7 | decode a clip range → fixed-rate mono f32 (P46 audio align) |
| `THUMB` | 6 | pool thumbnail |
| `SCOPE` | 3 | run a scope kernel on the last composed buffer |
| `CLOSE` | — | finalize the encoder / end the session |

**PREVIEW vs ENC** differ by one slot: PREVIEW carries the `out` path as its LAST token; ENC has no
out path. When you append a field: ENC appends after the current last field; PREVIEW **inserts before
`out`** (so `out` stays last). Both arities end up equal. The fields are documented inline in
`gcompose/src/main.rs` (the `//!` banner + the per-parser index comments) and emitted by
`worker.rs::format_enc` / `format_preview`.

---

## 6. The render pipeline (how a frame becomes pixels)

1. **`worker.rs::resolve_frame(project, t)`** picks the base clip (lowest visible video track covering
   `t`) and the over clip (next visible video track up), reads every per-clip param, evaluates
   keyframes, resolves the transition/look, and returns a `Resolved` struct. It is a **pure path/value
   builder** — it does NOT do worker round-trips.
2. **`format_enc(&Resolved)` / `format_preview(&Resolved, out)`** turn the `Resolved` into the wire
   line (the ONLY place wire lines are built — never hand-type them).
3. The engine composes: decode base+over → `compose_trans`/`compose_trans_f32` (`gcompose/src/ffi.rs`)
   runs the OpenCL pipeline (transition → PiP composite → grade → look → the OUTB filter chain →
   fade) → downloads RGBA (PREVIEW) or f32 (ENC, for the encoder).
4. **N-layer fold** (`build_layer_pipeline` + `build_enc_raw`): for >2 video layers, extra layers are
   composited via the **`RAW:<path>` sentinel** — a pre-rendered GVW×GVH RGBA file uploaded directly
   to a slot. `run_pipeline(lines, out)` runs the fold and reads back the RGBA.

**The `RAW:` trick is the reusable lever.** Titles (P5), subtitles (P48), and nested sequences (P49)
all work the same way: rasterize/compose something to a GVW×GVH RGBA temp, then feed it as a `RAW:`
layer. When you need "overlay X on the program," that's almost always: produce an RGBA temp + one more
`RAW:` layer in the fold, **gated so the no-X path stays byte-identical**.

- **Subtitles**: `rasterize_subtitle` (reuses `rasterize_title` + the bundled TTF) → top RAW layer
  when `active_subtitle_at(t)` is Some.
- **Nested sequences**: `prerender_compound_clips` composes `subseq_view(seq)`'s frame at
  `src_frame_at(c,t)` into `subseq_temp_path(idx,inner)` via the same fold; `resolve_frame` swaps a
  `seq>=0` clip's base/over path to `RAW:<temp>` frame 0. Called in `request_frame`, `render_program`,
  AND `scope()` (the scopes panel composes via `build_request` too).

**Entry points that compose video** (all must be kept consistent for a program-level overlay):
`request_frame` (preview), `render_program` (export, per-frame loop), `scope()` (scopes panel). Audio
entry points (`program_levels/spectrum/samples`, `play_program`) do NOT call `resolve_frame`.

---

## 7. The OpenCL engine (`gcompose/csrc/fpx_gpu.c`, ~1.6k lines)

- **`KSRC`** — one big concatenated string literal holding every kernel (`"__kernel void k_x(...){\n"
  ...`). Edit it as string-literal lines (each ending `\n"`).
- **26 kernels** at HEAD, bound once via `K("k_name")` into `static cl_kernel kX;` globals.
- **Init NULL-guards** — after binding, `if(!kX) return -NN;` for each kernel (codes up to **-55**).
  A nonzero init rc means a kernel failed to compile. **INIT-GATE every new kernel.**
- **Buffer pool** — `g_buf[]` slots; `OUTB` is the working composite; `g_tmp`/`g_tmp2` scratch.
  In-place OUTB filters do `d[i] op= ...`; spatial filters copy OUTB→g_tmp via `kCopy` then sample.
- `GVW=1280`, `GVH=856`, `IDX(x,y)=((y*VW+x)*4)`, `M_PI_F` and `clamp()` available.
- **Host→device uploads MUST be blocking** (`clEnqueueWriteBuffer(..., CL_TRUE, ...)`): the host
  source is a transient Rust `Vec` the caller drops on return. A non-blocking upload reads freed
  memory → intermittent corruption (this caused the "fold black-band" race; see git `c7067ae`).

**Recipe — add an in-place OUTB pixel filter** (the most common task; P41 solarize/temp, P45 fade are
templates):
1. `model.rs`: add the `Clip` field(s) + serde default (no-op identity) + ctor init + demo literal.
   *(Integrator pre-adds these so triads stay file-disjoint.)*
2. `fpx_gpu.c`: add `k_x` to KSRC, a `static cl_kernel kX;`, `kX=K("k_x");`, `if(!kX) return -NN;`
   (next unused code), and a `void fpx_gpu_x(...)` wrapper that **skips at the no-op default**.
3. `ffi.rs`: extern decl; thread the param through **both** `compose_trans` and `compose_trans_f32`;
   call `fpx_gpu_x(...)` after the prior OUTB filters.
4. `main.rs`: bump the ENC + PREVIEW arity guards by N; parse the new field(s) (`.unwrap_or(default)`);
   pass to both compose calls; the PREVIEW `out` index shifts.
5. `worker.rs`: add to `Resolved`; set in `resolve_frame` (base + the transition out-clip override);
   set the identity in **every** other `Resolved {…}` literal (`build_layer_resolved`,
   `build_enc_raw`); append to `format_enc` (after the last field) and `format_preview` (before `out`).
6. `panels.rs`: the properties slider.
7. **Gate** (§9): build 0-warn → init rc=0 → tests → render a frame with the filter active and
   **measure the per-pixel transform** → fold regression `0.000000`.

---

## 8. Headless gate hooks (env-driven; the measurement surface)

Set on the `genesis` binary; each composes/computes then exits, so gates need no GUI interaction:

| Env var | Effect |
|---|---|
| `GENESIS_OPEN=<project.json>` | load a project at launch (used by every render/shot gate) |
| `GENESIS_RENDER=<out.mp4>` | render the program then exit |
| `GENESIS_SHOT=<ppm>` | capture egui's framebuffer (no-crash UI gate) then exit |
| `GENESIS_SOURCE=<idx>` | open the source monitor on a pool item |
| `GENESIS_SPECTRUM` / `GENESIS_SAMPLES=<out>` | dump the audio spectrum / waveform buffers |
| `GENESIS_ALIGN=<out>` | run audio-align on clips 0 & 1, write the recovered frame delta |
| `GENESIS_AUTOSAVE`, `GENESIS_LUT_DIR`, `GENESIS_RECENT`, `GENESIS_RECOVERED` | feature config/hooks |

`GENESIS_OPEN` **disables** crash-recovery loading (so gates are deterministic). To set clip fields a
gate needs, write them into the project JSON (serde) — every model field is reachable that way.

---

## 9. Development workflow — how every feature shipped (and must keep shipping)

This codebase was built by **gated multi-agent waves**. The discipline is what kept it correct; follow
it.

**The triad pattern** (one work-slice): Builder → Skeptic (read-only, adversarial) → Bug-fixer, run
via the `Workflow` tool. For a UI-coupled feature it's a **single triad** over the touched files; for
filters it's often **2 file-disjoint triads** (engine `gcompose/*` + front `ui/*`). When several
features all touch the shared `ui` crate, run the triads **SEQUENTIALLY** (pipelined) — parallel
agents calling `Edit` on the same file race and clobber each other.

**Binding rules (these are why it works):**
- **Agents DO NOT build.** Only ONE process compiles at a time. The **main loop (integrator) owns
  every gate** — re-run the build + the render + the numeric comparison yourself. Agent self-reports
  ("PASS", "confirmed") are **never** the gate (Tenet 4: measurement beats assertion).
- **Integrator pre-adds shared model fields** before launching triads, so triad file-sets stay
  disjoint and old `.json` keeps loading.
- **Gate engine-protocol changes via the REAL Rust builders** (`format_enc`/`format_preview` +
  `GENESIS_RENDER`), never hand-typed wire lines.
- **`git add` ALL changed files**, not just the ones you pinned — a triad sometimes touches a file
  outside the planned set (this caused an incomplete commit once, fixed by amend).
- Commit after each gated wave with the trailer
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**The gate recipe (every wave):**
1. `cargo build` both crates → **0 warnings**.
2. `printf 'CLOSE\n' | gcompose --serve` → **`init rc=0`** (only matters if a kernel was added).
3. `cargo test` → all pass (model unit tests are the strongest, cheapest gate; add one per model op).
4. **Behavioral measurement** — render a frame/clip exercising the feature and measure the actual
   pixels/samples with ffmpeg + numpy/PIL (NOT "an image came out" — the *numbers*).
5. **Fold regression** — render `/tmp/p8_reg3.json` (a 3-layer composite) and diff against the prior
   wave's fold render: **mean-abs-diff must be `0.000000`** (proves the no-feature path is
   byte-identical). This caught the gray-frame bug in P49 and the speed-clamp bug in P43.

**Gate fixtures** live in `/tmp` (regenerate with the small python snippets in the commit history):
`p8_reg3.json` (fold regression), `m_lr.mp4` (0–1 gradient), `p47_red.mp4`/`p49_green.mp4` (solid
colors for swap/nested gates), `align_src.mp4` (pink noise for cross-correlation), `tone2.mp4` (audio
tone), `p47_luts/*.cube` (LUT library), `swap_rb.cube` (R↔B LUT). These are NOT in the repo (they're
synthesized test inputs); recreate them if `/tmp` is cleared.

---

## 10. Recurring traps (each cost real debugging time — measured, then fixed)

1. **`..Project::default()` ≠ identity.** `contrast`/`sat` derive-default to `0.0`, not `1.0` → grey.
   Any transient Project for reusing the compose path must set the neutral grade explicitly. (P49)
2. **egui `Slider` clamps its bound value EVERY draw**, even with no interaction. A slider range whose
   min excludes a sentinel (e.g. `0.25..=4.0` excluding the freeze `speed==0`) silently rewrites the
   model value when the panel draws. (P44 freeze break → fixed slider min to `0.0`.)
3. **A model field that "exists + has a UI control" is not necessarily rendered.** `fade_in/out`
   existed for ages (audio fade + timeline wedge) but the *video* never faded — only the end-to-end
   pixel-ramp gate exposed it. (P45 added the video fade.)
4. **Cross-correlation needs non-periodic content to gate** (pink noise, not a steady tone — a tone's
   correlation peak is ambiguous). (P46)
5. **The UI has no decoder** → media-dependent features need a `gcompose` wire query (`CLIPAUD`). (P46)
6. **Audio EOF flush is video-only.** The H.264 reorder-delay drain fix (`b223433`) does NOT
   generalize to audio decoders (no reorder delay); don't pattern-match "same bug in sibling code".
7. **Non-blocking CL upload reads freed host memory** → intermittent corruption (the fold black-band).
   Always `CL_TRUE`. (`c7067ae`)
8. **A wrong test expectation is also a bug.** A few times the impl was right and the *test*/the
   prompt's expected value was wrong (P36 transition clamp, P43 slide `next.t0`). Verify both sides.

---

## 11. Current state & what's left

**Complete + gated** (see `PARITY_GAPS.md` for per-wave numbers): full timeline editing (incl
slide/razor-all/nudge/detach-audio/paste-filters/freeze/export-region + on-clip fade & transition drag
handles + audio align + nested sequences), N-layer compositing + blend modes + masks + chroma key +
video fades, 11 transitions, ~36 video + ~18 audio filters, per-track mixer, 36 keyframe interp types,
the scope set (histogram/waveform/vectorscope/parade/peak+RMS/spectrum/audio-waveform), Program +
Source monitors, subtitles render, media management (LUT library / recents / bins / relink), export
depth (codec/CRF/GOP/preset/audio-codec/bitrate/region), save/load + auto-save.

**Remaining Shotcut gaps** — each needs a NEW subsystem or hardware this box lacks (i.e. NOT a
clean-gate pure-software wave):
- **Proxy editing** — a background transcode pipeline + proxy/full-res swap.
- **Speech-to-text** auto-captions — needs a model (e.g. Whisper) on the machine.
- **HW encoders** (NVENC/VAAPI), **RNN denoise** (no `.rnnn` model), **audio sample-rate/channels
  export** (encoder audio-feed assumes fed==output) — env-blocked here.
- **Low value**: two-pass / B-frames / 10-bit export, rich-text styling.

---

## 12. Pointers

- Memory: `~/.claude/projects/-home-alex-FramePFX/memory/genesis-shotcut-parity.md` (consolidated
  per-wave detail), `genesis-rust-egui-pivot.md`, `fix-bugs-before-next-phase.md`.
- The C engine is vendored from `MojoMedia/ffi`; `MojoMedia` (`/home/alex/MojoMedia`) remains the
  reference for numeric targets.
- Each wave's workflow script is kept at repo root as `.wave-pNN.wf.js` (the exact triad prompts).
