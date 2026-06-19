# Genesis → Shotcut parity gaps (source-grounded audit, 2026-06-19)

Auditor: 6 read-only agents diffing the Shotcut source (`/home/alex/shotcut`) against
the Genesis source (`/home/alex/Genesis`). Coverage is measured against Shotcut's
*entire* surface (a mature 10+yr NLE), so percentages are honest, not flattering.

## Coverage by area (measured)
| Area | Genesis / Shotcut | Coverage |
|------|-------------------|----------|
| Filters/effects | 54 / 154 | ~35% (video 36/117, audio 18/37) |
| Scopes | 6 / 12 | ~50% |
| Editing ops | 18 / 48 | ~38% |
| Export/encode | ~10 / 50+ | ~20% |
| Keyframes + transitions | — | ~65% |
| Core/structural subsystems | 8–9 / 15+ | ~55% |

**Reality check:** the core *editing workflow* (timeline, 3-point, ripple/roll/slip,
markers, snap, multitrack, transitions, undo, dual monitors) is solid. The gaps are
**breadth** (filter catalog), **pro finishing** (masking/keying/per-filter keyframing),
**media management** (bins), and **export depth**. My earlier "backlog is marginal" was
too optimistic — the audit shows several genuinely high-value gaps remain.

## HIGH value (the ones that actually matter for an editor)
| Gap | Area | Effort | Why it matters |
|-----|------|--------|----------------|
| Per-filter parameter keyframing | animation | L | Shotcut can keyframe ANY filter param; Genesis only keyframes 4 project params + PiP + master gain. Shotcut's signature capability. |
| Track operations UI (mute/hide/lock/rename/add/remove, composite) | editing/core | **S–M** | Model already has the state fields — just need header toggle buttons + undoable commands. **Best value/effort ratio.** |
| Export: format/container + audio codec + export region (in/out) | export | M | Genesis infers container from extension, hardcodes AAC stereo 48k, exports the whole timeline only. |
| Masking suite (shape/animated/keying masks) | filters | L | No mask shapes or animated mask paths. |
| Blend modes (multiply/screen/overlay/add…) | filters/editing | M | Genesis composites alpha-over only. |
| Media bins / management | editing/core | M | Flat read-only media pool; no bins/organization/smart filters. |
| Auto-save & crash recovery | core | M | No auto-save at all today — real data-loss risk. |
| Parametric / 15-band audio EQ | filters | M | Genesis has fixed 3-band EQ. |
| Subtitles (SRT/VTT import/edit/export + render) | core/editing | L | None today. |
| Pro keying: spill suppression / chromahold | filters | L | Basic chroma key only. |
| RNN audio denoise (librnnoise) | filters | M | High-value cleanup filter. |
| 3D LUT roundtrip + library (.cube) | filters | M | Can load LUTs; no library/export. |

## MEDIUM value
- Keyframed time-remapping (variable speed) + strobe; transition trimming + luma-wipe library (Shotcut has 23 wipes vs Genesis 8); transition-parameter keyframing.
- Audio scopes: loudness (LUFS), audio vectorscope (L/R correlation), time-domain audio waveform; video RGB waveform, pixel-inspector zoom scope.
- Clip grouping/ungroup; detach-audio; replace-clip; batch apply-filter; fade-point trim handles on the timeline.
- Audio multichannel ops (mono/swap/stereo-width/mid-side); expander + 2-pass normalize.
- Export: two-pass, B-frames, pixel format/10-bit, deinterlace, preset save/load, advanced FFmpeg options field; HW encoders (NVENC/VAAPI — env-dependent, hard to gate here).
- Distortion suite (fisheye/lens-correction/corners/reflect/…); selective-color (per-hue) adjust; rich text (bold/italic/outline)/typewriter.
- Proxy editing; nested sequences/compound clips; media relink for missing files; audio mixer UI (per-track meters/pan/solo).

## LOW value / niche (likely skip)
Ambisonics, VST host, GPS overlay, VFX artistic suite (choppy/nervous/dance/lightshow),
surround scope, HDR10 metadata, parallel multi-file export, plugin/extension store,
saved dock layouts, cover-art/ID3 metadata, motion tracking, clip auto-align,
speech-to-text (Whisper), full bigsh0t 360 suite, timecode/drop-frame display.

## Recommended next 3 (value × effort)
1. ~~**Track operations UI**~~ — DONE (P28, commit d6f7ba5): interactive headers (name/hide/mute/lock/add/remove + undo).
2. **Export depth: format + audio codec + export-region** (M) — removes real delivery limits. ← IN PROGRESS (P29)
3. **Per-filter parameter keyframing** (L) — the biggest capability gap; Shotcut's headline feature.
