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
| ~~Per-filter parameter keyframing~~ DONE (P30 e679197) | animation | L | bright/contrast/sat/blur/rot/scale keyframable per clip; arch extends to more. |
| Track operations UI (mute/hide/lock/rename/add/remove, composite) | editing/core | **S–M** | Model already has the state fields — just need header toggle buttons + undoable commands. **Best value/effort ratio.** |
| Export: format/container + audio codec + export region (in/out) | export | M | Genesis infers container from extension, hardcodes AAC stereo 48k, exports the whole timeline only. |
| Masking suite (shape/animated/keying masks) | filters | L | No mask shapes or animated mask paths. |
| Blend modes (multiply/screen/overlay/add…) ← IN PROGRESS P31 | filters/editing | M | Genesis composites alpha-over only. |
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
1. ~~**Track operations UI**~~ — DONE (P28, d6f7ba5): interactive headers (name/hide/mute/lock/add/remove + undo).
2. ~~**Export depth: audio codec**~~ — DONE (P29, 11307c3): aac/mp3/ac3/pcm selector. (still open: audio sample-rate/channels [needs encoder audio-feed rework], container/format selector, export-region in/out, two-pass.)
3. **Per-filter parameter keyframing** (L) — the biggest capability gap; Shotcut's headline feature. ← IN PROGRESS (P30)


## SHIPPED (P28–P41, all gated + pushed, head 9420a98)
- P28 track-ops UI · P29 export audio-codec · P30 per-clip filter keyframing · P31 blend modes
- P32 graphic 10-band EQ · P33 auto-save+recovery · P34 shape mask · P35 clip editing (replace+group)
- P36 luma-wipe transitions (iris/clock/barn) · P37 chroma spill suppression · P38 distort (mirror/kaleido/dither)
- P39 selective color (per-hue-band) · P40 audio waveform scope (time-domain oscilloscope)
- P41 solarize + colour temperature (in-place OUTB pixel filters)
  NOTE: P41 `temp` (additive OUTB warm/cool) OVERLAPS the existing white-balance `wb_temp`
  (multiplicative grade-fold, "Temp (cool↔warm)" slider). Different math, but user-facing-redundant
  — two temperature controls. Works + gated; flagged for possible removal (taste/UX call).

## P46 audio align — DONE + gated + pushed ca6c164 (user-requested, 2026-06-19)
Shotcut-style align-to-reference: select 2 clips, "Align audio" cross-correlates their audio and
shifts the 2nd to sync. model::cross_correlation_offset (pure, unit-tested) + new CLIPAUD wire query
(decode a clip's source range to fixed-rate mono — needed because the UI links no decoder, only
gcompose does) + worker::align_audio_offset_frames (4000Hz, bounded window/lag) + GENESIS_ALIGN gate
hook. Gate: end-to-end recovers a known src_in offset EXACTLY across 0/30/45/60 frames; fold 0.0.
THIS CLOSES THE EDITING-OP LONG TAIL — no measurable editing gap remains; the rest of PARITY_GAPS is
UI-only / big-architecture / env-blocked.

## P44+P45 fade/transition drag handles + video fade render — DONE + gated + pushed (user-requested, 2026-06-19)
P44 (f86d43a): on-clip drag handles — fade-in/out corner handles set Clip.fade_in/out; transition
bowtie gains drag-to-resize Transition.dur (click add/cycle/remove preserved). Gateable model layer
set_fade_in/out (clamp [0,len]) + set_transition_dur (floor 2), unit-tested. P45 (e692284): the
fades previously only ramped AUDIO + drew the wedge — the exported VIDEO never darkened (base had no
fade factor). Added Clip::fade_factor(t) + a k_fade OUTB kernel (rgb*=f, guard -55) on the wire
(ENC/PREVIEW 102→103); now fade_in=15 renders a measured 5.17→84.5 ramp. EDITING-OP GAPS now fully
closed (slide/razor/nudge/detach/paste/region/freeze + working fade & transition handles). Remaining
editing long-tail: waveform audio-align (cross-correlation) only.

## P43 editing batch — DONE + gated + pushed 596b352 (user-requested, 2026-06-19)
Closed the cleanly-gateable editing-op gaps (4 sequential triads, 12 agents): slide edit, razor-all-
tracks, frame-nudge, detach-audio, paste-filters, export in/out region, freeze-frame. Gate: 45 model
unit tests (the 7 new ops) + render measurements (export-region 60→30 frames; freeze held). Two
integrator-found bugs the gates caught (Tenet 4): a wrong slide test-expected value (13→23), and the
P24 speed Slider (0.25 min) silently clamping a freeze clip's speed 0→0.25 every draw → freeze broken
on export; fixed slider min to 0.0. EDITING OPS now ~closed; remaining editing long-tail = on-clip
fade/transition trim HANDLES (values exist in Properties) + waveform audio-align (needs cross-corr).

## P42 audio mixer — DONE + gated + pushed 6668fbd (user-requested, 2026-06-19)
Per-track FADER + PAN + SOLO, folded entirely in the worker audio-emit path (ZERO wire/engine change).
Track.{gain,pan,solo} + Project.{track_gain,track_pan,is_solo,any_solo}; applied to all 5 audio-emit
loops (render/levels/spectrum/samples/playback); panels mixer_ui per-audio-track strip. Gate (rendered
PCM): fader 0.5→RMS 0.499×; pan hard-right L 0.00000/R 0.06218; solo: off audible, solo-empty-track
0.00000 (silenced), solo-tone-track unchanged; video fold 0.000000 vs P41.
P42b (97084f1): mixer_ui now lists EVERY track (video + audio), resolving the earlier audio-kind-only
limit — a clip's audio on a video track is now fader/pan/solo-controllable from the mixer. Gate: ui
0-warn; mixer renders on a V1/V2/A1 project headless (no panic); a VIDEO track's mixer gain 0.5 →
rendered RMS 0.499x (the worker fold reaches any c.track).

## STOP POINT — cleanly-gateable set EXHAUSTED (2026-06-19)
Per "only the cleanly-gateable (audio scopes + a couple filters), then stop": delivered P40 (audio
waveform scope) + P41 (solarize + temperature). Remaining candidates are NOT cleanly-gateable:
- Audio loudness/LUFS  → REDUNDANT (LEVELS already returns per-channel peak + RMS dBFS).
- Audio vectorscope/correlation → weak gate (mono sources only give the +1/in-phase case; can't
  exercise anti-phase crisply here).
- New crisp filters → the simple deterministic ones are already shipped; remaining ones overlap
  existing controls or gate only fuzzily.
Everything below is UI-only / weak-gate / big-architecture / env-blocked — out of the agreed scope.

## SKIPPED — env-blocked / not verifiable here (Tenet 4, won't fake)
- RNN denoise (arnndn present, NO .rnnn model on system) · HW encoders NVENC/VAAPI (no GPU encoder)
- audio sample-rate/channels export (encoder audio-feed assumes fed==output; needs rework)

## REMAINING — genuine long-tail (low ROI and/or weak-to-gate and/or big-architecture)
- WEAK GATE (UI organization, no measurable behaviour): media bins, audio-mixer UI, 3D-LUT library, media relink, recent-files, notes
- BIG (days, pipeline/fold surgery): subtitles SRT/VTT render, proxy editing, nested sequences/compound clips, speech-to-text
- LOW VALUE: two-pass/B-frames/10-bit export, more exotic filters (fisheye/reflect/vertigo/...), rich text (bold/outline), detach-audio (model treats A/V as separate already), transition-trim + fade-point-trim drag handles
- SCOPE COMPLETENESS: audio loudness(LUFS)/vectorscope/time-domain-waveform scopes
