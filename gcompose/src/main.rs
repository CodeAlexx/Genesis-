//! gcompose — the Genesis engine worker (a separate process from the egui UI).
//!
//! Two modes:
//!
//!  1. One-shot (back-compat, unchanged):
//!       `gcompose <base> <over|-> <out.rgba>`
//!     decode `base` frame 60 + `over` frame 0 (letterboxed to GVW×GVH), run the OpenCL
//!     PiP composite + grade, write the GVW×GVH RGBA8 result to `out.rgba`.
//!     Falls back to a plain decoded `base` frame if OpenCL is unavailable.
//!
//!  2. Persistent serve mode (P1):
//!       `gcompose --serve`
//!     call fpx_gpu_init() ONCE, then read one request per line from stdin, compose the
//!     requested program frame, write raw RGBA to the per-request out path, and print
//!     "DONE <out_path>\n" (or "ERR\n") to stdout, flushing after each. Decoders are cached
//!     per media path (HashMap) so a held playhead / repeated frame reuses the open handle.
//!
//! Serve commands (one per line; reply "DONE..."/"ERR\n", always flushed):
//!
//!   Preview frame (PREVIEW keyword + 21 positional fields; a keyword-less line is still
//!   accepted for back-compat with one-shot/older clients):
//!     PREVIEW <base> <over|-> <bf> <of> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat>
//!             <look_kind> <look_amt> <lut_path|->
//!             <trans_kind> <trans_prog> <trans_param> <trans_path|-> <trans_frame> <out>
//!     -> compose program frame (incl. the per-clip LOOK + any per-boundary TRANSITION), write RGBA
//!     to <out>; reply "DONE <out>". A "-" base path renders a black frame (timeline gap). look_kind:
//!     0=none, 1=VHS, 2=LUT3D (loads <lut_path> .cube, cached); a missing/failed LUT degrades to no
//!     look. trans_kind: -1 = no transition (no slot-2 upload, track1(-1,0,4) copies base); 0..7 = a
//!     transition kernel (0=crossfade..7=dissolve): decode <trans_path>@<trans_frame> (cached) into
//!     slot 2 and run fpx_gpu_track1(trans_kind, trans_prog, trans_param) at the START of the
//!     pipeline (before pip/grade/look). The PREVIEW also records which buffer (OUTB/look-none vs
//!     LOOKB/look) the frame ended in, so a following SCOPE reads the POST-LOOK frame.
//!
//!   Render/export (Slice A — video + TIMELINE-SYNCED program audio):
//!     OPEN <out> <w> <h> <fps> <total_s>
//!        -> open + config_video(mpeg4,w,h@fps) + config_audio(aac,2ch,48000) + start; reply
//!           DONE/ERR. ALSO allocates the PROGRAM-AUDIO ACCUMULATOR: an f32 stereo @ 48000 buffer
//!           sized to <total_s> seconds (the timeline duration), zero-filled (silence). The
//!           encoder is ready for BOTH streams: ENC feeds video, AUDIO MIXES into the accumulator
//!           at a destination offset, CLOSE feeds the WHOLE accumulator to the encoder then
//!           finalizes — so the rendered audio is timeline-positioned and its length matches the
//!           video. Gaps stay silent; overlaps mix (sample-add, clamped to [-1,1]).
//!     ENC <base> <over|-> <bf> <of> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat>
//!         <look_kind> <look_amt> <lut_path|->
//!         <trans_kind> <trans_prog> <trans_param> <trans_path|-> <trans_frame>
//!        -> decode(cached) + compose(track1(trans_kind,prog,param)->pip->grade->look(kind,amt,lut_n))
//!           + feed the composited f32 frame to the encoder at ts = enc_count/fps; reply DONE/ERR; no
//!           file. look_kind: 0=none, 1=VHS, 2=LUT3D (loads <lut_path> .cube, cached per path); a
//!           missing/failed LUT degrades to no look (the frame still encodes). trans_kind: -1 = no
//!           transition (track1(-1,0,4) copies base); 0..7 = a transition kernel — decode
//!           <trans_path>@<trans_frame> (cached) into slot 2 and blend base→trans by <trans_prog> at
//!           the START of the pipeline (matching the PREVIEW path). A "-"/failed trans_path degrades
//!           to no transition (the frame still encodes the base).
//!     AUDIO <path> <src_in_s> <dur_s> <dst_offset_s> <gain>
//!        -> decode that SOURCE audio range [src_in_s, src_in_s+dur_s) (fpx_decode_audio_range ->
//!           2ch @ 48000 interleaved f32), apply <gain>, and MIX (sample-add, clamp) it into the
//!           active accumulator (render OR playback) starting at <dst_offset_s> seconds. Replies
//!           DONE/ERR; a range with no audio (or a decode failure) replies ERR so the client can
//!           skip that clip without aborting. NOTHING is fed to the encoder here (deferred to
//!           CLOSE), so AUDIO is also valid in a playback-WAV session that has no encoder.
//!     CLOSE
//!        -> feed the ENTIRE accumulator to the encoder (fpx_enc_audio_samples_f32 in chunks),
//!           then finish + close (flushes + writes BOTH video and audio); reply DONE.
//!     WAVE <out_wav> <total_s>
//!        -> begin a PLAYBACK accumulator session (no encoder): allocate an f32 stereo @ 48000
//!           accumulator sized to <total_s>; subsequent AUDIO lines mix into it; reply DONE/ERR.
//!     WAVECLOSE <out_wav>
//!        -> write the playback accumulator to <out_wav> as a 16-bit PCM stereo @ 48000 WAV and
//!           clear it; reply DONE/ERR. The UI then spawns a system player (paplay/aplay) on it.
//!     THUMB <path> <frame> <w> <h> <out>
//!        -> decode <frame> letterboxed to w×h -> write RGBA8 to <out>; reply DONE/ERR.
//!     ENV <path> <buckets> <out>
//!        -> fpx_audio_envelope -> write <buckets> little-endian f32 to <out>; reply DONE/ERR.
//!     SCOPE <kind> <out>
//!        -> run the kind-selected scope kernel on the LAST composed GPU buffer (the frame left in
//!           g_buf[OUTB] by the most recent PREVIEW — NOT re-composed here, so the client sends a
//!           PREVIEW for the wanted frame first), produce a 256×256 RGBA8 image, write it to <out>;
//!           reply DONE/ERR. kind 0 = histogram (the 768 R/G/B bins are RASTERIZED into a 256×256
//!           bar graph on a dark bg, since the histogram kernel returns raw bins not an image),
//!           kind 1 = luma waveform (kernel renders the image directly), kind 2 = vectorscope
//!           (kernel renders the image directly). The scope reads the buffer the most recent
//!           PREVIEW left the frame in — g_buf[OUTB] when that PREVIEW had look=none, g_buf[LOOKB]
//!           when a VHS/LUT look ran — so the scope reflects the POST-LOOK displayed frame.
//!
//! This binary links the C engine (FFmpeg + OpenCL) but NO GUI libraries, so it owns the
//! OpenCL driver init in isolation — the egui process never touches OpenCL (see workspace
//! Cargo.toml for why).

mod ffi;

use std::collections::HashMap;
use std::io::{BufRead, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Persistent serve mode: `gcompose --serve`.
    if args.len() >= 2 && args[1] == "--serve" {
        serve();
        return;
    }

    // One-shot mode (back-compat): `gcompose <base> <over|-> <out.rgba>`.
    if args.len() < 4 {
        eprintln!("usage: gcompose <base> <over|-> <out.rgba>   |   gcompose --serve");
        std::process::exit(2);
    }
    let (base, over, out) = (&args[1], &args[2], &args[3]);

    let buf = compose(base, over).or_else(|| decode_only(base));
    match buf {
        Some(b) => {
            std::fs::write(out, &b).expect("write rgba");
            println!("OK {} {} bytes={}", ffi::GVW, ffi::GVH, b.len());
        }
        None => {
            eprintln!("FAIL: could not decode {base}");
            std::process::exit(3);
        }
    }
}

/// Persistent serve loop: init OpenCL once, then service one request per stdin line.
///
/// The authoritative wire protocol (all commands, with the Wave-8 transition fields) is documented
/// in the module header above; see the `PREVIEW`/`ENC` entries there for the full field list.
fn serve() {
    // Initialize OpenCL exactly once for the lifetime of the process. A SOFT init failure (rc != 0:
    // transient driver/device-busy) is retried a couple of times in-process (init_retry); the HARD
    // flake is a driver segfault that kills the process outright, which only the client's respawn can
    // fix (see worker.rs render retry). If init still fails after the retries the worker is useless;
    // exit non-zero so the client's restart logic can react.
    const GPU_INIT_ATTEMPTS: usize = 3;
    let gpu = match ffi::Gpu::init_retry(GPU_INIT_ATTEMPTS) {
        Some(g) => g,
        None => {
            eprintln!("FAIL: fpx_gpu_init failed in --serve (after {GPU_INIT_ATTEMPTS} attempts)");
            std::process::exit(4);
        }
    };

    // Hardening (Slice A): verify OpenCL is actually USABLE — not merely init'd — before announcing
    // readiness. A tiny end-to-end self-check (upload a black frame + a no-op compose + confirm the
    // download round-trips a full-size buffer) catches a compositor that init'd but can't run the
    // kernels. On failure, exit non-zero IMMEDIATELY so the client respawns a fresh worker rather
    // than us serving a broken one (whose first real PREVIEW/ENC would fail). This converts a SOFT
    // broken-init into a fast, clean respawn; it cannot catch the hard mid-run segfault (that is the
    // client's render-retry job).
    if !gpu.self_check() {
        eprintln!("FAIL: OpenCL self-check failed in --serve (init ok but compose round-trip broken)");
        std::process::exit(5);
    }

    // One open decoder per media path, reused across requests (held playhead / repeated frames).
    let mut decoders: HashMap<String, ffi::Decoder> = HashMap::new();

    // LUT cache (Slice A — per-clip LOOK / LUT3D): one parsed `.cube` per path, reused across
    // frames so a held playhead / a long render with the same look does NOT reparse the file every
    // frame. The value is `Option<Lut>` so a file that FAILED to parse is cached as a negative result
    // (None) too — a broken LUT degrades to no look without re-attempting (and re-logging) the parse
    // every frame. `last_uploaded_lut` tracks which path is currently resident on the GPU so we skip
    // re-uploading the same LUT on consecutive same-look frames (mirrors MojoMedia's lut_loaded_idx).
    let mut lut_cache: HashMap<String, Option<ffi::Lut>> = HashMap::new();
    let mut last_uploaded_lut: Option<String> = None;

    // Which device buffer the MOST RECENT PREVIEW left the composed frame in: false = OUTB (look
    // none), true = LOOKB (a VHS/LUT look ran). SCOPE reads this so its scope kernels run on the
    // POST-LOOK composed buffer — the exact frame the UI is showing — rather than always OUTB
    // (which, after a look, holds the pre-look grade result). Updated by every PREVIEW; SCOPE does
    // not re-compose, so this faithfully tracks the displayed frame's final buffer.
    let mut last_final_is_look: bool = false;

    // Active render encoder (set by OPEN, fed by ENC, torn down by CLOSE). Holds the fps so
    // ENC can stamp ts = enc_count / fps; enc_count is the running frame counter for the job.
    let mut enc: Option<ffi::Encoder> = None;
    let mut enc_fps: f64 = 30.0;
    let mut enc_count: i64 = 0;
    // Whether the current encoder has a usable audio stream (finding #6). OPEN sets this true only
    // if config_audio succeeded; on a minimal FFmpeg build with no aac encoder it stays false and
    // OPEN still succeeds video-only, with AUDIO commands replying ERR instead of failing OPEN.
    let mut enc_audio_ok: bool = false;

    // PROGRAM-AUDIO ACCUMULATOR (Slice A): interleaved stereo f32 @ 48000, the full timeline
    // duration. OPEN (render) and WAVE (playback) allocate+zero it; each AUDIO line MIXES one
    // decoded clip range into it at the clip's destination offset (sample-add, clamped); CLOSE
    // feeds the whole buffer to the encoder, WAVECLOSE writes it as a PCM WAV. `prog_active`
    // gates AUDIO so a stray AUDIO with no open session replies ERR rather than silently dropping.
    let mut prog: ProgAudio = ProgAudio::default();

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    eprintln!("[gcompose] serve ready");

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // stdin closed / broken: shut down.
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Dispatch on the leading keyword; a line with no keyword is the legacy preview frame.
        let kw = line.split_whitespace().next().unwrap_or("");
        let reply: Reply = match kw {
            "OPEN" => {
                if open_render(
                    line,
                    &mut enc,
                    &mut enc_fps,
                    &mut enc_count,
                    &mut enc_audio_ok,
                    &mut prog,
                ) {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            "ENC" => {
                if enc_frame(
                    &gpu,
                    &mut decoders,
                    &mut lut_cache,
                    &mut last_uploaded_lut,
                    enc.as_mut(),
                    enc_fps,
                    &mut enc_count,
                    line,
                ) {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            // AUDIO no longer feeds the encoder directly: it MIXES one decoded clip range into the
            // active program-audio accumulator (render or playback) at a destination offset.
            "AUDIO" => {
                if audio_mix(&mut prog, line) {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            // WAVE begins a playback-only accumulator session (no encoder).
            "WAVE" => {
                if wave_open(line, &mut prog) {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            // WAVECLOSE writes the playback accumulator to a PCM WAV and clears it.
            "WAVECLOSE" => match wave_close(line, &mut prog) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            "CLOSE" => {
                // Drain the program-audio accumulator into the encoder (timeline-synced audio),
                // then finish + close. The accumulator is the FULL timeline duration, so the audio
                // stream length matches the video. A video-only encoder (no aac) just skips the
                // drain and finishes video-only.
                let ok = match enc.take() {
                    Some(mut e) => {
                        if enc_audio_ok {
                            prog.drain_into_encoder(&mut e);
                        }
                        e.finish() // drop after this scope closes the encoder
                    }
                    None => false,
                };
                enc_audio_ok = false; // encoder torn down: no audio stream until the next OPEN.
                prog.clear(); // accumulator consumed; next OPEN/WAVE reallocates.
                if ok {
                    Reply::Done(None)
                } else {
                    Reply::Err
                }
            }
            "THUMB" => match thumb(&mut decoders, line) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            "ENV" => match envelope(line) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            // SCOPE runs a scope kernel on the LAST composed buffer left by the most recent PREVIEW
            // (NOT cleared between requests, NO re-compose here). `last_final_is_look` selects OUTB
            // (look none) vs LOOKB (a look ran) so the scope reads the POST-LOOK frame the UI shows.
            "SCOPE" => match scope(&gpu, line, last_final_is_look) {
                Some(out) => Reply::Done(Some(out)),
                None => Reply::Err,
            },
            // Explicit preview-frame keyword (finding #3): a PREVIEW line carries the 21
            // positional fields after the keyword (15 composite+look fields, 5 Wave-8 transition
            // fields, then the out path). The keyword disambiguates it from an ENC line of similar
            // arity, so a media path can never be mistaken for a command.
            //
            // Slice A: a PREVIEW leaves the composed frame in the persistent GPU buffer — g_buf[OUTB]
            // when the clip's look is none, g_buf[LOOKB] when a VHS/LUT look ran (handle_request
            // records which in `last_final_is_look`). That buffer is a static cl_mem and is NEVER
            // cleared between requests, so a subsequent SCOPE reads exactly the (post-look) frame the
            // UI is showing. The only way it changes is another compose (a later PREVIEW/ENC) — which
            // is why the worker.rs `scope()` sends its PREVIEW immediately before SCOPE under one held
            // mutex.
            "PREVIEW" => match handle_request(
                &gpu,
                &mut decoders,
                &mut lut_cache,
                &mut last_uploaded_lut,
                &mut last_final_is_look,
                line,
            ) {
                Some(out_path) => Reply::Done(Some(out_path)),
                None => Reply::Err,
            },
            // Back-compat: a keyword-less line is still treated as a legacy positional preview
            // request (one-shot tools / older clients). New UI clients always send PREVIEW.
            _ => match handle_request(
                &gpu,
                &mut decoders,
                &mut lut_cache,
                &mut last_uploaded_lut,
                &mut last_final_is_look,
                line,
            ) {
                Some(out_path) => Reply::Done(Some(out_path)),
                None => Reply::Err,
            },
        };

        match reply {
            Reply::Done(Some(out)) => {
                let _ = writeln!(stdout, "DONE {out}");
            }
            Reply::Done(None) => {
                let _ = writeln!(stdout, "DONE");
            }
            Reply::Err => {
                let _ = writeln!(stdout, "ERR");
            }
        }
        // Always flush so the client (blocking on a single response line) unblocks promptly.
        let _ = stdout.flush();
    }
}

/// A serve reply: DONE (optionally echoing an out path) or ERR.
enum Reply {
    Done(Option<String>),
    Err,
}

/// `OPEN <out> <w> <h> <fps>` — (re)create the render encoder. Any prior encoder is dropped
/// (closed) without finishing, since a fresh OPEN supersedes it. Configures video (mpeg4) AND, when
/// the local FFmpeg build supports it, audio (aac, 2ch, 48000) so the encoder is ready for both the
/// ENC video feed and the AUDIO program-audio feed, then writes the header. Resets the counter.
///
/// `enc_audio_ok` is set true only if `config_audio` succeeded. config_audio failure (finding #6:
/// a minimal FFmpeg build with no aac encoder) is NON-FATAL — OPEN still succeeds video-only and
/// `enc_audio_ok` stays false, so later AUDIO commands reply ERR (the client skips audio) instead
/// of failing the whole render. This restores wave-2 behavior where a missing aac never broke video.
fn open_render(
    line: &str,
    enc: &mut Option<ffi::Encoder>,
    enc_fps: &mut f64,
    enc_count: &mut i64,
    enc_audio_ok: &mut bool,
    prog: &mut ProgAudio,
) -> bool {
    *enc_audio_ok = false;
    let f: Vec<&str> = line.split_whitespace().collect();
    // OPEN <out> <w> <h> <fps> <total_s>   (total_s = timeline duration, sizes the audio accumulator)
    if f.len() != 6 {
        eprintln!("[gcompose] bad OPEN ({} fields): {line}", f.len());
        return false;
    }
    let out = f[1];
    // The wire w/h are parsed/validated for protocol sanity but DELIBERATELY IGNORED for the
    // encoder dims (finding #7): every ENC frame is produced by `compose_f32`, which always
    // emits GVW×GVH (the OpenCL shim's fixed working resolution). Configuring the encoder at the
    // client's w/h instead would make `fpx_enc_video_frame_f32`'s in_w != vw check reject every
    // frame (-3) the moment preview resolution (PVW/PVH) ever diverged from GVW/GVH. So the
    // encoder input dims are pinned to the engine's GVW/GVH and the wire dims are decoupled.
    let _w: usize = match f[2].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let _h: usize = match f[3].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let fps: i32 = match f[4].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    if _w == 0 || _h == 0 || fps <= 0 {
        return false;
    }
    // Timeline duration in seconds; sizes the program-audio accumulator. A non-finite/negative
    // value is a protocol error; 0 is allowed (empty timeline → empty audio).
    let total_s: f64 = match f[5].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !total_s.is_finite() || total_s < 0.0 {
        eprintln!("[gcompose] bad OPEN total_s={total_s}");
        return false;
    }

    // Encoder input/output dims are the engine's fixed compose resolution, NOT the wire w/h.
    let w = ffi::GVW;
    let h = ffi::GVH;

    // Drop any previous (unfinished) encoder before starting a new job.
    *enc = None;

    let mut e = match ffi::Encoder::open(out) {
        Some(e) => e,
        None => {
            eprintln!("[gcompose] enc_open failed: {out}");
            return false;
        }
    };
    // mpeg4 video matches MojoMedia's render config (codec, dims, fps, bitrate).
    if !e.config_video("mpeg4", w, h, w, h, fps, 1, 4_000_000) {
        eprintln!("[gcompose] config_video failed");
        return false;
    }
    // Program audio (Slice A): configure an aac stream (2ch @ 48000, 128 kbps) matching
    // MojoMedia's render config. The AUDIO command feeds per-clip ranges decoded at this same
    // 2ch/48000 layout; CLOSE (enc_finish) flushes and writes both streams. config_audio is SAFE
    // to enable because the protocol has a real audio-feed command (no more zero-sample track): if
    // every AUDIO clip happens to fail/skip, enc_finish still produces a valid (empty) aac stream
    // rather than a malformed one.
    //
    // Finding #6: a config_audio failure (e.g. an FFmpeg build with no aac encoder) is NON-FATAL.
    // We log it and continue VIDEO-ONLY rather than failing OPEN — otherwise a minimal-FFmpeg
    // environment would lose the ability to render video at all (a regression vs wave-2). The
    // encoder header is then written without an audio stream, and AUDIO commands reply ERR.
    *enc_audio_ok = e.config_audio("aac", 2, 48_000, 128_000);
    if !*enc_audio_ok {
        eprintln!("[gcompose] config_audio failed; rendering video-only (no aac stream)");
    }
    if !e.start() {
        eprintln!("[gcompose] enc_start failed");
        return false;
    }

    *enc = Some(e);
    *enc_fps = fps as f64;
    *enc_count = 0;

    // Allocate the program-audio accumulator for this render's full timeline duration (silence).
    // Each AUDIO line mixes a clip range into it; CLOSE drains it into the encoder.
    prog.alloc(total_s);
    true
}

/// Resolve the three wire LOOK fields (`<look_kind> <look_amt> <lut_path|->`) into the
/// `(look_kind, look_amt, lut_n)` triple the compose pipeline passes to `fpx_gpu_look`, performing
/// any LUT load + upload as a SIDE EFFECT (Slice A).
///
/// Semantics (mirrors MojoMedia main_editor.mojo ~673-703):
///   - kind 0 (none): returns (0, 0.0, 0); no LUT touched. The compose then runs `fpx_gpu_look(0,..)`
///     (a no-op that leaves the frame in OUTB).
///   - kind 1 (VHS): returns (1, amt, 0); the procedural VHS kernel needs no LUT (lut_n=0).
///   - kind 2 (LUT3D): parse `lut_path` (CACHED per path in `lut_cache` — including NEGATIVE results,
///     so a broken .cube isn't reparsed every frame), upload it to the GPU only when it isn't already
///     resident (`last_uploaded_lut`, mirroring MojoMedia's lut_loaded_idx), and return (2, amt, N).
///     A MISSING / unparsable / failed-upload LUT DEGRADES TO NO LOOK: returns (0, 0.0, 0) so the
///     frame still composes (never fails) — exactly the contract's "missing/failed LUT degrades to no
///     look (do not fail the frame)".
///
/// `lut_path` is the "-" sentinel (or empty) when there is no LUT; only kind==2 ever consults it.
fn resolve_look(
    gpu: &ffi::Gpu,
    lut_cache: &mut HashMap<String, Option<ffi::Lut>>,
    last_uploaded_lut: &mut Option<String>,
    look_kind: i32,
    look_amt: f32,
    lut_path: &str,
) -> (i32, f32, i32) {
    match look_kind {
        1 => (1, look_amt, 0), // VHS: procedural, no LUT.
        2 => {
            if lut_path.is_empty() || lut_path == "-" {
                return (0, 0.0, 0); // LUT3D requested but no path: degrade to none.
            }
            // Parse the .cube once per path (cache positive AND negative results). The mutable
            // borrow is confined to the insert; we then re-borrow `lut_cache` immutably to read the
            // cached Lut (no overlapping borrows — the insert's borrow ends before the get).
            if !lut_cache.contains_key(lut_path) {
                let loaded = ffi::load_cube(lut_path);
                if loaded.is_none() {
                    eprintln!("[gcompose] LUT load failed (degrading to no look): {lut_path}");
                }
                lut_cache.insert(lut_path.to_string(), loaded);
            }
            let lut = match lut_cache.get(lut_path).and_then(|o| o.as_ref()) {
                Some(l) => l,
                None => return (0, 0.0, 0), // cached negative: no look.
            };
            // Upload only when this path isn't already the resident GPU LUT (skip the re-upload on
            // consecutive same-look frames). On a fresh/changed path, upload and record it; an upload
            // failure also degrades to no look and forgets the resident path so a later retry re-tries.
            if last_uploaded_lut.as_deref() != Some(lut_path) {
                if gpu.upload_lut(lut) {
                    *last_uploaded_lut = Some(lut_path.to_string());
                } else {
                    eprintln!("[gcompose] LUT upload failed (degrading to no look): {lut_path}");
                    *last_uploaded_lut = None;
                    return (0, 0.0, 0);
                }
            }
            (2, look_amt, lut.n as i32)
        }
        _ => (0, 0.0, 0), // 0 / unknown: no look.
    }
}

/// `ENC <base> <over|-> <bf> <of> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat> <look_kind>
/// <look_amt> <lut_path|-> <trans_kind> <trans_prog> <trans_param> <trans_path|-> <trans_frame>` —
/// decode the base (and optional overlay + optional transition partner), run the same OpenCL
/// composite + transition + look the preview uses, and feed the composited RGBA f32 frame to the
/// active encoder at ts = enc_count / fps. No file is written.
fn enc_frame(
    gpu: &ffi::Gpu,
    decoders: &mut HashMap<String, ffi::Decoder>,
    lut_cache: &mut HashMap<String, Option<ffi::Lut>>,
    last_uploaded_lut: &mut Option<String>,
    enc: Option<&mut ffi::Encoder>,
    fps: f64,
    enc_count: &mut i64,
    line: &str,
) -> bool {
    let e = match enc {
        Some(e) => e,
        None => {
            eprintln!("[gcompose] ENC with no open encoder");
            return false;
        }
    };

    let f: Vec<&str> = line.split_whitespace().collect();
    // ENC + 12 composite fields + 3 LOOK fields + 5 TRANSITION fields = 21 tokens (Wave 8).
    if f.len() != 21 {
        eprintln!("[gcompose] bad ENC ({} fields): {line}", f.len());
        return false;
    }
    let base_path = f[1];
    let over_path = f[2]; // "-" means no overlay
    let parsed = (|| {
        Some((
            f[3].parse::<i32>().ok()?,  // base_frame
            f[4].parse::<i32>().ok()?,  // over_frame
            f[5].parse::<f32>().ok()?,  // op
            f[6].parse::<f32>().ok()?,  // px
            f[7].parse::<f32>().ok()?,  // py
            f[8].parse::<f32>().ok()?,  // pw
            f[9].parse::<f32>().ok()?,  // ph
            f[10].parse::<f32>().ok()?, // bright
            f[11].parse::<f32>().ok()?, // contrast
            f[12].parse::<f32>().ok()?, // sat
            f[13].parse::<i32>().ok()?, // look_kind
            f[14].parse::<f32>().ok()?, // look_amt
        ))
    })();
    let (base_frame, over_frame, op, px, py, pw, ph, bright, contrast, sat, look_kind, look_amt) =
        match parsed {
            Some(v) => v,
            None => return false,
        };
    let lut_path = f[15]; // "-" / empty when no LUT (only used by LUT3D look_kind==2)

    // Transition fields (Wave 8): kind (-1 none, 0..7 kernel), progress, param, partner path+frame.
    let trans_parsed = (|| {
        Some((
            f[16].parse::<i32>().ok()?, // trans_kind
            f[17].parse::<f32>().ok()?, // trans_prog
            f[18].parse::<f32>().ok()?, // trans_param
            f[20].parse::<i32>().ok()?, // trans_frame
        ))
    })();
    let (trans_kind, trans_prog, trans_param, trans_frame) = match trans_parsed {
        Some(v) => v,
        None => return false,
    };
    let trans_path = f[19]; // "-" when no transition partner

    // Decode base @ base_frame (cached), upload to slot 0. A "-" base is an explicit timeline
    // gap (finding #5): fill slot 0 with black (matching MojoMedia's black-gap behavior) and
    // skip decoding entirely. A black frame also keeps timing if a real base can't be decoded.
    if base_path == "-" {
        gpu.upload(0, &vec![0u8; ffi::GVW * ffi::GVH * 4]);
    } else {
        match decode_cached(decoders, base_path, base_frame) {
            Some(rgba) => gpu.upload(0, &rgba),
            None => gpu.upload(0, &vec![0u8; ffi::GVW * ffi::GVH * 4]),
        }
    }

    // Decode overlay if present and op>0; otherwise disable the composite.
    let mut eff_op = op;
    if over_path != "-" && op > 0.0 {
        match decode_cached(decoders, over_path, over_frame) {
            Some(ov) => gpu.upload(1, &ov),
            None => eff_op = 0.0,
        }
    } else {
        eff_op = 0.0;
    }

    // Resolve the per-boundary TRANSITION (Wave 8): when active (kind 0..7 AND a real partner path),
    // decode the INCOMING clip's frame into slot 2 so track1 can blend base→trans; a "-"/failed
    // partner degrades to no transition (the base still encodes). Mirrors MojoMedia's render loop
    // (~1286-1300): decode the boundary partner, upload slot 2, then track1(tt_id, rtt, tt_p).
    let eff_tt = resolve_trans(gpu, decoders, trans_kind, trans_path, trans_frame);

    // Resolve the per-clip LOOK (load + upload the .cube for LUT3D, cached; VHS needs no LUT; a
    // missing/failed LUT degrades to no look). Then run the same transition→composite→look the
    // preview uses, downloading f32 for the encoder.
    let (lk, la, ln) =
        resolve_look(gpu, lut_cache, last_uploaded_lut, look_kind, look_amt, lut_path);
    let (frame, _fin) = gpu.compose_trans_f32(
        eff_tt, trans_prog, trans_param, eff_op, px, py, pw, ph, bright, contrast, sat, lk, la, ln,
    );
    let ts = (*enc_count as f64) / fps;
    if !e.video_frame(&frame, ts) {
        eprintln!("[gcompose] enc video_frame failed @ {}", *enc_count);
        return false;
    }
    *enc_count += 1;
    true
}

/// Resolve the wire TRANSITION fields into the effective transition kind for `compose_trans*`,
/// performing the slot-2 upload as a SIDE EFFECT (Wave 8).
///
/// Returns the kind to pass to `track1`/`compose_trans*`:
///   - `trans_kind == -1` (or out of the 0..7 range, or a "-"/empty partner path): returns -1 so
///     the pipeline runs `track1(-1, ..)` (copy the slot-0 base — today's no-transition behavior).
///     Slot 2 is NOT touched.
///   - `trans_kind` in 0..7 with a real partner path: decode `trans_path`@`trans_frame` (cached
///     decoder) and upload it to slot 2, then return `trans_kind`. If the decode FAILS, degrade to
///     no transition (return -1) so the frame still composes the base rather than failing.
///
/// Mirrors MojoMedia (main_editor.mojo ~1286-1300): `fpx_decode_frame_letterbox(...rgba8_trans...)`
/// → `fpx_gpu_upload_u8(2, rgba8_trans)` → the kind fed to `fpx_gpu_track1`. Unlike MojoMedia we do
/// NOT track a "currently-resident slot-2 frame" to skip re-uploads: each transition frame's partner
/// frame differs (the incoming clip advances with the playhead), so a cache key would rarely hit;
/// the decoder cache already avoids re-opening the file.
fn resolve_trans(
    gpu: &ffi::Gpu,
    decoders: &mut HashMap<String, ffi::Decoder>,
    trans_kind: i32,
    trans_path: &str,
    trans_frame: i32,
) -> i32 {
    if !(0..=7).contains(&trans_kind) || trans_path == "-" || trans_path.is_empty() {
        return -1; // no transition: track1(-1,..) copies the base.
    }
    match decode_cached(decoders, trans_path, trans_frame) {
        Some(rgba) => {
            gpu.upload(2, &rgba);
            trans_kind
        }
        None => {
            // Partner frame couldn't be decoded: degrade to no transition (don't fail the frame).
            eprintln!("[gcompose] transition partner decode failed (degrading to no transition): {trans_path}@{trans_frame}");
            -1
        }
    }
}

/// Program-audio sample rate / channel layout. The accumulator and every decoded clip range use
/// this fixed interleaved-stereo-48k layout (matches OPEN's `config_audio("aac", 2, 48000, ...)`
/// and MojoMedia's render config). `SR*CH` floats == one second of program audio.
const PROG_SR: usize = 48_000;
const PROG_CH: usize = 2;

/// The timeline-synced program-audio accumulator (Slice A).
///
/// `buf` is interleaved stereo f32 @ 48000 sized to the timeline duration; `active` is true between
/// an OPEN/WAVE (alloc) and its CLOSE/WAVECLOSE (clear). AUDIO lines MIX one decoded clip range into
/// `buf` at a sample offset (sample-add, clamped to [-1,1]); positions with no clip stay silent and
/// overlapping clips sum — this is the fix for the old back-to-back concatenation (a clip at t0=70
/// now starts at 70/FPS s, gaps are silence, overlaps mix).
struct ProgAudio {
    buf: Vec<f32>, // interleaved L,R,L,R... ; len = frames * PROG_CH
    active: bool,
}

impl Default for ProgAudio {
    fn default() -> Self {
        ProgAudio { buf: Vec::new(), active: false }
    }
}

impl ProgAudio {
    /// (Re)allocate the accumulator to `total_s` seconds of silence and mark it active. A prior
    /// (unconsumed) buffer is dropped. `total_s` is clamped to a sane ceiling so a pathological
    /// duration can't blow up the allocation.
    fn alloc(&mut self, total_s: f64) {
        // Ceiling on the WHOLE-program accumulator (24h stereo 48k ≈ 33 GB floats — far past any
        // real timeline; this only guards against a corrupt/huge total_s, not normal use).
        const MAX_FRAMES: usize = 24 * 3600 * PROG_SR;
        let frames = if total_s.is_finite() && total_s > 0.0 {
            (total_s * PROG_SR as f64).ceil() as usize
        } else {
            0
        };
        let frames = frames.min(MAX_FRAMES);
        self.buf = vec![0.0f32; frames.saturating_mul(PROG_CH)];
        self.active = true;
    }

    /// Drop the accumulator and mark inactive (after CLOSE/WAVECLOSE consumes it).
    fn clear(&mut self) {
        self.buf = Vec::new();
        self.active = false;
    }

    /// Mix `samples` (interleaved stereo @ 48000, already gain-applied) into the accumulator at
    /// frame offset `dst_frame` (samples-per-channel offset). Sample-adds and clamps to [-1,1].
    /// Samples that would land past the end of the accumulator are dropped (the accumulator is the
    /// authoritative timeline length, so a clip running slightly long is truncated to it).
    ///
    /// SUB-SAMPLE BOUNDARY (finding #5, INTEGRATOR NOTE): `alloc` sizes the accumulator with `.ceil()`
    /// while `dst_frame` here is `.round()`ed and the C decoder trims each range to
    /// `(int)(dur_sec*sr+0.5)` — three independent roundings. A clip whose end lands a fraction above
    /// the ceil'd accumulator end therefore has its final ≤1 interleaved frame clamped off by `room`.
    /// This is inaudible (≤1/48000 s ≈ 21 µs) and intentional (the accumulator is authoritative), but
    /// it means the rendered AUDIO duration can be ≤ the video by up to one sample. Any gate that
    /// compares audio-vs-video duration must allow ≥1 sample / ~21 µs of slack, NOT bit-exact equality.
    fn mix(&mut self, samples: &[f32], dst_frame: usize) {
        if samples.is_empty() || self.buf.is_empty() {
            return;
        }
        let start = dst_frame.saturating_mul(PROG_CH);
        if start >= self.buf.len() {
            return; // entirely past the timeline end.
        }
        // Whole interleaved frames only (drop a stray odd tail float).
        let n = (samples.len() / PROG_CH) * PROG_CH;
        let room = self.buf.len() - start;
        let take = n.min(room);
        for i in 0..take {
            let v = self.buf[start + i] + samples[i];
            self.buf[start + i] = v.clamp(-1.0, 1.0);
        }
    }

    /// Feed the WHOLE accumulator to the encoder's audio stream in chunks. The encoder packs the
    /// interleaved floats into proper codec frames internally (fpx_enc_audio_samples_f32 buffers
    /// across calls), so chunking is purely to bound the per-call slice — the muxed result is the
    /// single continuous timeline-length program audio. Best-effort: a rejected chunk is logged and
    /// the drain stops (the already-fed audio still muxes on finish).
    fn drain_into_encoder(&self, e: &mut ffi::Encoder) {
        if self.buf.is_empty() {
            return;
        }
        // ~0.25 s of stereo @ 48k per chunk (12000 frames). Aligned to whole interleaved samples.
        const CHUNK_FRAMES: usize = 12_000;
        let total_frames = self.buf.len() / PROG_CH;
        let mut f = 0usize;
        while f < total_frames {
            let nb = CHUNK_FRAMES.min(total_frames - f);
            let off = f * PROG_CH;
            let slice = &self.buf[off..off + nb * PROG_CH];
            if !e.audio_samples(slice, nb) {
                eprintln!("[gcompose] enc audio_samples failed draining accumulator @ frame {f}");
                return;
            }
            f += nb;
        }
    }
}

/// Per-clip audio-decode capacity ceiling (in FLOATS), shared by AUDIO mixing. Mirrors MojoMedia's
/// AUF (180 s stereo 48k + headroom) used as a per-decode cap. Bounds the temp decode buffer and
/// guarantees the `cap as c_int` narrowing in ffi::decode_audio_range is lossless & positive.
const AUDIO_CAP_MAX: usize = 180 * PROG_SR * PROG_CH + 8192;

/// `AUDIO <path> <src_in_s> <dur_s> <dst_offset_s> <gain>` — decode the SOURCE audio range
/// [src_in_s, src_in_s+dur_s) of `path` to interleaved 2ch @ 48000 f32, apply `gain`, and MIX it
/// into the active program-audio accumulator starting at `dst_offset_s` seconds (sample-add,
/// clamped). This is the timeline-sync fix: the clip is positioned at its timeline offset, not
/// concatenated. Returns false (-> ERR) if there is no active accumulator (no OPEN/WAVE), the line
/// is malformed, or the range has no decodable audio — the client treats ERR as "skip this clip"
/// and continues. Mirrors MojoMedia's fpx_decode_audio_range program-audio assembly, but with a
/// destination offset + gain instead of back-to-back concatenation.
fn audio_mix(prog: &mut ProgAudio, line: &str) -> bool {
    // No active accumulator: a stray AUDIO outside an OPEN/WAVE session. ERR (client skips).
    if !prog.active {
        eprintln!("[gcompose] AUDIO with no active accumulator (no OPEN/WAVE)");
        return false;
    }

    let f: Vec<&str> = line.split_whitespace().collect();
    // AUDIO <path> <src_in_s> <dur_s> <dst_offset_s> <gain> = 6 tokens. The path is whitespace-free
    // (the UI only ever sends pool media paths, same as ENC/THUMB), so a fixed-arity split is safe.
    if f.len() != 6 {
        eprintln!("[gcompose] bad AUDIO ({} fields): {line}", f.len());
        return false;
    }
    let path = f[1];
    let src_in_s: f64 = match f[2].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let dur_s: f64 = match f[3].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let dst_off_s: f64 = match f[4].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let gain: f32 = match f[5].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !(src_in_s.is_finite() && dur_s.is_finite() && dst_off_s.is_finite())
        || dur_s <= 0.0
        || src_in_s < 0.0
        || dst_off_s < 0.0
        || !gain.is_finite()
    {
        eprintln!("[gcompose] bad AUDIO src_in={src_in_s} dur={dur_s} dst={dst_off_s} gain={gain}");
        return false;
    }

    let sr = PROG_SR as i32;
    let ch = PROG_CH as i32;
    // Capacity: dur seconds * sr * ch, + a codec-frame of slack, clamped to AUDIO_CAP_MAX (finding
    // #4: saturating math + clamp keeps the `as c_int` narrowing lossless and positive).
    let want = (dur_s * PROG_SR as f64).ceil();
    let want = if want.is_finite() && want >= 0.0 { want as usize } else { 0 };
    let cap = want
        .saturating_mul(PROG_CH)
        .saturating_add(8192)
        .min(AUDIO_CAP_MAX);

    let mut samples = match ffi::decode_audio_range(path, src_in_s, dur_s, sr, ch, cap) {
        Some(s) => s,
        None => {
            eprintln!("[gcompose] AUDIO decode failed: {path} @ {src_in_s}+{dur_s}");
            return false; // hard decode error -> ERR (client skips this clip).
        }
    };
    // No audio in the range: nothing to mix. ERR so the client logs it; accumulator is unchanged.
    if samples.is_empty() {
        return false;
    }

    // Apply per-clip gain in place (1.0 = unity, the common case skips the multiply loop).
    if gain != 1.0 {
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }

    // Destination sample-per-channel offset. Round to nearest frame; the accumulator's mix() drops
    // anything past the end so an offset at/after the timeline end is a harmless no-op.
    let dst_frame = (dst_off_s * PROG_SR as f64).round();
    let dst_frame = if dst_frame.is_finite() && dst_frame >= 0.0 { dst_frame as usize } else { 0 };
    prog.mix(&samples, dst_frame);
    true
}

/// `WAVE <out_wav> <total_s>` — begin a PLAYBACK-ONLY accumulator session (no encoder). Allocates
/// the program-audio accumulator to `total_s` seconds of silence so subsequent AUDIO lines mix into
/// it exactly like the render path; `WAVECLOSE` then writes it to `<out_wav>`. The out path is
/// parsed here only for arity validation (WAVECLOSE carries the real write target).
fn wave_open(line: &str, prog: &mut ProgAudio) -> bool {
    let f: Vec<&str> = line.split_whitespace().collect();
    // WAVE <out_wav> <total_s>
    if f.len() != 3 {
        eprintln!("[gcompose] bad WAVE ({} fields): {line}", f.len());
        return false;
    }
    let total_s: f64 = match f[2].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !total_s.is_finite() || total_s < 0.0 {
        return false;
    }
    prog.alloc(total_s);
    true
}

/// `WAVECLOSE <out_wav>` — write the playback accumulator to `<out_wav>` as a 16-bit PCM stereo @
/// 48000 WAV, then clear it. Returns the out path on success. The UI spawns a system player on the
/// file. ERR if there is no active accumulator or the write fails.
fn wave_close(line: &str, prog: &mut ProgAudio) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // WAVECLOSE <out_wav>
    if f.len() != 2 {
        eprintln!("[gcompose] bad WAVECLOSE ({} fields): {line}", f.len());
        return None;
    }
    let out = f[1];
    if !prog.active {
        eprintln!("[gcompose] WAVECLOSE with no active accumulator");
        return None;
    }
    let ok = write_wav_pcm16(out, &prog.buf, PROG_SR as u32, PROG_CH as u16);
    prog.clear();
    if ok {
        Some(out.to_string())
    } else {
        eprintln!("[gcompose] WAVECLOSE write failed: {out}");
        None
    }
}

/// Write interleaved f32 [-1,1] `samples` as a 16-bit PCM WAV (`sr` Hz, `ch` channels). Standard
/// 44-byte canonical WAV header + little-endian s16 samples. Returns true on success. No external
/// deps — paplay/aplay both play canonical PCM WAV. (Best-effort playback path for Slice A.)
fn write_wav_pcm16(path: &str, samples: &[f32], sr: u32, ch: u16) -> bool {
    let bits_per_sample: u16 = 16;
    let block_align: u16 = ch * bits_per_sample / 8;
    let byte_rate: u32 = sr * block_align as u32;
    let data_bytes: u32 = (samples.len() as u32).saturating_mul(2); // 2 bytes per s16 sample
    let riff_size: u32 = 36u32.saturating_add(data_bytes);

    let mut out: Vec<u8> = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&ch.to_le_bytes());
    out.extend_from_slice(&sr.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in samples {
        // Clamp + scale to s16. (Mix already clamps, but re-clamp defensively.)
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i32;
        out.extend_from_slice(&(v as i16).to_le_bytes());
    }
    std::fs::write(path, &out).is_ok()
}

/// `THUMB <path> <frame> <w> <h> <out>` — decode one frame letterboxed to w×h and write the
/// RGBA8 buffer to <out>. Uses the cached decoders (a thumbnail of an already-open media reuses
/// the handle). Returns the out path on success.
fn thumb(decoders: &mut HashMap<String, ffi::Decoder>, line: &str) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // THUMB <path> <frame> <w> <h> <out>
    if f.len() != 6 {
        eprintln!("[gcompose] bad THUMB ({} fields): {line}", f.len());
        return None;
    }
    let path = f[1];
    let frame: i32 = f[2].parse().ok()?;
    let w: usize = f[3].parse().ok()?;
    let h: usize = f[4].parse().ok()?;
    let out = f[5];
    if w == 0 || h == 0 {
        return None;
    }

    if !decoders.contains_key(path) {
        let d = ffi::Decoder::open(path)?;
        decoders.insert(path.to_string(), d);
    }
    let dec = decoders.get_mut(path)?;
    let buf = dec.decode_rgba(frame.max(0), w, h)?;
    if std::fs::write(out, &buf).is_err() {
        eprintln!("[gcompose] THUMB write failed: {out}");
        return None;
    }
    Some(out.to_string())
}

/// `ENV <path> <buckets> <out>` — compute the whole-track peak envelope and write <buckets>
/// little-endian f32 to <out>. Returns the out path on success.
fn envelope(line: &str) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // ENV <path> <buckets> <out>
    if f.len() != 4 {
        eprintln!("[gcompose] bad ENV ({} fields): {line}", f.len());
        return None;
    }
    let path = f[1];
    let buckets: usize = f[2].parse().ok()?;
    let out = f[3];
    if buckets == 0 {
        return None;
    }

    let env = ffi::audio_envelope(path, buckets)?;
    // Serialize as little-endian f32 (the UI reads it back the same way).
    let mut bytes = Vec::with_capacity(buckets * 4);
    for v in &env {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if std::fs::write(out, &bytes).is_err() {
        eprintln!("[gcompose] ENV write failed: {out}");
        return None;
    }
    Some(out.to_string())
}

/// `SCOPE <kind> <out>` — run the kind-selected scope kernel on the LAST composed GPU buffer (the
/// frame left in g_buf[OUTB] by the most recent PREVIEW; it is NOT re-composed or cleared here),
/// produce a 256×256 RGBA8 image, write it to <out>, and return the out path. Returns None (-> ERR)
/// on a malformed line, an unknown kind, or a write failure.
///
/// `final_is_look` selects which composed buffer the scope reads: false = OUTB (the most recent
/// PREVIEW had look=none), true = LOOKB (a VHS/LUT look ran). It is passed by the serve loop from
/// the PREVIEW that composed the displayed frame (Slice A), so the scope ALWAYS reads the POST-LOOK
/// frame the UI is showing — not the pre-look grade buffer. kinds 1/2 (waveform/vectorscope) come
/// back as a rendered RGBA8 image from the kernel; kind 0 (histogram) comes back as 768 R/G/B int
/// bins which `render_histogram` rasterizes into a 256×256 RGBA bar graph.
fn scope(gpu: &ffi::Gpu, line: &str, final_is_look: bool) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    // SCOPE <kind> <out>
    if f.len() != 3 {
        eprintln!("[gcompose] bad SCOPE ({} fields): {line}", f.len());
        return None;
    }
    let kind: u8 = f[1].parse().ok()?;
    let out = f[2];

    // Read the buffer the last PREVIEW left the displayed frame in (OUTB when look=none, LOOKB when
    // a look ran) so the scope reflects the post-look picture.
    let img: Vec<u8> = match kind {
        0 => render_histogram(&gpu.histogram(final_is_look)),
        1 => gpu.waveform(final_is_look),
        2 => gpu.vectorscope(final_is_look),
        _ => {
            eprintln!("[gcompose] bad SCOPE kind: {kind}");
            return None;
        }
    };

    // Every scope image is exactly SVW×SVH×4 bytes (the UI length-checks SW*SH*4 on read-back).
    debug_assert_eq!(img.len(), ffi::SVW * ffi::SVH * 4);
    if std::fs::write(out, &img).is_err() {
        eprintln!("[gcompose] SCOPE write failed: {out}");
        return None;
    }
    Some(out.to_string())
}

/// Rasterize the 768 R/G/B histogram bins into a 256×256 RGBA8 bar graph on a dark background
/// (the histogram kernel returns raw counts, not an image — unlike waveform/vectorscope). Mirrors
/// MojoMedia main_editor.mojo (~880-894): scale to the tallest NON-black bin (bin index 0 is the
/// pure-black/letterbox spike and is excluded so it doesn't flatten everything else), then draw one
/// column per bin bottom-up, with the three channels overlaid translucently so overlap blends.
///
/// `bins` is `R[0..256] | G[256..512] | B[512..768]`. Output is row-major top-to-bottom RGBA8
/// (row 0 = top of the image), so a taller bar fills MORE rows toward the bottom — matching the
/// "bars rise from the baseline" look of a standard histogram scope.
fn render_histogram(bins: &[i32]) -> Vec<u8> {
    const W: usize = 256; // == ffi::SVW
    const H: usize = 256; // == ffi::SVH
    let mut img = vec![0u8; W * H * 4];

    // Dark background fill (matches MojoMedia's Col4(0.06,0.07,0.10)).
    const BG: [u8; 4] = [15, 18, 26, 255];
    for px in img.chunks_exact_mut(4) {
        px.copy_from_slice(&BG);
    }
    // A defensive guard: a short/empty bins slice (kernel no-op on a not-ready GPU) yields the
    // plain dark image rather than indexing out of bounds.
    if bins.len() < 768 {
        return img;
    }

    // Scale to the tallest non-black bin across all three channels (skip bin 0 of each channel:
    // the pure-black letterbox spike, which would otherwise dwarf the real content).
    let mut hmax: i32 = 1;
    for i in 1..256 {
        hmax = hmax.max(bins[i]).max(bins[256 + i]).max(bins[512 + i]);
    }
    let hmaxf = hmax as f32;

    // Per-column bar heights (in rows) for each channel, then composite the three translucent bars.
    // alpha ≈ 0.5 over the bg, additive-ish so overlapping channels brighten toward white.
    //
    // COLUMN 0 (finding #2): bin index 0 of each channel (`bins[0]`/`bins[256]`/`bins[512]`) is the
    // per-channel VALUE-0 count — the pure-black/letterbox spike. It is excluded from `hmax` above so
    // it doesn't flatten the real content; if we then DREW it, that spike would still paint column 0
    // at full height (its count ≫ hmax → clamped to H), just relocating the artifact instead of
    // removing it. So column 0 is drawn at zero height to match the hmax exclusion — the histogram
    // shows only the non-black tonal distribution. (The black-level information is intentionally
    // dropped; a scope reads tonal SHAPE, not the absolute black count.)
    for x in 0..W {
        let (rh, gh, bh) = if x == 0 {
            (0usize, 0usize, 0usize)
        } else {
            let rh = ((bins[x] as f32 / hmaxf) * H as f32).round() as usize;
            let gh = ((bins[256 + x] as f32 / hmaxf) * H as f32).round() as usize;
            let bh = ((bins[512 + x] as f32 / hmaxf) * H as f32).round() as usize;
            (rh.min(H), gh.min(H), bh.min(H))
        };
        for y in 0..H {
            // Row y counts from the top; a bar of height `hb` fills the bottom `hb` rows, i.e. rows
            // with (H - y) <= hb  ⇔  y >= H - hb.
            let from_bottom = H - y; // 1..=H
            let r_on = from_bottom <= rh;
            let g_on = from_bottom <= gh;
            let b_on = from_bottom <= bh;
            if !(r_on || g_on || b_on) {
                continue; // leave the dark bg
            }
            let off = (y * W + x) * 4;
            // Start from bg and blend each active channel bar (src-over at a=0.5) so overlapping
            // channels read as a brighter mixed color (R+G -> yellowish, etc.).
            let mut c = [BG[0] as f32, BG[1] as f32, BG[2] as f32];
            if r_on {
                c = blend(c, [235.0, 64.0, 64.0], 0.5);
            }
            if g_on {
                c = blend(c, [64.0, 230.0, 90.0], 0.5);
            }
            if b_on {
                c = blend(c, [90.0, 140.0, 247.0], 0.5);
            }
            img[off] = c[0].round().clamp(0.0, 255.0) as u8;
            img[off + 1] = c[1].round().clamp(0.0, 255.0) as u8;
            img[off + 2] = c[2].round().clamp(0.0, 255.0) as u8;
            img[off + 3] = 255;
        }
    }
    img
}

/// Src-over blend of `src` (RGB 0..255) onto `dst` (RGB 0..255) at alpha `a` (0..1).
fn blend(dst: [f32; 3], src: [f32; 3], a: f32) -> [f32; 3] {
    [
        dst[0] * (1.0 - a) + src[0] * a,
        dst[1] * (1.0 - a) + src[1] * a,
        dst[2] * (1.0 - a) + src[2] * a,
    ]
}

/// Parse + execute one serve request line. Returns the out_path on success, None on any failure.
/// Also UPDATES `last_final_is_look` to the buffer the composed frame ended up in (OUTB=false /
/// LOOKB=true), so a subsequent SCOPE reads the POST-LOOK frame this PREVIEW just produced.
fn handle_request(
    gpu: &ffi::Gpu,
    decoders: &mut HashMap<String, ffi::Decoder>,
    lut_cache: &mut HashMap<String, Option<ffi::Lut>>,
    last_uploaded_lut: &mut Option<String>,
    last_final_is_look: &mut bool,
    line: &str,
) -> Option<String> {
    let mut f: Vec<&str> = line.split_whitespace().collect();
    // Accept both the new explicit form (`PREVIEW` + 21 fields) and the legacy keyword-less form
    // (21 positional fields). Strip a leading PREVIEW keyword so the positional indices below are
    // identical for both (finding #3). The 21 fields are the 12 composite fields + the 3 Slice A
    // LOOK fields (look_kind, look_amt, lut_path) + the 5 Wave 8 TRANSITION fields (trans_kind,
    // trans_prog, trans_param, trans_path, trans_frame) + the out path.
    if f.first() == Some(&"PREVIEW") {
        f.remove(0);
    }
    if f.len() != 21 {
        eprintln!("[gcompose] bad request ({} fields): {line}", f.len());
        return None;
    }

    let base_path = f[0];
    let over_path = f[1]; // "-" means no overlay
    let base_frame: i32 = f[2].parse().ok()?;
    let over_frame: i32 = f[3].parse().ok()?;
    let op: f32 = f[4].parse().ok()?;
    let px: f32 = f[5].parse().ok()?;
    let py: f32 = f[6].parse().ok()?;
    let pw: f32 = f[7].parse().ok()?;
    let ph: f32 = f[8].parse().ok()?;
    let bright: f32 = f[9].parse().ok()?;
    let contrast: f32 = f[10].parse().ok()?;
    let sat: f32 = f[11].parse().ok()?;
    let look_kind: i32 = f[12].parse().ok()?;
    let look_amt: f32 = f[13].parse().ok()?;
    let lut_path = f[14]; // "-" / empty when no LUT (only used by LUT3D look_kind==2)
    // Wave 8 TRANSITION fields (before the out path).
    let trans_kind: i32 = f[15].parse().ok()?;
    let trans_prog: f32 = f[16].parse().ok()?;
    let trans_param: f32 = f[17].parse().ok()?;
    let trans_path = f[18]; // "-" when no transition partner
    let trans_frame: i32 = f[19].parse().ok()?;
    let out_path = f[20];

    // Decode base @ base_frame (cached decoder per path), upload to slot 0. A "-" base is an
    // explicit timeline gap (finding #5): fill slot 0 with black, matching the ENC path and
    // MojoMedia's black-gap behavior, rather than failing the frame.
    if base_path == "-" {
        gpu.upload(0, &vec![0u8; ffi::GVW * ffi::GVH * 4]);
    } else {
        let base_rgba = decode_cached(decoders, base_path, base_frame)?;
        gpu.upload(0, &base_rgba);
    }

    // Decode over @ over_frame (if any), upload to slot 1. A failed/missing over just
    // disables the composite (op forced to 0) rather than failing the whole frame.
    let mut eff_op = op;
    if over_path != "-" && op > 0.0 {
        match decode_cached(decoders, over_path, over_frame) {
            Some(ov) => gpu.upload(1, &ov),
            None => eff_op = 0.0,
        }
    } else {
        eff_op = 0.0;
    }

    // Resolve the per-boundary TRANSITION (Wave 8): decode the incoming partner into slot 2 when
    // active (kind 0..7 + a real path), else -1 (track1 copies the base). Same side-effecting helper
    // the ENC path uses, so the preview and the export animate the transition identically.
    let eff_tt = resolve_trans(gpu, decoders, trans_kind, trans_path, trans_frame);

    // Resolve the per-clip LOOK (load + upload the .cube for LUT3D, cached; VHS needs no LUT; a
    // missing/failed LUT degrades to no look). Then run the OpenCL pipeline (transition or base-copy
    // first; PiP over; grade; LOOK). `fin` tells us which buffer the frame ended in (OUTB / LOOKB).
    let (lk, la, ln) =
        resolve_look(gpu, lut_cache, last_uploaded_lut, look_kind, look_amt, lut_path);
    let (out, fin) = gpu.compose_trans(
        eff_tt, trans_prog, trans_param, eff_op, px, py, pw, ph, bright, contrast, sat, lk, la, ln,
    );
    // Record the final buffer so a following SCOPE reads the POST-LOOK frame the UI is showing.
    *last_final_is_look = fin;

    if std::fs::write(out_path, &out).is_err() {
        eprintln!("[gcompose] write failed: {out_path}");
        return None;
    }
    Some(out_path.to_string())
}

/// Decode `path` @ `frame` to GVW×GVH RGBA8, reusing (or opening + caching) the decoder.
/// Returns None if the file can't be opened or the frame can't be decoded.
fn decode_cached(
    decoders: &mut HashMap<String, ffi::Decoder>,
    path: &str,
    frame: i32,
) -> Option<Vec<u8>> {
    if !decoders.contains_key(path) {
        let d = ffi::Decoder::open(path)?;
        decoders.insert(path.to_string(), d);
    }
    let dec = decoders.get_mut(path)?;
    let f = if frame < 0 { 0 } else { frame }; // never seek a negative frame (C-side guard).
    dec.decode_rgba(f, ffi::GVW, ffi::GVH)
}

/// One-shot: decode base+over, run the OpenCL composite (demo PiP inset + grade). None if no GPU.
fn compose(base: &str, over: &str) -> Option<Vec<u8>> {
    let gpu = ffi::Gpu::init()?;
    let mut b = ffi::Decoder::open(base)?;
    let base_rgba = b.decode_rgba(60, ffi::GVW, ffi::GVH)?;
    gpu.upload(0, &base_rgba);

    let mut has_over = false;
    if over != "-" {
        if let Some(mut o) = ffi::Decoder::open(over) {
            if let Some(ov) = o.decode_rgba(0, ffi::GVW, ffi::GVH) {
                gpu.upload(1, &ov);
                has_over = true;
            }
        }
    }
    let op = if has_over { 1.0 } else { 0.0 };
    // One-shot demo: no look (kind 0). compose now returns (rgba, final_is_look); we only want the
    // pixels here, so discard the buffer-selection flag.
    let (buf, _fin) = gpu.compose(op, 0.6, 0.1, 0.3, 0.3, 0.08, 1.1, 1.25, 0, 0.0, 0);
    Some(buf)
}

/// CPU/FFmpeg-only fallback: just the decoded base frame.
fn decode_only(base: &str) -> Option<Vec<u8>> {
    let mut b = ffi::Decoder::open(base)?;
    b.decode_rgba(60, ffi::GVW, ffi::GVH)
}
