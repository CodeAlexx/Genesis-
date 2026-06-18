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
//!   Preview frame (PREVIEW keyword + 13 positional fields; a keyword-less line is still
//!   accepted for back-compat with one-shot/older clients):
//!     PREVIEW <base> <over|-> <bf> <of> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat> <out>
//!     -> compose program frame, write RGBA to <out>; reply "DONE <out>".
//!     A "-" base path renders a black frame (timeline gap).
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
//!        -> decode(cached) + compose(track1(-1,0,4)->pip->grade->look(0,0,0)) + feed the
//!           composited f32 frame to the encoder at ts = enc_count/fps; reply DONE/ERR; no file.
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
/// Protocol (per line in / per line out):
///   in : <base> <over|-> <bf> <of> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat> <out>
///   out: "DONE <out>\n" on success, "ERR\n" on failure (always flushed).
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
                if enc_frame(&gpu, &mut decoders, enc.as_mut(), enc_fps, &mut enc_count, line) {
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
            // Explicit preview-frame keyword (finding #3): a PREVIEW line carries the 13
            // positional fields after the keyword. The keyword disambiguates it from an ENC
            // line of equal arity, so a media path can never be mistaken for a command.
            "PREVIEW" => match handle_request(&gpu, &mut decoders, line) {
                Some(out_path) => Reply::Done(Some(out_path)),
                None => Reply::Err,
            },
            // Back-compat: a keyword-less line is still treated as a legacy positional preview
            // request (one-shot tools / older clients). New UI clients always send PREVIEW.
            _ => match handle_request(&gpu, &mut decoders, line) {
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

/// `ENC <base> <over|-> <bf> <of> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat>` — decode
/// the base (and optional overlay), run the same OpenCL composite the preview uses, and feed the
/// composited RGBA f32 frame to the active encoder at ts = enc_count / fps. No file is written.
fn enc_frame(
    gpu: &ffi::Gpu,
    decoders: &mut HashMap<String, ffi::Decoder>,
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
    // ENC + 12 payload fields = 13 tokens.
    if f.len() != 13 {
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
        ))
    })();
    let (base_frame, over_frame, op, px, py, pw, ph, bright, contrast, sat) = match parsed {
        Some(v) => v,
        None => return false,
    };

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

    // Same pipeline as the preview, but download f32 for the encoder.
    let frame = gpu.compose_f32(eff_op, px, py, pw, ph, bright, contrast, sat);
    let ts = (*enc_count as f64) / fps;
    if !e.video_frame(&frame, ts) {
        eprintln!("[gcompose] enc video_frame failed @ {}", *enc_count);
        return false;
    }
    *enc_count += 1;
    true
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

/// Parse + execute one serve request line. Returns the out_path on success, None on any failure.
fn handle_request(
    gpu: &ffi::Gpu,
    decoders: &mut HashMap<String, ffi::Decoder>,
    line: &str,
) -> Option<String> {
    let mut f: Vec<&str> = line.split_whitespace().collect();
    // Accept both the new explicit form (`PREVIEW` + 13 fields) and the legacy keyword-less form
    // (13 positional fields). Strip a leading PREVIEW keyword so the positional indices below are
    // identical for both (finding #3).
    if f.first() == Some(&"PREVIEW") {
        f.remove(0);
    }
    if f.len() != 13 {
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
    let out_path = f[12];

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

    // Run the OpenCL pipeline (no transition -> base; PiP over; grade; look=none).
    let out = gpu.compose(eff_op, px, py, pw, ph, bright, contrast, sat);

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
    Some(gpu.compose(op, 0.6, 0.1, 0.3, 0.3, 0.08, 1.1, 1.25))
}

/// CPU/FFmpeg-only fallback: just the decoded base frame.
fn decode_only(base: &str) -> Option<Vec<u8>> {
    let mut b = ffi::Decoder::open(base)?;
    b.decode_rgba(60, ffi::GVW, ffi::GVH)
}
