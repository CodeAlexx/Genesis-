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
//!   Render/export (Slice A):
//!     OPEN <out> <w> <h> <fps>
//!        -> open + config_video(mpeg4,w,h@fps) + start; reply DONE/ERR. (Video-only this
//!           slice: no audio stream is configured — see open_render.)
//!     ENC <base> <over|-> <bf> <of> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat>
//!        -> decode(cached) + compose(track1(-1,0,4)->pip->grade->look(0,0,0)) + feed the
//!           composited f32 frame to the encoder at ts = enc_count/fps; reply DONE/ERR; no file.
//!     CLOSE
//!        -> finish + close the encoder; reply DONE.
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
    // Initialize OpenCL exactly once for the lifetime of the process. If this fails the
    // worker is useless; exit non-zero so the client's restart logic can react.
    let gpu = match ffi::Gpu::init() {
        Some(g) => g,
        None => {
            eprintln!("FAIL: fpx_gpu_init failed in --serve");
            std::process::exit(4);
        }
    };

    // One open decoder per media path, reused across requests (held playhead / repeated frames).
    let mut decoders: HashMap<String, ffi::Decoder> = HashMap::new();

    // Active render encoder (set by OPEN, fed by ENC, torn down by CLOSE). Holds the fps so
    // ENC can stamp ts = enc_count / fps; enc_count is the running frame counter for the job.
    let mut enc: Option<ffi::Encoder> = None;
    let mut enc_fps: f64 = 30.0;
    let mut enc_count: i64 = 0;

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
                if open_render(line, &mut enc, &mut enc_fps, &mut enc_count) {
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
            "CLOSE" => {
                let ok = match enc.take() {
                    Some(mut e) => e.finish(), // drop after this scope closes the encoder
                    None => false,
                };
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
/// (closed) without finishing, since a fresh OPEN supersedes it. Configures video (mpeg4) only
/// (no audio stream this slice — see findings #1/#2), then writes the header. Resets the counter.
fn open_render(
    line: &str,
    enc: &mut Option<ffi::Encoder>,
    enc_fps: &mut f64,
    enc_count: &mut i64,
) -> bool {
    let f: Vec<&str> = line.split_whitespace().collect();
    // OPEN <out> <w> <h> <fps>
    if f.len() != 5 {
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
    // VIDEO-ONLY this slice: the serve protocol has no audio-feed command, so configuring an
    // aac stream here would mux a ZERO-sample audio track that some players/tools choke on
    // (findings #1/#2). Deliberately skip config_audio for a clean video-only mp4; re-enable it
    // only alongside an AENC-style audio-feed command in the program-audio follow-up.
    if !e.start() {
        eprintln!("[gcompose] enc_start failed");
        return false;
    }

    *enc = Some(e);
    *enc_fps = fps as f64;
    *enc_count = 0;
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
