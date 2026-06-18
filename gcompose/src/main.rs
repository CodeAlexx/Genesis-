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
//! Request line (space-separated, exactly 13 fields):
//!   <base_path> <over_path|-> <base_frame> <over_frame> <op> <px> <py> <pw> <ph> <bright> <contrast> <sat> <out_path>
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

        match handle_request(&gpu, &mut decoders, line) {
            Some(out_path) => {
                let _ = writeln!(stdout, "DONE {out_path}");
            }
            None => {
                let _ = writeln!(stdout, "ERR");
            }
        }
        // Always flush so the client (blocking on a single response line) unblocks promptly.
        let _ = stdout.flush();
    }
}

/// Parse + execute one serve request line. Returns the out_path on success, None on any failure.
fn handle_request(
    gpu: &ffi::Gpu,
    decoders: &mut HashMap<String, ffi::Decoder>,
    line: &str,
) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
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

    // Decode base @ base_frame (cached decoder per path), upload to slot 0.
    let base_rgba = decode_cached(decoders, base_path, base_frame)?;
    gpu.upload(0, &base_rgba);

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
