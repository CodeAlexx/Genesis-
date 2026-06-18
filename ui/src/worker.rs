//! Client to the `gcompose` engine worker (separate process; owns OpenCL — the egui process
//! never links/calls OpenCL).
//!
//! P1: a single PERSISTENT `gcompose --serve` process is started once and reused for every
//! frame. `request_frame(project, t)` resolves the program at timeline frame `t` from the model
//! (V1 base + V2 overlay, mirroring MojoMedia's rs0/rs1 logic), sends one request line to the
//! worker's stdin, waits for a `DONE <out>` line, reads the RGBA file back, and returns it.
//! On any I/O / protocol failure it restarts the worker (up to a few times) to absorb the
//! worker's small OpenCL-init flake.

use crate::model::Project;
use eframe::egui;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Mutex, OnceLock};

pub const PVW: usize = 1280;
pub const PVH: usize = 856;
const PREVIEW_RGBA: &str = "/tmp/genesis_frame.rgba"; // per-request output path
const MAX_ATTEMPTS: usize = 3;

/// A live `gcompose --serve` process plus its piped stdin/stdout.
struct WorkerProc {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Drop for WorkerProc {
    fn drop(&mut self) {
        // Best-effort: closing stdin makes the serve loop exit; then reap the child.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Process-global persistent worker. `None` until first spawned / after a failed restart.
static WORKER: OnceLock<Mutex<Option<WorkerProc>>> = OnceLock::new();

fn worker_slot() -> &'static Mutex<Option<WorkerProc>> {
    WORKER.get_or_init(|| Mutex::new(None))
}

/// Tear down the persistent worker (kills + reaps the gcompose child via WorkerProc::Drop).
/// Call before `std::process::exit`, which otherwise skips destructors and leaks the child.
pub fn shutdown() {
    if let Ok(mut slot) = worker_slot().lock() {
        *slot = None;
    }
}

/// Path to the sibling `gcompose` binary (same dir as the running UI executable).
fn worker_path() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join("gcompose"))
}

/// Spawn a fresh `gcompose --serve` with piped stdin/stdout.
fn spawn_worker() -> Option<WorkerProc> {
    let w = worker_path()?;
    let mut child = Command::new(&w)
        .arg("--serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // stderr stays inherited so the worker's [gcompose]/[gpu] logs reach the terminal.
        .spawn()
        .ok()?;
    let stdin = child.stdin.take()?;
    let stdout = BufReader::new(child.stdout.take()?);
    Some(WorkerProc { child, stdin, stdout })
}

/// One request/response round-trip against an already-running worker. Returns the RGBA bytes.
/// Any failure (write error, EOF, "ERR", short read) returns None so the caller can restart.
fn try_once(proc: &mut WorkerProc, req: &str) -> Option<Vec<u8>> {
    // Send the request line.
    proc.stdin.write_all(req.as_bytes()).ok()?;
    proc.stdin.write_all(b"\n").ok()?;
    proc.stdin.flush().ok()?;

    // Read response lines until DONE/ERR (skip any stray worker chatter that reached stdout).
    let mut resp = String::new();
    loop {
        resp.clear();
        let n = proc.stdout.read_line(&mut resp).ok()?;
        if n == 0 {
            return None; // worker closed stdout (crashed/exited).
        }
        let r = resp.trim();
        if r.is_empty() {
            continue;
        }
        if let Some(out_path) = r.strip_prefix("DONE ") {
            let bytes = std::fs::read(out_path.trim()).ok()?;
            if bytes.len() == PVW * PVH * 4 {
                return Some(bytes);
            }
            return None;
        }
        if r == "ERR" {
            return None;
        }
        // Unknown line: ignore and keep reading for the real response.
    }
}

/// Compose the program at timeline frame `t` -> RGBA8 PVW*PVH, via the persistent worker.
/// Restarts the worker (up to MAX_ATTEMPTS) on any failure to absorb its OpenCL-init flake.
pub fn request_frame(project: &Project, t: i64) -> Option<Vec<u8>> {
    let req = build_request(project, t)?;

    let slot = worker_slot();
    let mut guard = slot.lock().ok()?;

    for attempt in 0..MAX_ATTEMPTS {
        // Ensure a worker exists.
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            if let Some(bytes) = try_once(proc, &req) {
                return Some(bytes);
            }
        }
        // Failed: drop (kills) the current worker so the next loop spawns a clean one.
        *guard = None;
        eprintln!("gcompose serve attempt {} failed; restarting worker", attempt + 1);
    }
    None
}

/// Resolve the program at timeline frame `t` from the model and format the worker request line.
///
/// Mirrors MojoMedia main_editor.mojo (lines ~592-631): the base is the topmost clip on track 0
/// (V1) covering `t`; the overlay is the clip on track 1 (V2) covering `t`. Source frame for a
/// clip is `clip.src_in + (t - clip.t0)`. The PiP composite is only enabled when BOTH a V1 base
/// and a V2 overlay cover `t`. If no clip covers `t`, fall back to media[0] @ frame 0.
fn build_request(project: &Project, t: i64) -> Option<String> {
    // Topmost (last-wins, matching Mojo's >= track scan) clip on track 0 and track 1 covering t.
    let mut s0: Option<&crate::model::Clip> = None;
    let mut s1: Option<&crate::model::Clip> = None;
    for c in &project.clips {
        if t >= c.t0 && t < c.end() {
            match c.track {
                0 => s0 = Some(c),
                1 => s1 = Some(c),
                _ => {} // track 2 = audio; ignored for the video program.
            }
        }
    }

    // Base = V1 if present, else V2 shown directly (matches Mojo: s = s0 else s1).
    let base_clip = s0.or(s1);

    let (base_path, base_frame) = match base_clip {
        Some(c) => {
            let path = project.media.get(c.media)?;
            let frame = (c.src_in + (t - c.t0)) as i32;
            (path.clone(), frame.max(0))
        }
        None => {
            // No clip covers t: fall back to the first media @ frame 0.
            let path = project.media.first()?;
            (path.clone(), 0)
        }
    };

    // Overlay only when V1 is the base AND V2 is present (Mojo: over_v2 = s0>=0 && s1>=0).
    let over_v2 = s0.is_some() && s1.is_some();
    let (over_path, over_frame, op, px, py, pw, ph) = if over_v2 {
        let c = s1.unwrap();
        match project.media.get(c.media) {
            Some(p) => {
                let frame = (c.src_in + (t - c.t0)) as i32;
                (p.clone(), frame.max(0), 1.0f32, c.px, c.py, c.pw, c.ph)
            }
            None => ("-".to_string(), 0, 0.0, 0.0, 0.0, 1.0, 1.0),
        }
    } else {
        ("-".to_string(), 0, 0.0, 0.0, 0.0, 1.0, 1.0)
    };

    // 13 space-separated fields, matching gcompose's serve protocol exactly.
    Some(format!(
        "{base} {over} {bf} {of} {op} {px} {py} {pw} {ph} {b} {c} {s} {out}",
        base = base_path,
        over = over_path,
        bf = base_frame,
        of = over_frame,
        op = op,
        px = px,
        py = py,
        pw = pw,
        ph = ph,
        b = project.bright,
        c = project.contrast,
        s = project.sat,
        out = PREVIEW_RGBA,
    ))
}

/// Upload an RGBA8 PVW×PVH buffer as an egui texture (GL — needs a live context).
pub fn rgba_to_texture(ctx: &egui::Context, buf: &[u8]) -> egui::TextureHandle {
    let img = egui::ColorImage::from_rgba_unmultiplied([PVW, PVH], buf);
    ctx.load_texture("preview", img, egui::TextureOptions::LINEAR)
}
