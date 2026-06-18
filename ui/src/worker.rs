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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Preview surface resolution. These MUST equal the engine's OpenCL working resolution
// (gcompose ffi::GVW/GVH = 1280×856): the worker always composes at GVW×GVH and returns exactly
// PVW*PVH*4 bytes (see `try_once`'s length check) / GVW*GVH*4 floats per ENC frame. The render
// path (`render_program`) is decoupled — the engine pins the encoder dims to its own GVW/GVH and
// ignores the OPEN wire w/h (finding #7) — but the PREVIEW path still length-checks against
// PVW/PVH, so if PVW/PVH ever change here they MUST be changed to match the engine's GVW/GVH too,
// or every preview frame fails the byte-count check. ffi lives in the gcompose crate and is not
// importable from ui, so this invariant is enforced by convention, not a static assert.
pub const PVW: usize = 1280;
pub const PVH: usize = 856;
const PREVIEW_RGBA: &str = "/tmp/genesis_frame.rgba"; // per-request output path
const MAX_ATTEMPTS: usize = 3;

/// After a worker spawn/handshake fails, suppress further (re)spawns for this long. A single
/// egui repaint can miss many thumbnails at once; without this, each cache-miss would pay up to
/// MAX_ATTEMPTS fresh `gcompose --serve` spawns (each re-initing OpenCL) — a spawn/init storm
/// within one frame (finding #6). During the cooldown, `command_with_restart` fails fast (None)
/// instead of re-spawning, so the dead worker is retried at most once per cooldown window.
const SPAWN_COOLDOWN: Duration = Duration::from_millis(750);

/// Unix-millis of the last failed worker spawn/handshake (0 = none yet). Used to gate respawns
/// across the stateless THUMB/ENV/OPEN path so one dead-worker repaint can't storm the OS.
static LAST_SPAWN_FAIL_MS: AtomicU64 = AtomicU64::new(0);

/// Current Unix time in millis (monotonic enough for a short cooldown; never panics).
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// True if we are still inside the post-failure spawn cooldown (so we should fail fast).
fn in_spawn_cooldown() -> bool {
    let last = LAST_SPAWN_FAIL_MS.load(Ordering::Relaxed);
    last != 0 && now_ms().saturating_sub(last) < SPAWN_COOLDOWN.as_millis() as u64
}

/// Mark a fresh spawn failure (starts/refreshes the cooldown window).
fn mark_spawn_fail() {
    LAST_SPAWN_FAIL_MS.store(now_ms(), Ordering::Relaxed);
}

/// Clear the cooldown after any successful round-trip (worker is healthy again).
fn clear_spawn_cooldown() {
    LAST_SPAWN_FAIL_MS.store(0, Ordering::Relaxed);
}

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

/// One request/response round-trip that resolves to the DONE-echoed payload (the text after
/// "DONE ", trimmed) — or an empty String when DONE carries no payload. Returns None on any
/// failure (write error, EOF, "ERR", protocol break) so the caller can restart the worker.
///
/// Unlike `try_once`, this does NOT read or length-check any RGBA file — it is the generic
/// command transport for OPEN/ENC/CLOSE/THUMB/ENV, whose outputs vary in size (or are absent).
fn try_command(proc: &mut WorkerProc, req: &str) -> Option<String> {
    proc.stdin.write_all(req.as_bytes()).ok()?;
    proc.stdin.write_all(b"\n").ok()?;
    proc.stdin.flush().ok()?;

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
        if r == "DONE" {
            return Some(String::new());
        }
        if let Some(payload) = r.strip_prefix("DONE ") {
            return Some(payload.trim().to_string());
        }
        if r == "ERR" {
            return None;
        }
        // Unknown chatter on stdout: ignore and keep reading for the real response.
    }
}

/// Run one command on the persistent worker WITHOUT auto-restart. Render (ENC) sequences must
/// NOT silently restart mid-job — a restarted worker would lose the open encoder, so a failed
/// ENC must abort the whole render. Returns the DONE payload (possibly empty) or None.
fn command_no_restart(req: &str) -> Option<String> {
    let slot = worker_slot();
    let mut guard = slot.lock().ok()?;
    if guard.is_none() {
        *guard = spawn_worker();
    }
    let proc = guard.as_mut()?;
    match try_command(proc, req) {
        Some(p) => {
            clear_spawn_cooldown(); // healthy round-trip: allow normal respawns again.
            Some(p)
        }
        None => {
            // The worker is now in an unknown state (possibly dead): drop it so the next
            // top-level call spawns a clean one, and start the cooldown.
            *guard = None;
            mark_spawn_fail();
            None
        }
    }
}

/// Run one stateless command on the persistent worker WITH auto-restart (up to MAX_ATTEMPTS).
/// Suitable for THUMB / ENV / OPEN (idempotent, no in-flight encoder to lose).
///
/// Respects the post-failure spawn cooldown (finding #6): if a recent spawn failed and there is
/// no live worker to reuse, this fails fast (None) rather than paying another OpenCL-init spawn —
/// so one repaint that misses many thumbnails against a dead worker can't trigger a spawn storm.
fn command_with_restart(req: &str) -> Option<String> {
    let slot = worker_slot();
    let mut guard = slot.lock().ok()?;

    // Entry gate (finding #6): if a previous call already exhausted its retries against a
    // known-dead worker AND there is no live worker to reuse, fail fast for the cooldown window
    // instead of paying another full retry/OpenCL-init storm. A live worker (guard.is_some()) is
    // always tried regardless of cooldown — the cooldown only suppresses fresh respawn attempts.
    if guard.is_none() && in_spawn_cooldown() {
        return None;
    }

    // In-call retry loop: still does the full MAX_ATTEMPTS respawn-and-retry to absorb the
    // worker's known one-off OpenCL-init flake (the whole reason this machinery exists).
    for attempt in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            if let Some(payload) = try_command(proc, req) {
                clear_spawn_cooldown(); // healthy round-trip: allow normal respawns again.
                return Some(payload);
            }
        }
        // This attempt failed: drop the (now-suspect) worker so the next attempt spawns clean.
        *guard = None;
        eprintln!("gcompose command attempt {} failed; restarting worker", attempt + 1);
    }

    // All in-call attempts exhausted: arm the cooldown so subsequent misses in this same repaint
    // fail fast (no per-thumbnail respawn storm) until the window elapses and we retry once more.
    mark_spawn_fail();
    None
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
                clear_spawn_cooldown(); // healthy round-trip: allow normal respawns again.
                return Some(bytes);
            }
        }
        // Failed: drop (kills) the current worker so the next loop spawns a clean one.
        *guard = None;
        mark_spawn_fail();
        eprintln!("gcompose serve attempt {} failed; restarting worker", attempt + 1);
    }
    None
}

/// The resolved program at timeline frame `t`: base + optional overlay + composite params.
/// Shared by the preview request and the render ENC line so both bake the identical composite.
struct Resolved {
    base_path: String,
    base_frame: i32,
    over_path: String, // "-" when no overlay
    over_frame: i32,
    op: f32,
    px: f32,
    py: f32,
    pw: f32,
    ph: f32,
}

/// Resolve the program at timeline frame `t` from the model.
///
/// Mirrors MojoMedia main_editor.mojo (lines ~592-631 preview; ~1225-1301 render): the base is
/// the topmost clip on track 0 (V1) covering `t`; the overlay is the clip on track 1 (V2)
/// covering `t`. Source frame for a clip is `clip.src_in + (t - clip.t0)`. The PiP composite is
/// only enabled when BOTH a V1 base and a V2 overlay cover `t`. If no clip covers `t` (a timeline
/// gap), the base path is the "-" sentinel, which the engine fills with a black frame (matching
/// MojoMedia's black-gap behavior). Returns None only on a corrupt media index.
fn resolve_frame(project: &Project, t: i64) -> Option<Resolved> {
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
            // No clip covers t (timeline gap): emit the "-" base sentinel so the engine fills
            // the frame with black, matching MojoMedia's black-gap behavior — rather than
            // freeze-framing media[0]@0 (finding #5). The engine treats base "-" as the
            // all-black slot-0 frame; this never returns None for a gap on a valid project.
            ("-".to_string(), 0)
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

    Some(Resolved {
        base_path,
        base_frame,
        over_path,
        over_frame,
        op,
        px,
        py,
        pw,
        ph,
    })
}

/// Resolve frame `t` and format the preview request line: an explicit `PREVIEW` keyword followed
/// by the 13 positional fields (with out path). The keyword removes the latent dispatch ambiguity
/// where a media path whose first token happened to equal a command keyword (OPEN/ENC/...) could
/// misroute a preview frame to the wrong handler (finding #3); the engine now matches `PREVIEW`
/// explicitly and never falls through to keyword-guessing for a real frame request.
fn build_request(project: &Project, t: i64) -> Option<String> {
    let r = resolve_frame(project, t)?;
    // PREVIEW + 13 space-separated fields, matching gcompose's serve protocol exactly.
    Some(format!(
        "PREVIEW {base} {over} {bf} {of} {op} {px} {py} {pw} {ph} {b} {c} {s} {out}",
        base = r.base_path,
        over = r.over_path,
        bf = r.base_frame,
        of = r.over_frame,
        op = r.op,
        px = r.px,
        py = r.py,
        pw = r.pw,
        ph = r.ph,
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

/// Program render framerate. Matches gcompose's OPEN default and MojoMedia's render config (30).
const RENDER_FPS: i32 = 30;

/// Render the whole program to `out_path` (mp4) via the persistent worker.
///
/// Sequence (mirrors MojoMedia's render loop, driven over the serve protocol):
///   OPEN <out> PVW PVH 30
///   for t in 0..total_frames:  ENC <resolved frame fields>   (composite + feed encoder)
///   CLOSE
///
/// VIDEO-ONLY: the slice's serve protocol defines OPEN/ENC/CLOSE/THUMB/ENV but no audio-feed
/// command, so program audio (MojoMedia's per-clip fpx_decode_audio_range -> enc_audio_samples)
/// is NOT muxed here. The encoder still configures an aac stream in OPEN; adding audio is a
/// follow-up that needs a new ENV-style audio-feed command. Returns true if every step DONEd.
///
/// Does NOT auto-restart the worker mid-render: a restart would lose the open encoder, so any
/// failed step aborts the whole job. On abort, a best-effort CLOSE (`abort_render`) tears the
/// half-open encoder down immediately so its partial mp4 is dropped now, not lazily on the next
/// OPEN/exit (finding #8). RESILIENCE LIMITATION: a single worker flake anywhere in 0..total
/// aborts the entire render with the partial discarded — there is no per-frame retry / re-OPEN
/// (finding #4), so long renders have zero fault tolerance. Acceptable for this slice.
pub fn render_program(project: &Project, out_path: &str) -> bool {
    let total = project.total_frames();
    if total <= 0 {
        return false;
    }

    // OPEN — idempotent enough to allow a restart (no encoder in flight yet on first try).
    let open_req = format!("OPEN {out_path} {PVW} {PVH} {RENDER_FPS}");
    if command_with_restart(&open_req).is_none() {
        return false;
    }

    // ENC every frame, in order. No restart: a mid-render worker restart loses the encoder.
    for t in 0..total {
        let req = match build_enc_line(project, t) {
            Some(r) => r,
            None => {
                // Unresolved frame (corrupt media index) -> abort. Best-effort CLOSE so the
                // worker tears the half-open encoder down NOW (finding #8) instead of lazily on
                // the next OPEN/exit; the partial mp4 is then dropped immediately.
                abort_render();
                return false;
            }
        };
        if command_no_restart(&req).is_none() {
            // ENC failed (worker flake / encoder error) -> abort. Best-effort CLOSE as above.
            abort_render();
            return false;
        }
    }

    // CLOSE — finish + close the encoder, write the trailer.
    command_no_restart("CLOSE").is_some()
}

/// Best-effort teardown of a half-open render encoder after `render_program` aborts mid-ENC.
/// Sends CLOSE without restart so the worker drops the encoder (and its partial mp4) immediately
/// rather than leaving it alive until the next OPEN or process exit (finding #8). If the worker
/// already died (slot is None / EOF), there is nothing to tear down and this is a no-op.
fn abort_render() {
    let _ = command_no_restart("CLOSE");
}

/// Format the `ENC ...` line for timeline frame `t` (12 payload fields, no out path), baking the
/// same composite as the preview. Returns None when the frame can't be resolved.
fn build_enc_line(project: &Project, t: i64) -> Option<String> {
    let r = resolve_frame(project, t)?;
    Some(format!(
        "ENC {base} {over} {bf} {of} {op} {px} {py} {pw} {ph} {b} {c} {s}",
        base = r.base_path,
        over = r.over_path,
        bf = r.base_frame,
        of = r.over_frame,
        op = r.op,
        px = r.px,
        py = r.py,
        pw = r.pw,
        ph = r.ph,
        b = project.bright,
        c = project.contrast,
        s = project.sat,
    ))
}

/// Decode one frame of `media_path` letterboxed to `w*h` -> RGBA8 (`w*h*4` bytes), via the
/// worker's THUMB command (no composite). Returns None on failure. Used for pool/clip thumbs.
pub fn thumbnail(media_path: &str, frame: i64, w: usize, h: usize) -> Option<Vec<u8>> {
    if w == 0 || h == 0 {
        return None;
    }
    let out = thumb_temp_path(media_path, frame, w, h);
    let req = format!("THUMB {media_path} {frame} {w} {h} {out}");
    let payload = command_with_restart(&req)?;
    // Worker echoes the out path on DONE; trust our own path if it echoes empty.
    let read_path = if payload.is_empty() { out.clone() } else { payload };
    let bytes = std::fs::read(&read_path).ok()?;
    if bytes.len() == w * h * 4 {
        Some(bytes)
    } else {
        None
    }
}

/// Per-track peak audio envelope (`buckets` peaks in 0..1) of `media_path`, via the worker's ENV
/// command. The worker writes `buckets` little-endian f32 to a temp file; we read them back.
/// Returns None if the file has no audio / on any failure. Used for waveform display.
pub fn audio_envelope(media_path: &str, buckets: usize) -> Option<Vec<f32>> {
    if buckets == 0 {
        return None;
    }
    let out = env_temp_path(media_path, buckets);
    let req = format!("ENV {media_path} {buckets} {out}");
    let payload = command_with_restart(&req)?;
    let read_path = if payload.is_empty() { out.clone() } else { payload };
    let bytes = std::fs::read(&read_path).ok()?;
    if bytes.len() != buckets * 4 {
        return None;
    }
    let mut env = Vec::with_capacity(buckets);
    for chunk in bytes.chunks_exact(4) {
        env.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(env)
}

/// A stable-ish temp path for a thumbnail of `path` @ `frame` at `w×h`. Hashing the inputs keeps
/// concurrent THUMB requests for different media/frames from clobbering each other's output file.
fn thumb_temp_path(path: &str, frame: i64, w: usize, h: usize) -> String {
    format!("/tmp/genesis_thumb_{:x}.rgba", small_hash(&format!("{path}|{frame}|{w}|{h}")))
}

/// A temp path for the envelope of `path` @ `buckets`.
fn env_temp_path(path: &str, buckets: usize) -> String {
    format!("/tmp/genesis_env_{:x}.f32", small_hash(&format!("{path}|{buckets}")))
}

/// Tiny FNV-1a hash for building collision-resistant temp filenames (no extra deps).
fn small_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
