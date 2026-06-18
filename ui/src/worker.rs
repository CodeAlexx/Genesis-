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

/// Outcome of a single command round-trip, distinguishing the worker's two failure shapes:
///   - `Done(payload)`   : worker replied "DONE [payload]" — success.
///   - `Err`             : worker replied "ERR" — the COMMAND failed, but the worker is alive and
///                          the encoder/session is intact (e.g. AUDIO range with no audio). The
///                          caller may continue the session without restarting.
///   - `Broken`          : write error / EOF / protocol break — the worker is in an unknown state
///                          (likely dead) and must be dropped + (maybe) respawned.
enum CmdStatus {
    Done(String),
    Err,
    Broken,
}

/// One request/response round-trip returning the full tri-state status (see `CmdStatus`). This is
/// the lowest-level transport; `try_command` wraps it for the common DONE/None callers, while
/// `render_program` drives it directly on the held proc for the whole OPEN→ENC→AUDIO→CLOSE
/// sequence (finding #1) so it can (a) keep one lock across the render and (b) treat a plain `ERR`
/// on AUDIO as skip-this-clip without tearing the live worker/encoder down mid-render.
fn try_command_status(proc: &mut WorkerProc, req: &str) -> CmdStatus {
    if proc.stdin.write_all(req.as_bytes()).is_err()
        || proc.stdin.write_all(b"\n").is_err()
        || proc.stdin.flush().is_err()
    {
        return CmdStatus::Broken;
    }

    let mut resp = String::new();
    loop {
        resp.clear();
        match proc.stdout.read_line(&mut resp) {
            Ok(0) => return CmdStatus::Broken, // worker closed stdout (crashed/exited).
            Ok(_) => {}
            Err(_) => return CmdStatus::Broken,
        }
        let r = resp.trim();
        if r.is_empty() {
            continue;
        }
        if r == "DONE" {
            return CmdStatus::Done(String::new());
        }
        if let Some(payload) = r.strip_prefix("DONE ") {
            return CmdStatus::Done(payload.trim().to_string());
        }
        if r == "ERR" {
            return CmdStatus::Err;
        }
        // Unknown chatter on stdout: ignore and keep reading for the real response.
    }
}

/// One request/response round-trip that resolves to the DONE-echoed payload (the text after
/// "DONE ", trimmed) — or an empty String when DONE carries no payload. Returns None on any
/// failure (write error, EOF, "ERR", protocol break) so the caller can restart the worker.
///
/// Unlike `try_once`, this does NOT read or length-check any RGBA file — it is the generic
/// command transport for OPEN/ENC/CLOSE/THUMB/ENV, whose outputs vary in size (or are absent).
fn try_command(proc: &mut WorkerProc, req: &str) -> Option<String> {
    match try_command_status(proc, req) {
        CmdStatus::Done(p) => Some(p),
        CmdStatus::Err | CmdStatus::Broken => None,
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
        // NOTE (finding #3): unlike `command_with_restart` (which arms the cooldown ONCE, only after
        // exhausting all attempts, so its known one-off OpenCL-init flake can retry cleanly within
        // the same call), the preview path arms it on EVERY failed attempt. That is intentional and
        // harmless: a success returns at the `return Some(bytes)` above before reaching here, so the
        // cooldown is only ever stamped on a genuine per-attempt failure — and stamping it eagerly
        // means a dead-worker preview burst (many cache-miss frames in one repaint) starts failing
        // fast a little sooner.
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
///   OPEN <out> PVW PVH 30 <total_s>                (config video + aac, alloc audio accumulator)
///   for t in 0..total_frames:  ENC <resolved frame fields>   (composite + feed video encoder)
///   for each audible clip:  AUDIO <media> <src_in/FPS> <len/FPS> <t0/FPS> 1.0   (mix at offset)
///   CLOSE                                          (drain accumulator -> encoder; write BOTH)
///
/// CONCURRENCY (finding #1): the ENTIRE OPEN→ENC*→AUDIO*→CLOSE sequence runs under ONE hold of the
/// worker mutex (`worker_slot().lock()`), driving the held `WorkerProc` via `try_command_status`
/// directly — it never re-enters the per-call `command_*` helpers (which each re-lock). This
/// guarantees that no concurrent worker consumer (an egui repaint calling `request_frame` /
/// `thumbnail`) can interleave PREVIEW/THUMB traffic on the worker's stdin/stdout *between* this
/// render's commands — which would otherwise let the render's `read_line` consume a preview's
/// `DONE` as its own ENC ack, or spawn a fresh encoder-less worker mid-render. The whole render is
/// now atomic with respect to the worker; concurrent callers simply block on the mutex until it
/// completes. The trade-off (the UI thread stalls for the render's duration if it shares the lock)
/// is acceptable for this slice — render is an explicit, blocking user action.
///
/// PROGRAM AUDIO (Slice A — TIMELINE-SYNCED): after all video frames are encoded, the program audio
/// is assembled timeline-positioned. For each AUDIBLE timeline clip, an AUDIO command tells the
/// worker to decode the clip's SOURCE range [src_in/FPS, (src_in+len)/FPS) and MIX it into the
/// worker's program-audio accumulator at dst_offset = t0/FPS seconds. The accumulator was sized to
/// the full timeline duration (total_s in OPEN), so on CLOSE the whole buffer is fed to the encoder
/// and the audio stream length MATCHES the video: a clip at t0=70 starts at 70/FPS s, gaps are
/// silence, and overlapping clips mix (sample-add, clamped). This replaces the old back-to-back
/// concatenation (which ignored t0 and made audio duration != video duration).
///
/// SILENT AUDIO-DROP CAVEAT (finding #6, INTEGRATOR NOTE): a render that returns `true` does NOT
/// guarantee an audio stream. On a minimal FFmpeg build with no aac encoder, OPEN's config_audio
/// fails NON-FATALLY in the worker (logged on the worker's stderr as "config_audio failed; rendering
/// video-only") and the render proceeds video-only — every AUDIO line still mixes into the
/// accumulator but CLOSE discards it. The serve protocol has no DONE field to report this back, so
/// `render_program` cannot distinguish a with-audio render from a video-only one; the only signal is
/// the worker's stderr. Acceptable for the minimal-FFmpeg fallback this slice, but the integrator
/// must not assume render-true ⇒ audio-present.
///
/// TRACK POLICY (NEW this slice — NOT a MojoMedia mirror): the AUDIBLE tracks are track 0 (V1, the
/// program base) and track 2 (A1, the dedicated audio track). Track 1 (V2 overlay) is INTENTIONALLY
/// SKIPPED — the assumption is its audio would duplicate the underlying V1 audio. NOTE for the
/// integrator: MojoMedia's own render audio assembly does NOT filter by track (it sums every
/// segment regardless of lane), so this V2-skip is a deliberate editorial decision here, and any
/// legitimately-distinct V2 audio is silently dropped — revisit if a different track model is wanted.
/// ADDITIONALLY (Slice A, pinned by the contract): any clip on a track for which
/// `project.is_muted(track)` is true contributes NO audio (the AUDIO line is not emitted) — that
/// mute-honoring IS required by the contract.
///
/// Does NOT auto-restart the worker mid-render: a restart would lose the open encoder, so any
/// fatal step (ENC failure, worker death during AUDIO/CLOSE) aborts the whole job. On abort, a
/// best-effort CLOSE (`abort_held`) tears the half-open encoder down immediately on the same held
/// proc so its partial mp4 is dropped now, not lazily on the next OPEN/exit (finding #8).
/// RESILIENCE LIMITATION: a single worker flake anywhere in the video pass aborts the entire render
/// with the partial discarded — there is no per-frame retry / re-OPEN (finding #4), so long renders
/// have zero fault tolerance. Acceptable for this slice. Returns true if video + CLOSE all
/// succeeded (a skipped clip's audio does not fail the render).
pub fn render_program(project: &Project, out_path: &str) -> bool {
    let total = project.total_frames();
    if total <= 0 {
        return false;
    }

    // Build every request line BEFORE taking the lock so a corrupt media index fails the render
    // up-front without ever opening an encoder (and without holding the worker mutex while we
    // touch the model). If any ENC line can't be resolved, abort before OPEN.
    let mut enc_lines: Vec<String> = Vec::with_capacity(total as usize);
    for t in 0..total {
        match build_enc_line(project, t) {
            Some(r) => enc_lines.push(r),
            None => return false, // corrupt media index: nothing opened yet, just bail.
        }
    }
    let audio_lines = build_audio_lines(project);

    // Acquire the worker for the WHOLE render (finding #1): one lock hold spanning OPEN→CLOSE, so
    // no concurrent preview/thumbnail can interleave on the worker's pipes mid-render.
    let slot = worker_slot();
    let mut guard = match slot.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };

    // Ensure a live worker (spawning if needed; respawn once on the known OpenCL-init flake). Done
    // INLINE on the held guard rather than via command_with_restart so we keep the lock the whole
    // render. The render's first command is OPEN, which has no encoder in flight yet, so a respawn
    // here is safe.
    // Total timeline duration in seconds (sizes the worker's program-audio accumulator so the
    // rendered audio is exactly the timeline length — see render_program docs / Slice A).
    let total_s = total as f64 / RENDER_FPS as f64;
    let mut opened = false;
    let open_req = format!("OPEN {out_path} {PVW} {PVH} {RENDER_FPS} {total_s}");
    for attempt in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            match try_command_status(proc, &open_req) {
                CmdStatus::Done(_) => {
                    clear_spawn_cooldown();
                    opened = true;
                    break;
                }
                CmdStatus::Err => {
                    // OPEN itself failed (e.g. bad out path) but the worker is alive: no encoder
                    // was created, so there is nothing to tear down. Don't respawn — fail clean.
                    return false;
                }
                CmdStatus::Broken => {
                    // Worker died on OPEN: drop it and retry the spawn (absorb the init flake).
                    *guard = None;
                    mark_spawn_fail();
                    eprintln!("gcompose OPEN attempt {} failed; restarting worker", attempt + 1);
                }
            }
        } else {
            // Spawn failed outright.
            *guard = None;
            mark_spawn_fail();
        }
    }
    if !opened {
        return false;
    }

    // From here an encoder is open: any Broken/Err on ENC or a Broken on AUDIO/CLOSE must tear the
    // encoder down on THIS proc (never respawn — that loses the encoder). `proc()` re-borrows the
    // live worker out of the guard each step; if it ever vanished (Broken set it to None) we abort.

    // ENC every frame, in order, on the held proc.
    for req in &enc_lines {
        let alive = match guard.as_mut() {
            Some(proc) => matches!(try_command_status(proc, req), CmdStatus::Done(_)),
            None => false,
        };
        if !alive {
            // ENC failed (encoder error -> Err) or the worker died (Broken). Tear the half-open
            // encoder down NOW (finding #8) on whatever proc is still in the guard, then abort.
            // `&mut *guard` derefs the MutexGuard to the inner Option the helper expects.
            eprintln!("[render] ENC aborted at: {req}");
            abort_held(&mut *guard);
            return false;
        }
    }

    // PROGRAM AUDIO: feed each AUDIBLE clip's source-audio range, in t0 (timeline) order. A clip
    // with no audio / a decode skip (worker ERR) is dropped but the render continues; only a worker
    // death (Broken / vanished proc) aborts. This uses MojoMedia's per-segment
    // fpx_decode_audio_range building block, but is NOT a 1:1 mirror of its assembly: MojoMedia's
    // render path concatenates EVERY segment's audio back-to-back with no track filtering, whereas
    // this path positions each clip at its timeline offset (dst_offset) AND applies the NEW
    // track-audibility policy below (track-2/track-0 only; track_mute honored). See TRACK POLICY.
    for line in &audio_lines {
        let outcome = match guard.as_mut() {
            Some(proc) => try_command_status(proc, line),
            None => CmdStatus::Broken,
        };
        match outcome {
            CmdStatus::Done(_) => clear_spawn_cooldown(),
            CmdStatus::Err => {} // worker alive; skip just this clip's audio and continue.
            CmdStatus::Broken => {
                // The worker died feeding audio: the encoder is gone. Drop the dead proc, arm the
                // cooldown, and fail the render (nothing left to CLOSE).
                *guard = None;
                mark_spawn_fail();
                return false;
            }
        }
    }

    // CLOSE — finish + close the encoder, flush + write BOTH the video and audio trailers.
    let ok = match guard.as_mut() {
        Some(proc) => match try_command_status(proc, "CLOSE") {
            CmdStatus::Done(_) => {
                clear_spawn_cooldown();
                true
            }
            CmdStatus::Err => false, // encoder reported a finish failure; worker still alive.
            CmdStatus::Broken => {
                *guard = None;
                mark_spawn_fail();
                false
            }
        },
        None => false,
    };
    ok
}

/// Temp WAV the worker writes program audio to for playback (see `play_program`).
const PLAY_WAV: &str = "/tmp/genesis_play.wav";

/// Best-effort, non-blocking timeline-audio playback from `start_frame` (Slice A).
///
/// CHOSEN PATH (stated for the integrator): rather than vendoring PulseAudio into the worker, this
/// reuses the SAME program-audio mixing as the render path to write a PCM WAV, then spawns the
/// system player (`paplay`, falling back to `aplay`) on it detached. This keeps gcompose free of any
/// new audio-device link (no libpulse), and survives a missing player binary gracefully. The
/// trade-off vs. fpx_aplay is no live position/VU feedback and no instant scrub-restart — acceptable
/// for a best-effort audition this slice.
///
/// THREADING (finding #1): the WAVE→AUDIO*→WAVECLOSE assembly decodes+mixes the ENTIRE timeline tail
/// from the playhead and holds the worker mutex for its full duration — seconds for a long timeline,
/// during which it also blocks any concurrent `request_frame`/`thumbnail` on the same mutex. This
/// function is called from the egui UI thread (app.rs owns the playhead), so doing that work inline
/// would stall the UI. Instead, only the CHEAP, model-touching part — resolving the audible clips
/// into owned `AUDIO ...` command strings — runs on the calling thread (no decode, no lock); the
/// owned lines are then moved onto a detached `std::thread::spawn` that takes the lock, drives the
/// WAVE session, writes the WAV, and spawns the detached player. So `play_program` RETURNS
/// IMMEDIATELY (returning `true` to mean "playback was dispatched", not "audio is already audible");
/// any worker/spawn failure is logged on the background thread, not surfaced to the caller. The
/// background thread owns its own `Vec<String>` + `String` (no borrow of `project`), so there is no
/// lifetime tie to the UI's project.
///
/// Mechanism: run a `WAVE <wav> <dur_s>` / `AUDIO*` / `WAVECLOSE <wav>` session on the persistent
/// worker. The accumulator is sized to the timeline tail from `start_frame` (so the WAV begins at
/// the playhead). Each clip is mapped into playhead-relative time: clips ending at/before the
/// playhead are dropped; a clip straddling the playhead has its source in-point ADVANCED and its
/// duration shortened by the already-played head so it plays from the source frame under the
/// playhead at dst_offset 0; clips after the playhead keep their source range at dst_offset =
/// (t0 - start)/FPS. Returns true if playback was dispatched (a background thread was spawned);
/// false only when there is nothing to play (empty timeline / playhead at/after the end).
pub fn play_program(project: &Project, start_frame: i64) -> bool {
    let total = project.total_frames();
    let start = start_frame.max(0);
    if total <= 0 || start >= total {
        return false; // nothing to play at/after the timeline end.
    }
    let fps = RENDER_FPS as f64;
    // Duration of the audio to assemble = timeline tail from the playhead.
    let tail_frames = total - start;
    let tail_s = tail_frames as f64 / fps;
    let start_s = start as f64 / fps;

    // Build the WAVE/AUDIO*/WAVECLOSE lines NOW, on the calling (UI) thread — this is the only part
    // that touches `project`, and it is cheap (no decode, no lock). dst_offset is shifted so the
    // playhead is t=0 in the WAV. The resulting owned Vec<String> is moved into the worker thread
    // below, so the heavy decode/mix never borrows `project`.
    let wave_open = format!("WAVE {PLAY_WAV} {tail_s}");
    let mut audio_lines: Vec<String> = Vec::new();
    {
        let mut idx: Vec<usize> = (0..project.clips.len()).collect();
        idx.sort_by_key(|&i| project.clips[i].t0);
        for i in idx {
            let c = &project.clips[i];
            if c.len <= 0 {
                continue;
            }
            if !track_is_audible(project, c.track) {
                continue;
            }
            // Skip clips that end at/before the playhead (entirely in the past).
            if c.end() <= start {
                continue;
            }
            let media_path = match project.media.get(c.media) {
                Some(p) => p,
                None => continue,
            };
            // Same whitespace-path filter (and now the same diagnostic) as build_audio_lines
            // (finding #8): a space in the path would shift the AUDIO line's fixed-arity fields, so
            // skip it — and LOG it, for parity with the render path which already logs the skip.
            if media_path.split_whitespace().count() != 1 {
                eprintln!("gcompose: skipping playback audio for media path with whitespace: {media_path}");
                continue;
            }
            // For a clip straddling the playhead (t0 < start), skip the already-played head: advance
            // the SOURCE in-point by `start - t0` frames and shorten the duration by the same, so the
            // clip plays from the source frame under the playhead at dst_offset 0. For a clip wholly
            // after the playhead, src_in/len are unchanged and dst_offset is its forward distance.
            let head_skip = (start - c.t0).max(0); // frames of this clip already behind the playhead
            let eff_src_in = c.src_in + head_skip; // frames
            let eff_len = c.len - head_skip; // frames remaining to play
            if eff_len <= 0 {
                continue;
            }
            let src_in_s = eff_src_in as f64 / fps;
            let dur_s = eff_len as f64 / fps;
            // Timeline position relative to the playhead (>= 0 by construction: head_skip clamps it).
            let dst_off_s = ((c.t0 + head_skip) as f64 / fps - start_s).max(0.0);
            audio_lines.push(format!("AUDIO {media_path} {src_in_s} {dur_s} {dst_off_s} 1.0"));
        }
    }

    // Hand the owned command lines to a detached background thread so the UI thread returns at once
    // (finding #1). The thread takes the worker lock, assembles the WAV, and spawns the player; its
    // failures are logged there, not returned here.
    std::thread::spawn(move || {
        if assemble_and_play(&wave_open, &audio_lines) {
            // WAV written: launch the detached system player on it (best-effort).
            spawn_player(PLAY_WAV);
        }
    });

    true // playback dispatched (the background thread owns the rest).
}

/// Worker side of `play_program`, run on a detached background thread (finding #1). Assembles the
/// playback WAV by driving a WAVE→AUDIO*→WAVECLOSE session on the persistent worker under ONE lock
/// hold (no auto-restart mid-session: a restart would lose the accumulator). Any `Broken` tears the
/// worker down + arms the cooldown; a per-clip `ERR` is skipped. Returns true if the WAV was written.
/// Takes only OWNED data (`&str` / `&[String]` into thread-local Strings) so it never borrows the
/// UI's `Project`.
fn assemble_and_play(wave_open: &str, audio_lines: &[String]) -> bool {
    let slot = worker_slot();
    let mut guard = match slot.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };

    // Ensure a live worker for the WAVE (no in-flight session yet, so a respawn here is safe).
    let mut opened = false;
    for _ in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            match try_command_status(proc, wave_open) {
                CmdStatus::Done(_) => {
                    clear_spawn_cooldown();
                    opened = true;
                    break;
                }
                CmdStatus::Err => return false, // WAVE rejected (bad dur): worker alive, bail.
                CmdStatus::Broken => {
                    *guard = None;
                    mark_spawn_fail();
                }
            }
        } else {
            *guard = None;
            mark_spawn_fail();
        }
    }
    if !opened {
        return false;
    }

    // Mix each clip (skip ERR clips; a Broken kills the session).
    for line in audio_lines {
        match guard.as_mut() {
            Some(proc) => match try_command_status(proc, line) {
                CmdStatus::Done(_) => {}
                CmdStatus::Err => {} // no audio in range / decode skip: drop this clip.
                CmdStatus::Broken => {
                    *guard = None;
                    mark_spawn_fail();
                    return false;
                }
            },
            None => return false,
        }
    }

    // WAVECLOSE -> write the WAV.
    match guard.as_mut() {
        Some(proc) => match try_command_status(proc, &format!("WAVECLOSE {PLAY_WAV}")) {
            CmdStatus::Done(_) => {
                clear_spawn_cooldown();
                true
            }
            CmdStatus::Err => false,
            CmdStatus::Broken => {
                *guard = None;
                mark_spawn_fail();
                false
            }
        },
        None => false,
    }
}

/// Spawn a detached audio player on `wav`. Tries `paplay` then `aplay`. Returns true if one was
/// launched. The child is fully detached (its stdio is /dev/null) and not waited on — playback runs
/// independently of the UI. Best-effort: false if neither binary exists.
fn spawn_player(wav: &str) -> bool {
    for bin in ["paplay", "aplay"] {
        let spawned = Command::new(bin)
            .arg(wav)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        if spawned.is_ok() {
            return true;
        }
    }
    eprintln!("gcompose: no audio player (paplay/aplay) found; playback skipped");
    false
}

/// Best-effort teardown of a half-open render encoder after `render_program` aborts mid-ENC, using
/// the ALREADY-HELD worker guard (finding #1 + #8): we never re-lock and never spawn a worker just
/// to send CLOSE (finding #8). If the proc is still live we send a single CLOSE so the worker drops
/// the encoder (and its partial mp4) immediately; if the worker already died we just clear the slot
/// so the next top-level call spawns clean.
fn abort_held(guard: &mut Option<WorkerProc>) {
    if let Some(proc) = guard.as_mut() {
        match try_command_status(proc, "CLOSE") {
            CmdStatus::Done(_) | CmdStatus::Err => {} // encoder torn down; worker stays alive.
            CmdStatus::Broken => {
                // Worker died during teardown: drop it + arm the cooldown.
                *guard = None;
                mark_spawn_fail();
            }
        }
    }
}

/// True if track `t` contributes to the program audio given the project's mute flags. Track 0 (V1)
/// and track 2 (A1) are AUDIBLE by default; track 1 (V2 overlay) is NEVER audible (its audio would
/// duplicate V1). A track muted via `project.is_muted(t)` (Team C's per-track mute, Slice A)
/// contributes nothing. See the TRACK POLICY note on `render_program`.
fn track_is_audible(project: &Project, t: u8) -> bool {
    if !matches!(t, 0 | 2) {
        return false; // V2 overlay never contributes program audio.
    }
    !project.is_muted(t) // Team C accessor (bounds-checked; 0=V1,1=V2,2=A1).
}

/// Build the timeline-synced `AUDIO <media_path> <src_in_s> <dur_s> <dst_offset_s> <gain>` lines for
/// the program audio, one per AUDIBLE clip (track 0 V1 + track 2 A1, honoring `track_mute`; track 1
/// V2 skipped). Emitted in timeline (`t0`) order for determinism, though order no longer affects the
/// result now that the worker mixes by destination offset rather than concatenating.
///
/// Per the slice contract: `src_in_s = clip.src_in / FPS` (source in-point), `dur_s = clip.len / FPS`
/// (timeline length), `dst_offset_s = clip.t0 / FPS` (timeline position), `gain = 1.0`. FPS is the
/// render framerate (30), matching `RENDER_FPS`. Clips with non-positive length, a corrupt media
/// index, a non-audible/muted track, or whitespace in the media path are skipped (no line emitted —
/// a whitespace path would break the worker's fixed-arity AUDIO parse, so it is filtered here).
fn build_audio_lines(project: &Project) -> Vec<String> {
    // Sort clip indices by timeline start for deterministic, readable output (order-independent now
    // that the worker positions by dst_offset). Stable on t0; ties keep the project's clip order.
    let mut idx: Vec<usize> = (0..project.clips.len()).collect();
    idx.sort_by_key(|&i| project.clips[i].t0);

    let fps = RENDER_FPS as f64;
    let mut lines = Vec::new();
    for i in idx {
        let c = &project.clips[i];
        if c.len <= 0 {
            continue;
        }
        // Track policy + Slice A mute: skip V2 overlay and any muted track.
        if !track_is_audible(project, c.track) {
            continue;
        }
        let media_path = match project.media.get(c.media) {
            Some(p) => p,
            None => continue, // corrupt media index: skip this clip's audio (don't abort).
        };
        // The AUDIO line is whitespace-split with fixed arity on the worker; a path containing a
        // space would shift the numeric fields. Pool paths are space-free in practice, but skip
        // (rather than corrupt the render) if one ever isn't.
        if media_path.split_whitespace().count() != 1 {
            eprintln!("gcompose: skipping audio for media path with whitespace: {media_path}");
            continue;
        }
        let src_in_s = c.src_in as f64 / fps;
        let dur_s = c.len as f64 / fps;
        let dst_off_s = c.t0 as f64 / fps;
        let gain = 1.0f32;
        lines.push(format!("AUDIO {media_path} {src_in_s} {dur_s} {dst_off_s} {gain}"));
    }
    lines
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
