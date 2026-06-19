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
use ab_glyph::{point, Font, FontVec, PxScale, ScaleFont};
use eframe::egui;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================ P5 TITLE TEXT RASTERIZER ============================
//
// A per-clip TEXT overlay (Shotcut "Text: Simple" / dynamictext). A clip whose `Clip.title` is
// non-empty has its text rasterized into a full-frame GVW×GVH transparent RGBA8 layer here, written
// to a cached temp file, and fed to the engine on the existing base/over path as the `RAW:<path>`
// sentinel (see `resolve_frame`). The engine reads that raw buffer straight into the slot (skipping
// decode) and composites it with OVER alpha — so a V2 title clip shows its text over the V1 base,
// and a title-only clip shows text over black. A project with NO titles never produces a `RAW:`
// path, so the render is byte-identical to the pre-P5 output (identity).
//
// FONT: loaded ONCE (lazy `OnceLock`) from the bundled `ui/assets/title_font.ttf` — located by
// mirroring icons.rs's asset-path search (beside the exe, `<exe>/assets`, a few parents up, and the
// dev `ui/assets`). If the bundled asset can't be found/parsed we fall back to the system
// LiberationSans. When NO font can be loaded, `rasterize_title` returns None (the title is dropped;
// the clip composites normally) rather than failing the frame.

/// The engine compose canvas (== gcompose ffi::GVW/GVH and worker PVW/PVH). The title is rasterized
/// into a GVW×GVH×4 RGBA8 buffer so the engine can upload it directly to a slot via the `RAW:` path.
const TITLE_W: usize = PVW; // 1280
const TITLE_H: usize = PVH; // 856

/// System-font fallback when the bundled `title_font.ttf` asset is missing (matches the contract's
/// pinned fallback path). LiberationSans is the same family bundled at ui/assets/title_font.ttf.
const FALLBACK_FONT_PATH: &str = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf";

/// Process-global, lazily-loaded title font. `None` only if neither the bundled asset nor the system
/// fallback could be read/parsed; the rasterizer then yields None (the clip composites without text).
static TITLE_FONT: OnceLock<Option<FontVec>> = OnceLock::new();

/// Candidate filesystem paths for the bundled `title_font.ttf`, in priority order — mirrors
/// `icons.rs::candidate_paths` (beside the running exe, `<exe>/assets`, a couple of parents up into a
/// sibling `assets`/`ui/assets`, then the dev-tree `ui/assets`/`assets` relative to the cwd). The
/// system fallback is appended LAST so a deployed build with no bundled asset still finds a font.
fn title_font_candidates() -> Vec<std::path::PathBuf> {
    const FONT_NAME: &str = "title_font.ttf";
    let mut dirs: Vec<std::path::PathBuf> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            dirs.push(exe_dir.to_path_buf());
            dirs.push(exe_dir.join("assets"));
            let mut up = exe_dir.to_path_buf();
            for _ in 0..3 {
                if let Some(parent) = up.parent() {
                    up = parent.to_path_buf();
                    dirs.push(up.join("assets"));
                    dirs.push(up.join("ui").join("assets"));
                } else {
                    break;
                }
            }
        }
    }
    // Dev fallbacks relative to the working directory.
    dirs.push(std::path::PathBuf::from("ui/assets"));
    dirs.push(std::path::PathBuf::from("assets"));

    let mut paths: Vec<std::path::PathBuf> = dirs.into_iter().map(|d| d.join(FONT_NAME)).collect();
    // System-font fallback last (the pinned LiberationSans path).
    paths.push(std::path::PathBuf::from(FALLBACK_FONT_PATH));
    paths
}

/// Load (once) and return the bundled title font, or None if no font file could be read/parsed.
/// Tries each candidate in turn; the first that both reads AND parses as a valid font wins.
fn title_font() -> Option<&'static FontVec> {
    TITLE_FONT
        .get_or_init(|| {
            for p in title_font_candidates() {
                match std::fs::read(&p) {
                    Ok(bytes) => match FontVec::try_from_vec(bytes) {
                        Ok(font) => {
                            eprintln!("[title] loaded font: {}", p.display());
                            return Some(font);
                        }
                        Err(_) => {
                            eprintln!("[title] font parse failed (trying next): {}", p.display());
                        }
                    },
                    Err(_) => {} // not at this candidate; try the next.
                }
            }
            eprintln!("[title] no title font found (bundled or system); titles will not render");
            None
        })
        .as_ref()
}

/// Stable hash of a title's RENDERED inputs, for the cache filename. Two titles that rasterize to the
/// same pixels share the same `/tmp/genesis_title_<hash>.rgba` file (and the same upload). Uses the
/// same FNV-1a as the thumbnail temp paths. The text + every layout/colour field is folded in so a
/// change to any of them keys a fresh file.
fn title_hash(title: &crate::model::Title) -> u64 {
    let key = format!(
        "{}|{}|{}|{}|{}|{}|{}",
        title.text,
        title.size_frac.to_bits(),
        title.x.to_bits(),
        title.y.to_bits(),
        title.rgb[0].to_bits(),
        title.rgb[1].to_bits(),
        title.rgb[2].to_bits(),
    );
    small_hash(&key)
}

/// Rasterize `title.text` into a full-frame (`TITLE_W`×`TITLE_H`) transparent RGBA8 layer and write
/// it to a CACHED temp file `/tmp/genesis_title_<hash>.rgba`, returning that path. The engine reads
/// the file via the `RAW:` sentinel and composites it (OVER alpha) over the program.
///
/// Layout (Shotcut Text: Simple-ish): the font pixel height is `title.size_frac * TITLE_H`; the pen
/// origin is at `(title.x*TITLE_W, title.y*TITLE_H + ascent)` so `title.y` anchors the TOP of the
/// text. Glyphs advance the pen by their scaled `h_advance`; a `\n` starts a new line (pen x reset,
/// y advanced by the scaled line height). Each glyph is outlined and `draw`n: every covered pixel is
/// written `rgb = title.rgb*255`, `alpha = coverage*255`, MAX-blended so overlapping glyph outlines
/// don't darken the overlap. Pixels outside the frame are clipped.
///
/// Returns None when the text is empty (the caller then composites the clip normally — no `RAW:`
/// path, identity render) or when no font could be loaded.
///
/// CACHING: if the keyed file already exists, the raster is skipped and the existing path returned —
/// a held playhead / a long render on the same title reuses the file (the engine re-reads it cheaply;
/// it never changes for the same inputs). A write failure returns None (the title is dropped, never
/// fails the frame).
fn rasterize_title(title: &crate::model::Title) -> Option<String> {
    if title.is_empty() {
        return None; // no text: clip composites normally (identity).
    }
    let path = format!("/tmp/genesis_title_{:x}.rgba", title_hash(title));
    // Cache hit: the keyed file already holds this exact raster — reuse it (skip re-rasterizing).
    if std::path::Path::new(&path).exists() {
        return Some(path);
    }

    let font = title_font()?; // no font available: drop the title (composite normally).

    // Font pixel height from the normalized size fraction; clamp to a sane positive range so a stray
    // 0/huge size_frac can't produce a zero-area or runaway raster.
    let px = (title.size_frac * TITLE_H as f32).clamp(4.0, TITLE_H as f32);
    let scale = PxScale::from(px);
    let scaled = font.as_scaled(scale);
    let ascent = scaled.ascent();
    let line_height = scaled.height(); // ascent - descent + line_gap (scaled)

    // Full-frame, fully TRANSPARENT RGBA8 (alpha 0 everywhere) — the engine composites it OVER the
    // program, so only the drawn glyph pixels (alpha > 0) show.
    let mut buf = vec![0u8; TITLE_W * TITLE_H * 4];

    let r = (title.rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (title.rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (title.rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;

    // Pen origin: x = title.x*W (left of the text), y = title.y*H + ascent (so title.y anchors the
    // TEXT TOP, with the baseline one ascent below it). Multi-line: each '\n' resets pen_x and
    // advances pen_y by the scaled line height.
    let origin_x = title.x * TITLE_W as f32;
    let top_y = title.y * TITLE_H as f32;
    let mut pen_x = origin_x;
    let mut pen_y = top_y + ascent;

    for ch in title.text.chars() {
        if ch == '\n' {
            pen_x = origin_x;
            pen_y += line_height;
            continue;
        }
        let glyph_id = scaled.glyph_id(ch);
        let advance = scaled.h_advance(glyph_id);
        // Build a positioned glyph at the current pen, then outline it. Whitespace / glyphs with no
        // outline (e.g. ' ') just advance the pen.
        let mut glyph = glyph_id.with_scale(scale);
        glyph.position = point(pen_x, pen_y);
        if let Some(outline) = font.outline_glyph(glyph) {
            // px_bounds gives the integer pixel rect of the outline in the buffer's coordinate space;
            // draw yields (gx, gy) RELATIVE to that rect's top-left + a coverage in [0,1].
            let bounds = outline.px_bounds();
            let ox = bounds.min.x as i32;
            let oy = bounds.min.y as i32;
            outline.draw(|gx, gy, coverage| {
                if coverage <= 0.0 {
                    return;
                }
                let x = ox + gx as i32;
                let y = oy + gy as i32;
                if x < 0 || y < 0 || x >= TITLE_W as i32 || y >= TITLE_H as i32 {
                    return; // off-frame: clip.
                }
                let idx = (y as usize * TITLE_W + x as usize) * 4;
                let a = (coverage.clamp(0.0, 1.0) * 255.0).round() as u8;
                // MAX-blend the alpha so overlapping glyph outlines never darken the overlap; write
                // the (solid) colour wherever this glyph contributes more alpha than what's there.
                if a > buf[idx + 3] {
                    buf[idx] = r;
                    buf[idx + 1] = g;
                    buf[idx + 2] = b;
                    buf[idx + 3] = a;
                }
            });
        }
        pen_x += advance;
    }

    if std::fs::write(&path, &buf).is_err() {
        eprintln!("[title] failed to write raster cache: {path}");
        return None; // drop the title rather than fail the frame.
    }
    Some(path)
}

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

/// Scope image dimensions (PINNED, Slice A). The engine's scope kernels always render a fixed
/// 256×256 RGBA8 image (gcompose ffi::SVW/SVH; the histogram is rasterized into the same size), so
/// `scope()` always reads back exactly SW*SH*4 bytes. Team C consumes these for the scopes panel.
pub const SW: usize = 256;
pub const SH: usize = 256;

const PREVIEW_RGBA: &str = "/tmp/genesis_frame.rgba"; // per-request output path
/// Per-request output path for a rendered scope image (256×256 RGBA8). Reused each call — `scope()`
/// holds the worker mutex across its PREVIEW+SCOPE round-trip, so there is never a concurrent writer.
const SCOPE_RGBA: &str = "/tmp/genesis_scope.rgba";
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

/// True while a `play_program` background thread is assembling + playing the audition WAV
/// (finding #9). Set before the thread is spawned and cleared when it finishes. A second Space
/// press while one playback is already in flight is dropped (no-op) rather than stacking another
/// detached thread that would block on the worker mutex and then spawn a duplicate `paplay`. This
/// dedups rapid presses and bounds the number of concurrent background players to one.
static PLAYBACK_IN_FLIGHT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Stop-edge coordination flag (findings #7 + #8). `stop_playback` sets this true; `play_program`
/// clears it at the moment it dispatches a fresh audition. The detached assembly thread checks it
/// right BEFORE spawning the system player and SKIPS the spawn if a stop arrived during the
/// (seconds-long) WAVE/AUDIO assembly window — so a `stop_playback` fired before `spawn_player` ran
/// no longer leaks a late, unkillable player. Without this, the only stop mechanism was killing an
/// already-spawned child (finding #7), which does nothing for a player that has not been spawned yet.
static STOP_REQUESTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The currently-playing detached audition player child (`paplay`/`aplay`), if any. `spawn_player`
/// stores the spawned `Child` here so `stop_playback` (the PINNED API) can kill it on demand —
/// instead of leaving the audio to run to its natural EOF. `None` when nothing is playing (no child
/// spawned yet, or the previous one was already killed/finished). Guarded by its own mutex,
/// independent of the worker mutex, so `stop_playback` can fire instantly even while the worker is
/// busy composing/assembling. See `stop_playback` for the kill mechanism + pkill fallback.
static PLAYER_CHILD: OnceLock<Mutex<Option<Child>>> = OnceLock::new();

fn player_slot() -> &'static Mutex<Option<Child>> {
    PLAYER_CHILD.get_or_init(|| Mutex::new(None))
}

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

/// How long a single worker reply may take before we treat the worker as WEDGED (finding #4).
///
/// The ~10% flake the slice targets is a worker process CRASH (the driver segfaults), which the OS
/// surfaces as an immediate EOF on the pipe → `Broken` → retry. But a flaky OpenCL driver can also
/// HANG (deadlock in the queue) rather than crash; a blocking `read_line` would then wait FOREVER,
/// holding the worker mutex and freezing the egui UI thread (which blocks on the same mutex for the
/// whole render). To bound that, every reply is read with this timeout: if no line arrives in time,
/// the read returns `Broken` so the worker is torn down + retried instead of hanging the export.
///
/// The value must comfortably exceed the slowest LEGITIMATE single reply. The heaviest per-command
/// work is one ENC frame (decode + OpenCL composite + one encoded frame) or the CLOSE drain (mux +
/// trailer); both are well under a second on any working GPU. 30 s is generous headroom so a merely
/// slow-but-alive worker is never killed, while a true deadlock is caught in bounded time.
const REPLY_TIMEOUT: Duration = Duration::from_secs(30);

/// A line read from the worker's stdout, or the EOF marker.
enum WorkerLine {
    Line(String),
    Eof,
}

/// A live `gcompose --serve` process plus its piped stdin and a background reader.
///
/// READER THREAD (finding #4): the worker's stdout is drained by a dedicated thread that pushes each
/// line (trimmed) onto `rx` and finally an `Eof` marker when the pipe closes. The request path then
/// reads replies with `recv_timeout(REPLY_TIMEOUT)` so a WEDGED worker (driver deadlock — no crash,
/// no EOF) is detected as `Broken` in bounded time rather than blocking the held worker mutex (and
/// the egui UI) forever. A crash still surfaces promptly as `Eof` (pipe closed). The thread owns the
/// `BufReader<ChildStdout>`; it exits on EOF or when the child is killed in `Drop`.
struct WorkerProc {
    child: Child,
    stdin: ChildStdin,
    rx: Receiver<WorkerLine>,
}

impl Drop for WorkerProc {
    fn drop(&mut self) {
        // Best-effort: closing stdin makes the serve loop exit, killing the child closes its stdout
        // which unblocks + ends the reader thread (it then drops its channel sender). We do not join
        // the reader thread (it is detached): once the child's stdout closes, its blocking read
        // returns and the thread exits on its own.
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

/// Spawn a fresh `gcompose --serve` with piped stdin/stdout, plus a background reader thread that
/// feeds stdout lines over a channel so the request path can read replies WITH a timeout (finding
/// #4: a wedged worker must not block the held mutex / UI forever).
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
    let stdout = child.stdout.take()?;

    // Reader thread: owns the BufReader, pushes each line onto the channel, then an Eof marker when
    // the pipe closes (worker exited/crashed). It exits when the child's stdout closes — which Drop
    // forces by killing the child. The send may fail if the receiver was dropped (proc torn down);
    // in that case the thread just stops. Detached: never explicitly joined.
    let (tx, rx) = mpsc::channel::<WorkerLine>();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut buf = String::new();
        loop {
            buf.clear();
            match reader.read_line(&mut buf) {
                Ok(0) => {
                    let _ = tx.send(WorkerLine::Eof);
                    break; // worker closed stdout (crashed/exited).
                }
                Ok(_) => {
                    if tx.send(WorkerLine::Line(buf.trim().to_string())).is_err() {
                        break; // receiver gone (proc dropped): stop reading.
                    }
                }
                Err(_) => {
                    let _ = tx.send(WorkerLine::Eof);
                    break; // read error: treat as EOF.
                }
            }
        }
    });

    Some(WorkerProc { child, stdin, rx })
}

/// Read one reply line from the worker with `REPLY_TIMEOUT` (finding #4). Returns the line, or None
/// if the worker hit EOF (crash/exit), the reader channel disconnected, OR no line arrived in time
/// (a wedged worker — driver deadlock). In every None case the worker is considered Broken and the
/// caller tears it down + retries instead of blocking forever.
fn read_reply(proc: &mut WorkerProc) -> Option<String> {
    match proc.rx.recv_timeout(REPLY_TIMEOUT) {
        Ok(WorkerLine::Line(l)) => Some(l),
        Ok(WorkerLine::Eof) => None,             // worker closed stdout (crashed/exited).
        Err(RecvTimeoutError::Timeout) => {
            eprintln!("gcompose: worker reply timed out after {REPLY_TIMEOUT:?} (wedged worker?)");
            None
        }
        Err(RecvTimeoutError::Disconnected) => None, // reader thread gone.
    }
}

/// One request/response round-trip against an already-running worker. Returns the RGBA bytes.
/// Any failure (write error, EOF, "ERR", short read) returns None so the caller can restart.
fn try_once(proc: &mut WorkerProc, req: &str) -> Option<Vec<u8>> {
    // Send the request line.
    proc.stdin.write_all(req.as_bytes()).ok()?;
    proc.stdin.write_all(b"\n").ok()?;
    proc.stdin.flush().ok()?;

    // Read response lines until DONE/ERR (skip any stray worker chatter that reached stdout). Reads
    // are bounded by REPLY_TIMEOUT (finding #4): a wedged worker yields None instead of blocking.
    loop {
        let r = read_reply(proc)?; // None on EOF/timeout/disconnect -> caller restarts.
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

    // Reads are bounded by REPLY_TIMEOUT (finding #4): EOF (crash), a read error, OR a reply that
    // never arrives in time (a wedged/deadlocked worker) all map to `Broken` so the render-retry
    // machinery tears the worker down + respawns rather than blocking the held mutex forever.
    loop {
        let r = match read_reply(proc) {
            Some(l) => l,
            None => return CmdStatus::Broken,
        };
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

/// PINNED (Slice A): render a scope of the program frame at timeline frame `t`.
///   kind 0 = histogram, 1 = luma waveform, 2 = vectorscope, 3 = RGB parade (Triad-B P1).
/// Returns the rendered scope as `SW*SH*4` (256×256×4) RGBA8 bytes, or None on any failure.
///
/// Mechanism: send a `PREVIEW` line for frame `t` (composites the frame on the GPU — identical to
/// `request_frame`, leaving the result in the worker's persistent g_buf[OUTB]), then a
/// `SCOPE <kind> <out>` line (runs the scope kernel on that buffer and writes a 256×256 RGBA image),
/// then read the image back. BOTH lines run under ONE hold of the worker mutex (finding #1 style):
/// the lock guarantees no concurrent `request_frame`/`thumbnail`/render can interleave another
/// compose between our PREVIEW and our SCOPE — which would otherwise make the scope read a different
/// frame than the one we composed. On a worker failure the whole PREVIEW+SCOPE pair is retried on a
/// fresh worker (up to MAX_ATTEMPTS), matching the preview path's OpenCL-init-flake absorption.
///
/// The PREVIEW's own RGBA output (PREVIEW_RGBA) is composed-and-discarded here — we only need its
/// side effect (the composed GPU buffer); the bytes we return are the SCOPE image, not the frame.
///
/// NON-BLOCKING LOCK (finding #4): this is called from the egui UI thread (panels::scopes_ui) on
/// every repaint. It takes the worker lock with `try_lock`, NOT a blocking `lock`: while a
/// background audio assembly (`assemble_and_play`) or a render holds the worker for seconds, a
/// blocking acquire here would FREEZE the UI for that whole duration on every repaint. With
/// `try_lock`, a contended scope simply returns None this frame and the caller keeps showing its
/// last scope image — the panel goes momentarily stale instead of the whole UI hanging. The 30 s
/// `REPLY_TIMEOUT` only bounds a wedged WORKER; it does nothing for lock contention, so this is the
/// piece that actually protects the UI thread during a long-running background worker operation.
pub fn scope(project: &Project, t: i64, kind: u8) -> Option<Vec<u8>> {
    if kind > 3 {
        return None; // unknown scope kind: nothing to render (0=hist,1=wave,2=vec,3=parade).
    }
    let preview_req = build_request(project, t)?;
    let scope_req = format!("SCOPE {kind} {SCOPE_RGBA}");

    let slot = worker_slot();
    // try_lock (finding #4): do NOT block the UI thread behind a background audio assembly / render.
    let mut guard = match slot.try_lock() {
        Ok(g) => g,
        Err(_) => return None, // worker busy: skip this frame's scope (caller keeps last image).
    };

    for attempt in 0..MAX_ATTEMPTS {
        // Ensure a worker exists.
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            // 1) PREVIEW: compose frame t (discard its RGBA file — we only want the GPU side effect).
            //    try_once reads PREVIEW's DONE and length-checks PVW*PVH*4; a None means the worker
            //    is broken, so we fall through to the restart below.
            if try_once(proc, &preview_req).is_some() {
                // 2) SCOPE: run the kernel on the just-composed buffer and read back the image.
                if let Some(img) = read_scope(proc, &scope_req) {
                    clear_spawn_cooldown(); // healthy round-trip.
                    return Some(img);
                }
            }
        }
        // Either PREVIEW or SCOPE failed: drop (kills) the worker so the next loop spawns clean.
        *guard = None;
        mark_spawn_fail();
        eprintln!("gcompose scope attempt {} failed; restarting worker", attempt + 1);
    }
    None
}

/// PINNED-adjacent (Slice A, finding #5): render ALL THREE scopes for the program frame at timeline
/// frame `t` in ONE round-trip — one PREVIEW compose followed by three SCOPE reads — returning
/// `[histogram, luma_waveform, vectorscope]`, each `SW*SH*4` RGBA8 bytes, or None on any failure.
///
/// WHY: a scopes panel showing all three scopes that called `scope()` three times per repaint would
/// trigger THREE identical PREVIEW composes of the same frame (plus three PREVIEW_RGBA file
/// write+read round-trips of ~4.4 MB each), since each `scope()` re-composes before its SCOPE. The
/// composed GPU buffer (g_buf[OUTB]) is stable between requests, so the frame only needs composing
/// ONCE; the three scope kernels all read that same buffer. This composes once then issues SCOPE 0,
/// 1, 2 back-to-back under a single lock hold, cutting the per-frame work from 3 composes to 1.
///
/// Uses `try_lock` for the same UI-freeze reason as `scope()` (finding #4): a contended call returns
/// None and the caller keeps its last images. On any worker failure mid-sequence the WHOLE
/// PREVIEW+3×SCOPE is retried on a fresh worker (a respawn re-composes from scratch, so partial
/// progress is never reused). Team C should prefer this over three separate `scope()` calls.
pub fn scope_all(project: &Project, t: i64) -> Option<[Vec<u8>; 3]> {
    let preview_req = build_request(project, t)?;
    let scope_reqs = [
        format!("SCOPE 0 {SCOPE_RGBA}"),
        format!("SCOPE 1 {SCOPE_RGBA}"),
        format!("SCOPE 2 {SCOPE_RGBA}"),
    ];

    let slot = worker_slot();
    // try_lock (finding #4): never block the UI thread behind a background assembly / render.
    let mut guard = match slot.try_lock() {
        Ok(g) => g,
        Err(_) => return None, // worker busy: skip this frame's scopes (caller keeps last images).
    };

    for attempt in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            // 1) ONE PREVIEW compose of frame t (finding #5): all three scopes read this buffer.
            if try_once(proc, &preview_req).is_some() {
                // 2) Three SCOPE reads on the same composed buffer. Read them into a fixed-size
                //    array; any short/failed read aborts the whole attempt (restart on a fresh
                //    worker rather than returning a torn mix of old/new scope images).
                let mut imgs: [Option<Vec<u8>>; 3] = [None, None, None];
                let mut ok = true;
                for (i, req) in scope_reqs.iter().enumerate() {
                    match read_scope(proc, req) {
                        Some(img) => imgs[i] = Some(img),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    // SAFETY: every slot is Some when ok (the loop only completes without break in
                    // that case). Unwraps are infallible here.
                    let [h, w, v] = imgs;
                    clear_spawn_cooldown(); // healthy round-trip.
                    return Some([h.unwrap(), w.unwrap(), v.unwrap()]);
                }
            }
        }
        // PREVIEW or one of the SCOPE reads failed: drop (kills) the worker so the next loop spawns
        // clean and re-composes from scratch.
        *guard = None;
        mark_spawn_fail();
        eprintln!("gcompose scope_all attempt {} failed; restarting worker", attempt + 1);
    }
    None
}

/// Run one `SCOPE` round-trip on an already-running worker and read back the 256×256 RGBA image.
/// Mirrors `try_once` but length-checks against the SCOPE image size (SW*SH*4) instead of the
/// preview frame size. Returns None on any failure (write error, EOF/timeout, "ERR", short read).
fn read_scope(proc: &mut WorkerProc, req: &str) -> Option<Vec<u8>> {
    proc.stdin.write_all(req.as_bytes()).ok()?;
    proc.stdin.write_all(b"\n").ok()?;
    proc.stdin.flush().ok()?;

    loop {
        let r = read_reply(proc)?; // None on EOF/timeout/disconnect -> caller restarts.
        if r.is_empty() {
            continue;
        }
        if let Some(out_path) = r.strip_prefix("DONE ") {
            let bytes = std::fs::read(out_path.trim()).ok()?;
            if bytes.len() == SW * SH * 4 {
                return Some(bytes);
            }
            return None; // wrong size: treat as a failed render.
        }
        if r == "ERR" {
            return None;
        }
        // Unknown chatter on stdout: ignore and keep reading for the real response.
    }
}

/// The resolved program at timeline frame `t`: base + optional overlay + composite params.
/// Shared by the preview request and the render ENC line so both bake the identical composite.
///
/// `px/py/pw/ph` are the KEYFRAMED over-clip PiP rect (from `project.pip_at`) and `bright/contrast/
/// sat` are the KEYFRAMED grade (from `project.grade_at`) at this timeline frame — both evaluated
/// once here so the preview and the render emit byte-identical composite params (Slice A).
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
    bright: f32,
    contrast: f32,
    sat: f32,
    // Per-clip COLOR grade from the BASE (visible) clip (Triad-B P1), ADDITIVE on top of the program
    // grade above. Same kernel semantics: `cbright` added (−1..1), `ccontrast`/`csat` multipliers
    // (0..2, 1.0 = identity). DOCUMENTED COMBINE ORDER: gcompose runs the PER-CLIP grade FIRST (right
    // after the PiP composite), then the PROGRAM grade — so a clip with bright +0.3 lifts the picture
    // before the program grade is applied. A timeline gap (no base clip) is neutral (0/1/1). When a
    // transition is active the per-clip grade travels with the OUTGOING clip (matching the look).
    cbright: f32,
    ccontrast: f32,
    csat: f32,
    // Per-clip LOOK from the BASE (track-0/visible) clip (Slice A). `look_kind`: 0=None, 1=VHS,
    // 2=LUT3D. `look_amt` is the look mix (0..1). `lut_path` is the clip's `.cube` path for LUT3D,
    // or "-" when there is no LUT (look != 2 or an empty path). Both the preview (`build_request`)
    // and the render (`build_enc_line`) emit these, so a clip's look animates identically in the
    // preview and the export. Mirrors MojoMedia's playhead-segment look driving the grade pipeline.
    look_kind: i32,
    look_amt: f32,
    lut_path: String,
    // Per-boundary TRANSITION fields (Wave 8). Set by `resolve_frame` from `project.transition_at`
    // on the base track and the incoming clip the model's `boundaries()` pairs with the outgoing
    // (base) clip at this boundary. Forwarded to the engine (via `build_request`/`build_enc_line`)
    // as the 5 trailing wire fields BEFORE the out:
    //   trans_kind: -1 = no transition (engine skips slot 2, track1(-1,..) copies the base);
    //               0..7 = fpx_gpu transition id (0=crossfade..7=dissolve).
    //   trans_prog: progress in [0,1] across the transition window [center-dur/2, center+dur/2).
    //   trans_param: per-transition parameter (default 4.0, mirrors MojoMedia's tt_p default).
    //   trans_path: the INCOMING clip's media path, or "-" when there is no transition/incoming clip.
    //   trans_frame: the incoming clip's source frame at `t`, clamped to the incoming clip's valid
    //                source range; 0 when there is no transition.
    // So preview AND render bake the same animated transition.
    trans_kind: i32,
    trans_prog: f32,
    trans_param: f32,
    trans_path: String,
    trans_frame: i32,
    // ----- Triad-B P2 per-clip COLOR-WHEELS (LIFT/GAMMA/GAIN) + TRANSFORM + BLUR -----
    // Read from the BASE (visible) clip (or, during a transition, the OUTGOING clip — they travel
    // with the look/grade like the per-clip P1 grade). Forwarded to the engine as the 12 TRAILING
    // wire fields appended AFTER csat (PREVIEW: before `out`) in the PINNED order
    //   lift_r lift_g lift_b  gamma_r gamma_g gamma_b  gain_r gain_g gain_b  rot scale blur
    // so the engine's fpx_gpu_lgg / fpx_gpu_transform / fpx_gpu_blur kernels apply them identically
    // in preview and render. The 9 lift/gamma/gain values ALREADY HAVE white balance folded in (see
    // resolve_frame's wb fold) — the engine only ever sees lift/gamma/gain, never wb_temp/wb_tint.
    // A timeline gap (no base clip) sends IDENTITY (lift 0, gamma 1, gain 1, rot 0, scale 1, blur 0)
    // so the engine no-ops and reproduces the current output.
    lift: [f32; 3],
    gamma: [f32; 3],
    gain_rgb: [f32; 3],
    rot: f32,
    scale: f32,
    blur: f32,
    // ----- P4 per-clip CHROMA KEY (green-screen) on the OVER (V2) clip -----
    // Read from the OVER (V2/track-1) overlay clip's `Clip.chroma` (Team A reads it; never edits
    // model.rs). The engine keys the OVER buffer's alpha where the pixel matches the key colour, so
    // pip then shows V1 through the keyed pixels. Forwarded to the engine as the 6 TRAILING wire
    // fields appended AFTER blur (PREVIEW: before `out`) in the PINNED order
    //   ck_on ck_r ck_g ck_b ck_sim ck_smooth
    // so preview + render key identically. `ck_on` is 1 only when an over clip exists AND its
    // chroma.enabled is true; otherwise IDENTITY (ck_on=0, key green, sim 0.4, smooth 0.1) so the
    // engine no-ops and reproduces the P3 composite byte-for-byte. NB the chroma describes the OVER
    // clip, NOT the base/outgoing clip (it is the green-screen layer being keyed over V1).
    ck_on: i32,
    ck_key: [f32; 3],
    ck_sim: f32,
    ck_smooth: f32,
}

/// Fold a clip's white balance (`wb_temp`, `wb_tint`) INTO its 9 lift/gamma/gain values, returning
/// the white-balanced `(lift, gamma, gain_rgb)` triples. White balance is NOT a wire field (PINNED):
/// the engine only ever sees lift/gamma/gain, so the warm/cool + green/magenta bias is baked into the
/// GAIN channels here (multiplicative highlight gains are the natural place for a colour cast, matching
/// how Shotcut's white-balance maps a temperature onto the channel gains). `lift` and `gamma` pass
/// through unchanged — only `gain_rgb` is modulated.
///
/// FORMULA (both biases in [−1,1], 0 = neutral; `K_TEMP`/`K_TINT` keep a full-scale bias to a sane
/// ±gain so a maxed slider tints rather than blowing the channel out):
///   temp > 0 (WARMER): gain_r *= 1 + K_TEMP*temp ; gain_b *= 1 − K_TEMP*temp   (red up, blue down)
///   temp < 0 (COOLER): symmetric (red down, blue up — the same expression with temp negative)
///   tint > 0 (GREENER): gain_g *= 1 + K_TINT*tint ; gain_r *= 1 − 0.5*K_TINT*tint ;
///                       gain_b *= 1 − 0.5*K_TINT*tint   (green up, red+blue down → magenta drops)
///   tint < 0 (MAGENTA): symmetric.
/// Each resulting gain is clamped to [0, 4] (the engine/Shotcut gain range) so a combined
/// temp+tint push can never produce a negative or runaway multiplier.
fn fold_white_balance(
    lift: [f32; 3],
    gamma: [f32; 3],
    gain: [f32; 3],
    wb_temp: f32,
    wb_tint: f32,
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    const K_TEMP: f32 = 0.5; // full warm/cool bias scales a channel gain by up to ±50%.
    const K_TINT: f32 = 0.4; // full green/magenta bias scales the green gain by up to ±40%.
    let temp = wb_temp.clamp(-1.0, 1.0);
    let tint = wb_tint.clamp(-1.0, 1.0);

    let mut gr = gain[0] * (1.0 + K_TEMP * temp);
    let mut gg = gain[1];
    let mut gb = gain[2] * (1.0 - K_TEMP * temp);

    gg *= 1.0 + K_TINT * tint;
    gr *= 1.0 - 0.5 * K_TINT * tint;
    gb *= 1.0 - 0.5 * K_TINT * tint;

    let gain_wb = [gr.clamp(0.0, 4.0), gg.clamp(0.0, 4.0), gb.clamp(0.0, 4.0)];
    // lift + gamma pass through unchanged; only the highlight gains carry the colour cast.
    (lift, gamma, gain_wb)
}

/// Resolve the program at timeline frame `t` from the model.
///
/// Mirrors MojoMedia main_editor.mojo (lines ~592-631 preview; ~1225-1301 render): the base is
/// the topmost clip on track 0 (V1) covering `t`; the overlay is the clip on track 1 (V2)
/// covering `t`. Source frame for a clip is `clip.src_in + (t - clip.t0)`. The PiP composite is
/// only enabled when BOTH a V1 base and a V2 overlay cover `t`. If no clip covers `t` (a timeline
/// gap), the base path is the "-" sentinel, which the engine fills with a black frame (matching
/// MojoMedia's black-gap behavior). Returns None only on a corrupt media index.
///
/// KEYFRAME-AWARE (Slice A, consuming Team C's model API): the grade (bright/contrast/sat) is read
/// from `project.grade_at(t)` — the keyframed grade at timeline frame `t`, mirroring MojoMedia's
/// `kf_eval(...)` at the playhead/output frame (main_editor.mojo ~692-694 preview, ~1302-1304
/// render). The over-clip PiP rect is read from `project.pip_at(over_clip_idx, t - over_clip.t0)` —
/// the keyframed (px,py,pw,ph) at the CLIP-LOCAL frame, mirroring `pip_eval(... pip_lf ...)`
/// (~642-645 preview, ~1293-1296 render). Both accessors fall back to the clip/project STATIC
/// values when no keyframes exist for that track/clip, so a project with no keyframes resolves
/// identically to before. Because both the preview line (`build_request`) and the render line
/// (`build_enc_line`) flow through this one resolver, keyframed grade + PiP animate identically in
/// the preview and the export.
///
/// PER-CLIP LOOK (Slice A): the look is taken from the BASE (track-0/visible) clip — `look_kind =
/// clip.look` (0=None, 1=VHS, 2=LUT3D), `look_amt = clip.look_amt`, and `lut_path = clip.lut` for a
/// LUT3D look (else the "-" sentinel). Mirrors MojoMedia, whose playhead-segment look drives the
/// look pipeline. Flowing through this one resolver means the look applies in BOTH the preview and
/// the export. A timeline gap (no base clip) has no look.
fn resolve_frame(project: &Project, t: i64) -> Option<Resolved> {
    // Topmost (last-wins, matching Mojo's >= track scan) clip on track 0 and track 1 covering t.
    // We also remember the V2 (overlay) clip's INDEX, because `pip_at` keys PiP keyframes by clip
    // index (not by reference); the index mirrors MojoMedia's `s1` segment index fed to pip_eval.
    //
    // `s1` carries the V2 (overlay) clip AND its index together (finding #7): `pip_at` keys PiP
    // keyframes by clip index, so the index must travel with the clip. Storing them as one
    // `Option<(&Clip, usize)>` makes it impossible for the clip and its index to desync (the old
    // two-variable form relied on the `s1`/`s1_idx` assignments staying in lockstep, and used two
    // `.unwrap()`s downstream that a future edit could break).
    // P5 ARBITRARY TRACKS: generalize the old fixed V1=base / V2=over scan. The BASE is the covering
    // clip on the LOWEST visible VIDEO track; the OVERLAY is the covering clip on the HIGHEST visible
    // video track ABOVE the base. For the default V1/V2 layout this is exactly base=track0,
    // over=track1. Hidden video tracks + all audio tracks are skipped. (Engine is single-base +
    // single-over, so video layers strictly between base and over are not composited — true N-layer
    // compositing is a Stage-2 follow-up; bottom+top covers the overwhelmingly common case.)
    let mut base_tv: Option<(&crate::model::Clip, usize, u8)> = None; // lowest visible video
    let mut over_tv: Option<(&crate::model::Clip, usize, u8)> = None; // highest visible video
    for (i, c) in project.clips.iter().enumerate() {
        if t >= c.t0 && t < c.end() && !project.is_audio(c.track) && !project.is_hidden(c.track) {
            if base_tv.is_none_or(|(_, _, bt)| c.track <= bt) {
                base_tv = Some((c, i, c.track)); // <= => last-wins on a tie (transition overlap)
            }
            if over_tv.is_none_or(|(_, _, ot)| c.track >= ot) {
                over_tv = Some((c, i, c.track));
            }
        }
    }
    let s0: Option<&crate::model::Clip> = base_tv.map(|(c, _, _)| c);
    // The overlay only exists when the top visible video track is strictly ABOVE the base track.
    let s1: Option<(&crate::model::Clip, usize)> = match (base_tv, over_tv) {
        (Some((_, _, bt)), Some((c, i, ot))) if ot > bt => Some((c, i)),
        _ => None,
    };

    // Base = the lowest visible video clip if present, else the overlay shown directly.
    let base_clip = s0.or(s1.map(|(c, _)| c));

    // Per-clip LOOK from the BASE (visible) clip (Slice A). Mirrors MojoMedia, whose playhead-
    // segment look (seg_look/seg_look_amt) drives the look pipeline. look_kind: 0=None, 1=VHS,
    // 2=LUT3D. For LUT3D we forward the clip's `.cube` path; for every other case (None/VHS, or an
    // empty LUT path) we send the "-" sentinel so the engine never tries to load a LUT. A timeline
    // gap (no base clip) has no look. The amount is the clip's `look_amt`.
    // `mut` because an active transition overrides the look to the OUTGOING clip's (it fades out
    // carrying its own look) — see the transition block below.
    let (mut look_kind, mut look_amt, mut lut_path) = match base_clip {
        Some(c) => {
            // LUT path only travels with a LUT3D look (kind 2) AND a non-empty path; otherwise "-".
            // WHITESPACE GUARD: the ENC/PREVIEW lines are whitespace-split with fixed arity on the
            // worker, so a LUT path containing a space would shift every following field. A space-
            // bearing path degrades to no LUT ("-") here — the frame still composes (look none)
            // rather than corrupting the request. This matches the media-path whitespace policy.
            let lut = if c.look == 2
                && !c.lut.is_empty()
                && c.lut.split_whitespace().count() == 1
            {
                c.lut.clone()
            } else {
                if c.look == 2 && c.lut.split_whitespace().count() > 1 {
                    eprintln!("gcompose: LUT path has whitespace; ignoring look for this clip: {}", c.lut);
                }
                "-".to_string()
            };
            (c.look, c.look_amt, lut)
        }
        None => (0, 0.0, "-".to_string()), // gap: no look.
    };

    // `mut` because an active transition forces base = OUTGOING clip for the whole window (a
    // symmetric crossfade) rather than the cover-scan winner — see the transition block below.
    let (mut base_path, mut base_frame) = match base_clip {
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

    // Overlay only when V1 is the base AND V2 is present (Mojo: over_v2 = s0>=0 && s1>=0). The
    // clip + its index come from the single `s1` tuple (finding #7), so there are no unwraps and no
    // way for the index to desync from the clip.
    let (over_path, over_frame, op, px, py, pw, ph) = if let (Some(_), Some((c, idx))) = (s0, s1) {
        match project.media.get(c.media) {
            Some(p) => {
                let frame = (c.src_in + (t - c.t0)) as i32;
                // Clip-LOCAL frame for the PiP keyframe track: t - over_clip.t0 (matches Mojo's
                // pip_lf / qlf = the frame offset into the overlay clip). pip_at returns the clip's
                // static px/py/pw/ph when this clip has no PiP keyframes, so this is a drop-in
                // upgrade of the previous static `(c.px, c.py, c.pw, c.ph)`.
                let t_local = t - c.t0;
                let (px, py, pw, ph) = project.pip_at(idx, t_local);
                // OPACITY KEYFRAME (P1): the base overlay opacity (1.0) is scaled by the keyframed
                // opacity_kf at this timeline frame. An empty opacity_kf track → 1.0 (unchanged), so
                // composites without an opacity animation behave exactly as before; a keyframed fade
                // now drops `op` toward 0, fading the V2 overlay out in preview + render. Clamp to a
                // sane [0,1] so a stray keyframe value can't push the composite weight out of range.
                let op = project.opacity_at(t).clamp(0.0, 1.0);
                (p.clone(), frame.max(0), op, px, py, pw, ph)
            }
            None => ("-".to_string(), 0, 0.0, 0.0, 0.0, 1.0, 1.0),
        }
    } else {
        ("-".to_string(), 0, 0.0, 0.0, 0.0, 1.0, 1.0)
    };

    // P4 CHROMA KEY (green-screen) from the OVER (V2) overlay clip. The key applies to the OVERLAY
    // (the green-screen layer composited over V1), so it is read from `s1` (the V2 clip), NOT the
    // base/outgoing clip — and only when a PiP composite is actually active (BOTH a V1 base AND a V2
    // overlay cover `t`, i.e. `op > 0`). A disabled chroma (or no overlay) sends the IDENTITY sentinel
    // (ck_on=0 + the default key/sim/smooth), so the engine no-ops and the composite is byte-identical
    // to P3. Team A READS `Clip.chroma` here (pre-added by Team C) but NEVER edits model.rs.
    let (ck_on, ck_key, ck_sim, ck_smooth) = match (s0, s1) {
        (Some(_), Some((c, _))) if c.chroma.enabled && op > 0.0 => {
            (1, c.chroma.key, c.chroma.similarity, c.chroma.smoothness)
        }
        // No overlay / chroma disabled: identity (engine skips keying). Defaults mirror ChromaKey.
        _ => (0, [0.0, 1.0, 0.0], 0.4, 0.1),
    };

    // Keyframed grade at the timeline frame (falls back to project.bright/contrast/sat when there
    // are no grade keyframes — Team C's grade_at contract). Replaces the old static grade that was
    // previously read directly off the project in build_request/build_enc_line.
    let (bright, contrast, sat) = project.grade_at(t);

    // PER-CLIP COLOR grade (P1) from the BASE (visible) clip — ADDITIVE on top of the program grade
    // (gcompose runs the per-clip grade FIRST, then the program grade). A timeline gap (no base clip)
    // is neutral (0/1/1). `mut` because an active transition overrides this to the OUTGOING clip's
    // grade for the whole window (it travels with the clip as it fades out, like the look).
    let (mut cbright, mut ccontrast, mut csat) = match base_clip {
        Some(c) => (c.bright, c.contrast, c.sat),
        None => (0.0, 1.0, 1.0),
    };

    // PER-CLIP COLOR-WHEELS (LIFT/GAMMA/GAIN) + TRANSFORM + BLUR (P2) from the BASE (visible) clip.
    // White balance (wb_temp/wb_tint) is FOLDED into the 9 lift/gamma/gain values here (the engine
    // never sees wb — PINNED), so `lgg` below carries the white-balanced gains. A timeline gap (no
    // base clip) is IDENTITY (lift 0 / gamma 1 / gain 1 / rot 0 / scale 1 / blur 0) so the engine
    // no-ops and reproduces the current output. `mut` because an active transition overrides these
    // to the OUTGOING clip's for the whole window (they travel with the look/grade as it fades out).
    let (mut lift, mut gamma, mut gain_rgb, mut rot, mut scale, mut blur) = match base_clip {
        Some(c) => {
            let (l, g, gn) = fold_white_balance(c.lift, c.gamma, c.gain_rgb, c.wb_temp, c.wb_tint);
            (l, g, gn, c.rot, c.scale, c.blur)
        }
        None => ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.0, 1.0, 0.0),
    };

    // ----- Per-boundary TRANSITION (Wave 8) -------------------------------------------------------
    // Consult the transition (if any) on the BASE track whose window contains `t`, then blend the
    // OUTGOING clip (slot 0) -> INCOMING clip (slot 2) by `trans_prog` over the CENTERED window
    // [center - dur/2, center + dur/2). The progress ramp + window come from `Transition::progress`
    // so preview/render never desync from the model. `trans_param` defaults to 4.0 (MojoMedia tt_p).
    //
    // SYMMETRIC crossfade (the wave-8 follow-up): when a transition is active we OVERRIDE the base
    // (slot 0) + look to the OUTGOING clip for the WHOLE window — not the cover-scan winner. Without
    // this, once `t` reaches the incoming clip's span the cover scan makes the INCOMING clip the
    // base, so slot 0 == slot 2 and the window's second half degenerates to incoming-vs-incoming
    // (effectively just the incoming clip). Forcing base = outgoing makes prog=0 -> 100% outgoing,
    // prog=1 -> 100% incoming, a true crossfade straddling the seam. Both clips' source frames are
    // clamped into their own `[src_in, src_in+len-1]` range, so past a clip's end it freezes on its
    // last frame — the standard crossfade hold for adjacent, non-overlapping media.
    //
    // base_track is the track the chosen base clip lives on (V1 if present, else V2). A timeline gap
    // (no base clip) has no track to query, so there is no transition.
    let (trans_kind, trans_prog, trans_param, trans_path, trans_frame) = {
        let mut tk: i32 = -1;
        let mut tp: f32 = 0.0;
        let mut tparam: f32 = 4.0;
        let mut tpath = "-".to_string();
        let mut tframe: i32 = 0;

        // A media path is wire-safe only when non-empty + single-token (ENC/PREVIEW are
        // whitespace-split with fixed arity, so a space would shift every following field).
        let clean = |s: &str| !s.is_empty() && s.split_whitespace().count() == 1;

        if let Some(bc) = base_clip {
            let base_track = bc.track;
            // `transition_at` copies-out so we don't hold a `&project` borrow across the
            // `project.boundaries(..)` call below (which also borrows `project` immutably).
            if let Some(tr) = project.transition_at(base_track, t).copied() {
                // `boundaries()` pairs each OUTGOING clip with its successor (the INCOMING clip) and
                // reports the seam frame (the overlap midpoint for overlapping pairs); match the pair
                // whose seam == the transition center (finding #1 — never a `t0 >= center` scan that
                // would skip the real incoming clip).
                let pair = project
                    .boundaries(base_track)
                    .into_iter()
                    .find(|&(_outgoing, _incoming, boundary)| boundary == tr.center);
                if let Some((out_idx, in_idx, _boundary)) = pair {
                    if let (Some(out_clip), Some(inc)) =
                        (project.clips.get(out_idx), project.clips.get(in_idx))
                    {
                        match (project.media.get(out_clip.media), project.media.get(inc.media)) {
                            (Some(op), Some(ip)) if clean(op) && clean(ip) => {
                                let prog = tr.progress(t);
                                // OUTGOING -> slot 0 base (forced for the whole window). Clamp into
                                // the outgoing clip's valid source range so past its end it freezes
                                // on its last frame instead of decoding a frame it doesn't have.
                                let raw_out = out_clip.src_in + (t - out_clip.t0);
                                let last_out =
                                    (out_clip.src_in + out_clip.len - 1).max(out_clip.src_in);
                                base_path = op.clone();
                                base_frame = raw_out.clamp(out_clip.src_in, last_out) as i32;
                                // The LOOK travels with the OUTGOING clip while it fades out.
                                look_kind = out_clip.look;
                                look_amt = out_clip.look_amt;
                                lut_path = if out_clip.look == 2 && clean(&out_clip.lut) {
                                    out_clip.lut.clone()
                                } else {
                                    "-".to_string()
                                };
                                // The per-clip GRADE likewise travels with the OUTGOING clip (P1).
                                cbright = out_clip.bright;
                                ccontrast = out_clip.contrast;
                                csat = out_clip.sat;
                                // The P2 color-wheels (white-balanced) + transform + blur ALSO
                                // travel with the OUTGOING clip while it fades out (matching the
                                // look/grade), so a graded/transformed clip keeps its grade through
                                // the whole transition window rather than snapping at the seam.
                                let (ol, og, ogn) = fold_white_balance(
                                    out_clip.lift,
                                    out_clip.gamma,
                                    out_clip.gain_rgb,
                                    out_clip.wb_temp,
                                    out_clip.wb_tint,
                                );
                                lift = ol;
                                gamma = og;
                                gain_rgb = ogn;
                                rot = out_clip.rot;
                                scale = out_clip.scale;
                                blur = out_clip.blur;
                                // INCOMING -> slot 2 (the partner the kernel blends the base toward),
                                // its source frame likewise clamped into its valid range.
                                let raw_in = inc.src_in + (t - inc.t0);
                                let last_in = (inc.src_in + inc.len - 1).max(inc.src_in);
                                tk = tr.kind;
                                tp = prog;
                                tparam = 4.0;
                                tpath = ip.clone();
                                tframe = raw_in.clamp(inc.src_in, last_in) as i32;
                            }
                            _ => {
                                // A missing/whitespace media path on either side degrades to no
                                // transition: the base (cover-scan winner) still composes and the
                                // fixed-arity wire line stays intact.
                                eprintln!("gcompose: transition clip media path missing/has whitespace; skipping transition at center {}", tr.center);
                            }
                        }
                    }
                }
                // No matching boundary / clip pair: leave tk = -1 (no transition).
            }
        }
        (tk, tp, tparam, tpath, tframe)
    };

    // ----- P5 TITLE TEXT OVERLAY (Slice A) --------------------------------------------------------
    // When the BASE clip OR the OVER clip carries a non-empty title, rasterize that title to a
    // full-frame transparent RGBA8 layer (cached temp file) and SUBSTITUTE that clip's media path
    // with the `RAW:<raster_path>` sentinel + frame 0 (a title is STATIC — its source frame is
    // ignored). The engine then reads the raw RGBA straight into the slot (skipping decode) and
    // composites it: a V2 title shows its text OVER the V1 base; a title-only clip (no V1) shows the
    // text over black (the "-" base). All other per-clip effects (grade/pip/chroma/look/transition)
    // still apply to the rasterized text exactly as they would to decoded media — by design.
    //
    // IDENTITY: a clip with an empty title yields no raster (rasterize_title -> None), so its path is
    // left untouched and a project with no titles never emits a `RAW:` path → byte-identical render.
    // No new wire field is needed: the title rides on the EXISTING base/over path as the `RAW:`
    // sentinel, so build_request / build_enc_line are unchanged.
    let mut over_path = over_path;
    let mut over_frame = over_frame;
    // BASE clip title (covers both a title-only V1 clip and a V1 clip that itself has a title).
    if let Some(c) = base_clip {
        if !c.title.is_empty() {
            if let Some(raster) = rasterize_title(&c.title) {
                base_path = format!("RAW:{raster}");
                base_frame = 0;
            }
        }
    }
    // OVER (V2) clip title: substitute the overlay path so the title composites over the base. Only
    // meaningful when an overlay actually exists (s1.is_some()); when it does, force op>0 so the
    // composite runs even if the model's opacity happened to resolve to 0 for a pure title layer.
    let mut op = op;
    if let Some((c, _)) = s1 {
        if !c.title.is_empty() {
            if let Some(raster) = rasterize_title(&c.title) {
                over_path = format!("RAW:{raster}");
                over_frame = 0;
                if op <= 0.0 {
                    op = 1.0; // a title overlay with zero opacity would never show; make it visible.
                }
            }
        }
    }

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
        bright,
        contrast,
        sat,
        cbright,
        ccontrast,
        csat,
        look_kind,
        look_amt,
        lut_path,
        trans_kind,
        trans_prog,
        trans_param,
        trans_path,
        trans_frame,
        lift,
        gamma,
        gain_rgb,
        rot,
        scale,
        blur,
        ck_on,
        ck_key,
        ck_sim,
        ck_smooth,
    })
}

/// Resolve frame `t` and format the preview request line: an explicit `PREVIEW` keyword followed
/// by the 36 positional payload fields, the LAST of which is the out path (P2: was 24+out). The
/// keyword removes the latent dispatch ambiguity where a media path whose first token happened to
/// equal a command keyword (OPEN/ENC/...) could misroute a preview frame to the wrong handler
/// (finding #3); the engine now matches `PREVIEW` explicitly and never falls through to keyword-
/// guessing for a real frame request. Total PREVIEW token count = 37 (keyword + 36 payload fields).
fn build_request(project: &Project, t: i64) -> Option<String> {
    let r = resolve_frame(project, t)?;
    // PREVIEW + 36 space-separated payload fields + out path (P2: was 24 + out): the 12 composite
    // fields, then the 3 Slice A LOOK fields (look_kind, look_amt, lut_path), then the 5 Wave 8
    // TRANSITION fields (trans_kind, trans_prog, trans_param, trans_path, trans_frame), then the 3
    // Triad-B P1 PER-CLIP GRADE fields (cbright, ccontrast, csat), then the 12 Triad-B P2 fields
    // (lift_r lift_g lift_b  gamma_r gamma_g gamma_b  gain_r gain_g gain_b  rot scale blur) in the
    // PINNED order, then the out path. The program grade (b/c/s) comes from the RESOLVED (keyframed)
    // values, the per-clip grade/wheels/transform/blur from the BASE clip (white-balance already
    // folded into the 9 lift/gamma/gain), the look from the BASE clip, and the transition from the
    // base-track boundary — so the preview reflects keyframed grade, per-clip grade, color-wheels,
    // transform, blur, per-clip look, AND the animated transition, identical to the render
    // (`build_enc_line`), then the 6 Triad-A P4 CHROMA-KEY fields (ck_on ck_r ck_g ck_b ck_sim
    // ck_smooth) describing the OVER (V2) clip — identity (ck_on=0) when there is no overlay / the
    // chroma is disabled, so a project with no chroma renders byte-identically to P3. PREVIEW token
    // count = 43 (the PREVIEW keyword + 42 fields, last = out; P2 was 37).
    Some(format!(
        "PREVIEW {base} {over} {bf} {of} {op} {px} {py} {pw} {ph} {b} {c} {s} {lk} {la} {lut} \
         {tk} {tp} {tparam} {tpath} {tframe} {cb} {cc} {cs} \
         {lr} {lg} {lb} {gmr} {gmg} {gmb} {gnr} {gng} {gnb} {rot} {scl} {blr} \
         {ckon} {ckr} {ckg} {ckb} {cksim} {cksm} {out}",
        base = r.base_path,
        over = r.over_path,
        bf = r.base_frame,
        of = r.over_frame,
        op = r.op,
        px = r.px,
        py = r.py,
        pw = r.pw,
        ph = r.ph,
        b = r.bright,
        c = r.contrast,
        s = r.sat,
        lk = r.look_kind,
        la = r.look_amt,
        lut = r.lut_path,
        tk = r.trans_kind,
        tp = r.trans_prog,
        tparam = r.trans_param,
        tpath = r.trans_path,
        tframe = r.trans_frame,
        cb = r.cbright,
        cc = r.ccontrast,
        cs = r.csat,
        lr = r.lift[0],
        lg = r.lift[1],
        lb = r.lift[2],
        gmr = r.gamma[0],
        gmg = r.gamma[1],
        gmb = r.gamma[2],
        gnr = r.gain_rgb[0],
        gng = r.gain_rgb[1],
        gnb = r.gain_rgb[2],
        rot = r.rot,
        scl = r.scale,
        blr = r.blur,
        ckon = r.ck_on,
        ckr = r.ck_key[0],
        ckg = r.ck_key[1],
        ckb = r.ck_key[2],
        cksim = r.ck_sim,
        cksm = r.ck_smooth,
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
///   OPEN <out> <out_w> <out_h> <fps_num> <fps_den> <rate_mode> <rate_value> <vcodec> <total_s>
///                                                  (config video [scaled to out WxH] + aac, alloc audio accumulator)
///   for t in 0..total_frames:  ENC <resolved frame fields>   (composite + feed video encoder)
///   for each audible clip:  AUDIO <media> <src_in/FPS> <len/FPS> <t0/FPS> <gain> <fade_in_s> <fade_out_s> <clip_len_s> <range_local_s>
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
/// RETRY-FROM-SCRATCH (Slice A, finding #4 fixed): the WHOLE OPEN→ENC*→AUDIO*→CLOSE sequence is
/// wrapped in a retry loop (up to MAX_ATTEMPTS). A fresh OPEN supersedes any half-open encoder
/// (open_render drops the prior encoder), so re-running the entire render against a freshly spawned
/// worker is safe — this absorbs the intermittent (~10%) worker OpenCL-init/compose flake that used
/// to abort a long export with a partial/empty mp4. Only the worker-death shape (`CmdStatus::Broken`
/// mid-render, or a CLOSE that comes back Broken) triggers a respawn + full retry; a clean
/// `CmdStatus::Err` on OPEN or ENC (a genuine encoder/protocol error, worker still alive) is a clean
/// abort with NO retry (retrying a deterministic error would just burn attempts). Returns true only
/// when one full OPEN..CLOSE attempt completes.
///
/// Per attempt the worker is held under ONE lock for the whole OPEN→CLOSE sequence (finding #1): no
/// concurrent preview/thumbnail can interleave on the worker's pipes mid-render. On a worker death,
/// the dead proc is dropped + mark_spawn_fail'd and a NEW worker is spawned before the next attempt.
/// Keeps every other worker.rs API/behavior (request_frame, play_program, thumbnail, …) intact.
pub fn render_program(project: &Project, out_path: &str) -> bool {
    let total = project.total_frames();
    if total <= 0 {
        return false;
    }

    // WHITESPACE-PATH GUARD (finding #8): the ENC line is whitespace-split with fixed arity on the
    // worker, so a media path containing a space would shift every numeric field and the worker
    // would reject the frame ("bad ENC (N fields)" -> ERR -> the whole render Aborts with a partial
    // mp4 and a non-obvious cause). The AUDIO path already skips such paths; the VIDEO path cannot
    // silently skip a clip (that would drop its picture), so instead we FAIL FAST here with a clear
    // diagnostic BEFORE opening any encoder. Pool paths are space-free in practice, so this is a
    // latent guard — but it converts an opaque mid-render abort into an explicit up-front error.
    for m in &project.media {
        if m.split_whitespace().count() != 1 {
            eprintln!(
                "[render] aborting: media path contains whitespace (unsupported by the fixed-arity \
                 ENC protocol): {m}"
            );
            return false;
        }
    }

    // Build every request line BEFORE taking the lock so a corrupt media index fails the render
    // up-front without ever opening an encoder (and without holding the worker mutex while we
    // touch the model). The lines are reused across retry attempts (the model is immutable here).
    // If any ENC line can't be resolved, abort before OPEN.
    let mut enc_lines: Vec<String> = Vec::with_capacity(total as usize);
    for t in 0..total {
        match build_enc_line(project, t) {
            Some(r) => enc_lines.push(r),
            None => return false, // corrupt media index: nothing opened yet, just bail.
        }
    }
    let audio_lines = build_audio_lines(project);

    // Total timeline duration in seconds (sizes the worker's program-audio accumulator so the
    // rendered audio is exactly the timeline length — see Slice A). Computed from the TIMELINE fps
    // (RENDER_FPS=30, the rate at which clips are sampled), NOT the output fps: the audio buffer is
    // wall-clock seconds, and each composed frame is stamped at its timeline time, so audio/video
    // stay synced regardless of the chosen OUTPUT framerate. Same for every attempt.
    let total_s = total as f64 / RENDER_FPS as f64;

    // EXPORT CONTROLS (Triad-B P1): the OPEN line now carries the output resolution, fps, rate mode +
    // value, and codec from `project.export` — decoupling the ENCODER dims from the fixed GVW×GVH
    // OpenCL working canvas (gcompose swscales the composed GVW×GVH frame to out_w×out_h). DEFAULTS
    // (1280×856 @ 30/1, mpeg4, 4 Mbit/s bitrate, rate_mode 0) reproduce today's behavior so existing
    // render gates pass unchanged. A vcodec containing whitespace would shift the fixed-arity OPEN
    // parse, so it is sanitized to the default here (codec names are single-token in practice).
    let ex = &project.export;
    let out_w = if ex.out_w == 0 { PVW as u32 } else { ex.out_w };
    let out_h = if ex.out_h == 0 { PVH as u32 } else { ex.out_h };
    let fps_num = if ex.fps_num == 0 { RENDER_FPS as u32 } else { ex.fps_num };
    let fps_den = if ex.fps_den == 0 { 1 } else { ex.fps_den };
    let rate_mode = if ex.rate_mode > 1 { 0 } else { ex.rate_mode };
    // In bitrate mode (0) a non-positive value falls back to the 4 Mbit/s default; in CRF mode (1) the
    // value IS the CRF (0 = lossless is legitimate), so it is passed through clamped to a sane range.
    let rate_value = if rate_mode == 1 {
        ex.rate_value.clamp(0, 51)
    } else if ex.rate_value > 0 {
        ex.rate_value
    } else {
        4_000_000
    };
    let vcodec = if ex.vcodec.split_whitespace().count() == 1 && !ex.vcodec.is_empty() {
        ex.vcodec.as_str()
    } else {
        "mpeg4"
    };
    // OPEN <out> <out_w> <out_h> <fps_num> <fps_den> <rate_mode> <rate_value> <vcodec> <total_s> (10 tokens, was 6).
    let open_req = format!(
        "OPEN {out_path} {out_w} {out_h} {fps_num} {fps_den} {rate_mode} {rate_value} {vcodec} {total_s}"
    );

    // RETRY-FROM-SCRATCH loop: each iteration runs one full OPEN..CLOSE attempt. A worker-death
    // outcome (Retry) respawns and loops; a Success returns true; a clean Abort returns false.
    for attempt in 0..MAX_ATTEMPTS {
        match render_attempt(&open_req, &enc_lines, &audio_lines) {
            RenderOutcome::Success => return true,
            RenderOutcome::Abort => {
                // Deterministic error (bad OPEN, ENC/CLOSE encoder error) — retrying won't help.
                return false;
            }
            RenderOutcome::Retry => {
                // Worker died mid-render (the flake). render_attempt already dropped the dead proc
                // and armed the cooldown; spawn-from-scratch happens at the top of the next attempt.
                eprintln!(
                    "[render] worker died mid-render (attempt {} of {}); retrying whole render from a fresh OPEN",
                    attempt + 1,
                    MAX_ATTEMPTS
                );
            }
        }
    }
    eprintln!("[render] all {MAX_ATTEMPTS} render attempts hit a worker death; giving up");
    false
}

/// Outcome of one full `render_attempt` (one OPEN..CLOSE pass):
///   - `Success` : the whole sequence completed; the mp4 is written.
///   - `Abort`   : a CLEAN, deterministic failure (corrupt request, OPEN/ENC/CLOSE encoder error)
///                 with the worker STILL ALIVE — retrying the same render would just fail again.
///   - `Retry`   : the worker DIED mid-render (a `Broken` transport break, the ~10% flake). The dead
///                 proc has already been dropped + cooldown-armed; the caller should respawn + retry
///                 the whole render from a fresh OPEN (safe: a new OPEN supersedes any half encoder).
enum RenderOutcome {
    Success,
    Abort,
    Retry,
}

/// One full OPEN→ENC*→AUDIO*→CLOSE render pass under a single worker-lock hold (finding #1). All
/// request lines are pre-built and immutable, so this is safe to call repeatedly (retry-from-scratch,
/// finding #4): a fresh OPEN inside drops any prior half-open encoder on the worker side.
///
/// Failure mapping (drives the retry loop in `render_program`):
///   - OPEN  Broken  → drop+respawn-mark, return Retry  (worker died spawning/initialising: the flake)
///   - OPEN  Err     → return Abort                      (bad out path etc.; worker alive, no encoder)
///   - ENC   Broken  → abort_held (tear half encoder), return Retry  (worker died mid-video pass)
///   - ENC   Err     → abort_held, return Abort          (deterministic encoder error; worker alive)
///   - AUDIO Broken  → drop+mark, return Retry           (worker died feeding audio; encoder gone)
///   - AUDIO Err     → skip just this clip's audio, continue (worker alive)
///   - CLOSE Broken  → drop+mark, return Retry           (worker died finalising)
///   - CLOSE Err     → return Abort                      (encoder finish error; worker alive)
fn render_attempt(open_req: &str, enc_lines: &[String], audio_lines: &[String]) -> RenderOutcome {
    // Acquire the worker for the WHOLE attempt (finding #1): one lock hold spanning OPEN→CLOSE, so
    // no concurrent preview/thumbnail can interleave on the worker's pipes mid-render.
    let slot = worker_slot();
    let mut guard = match slot.lock() {
        Ok(g) => g,
        // A poisoned worker mutex is a clean, non-retryable failure (the process is in trouble).
        Err(_) => return RenderOutcome::Abort,
    };

    // OPEN: ensure a live worker, then start the encoder. A respawn here is safe (no encoder in
    // flight yet). Distinguish a worker death (Broken/spawn-fail → Retry) from a deterministic OPEN
    // rejection (Err → Abort). We only try to (re)spawn ONCE inside an attempt; the retry-from-
    // scratch loop in render_program provides the outer attempts.
    if guard.is_none() {
        *guard = spawn_worker();
    }
    let open_status = match guard.as_mut() {
        Some(proc) => try_command_status(proc, open_req),
        None => {
            // Spawn failed outright: treat as a worker death so the caller retries (respawn next).
            *guard = None;
            mark_spawn_fail();
            return RenderOutcome::Retry;
        }
    };
    match open_status {
        CmdStatus::Done(_) => clear_spawn_cooldown(),
        CmdStatus::Err => {
            // OPEN rejected (e.g. bad out path) but the worker is alive: no encoder was created, so
            // there is nothing to tear down — and a retry would deterministically fail. Abort clean.
            return RenderOutcome::Abort;
        }
        CmdStatus::Broken => {
            // Worker died on OPEN (the init flake): drop it + arm cooldown, ask the caller to retry.
            *guard = None;
            mark_spawn_fail();
            return RenderOutcome::Retry;
        }
    }

    // From here an encoder is open: a Broken on any step means the worker died (encoder lost) → the
    // caller respawns + retries the WHOLE render; an Err on ENC/CLOSE is a deterministic encoder
    // error (worker alive) → tear the half-open encoder down and Abort (no retry).

    // ENC every frame, in order, on the held proc.
    for req in enc_lines {
        let status = match guard.as_mut() {
            Some(proc) => try_command_status(proc, req),
            None => CmdStatus::Broken, // proc vanished (a prior Broken cleared it): worker died.
        };
        match status {
            CmdStatus::Done(_) => {}
            CmdStatus::Err => {
                // Deterministic encoder error: tear the half-open encoder down NOW (finding #8) so
                // its partial mp4 is dropped immediately, then Abort (retry would just fail again).
                eprintln!("[render] ENC error (worker alive) at: {req}");
                abort_held(&mut *guard);
                return RenderOutcome::Abort;
            }
            CmdStatus::Broken => {
                // Worker died mid-video: try a best-effort teardown on whatever's left (usually a
                // no-op since the proc is gone), then ask the caller to respawn + retry from OPEN.
                eprintln!("[render] ENC worker-death at: {req}");
                abort_held(&mut *guard);
                return RenderOutcome::Retry;
            }
        }
    }

    // PROGRAM AUDIO: feed each AUDIBLE clip's source-audio range, in t0 (timeline) order. A clip
    // with no audio / a decode skip (worker ERR) is dropped but the render continues; only a worker
    // death (Broken / vanished proc) aborts the attempt for retry. This uses MojoMedia's per-segment
    // fpx_decode_audio_range building block, but is NOT a 1:1 mirror of its assembly: MojoMedia's
    // render path concatenates EVERY segment's audio back-to-back with no track filtering, whereas
    // this path positions each clip at its timeline offset (dst_offset) AND applies the track-
    // audibility policy in build_audio_lines (track-2/track-0 only; track_mute honored).
    for line in audio_lines {
        let outcome = match guard.as_mut() {
            Some(proc) => try_command_status(proc, line),
            None => CmdStatus::Broken,
        };
        match outcome {
            CmdStatus::Done(_) => {}
            CmdStatus::Err => {} // worker alive; skip just this clip's audio and continue.
            CmdStatus::Broken => {
                // The worker died feeding audio: the encoder is gone. Drop the dead proc, arm the
                // cooldown, and ask the caller to retry the whole render from a fresh OPEN.
                *guard = None;
                mark_spawn_fail();
                return RenderOutcome::Retry;
            }
        }
    }

    // CLOSE — finish + close the encoder, flush + write BOTH the video and audio trailers.
    match guard.as_mut() {
        Some(proc) => match try_command_status(proc, "CLOSE") {
            CmdStatus::Done(_) => {
                clear_spawn_cooldown();
                RenderOutcome::Success
            }
            CmdStatus::Err => RenderOutcome::Abort, // encoder finish error; worker alive, no retry.
            CmdStatus::Broken => {
                // Worker died finalising: the trailer may be missing. Drop + retry from a fresh OPEN
                // (a re-render produces a clean, complete file rather than a truncated one).
                *guard = None;
                mark_spawn_fail();
                RenderOutcome::Retry
            }
        },
        None => RenderOutcome::Retry, // proc vanished before CLOSE: worker died, retry.
    }
}

/// Temp WAV the worker writes program audio to for playback (see `play_program`).
const PLAY_WAV: &str = "/tmp/genesis_play.wav";

/// Per-request output path for the program-audio LEVELS query (4 little-endian f32: peak_L, peak_R,
/// rms_L, rms_R, all in dBFS). Reused each call — `program_levels` holds the worker mutex across its
/// whole MEAS→AUDIO*→LEVELS round-trip, so there is never a concurrent writer.
const LEVELS_OUT: &str = "/tmp/genesis_levels.f32";

/// Floor (dBFS) the engine reports for digital silence (and the worst-case the UI meter draws). A
/// peak/RMS of 0 linear maps to this instead of −inf so the meter has a finite bottom of its scale.
pub const LEVELS_FLOOR_DB: f32 = -90.0;

/// Stereo program-audio levels (dBFS) of the assembled mix over a measurement window. `peak` is the
/// per-channel sample peak, `rms` the per-channel root-mean-square, both already in dBFS (0 dBFS =
/// full scale, `LEVELS_FLOOR_DB` = silence). Produced by `program_levels`; drawn by panels' meter.
#[derive(Clone, Copy)]
pub struct AudioLevels {
    pub peak_l: f32,
    pub peak_r: f32,
    pub rms_l: f32,
    pub rms_r: f32,
}

/// Measure the ASSEMBLED program-audio levels (peak + RMS, dBFS, per channel) over a short window of
/// the timeline starting at `start_frame` — the lightweight meter feed for panels.rs. Returns the
/// stereo `AudioLevels`, or None when there's nothing to measure (empty/past-end timeline) OR the
/// worker is busy (a contended `try_lock`, so the meter keeps its last reading rather than the UI
/// freezing — exactly like `scope`).
///
/// MECHANISM (no real-time device capture — we measure the ASSEMBLED mix, per the contract): run a
/// `MEAS <window_s>` / `AUDIO*` / `LEVELS <out>` session on the persistent worker. `MEAS` allocates a
/// playback-style accumulator (no encoder, no WAV) sized to a SHORT window; each AUDIO line mixes a
/// clip's filtered+gained range into it exactly like the render/playback path (so the meter reflects
/// the per-clip gain/fade/FX chain); `LEVELS` then computes peak+RMS over the accumulator, writes the
/// 4 f32 dBFS values to `<out>`, and clears the accumulator. The whole session runs under ONE held
/// mutex so no concurrent compose interleaves on the worker's pipes.
///
/// WINDOW: a fixed `LEVELS_WINDOW_S` slice from the playhead (not the whole tail) keeps the decode +
/// measure cheap enough to call at meter cadence — the meter shows the level "around the playhead".
///
/// NON-BLOCKING LOCK (finding #4 style): `try_lock`, called from the egui UI thread (panels), so a
/// background assembly/render holding the worker just yields None this frame.
pub fn program_levels(project: &Project, start_frame: i64) -> Option<AudioLevels> {
    /// Measurement window length (seconds) from the playhead. Short so the per-repaint decode is cheap.
    const LEVELS_WINDOW_S: f64 = 0.25;

    let total = project.total_frames();
    let start = start_frame.max(0);
    if total <= 0 || start >= total {
        return None; // nothing to measure at/after the timeline end.
    }
    let fps = RENDER_FPS as f64;
    let start_s = start as f64 / fps;
    // The window is the lesser of LEVELS_WINDOW_S and the remaining tail.
    let tail_s = (total - start) as f64 / fps;
    let window_s = LEVELS_WINDOW_S.min(tail_s).max(1.0 / fps);

    // Build the MEAS/AUDIO*/LEVELS lines on the CALLING thread (cheap, no decode/lock). dst offsets
    // are shifted so the playhead is t=0 in the measurement window; only clips overlapping the window
    // are emitted (a clip entirely before/after the window contributes nothing). This reuses the same
    // per-clip gain/fade/FX-chain build as playback so the meter reflects the audible mix.
    let meas_open = format!("MEAS {window_s}");
    let window_end = start_s + window_s;
    let mut audio_lines: Vec<String> = Vec::new();
    {
        let mut idx: Vec<usize> = (0..project.clips.len()).collect();
        idx.sort_by_key(|&i| project.clips[i].t0);
        for i in idx {
            let c = &project.clips[i];
            if c.len <= 0 || !track_is_audible(project, c.track) {
                continue;
            }
            if c.end() <= start {
                continue; // entirely in the past.
            }
            let clip_t0_s = c.t0 as f64 / fps;
            if clip_t0_s >= window_end {
                continue; // starts after the measurement window: nothing in range.
            }
            let media_path = match project.media.get(c.media) {
                Some(p) => p,
                None => continue,
            };
            if media_path.split_whitespace().count() != 1 {
                continue; // whitespace path: skip (same fixed-arity guard as the mix path).
            }
            // Head-trim the same way playback does (clip straddling the playhead plays from the
            // source frame under it), then clamp the decoded duration to the window so a long clip
            // doesn't decode its whole tail just to measure a 0.25 s window.
            let head_skip = (start - c.t0).max(0);
            let eff_src_in = c.src_in + head_skip;
            let mut eff_len = c.len - head_skip;
            if eff_len <= 0 {
                continue;
            }
            let dst_off_s = ((c.t0 + head_skip) as f64 / fps - start_s).max(0.0);
            // Cap the decoded range to what fits inside the window from this clip's dst offset.
            let max_len_frames = (((window_s - dst_off_s).max(0.0)) * fps).ceil() as i64;
            if max_len_frames <= 0 {
                continue;
            }
            eff_len = eff_len.min(max_len_frames);
            let src_in_s = eff_src_in as f64 / fps;
            let dur_s = eff_len as f64 / fps;
            let gain = c.gain;
            let fade_in_s = (c.fade_in.max(0)) as f64 / fps;
            let fade_out_s = (c.fade_out.max(0)) as f64 / fps;
            let clip_len_s = c.len as f64 / fps;
            let range_local_s = head_skip as f64 / fps;
            let fx_chain = build_audio_chain(&c.audio_fx);
            audio_lines.push(format!(
                "AUDIO {media_path} {src_in_s} {dur_s} {dst_off_s} {gain} {fade_in_s} {fade_out_s} {clip_len_s} {range_local_s} {fx_chain}"
            ));
        }
    }

    let slot = worker_slot();
    // try_lock (finding #4): never block the UI thread behind a background assembly / render. A
    // contended meter just returns None and the caller keeps its last reading.
    let mut guard = match slot.try_lock() {
        Ok(g) => g,
        Err(_) => return None,
    };

    for attempt in 0..MAX_ATTEMPTS {
        if guard.is_none() {
            *guard = spawn_worker();
        }
        if let Some(proc) = guard.as_mut() {
            // 1) MEAS: open a measurement accumulator (no encoder, no WAV). A clean ERR (bad window)
            //    is non-retryable — bail. Broken falls through to the respawn below.
            match try_command_status(proc, &meas_open) {
                CmdStatus::Done(_) => {
                    // 2) Mix each clip's filtered+gained range (skip ERR clips; Broken aborts).
                    let mut broke = false;
                    for line in &audio_lines {
                        match try_command_status(proc, line) {
                            CmdStatus::Done(_) | CmdStatus::Err => {}
                            CmdStatus::Broken => {
                                broke = true;
                                break;
                            }
                        }
                    }
                    if !broke {
                        // 3) LEVELS: measure + write 4 f32 dBFS, clear the accumulator. Read back.
                        if let Some(levels) = read_levels(proc) {
                            clear_spawn_cooldown();
                            return Some(levels);
                        }
                    }
                }
                CmdStatus::Err => return None, // MEAS rejected (worker alive): nothing to measure.
                CmdStatus::Broken => {}        // worker died: respawn below.
            }
        }
        // MEAS/AUDIO/LEVELS broke mid-session: drop the worker so the next loop spawns clean.
        *guard = None;
        mark_spawn_fail();
    }
    None
}

/// Run the `LEVELS <out>` round-trip on an already-running worker, then read back the 4 little-endian
/// f32 (peak_L, peak_R, rms_L, rms_R; dBFS). Returns None on any failure (write error, EOF/timeout,
/// "ERR", a short/oversized read). The worker clears its measurement accumulator inside LEVELS.
fn read_levels(proc: &mut WorkerProc) -> Option<AudioLevels> {
    let req = format!("LEVELS {LEVELS_OUT}");
    match try_command_status(proc, &req) {
        CmdStatus::Done(payload) => {
            // The worker echoes the out path on DONE; trust our own path if it echoed empty.
            let read_path = if payload.is_empty() { LEVELS_OUT } else { payload.as_str() };
            let bytes = std::fs::read(read_path).ok()?;
            if bytes.len() != 16 {
                return None; // exactly 4 f32 expected.
            }
            let f = |i: usize| {
                f32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]])
            };
            Some(AudioLevels { peak_l: f(0), peak_r: f(4), rms_l: f(8), rms_r: f(12) })
        }
        CmdStatus::Err | CmdStatus::Broken => None,
    }
}

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
/// false when there is nothing to play (empty timeline / playhead at/after the end) OR when a
/// previous playback is still in flight (finding #9: a Space press while one audition is assembling
/// is dropped rather than stacking a second background thread + duplicate player).
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
            // Per-clip gain + fade envelope (P1). The decoded range here is HEAD-TRIMMED by
            // `head_skip` frames, so the first decoded sample is at clip-local `head_skip/FPS` —
            // pass that as `range_local_s` so gcompose ramps the fade against the FULL clip edges
            // (a clip whose fade-in is entirely before the playhead plays at full gain, as it
            // should). fade/clip_len are the clip's own (untrimmed) frame counts in seconds.
            let gain = c.gain;
            let fade_in_s = (c.fade_in.max(0)) as f64 / fps;
            let fade_out_s = (c.fade_out.max(0)) as f64 / fps;
            let clip_len_s = c.len as f64 / fps;
            let range_local_s = head_skip as f64 / fps;
            // P3: the per-clip libavfilter chain (space-free) or "-" when neutral. Applied to the
            // HEAD-TRIMMED decoded range (the chain is per-clip, the fade/range_local handle the trim).
            let fx_chain = build_audio_chain(&c.audio_fx);
            audio_lines.push(format!(
                "AUDIO {media_path} {src_in_s} {dur_s} {dst_off_s} {gain} {fade_in_s} {fade_out_s} {clip_len_s} {range_local_s} {fx_chain}"
            ));
        }
    }

    // Dedup rapid presses (finding #9): if a previous play_program's background thread is still
    // assembling/playing, do NOT spawn another — it would block on the worker mutex behind the first
    // (and behind any in-progress render), then fire a duplicate, delayed `paplay`. compare_exchange
    // claims the in-flight slot; if it's already taken we drop this press as a no-op.
    if PLAYBACK_IN_FLIGHT
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .is_err()
    {
        // A playback is already in flight; ignore this press rather than stacking another thread.
        return false;
    }

    // Clear the stop edge for THIS dispatch (findings #7/#8): any stop requested before now belonged
    // to a previous audition; this fresh play_program supersedes it. Ordered AFTER the in-flight
    // claim above so a stop racing the claim is still observed by the assembly thread's pre-spawn
    // check below (it can only make us SKIP a late player, never spuriously spawn one).
    STOP_REQUESTED.store(false, std::sync::atomic::Ordering::Release);

    // Hand the owned command lines to a detached background thread so the UI thread returns at once
    // (finding #1). The thread takes the worker lock, assembles the WAV, and spawns the player; its
    // failures are logged there, not returned here. The in-flight guard is cleared when the thread
    // finishes (success OR failure) so the next Space press can dispatch again.
    std::thread::spawn(move || {
        if assemble_and_play(&wave_open, &audio_lines) {
            // STOP-DURING-ASSEMBLY (findings #7/#8): if `stop_playback` fired while we were
            // assembling the WAV (the seconds-long window before any player exists), do NOT spawn a
            // late player — the user asked for silence. Without this check the WAV would still play a
            // moment later, unkillable (there was no child to kill at stop time), and on a loop wrap
            // it would overlap the next cycle's audio. Acquire pairs with stop_playback's Release.
            if !STOP_REQUESTED.load(std::sync::atomic::Ordering::Acquire) {
                // WAV written and no stop pending: launch the detached system player (best-effort).
                spawn_player(PLAY_WAV);
            }
        }
        PLAYBACK_IN_FLIGHT.store(false, std::sync::atomic::Ordering::Release);
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
/// launched. The child's stdio is /dev/null and it is not WAITED on (playback runs independently of
/// the UI), but the `Child` handle IS stored in `PLAYER_CHILD` so `stop_playback` can kill it
/// on demand (Slice A pinned API). Any previously-tracked child is replaced here — by the time a new
/// audition is dispatched the old one is either finished or being superseded, so we best-effort kill
/// + reap the prior handle before storing the new one (avoids two players overlapping AND avoids
/// leaking a zombie). Best-effort: false if neither binary exists.
fn spawn_player(wav: &str) -> bool {
    for bin in ["paplay", "aplay"] {
        match Command::new(bin)
            .arg(wav)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => {
                if let Ok(mut slot) = player_slot().lock() {
                    // Replace any prior child: kill+reap it first so we don't stack two players or
                    // leak a zombie. A finished child's kill is a harmless no-op.
                    if let Some(mut old) = slot.take() {
                        let _ = old.kill();
                        let _ = old.wait();
                    }
                    *slot = Some(child);
                }
                return true;
            }
            Err(_) => continue,
        }
    }
    eprintln!("gcompose: no audio player (paplay/aplay) found; playback skipped");
    false
}

/// PINNED (Slice A): stop any in-flight audition player started by `play_program`.
///
/// CHOSEN MECHANISM (stated for the integrator): we track the spawned player `Child` (paplay/aplay)
/// in the process-global `PLAYER_CHILD` and `kill()`+`wait()` it here — the precise path that stops
/// exactly the player THIS process launched and reaps it so no zombie is left. `stop_playback` does
/// NOT touch the worker mutex, so it returns instantly even while a render/assemble holds the worker.
///
/// TWO COORDINATION FLAGS (findings #7/#8):
///   1. `STOP_REQUESTED` is set so the detached assembly thread, if it is still inside the
///      WAVE/AUDIO window (no player spawned yet), SKIPS spawning the late player when it finishes
///      (see `play_program`). This is what makes a stop fired DURING assembly actually take effect —
///      previously there was no child to kill yet, so the WAV played a moment later, unkillable.
///   2. `PLAYBACK_IN_FLIGHT` is force-cleared so an immediately-following `play_program` (the loop
///      wrap: `stop_playback(); play_program(0)`) can re-dispatch instead of being dropped as a
///      no-op by the in-flight guard while the previous assembly thread is still winding down.
///      Clearing it here can transiently allow TWO assembly threads to run concurrently, but the new
///      thread cleared `STOP_REQUESTED` for itself and the OLD thread's pre-spawn check now sees that
///      cleared flag — so the OLD thread might still spawn its (now-stale) player. That is bounded:
///      `spawn_player` kills+reaps any prior tracked child before storing the new one, so at most one
///      player is ever audible. The net effect is the loop wrap re-dispatches reliably rather than
///      going silent (the previous fragility) while never stacking audible players.
///
/// REMOVED (finding #7): the old `pkill -f /tmp/genesis_play.wav` fallback. A broad `pkill -f`
/// matches the pattern against the WHOLE command line of every process, so it could kill unrelated
/// processes that merely reference the path (a text editor, `tail`, a second Genesis instance's
/// player sharing the hard-coded WAV) — collateral damage for a marginal benefit. The tracked-child
/// kill plus the STOP_REQUESTED edge cover the real cases (audible player, and the assembly-window
/// race) precisely, with no risk of killing a bystander.
pub fn stop_playback() {
    // Edge flags first (findings #7/#8): suppress any late player from an in-progress assembly, and
    // free the in-flight slot so a loop-wrap re-dispatch isn't dropped. Release pairs with the
    // assembly thread's Acquire load and play_program's compare_exchange.
    STOP_REQUESTED.store(true, std::sync::atomic::Ordering::Release);
    PLAYBACK_IN_FLIGHT.store(false, std::sync::atomic::Ordering::Release);

    // Primary + only kill path: kill + reap the tracked player child (no broad pkill — finding #7).
    if let Ok(mut slot) = player_slot().lock() {
        if let Some(mut child) = slot.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
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

/// Build a libavfilter CHAIN STRING (P3 Triad-B audio depth) from a clip's `AudioFx`, or the "-"
/// sentinel when the FX are neutral. The returned string has NO SPACES — filters are joined with
/// COMMAS, and `=`/`:`/`|` are used inside a filter — so it rides the fixed-arity AUDIO wire line as
/// ONE token. gcompose runs `fpx_au_apply(sr, ch, chain, ...)` on the decoded clip range BEFORE the
/// per-clip gain + fade + dst-offset mix, but ONLY when the chain != "-".
///
/// NEUTRAL ⇒ "-" (PINNED contract): a clip whose `AudioFx::is_neutral()` is true sends "-", so the
/// engine skips `fpx_au_apply` entirely and the mix is BYTE-IDENTICAL to P2. This is the identity
/// guarantee — no FX dialed means the P2 audio path is reproduced exactly.
///
/// MAPPING (each emitted only when it changes the audio; mirrors Shotcut's audio_eq3band / audio_pan
/// / audio_compressor / audio_noisegate / audio_normalize_1p):
///   eq_low_db  (≠0) -> `equalizer=f=100:t=q:w=1:g=<low>`   (low band   @ 100 Hz)
///   eq_mid_db  (≠0) -> `equalizer=f=1000:t=q:w=1:g=<mid>`  (mid band   @ 1 kHz)
///   eq_high_db (≠0) -> `equalizer=f=8000:t=q:w=1:g=<high>` (high band  @ 8 kHz)
///   pan        (≠0) -> `stereotools=balance_out=<pan>`     (−1 = full L, +1 = full R; matches the
///                       model's −1..1 pan, which is exactly stereotools' balance_out range)
///   compress (true) -> `acompressor`     (libavfilter sensible defaults)
///   gate     (true) -> `agate`           (libavfilter sensible defaults)
///   normalize(true) -> `loudnorm`        (single-pass EBU R128 loudness normalization)
/// Filter ORDER is EQ → pan → compress → gate → normalize (tone-shape first, then dynamics, then a
/// final loudness pass). dB values are formatted WITHOUT a thousands separator / locale, so the
/// `{:.3}` float never contains a space.
///
/// SAFETY: `is_neutral()` is the single source of truth for "no FX"; if a future field is added to
/// AudioFx, neutral must keep returning "-". A non-finite slider value (shouldn't occur from the UI
/// sliders) is treated as 0 / off so the chain can never contain "NaN"/"inf" tokens.
fn build_audio_chain(fx: &crate::model::AudioFx) -> String {
    if fx.is_neutral() {
        return "-".to_string();
    }
    let mut parts: Vec<String> = Vec::new();

    // 3-band EQ — only emit a band whose gain is non-zero (and finite). `f` = center freq, `t=q` +
    // `w=1` give a moderate-Q peaking filter per band (Shotcut's 3-band bass/mid/treble equivalent).
    let band = |freq: i32, g: f32| -> Option<String> {
        if g != 0.0 && g.is_finite() {
            Some(format!("equalizer=f={freq}:t=q:w=1:g={:.3}", g))
        } else {
            None
        }
    };
    if let Some(s) = band(100, fx.eq_low_db) {
        parts.push(s);
    }
    if let Some(s) = band(1000, fx.eq_mid_db) {
        parts.push(s);
    }
    if let Some(s) = band(8000, fx.eq_high_db) {
        parts.push(s);
    }

    // Pan — stereotools balance_out (−1..1). Clamp defensively so a stray value can't exceed the
    // filter's accepted range (which would make the whole graph fail to parse → unfiltered fallback).
    if fx.pan != 0.0 && fx.pan.is_finite() {
        let p = fx.pan.clamp(-1.0, 1.0);
        parts.push(format!("stereotools=balance_out={:.3}", p));
    }

    // Dynamics + loudness — boolean toggles, libavfilter defaults (NO spaces in the bare names).
    if fx.compress {
        parts.push("acompressor".to_string());
    }
    if fx.gate {
        parts.push("agate".to_string());
    }
    if fx.normalize {
        parts.push("loudnorm".to_string());
    }

    // Defensive: if every "band" was actually 0/off (is_neutral was false only because of a
    // non-finite that we dropped), fall back to "-" so we never send an empty chain token.
    if parts.is_empty() {
        return "-".to_string();
    }
    parts.join(",")
}

/// True if track `t` contributes to the program audio (P5 arbitrary tracks): EVERY track — video or
/// audio — contributes its clips' audio unless it is MUTED. This replaces the old fixed policy (only
/// V1+A1 audible, V2 never) now that tracks are a typed list: a video clip carries audio that plays
/// like Shotcut, and a muted track (`project.is_muted(t)`) is silent. An out-of-range index is silent.
fn track_is_audible(project: &Project, t: u8) -> bool {
    (t as usize) < project.track_count() && !project.is_muted(t)
}

/// Build the timeline-synced AUDIO lines for the program audio, one per AUDIBLE clip (track 0 V1 +
/// track 2 A1, honoring `track_mute`; track 1 V2 skipped). Emitted in timeline (`t0`) order for
/// determinism, though order no longer affects the result now that the worker mixes by destination
/// offset rather than concatenating.
///
/// WIRE (Triad-B P3 — 11 tokens; was 10 in P1, 6 in wave-2):
///   AUDIO <media> <src_in_s> <dur_s> <dst_off_s> <gain> <fade_in_s> <fade_out_s> <clip_len_s>
///         <range_local_s> <fx_chain|->
///
/// `src_in_s = clip.src_in / FPS`, `dur_s = clip.len / FPS`, `dst_off_s = clip.t0 / FPS`. `gain` is
/// the per-clip LINEAR gain (`Clip.gain`, was hardcoded 1.0). The fade fields let gcompose apply the
/// clip's audio fades AT MIX TIME: `fade_in_s`/`fade_out_s` are the clip's fade_in/fade_out frame
/// counts in seconds, `clip_len_s` is the clip's FULL on-timeline length in seconds (so fade-out can
/// be measured from the clip end), and `range_local_s` is the clip-local seconds of the FIRST decoded
/// sample (0 for the full-clip render range; the head-trim for a playback clip straddling the
/// playhead). gcompose computes per-sample clip-local time = range_local_s + k/sr and ramps the gain
/// 0→1 over [0, fade_in_s) and 1→0 over [clip_len_s − fade_out_s, clip_len_s). FPS = RENDER_FPS (30).
///
/// `fx_chain` (P3, the 11th field) is the per-clip libavfilter chain from `build_audio_chain(clip.
/// audio_fx)` — a SPACE-FREE comma-joined filter expression, or "-" when the clip's AudioFx is
/// neutral. gcompose runs `fpx_au_apply` on the decoded range BEFORE the gain+fade+offset mix when
/// `fx_chain != "-"`; a "-" skips the filter entirely (byte-identical to P2). The chain is built
/// from `c.audio_fx` (Team B reads/writes audio_fx; never edits model.rs).
///
/// Clips with non-positive length, a corrupt media index, a non-audible/muted track, or whitespace in
/// the media path are skipped (a whitespace path would break the fixed-arity AUDIO parse).
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
        // Per-clip linear gain (P1) + fade envelope (frames → seconds). The render range is the WHOLE
        // clip, so the first decoded sample is at clip-local 0.
        let gain = c.gain;
        let fade_in_s = (c.fade_in.max(0)) as f64 / fps;
        let fade_out_s = (c.fade_out.max(0)) as f64 / fps;
        let clip_len_s = c.len as f64 / fps;
        let range_local_s = 0.0f64;
        // P3: the per-clip libavfilter chain (space-free) or "-" when the AudioFx is neutral.
        let fx_chain = build_audio_chain(&c.audio_fx);
        lines.push(format!(
            "AUDIO {media_path} {src_in_s} {dur_s} {dst_off_s} {gain} {fade_in_s} {fade_out_s} {clip_len_s} {range_local_s} {fx_chain}"
        ));
    }
    lines
}

/// Format the `ENC ...` line for timeline frame `t` (35 payload fields, no out path; P2 grew it
/// from 23), baking the same composite as the preview. ENC total token count = 36 (the `ENC`
/// keyword + 35 payload fields). Returns None when the frame can't be resolved.
fn build_enc_line(project: &Project, t: i64) -> Option<String> {
    let r = resolve_frame(project, t)?;
    // Program grade (b/c/s) comes from the RESOLVED (keyframed) values so the render bakes the SAME
    // keyframed grade the preview shows — not the static project.bright/contrast/sat (Slice A). The
    // 3 LOOK fields (look_kind, look_amt, lut_path), then the 5 Wave 8 TRANSITION fields (trans_kind,
    // trans_prog, trans_param, trans_path, trans_frame), then the 3 Triad-B P1 PER-CLIP GRADE fields
    // (cbright, ccontrast, csat), then the 12 Triad-B P2 fields (lift_r lift_g lift_b  gamma_r gamma_g
    // gamma_b  gain_r gain_g gain_b  rot scale blur) are appended in the PINNED order so the render
    // bakes the SAME per-clip look + grade + color-wheels + transform + blur + animated transition
    // the preview shows (white-balance already folded into the 9 lift/gamma/gain). Then the 6 Triad-A
    // P4 CHROMA-KEY fields (ck_on ck_r ck_g ck_b ck_sim ck_smooth) describing the OVER (V2) clip —
    // identity (ck_on=0) when there is no overlay / chroma disabled, so a no-chroma render is
    // byte-identical to P3. ENC has NO out path — the 6 chroma fields are the LAST 6 → ENC is now 42
    // tokens incl the keyword (P2 was 36).
    Some(format!(
        "ENC {base} {over} {bf} {of} {op} {px} {py} {pw} {ph} {b} {c} {s} {lk} {la} {lut} \
         {tk} {tp} {tparam} {tpath} {tframe} {cb} {cc} {cs} \
         {lr} {lg} {lb} {gmr} {gmg} {gmb} {gnr} {gng} {gnb} {rot} {scl} {blr} \
         {ckon} {ckr} {ckg} {ckb} {cksim} {cksm}",
        base = r.base_path,
        over = r.over_path,
        bf = r.base_frame,
        of = r.over_frame,
        op = r.op,
        px = r.px,
        py = r.py,
        pw = r.pw,
        ph = r.ph,
        b = r.bright,
        c = r.contrast,
        s = r.sat,
        lk = r.look_kind,
        la = r.look_amt,
        lut = r.lut_path,
        tk = r.trans_kind,
        tp = r.trans_prog,
        tparam = r.trans_param,
        tpath = r.trans_path,
        tframe = r.trans_frame,
        cb = r.cbright,
        cc = r.ccontrast,
        cs = r.csat,
        lr = r.lift[0],
        lg = r.lift[1],
        lb = r.lift[2],
        gmr = r.gamma[0],
        gmg = r.gamma[1],
        gmb = r.gamma[2],
        gnr = r.gain_rgb[0],
        gng = r.gain_rgb[1],
        gnb = r.gain_rgb[2],
        rot = r.rot,
        scl = r.scale,
        blr = r.blur,
        ckon = r.ck_on,
        ckr = r.ck_key[0],
        ckg = r.ck_key[1],
        ckb = r.ck_key[2],
        cksim = r.ck_sim,
        cksm = r.ck_smooth,
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
