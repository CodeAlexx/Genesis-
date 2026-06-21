//! App shell — eframe::App, toolbar, preview pane, panel/timeline layout, screenshot gate.
//!
//! Owned by the layout team. A Shotcut-style 3-column layout built from plain SidePanels
//! (left = media pool, right = properties + scopes, bottom = timeline, center = preview),
//! each topped by a thin labeled dock-header bar. Wires together model + worker + timeline
//! + pool + panels. The preview re-composites whenever the playhead moves off the last
//! frame we composited (`last_composed`), in addition to the initial frame-2 gate.

use crate::model::{Clip, History, Project};
use crate::{icons, panels, pool, project_io, theme, timeline, worker};
use eframe::egui::{self, Color32};

/// P18 SOURCE MONITOR: which clip the CENTER preview is showing.
///   `Program` — the composited TIMELINE frame at the program playhead (the existing, default
///               preview; `compose()` -> `request_frame`).
///   `Source`  — a RAW, effect-free frame of a single pool/source clip opened from the media pool,
///               decoded by `worker::thumbnail` at the source-local `src_playhead`. This mirrors
///               Shotcut's Source vs Project monitor tabs: the Source monitor scrubs an opened
///               clip with its own playhead + in/out marks, independent of the timeline.
/// Default is `Program`, so with no source opened (and no `GENESIS_SOURCE`) the app behaves exactly
/// as before — the Source feature is purely additive.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MonitorMode {
    Program,
    Source,
}

impl Default for MonitorMode {
    fn default() -> Self {
        MonitorMode::Program
    }
}

/// Which tab the right dock shows. Shotcut keeps Properties, the video Scopes, and the audio
/// meters/spectrum in separate dockable panels; in our single right `SidePanel` we tab between them
/// so each gets the full panel height/width instead of being stacked + crammed into ~260px.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RightTab {
    Properties,
    Scopes,
    Audio,
}

impl Default for RightTab {
    fn default() -> Self {
        RightTab::Properties
    }
}

pub struct Genesis {
    preview: Option<egui::TextureHandle>,
    project: Project,
    /// The PRIMARY (last-clicked) clip index — drives panels.rs `properties_ui` and stays the
    /// single-clip target for split/lift/slip. PINNED cross-team: this field STAYS as `usize`
    /// (Team B reads it); multi-select lives separately in `selection` below.
    selected: usize,
    /// MULTI-SELECT set (P3 editing): the clips Ctrl/Shift-click added, plus the primary. Copy/cut
    /// act on this set (falling back to `[selected]` when empty). A plain click resets it to just
    /// the clicked clip. Threaded into `timeline_ui` so the timeline can add/remove members and
    /// draw a distinct multi-select highlight. `selected` always remains the primary clip.
    selection: Vec<usize>,
    /// COPY/CUT clipboard (P3 editing): an OFFSET-PRESERVING, playhead-rebased snapshot of the
    /// copied clips (`Project::copy_clips` rebases the earliest to t0 = 0). Paste re-anchors it at
    /// the playhead via `Project::paste_clips`. Survives across edits + project loads (it is app
    /// state, independent of the model). Cleared on Open so a stale clip from another project can't
    /// paste media indices that no longer exist.
    clipboard: Vec<Clip>,
    /// FILTER CLIPBOARD (T3 detach/paste-fx): a TRANSIENT snapshot of a clip whose FILTER/GRADE/LOOK/
    /// audio-fx stack was "Copy filters"-ed, for "Paste filters" onto another clip
    /// (`Project::copy_filters_from`). Holds a full `Clip` (cloned) but only its filter fields are
    /// applied on paste — dst keeps its own media/src_in/len/t0/track/group/fades. App state only,
    /// NOT serialized; cleared on project Open (a stale clip from another project must not be pasted).
    filter_clip: Option<Clip>,
    ppf: f32,
    playhead: i64,
    /// The playhead value of the frame currently in `preview`. `-1` = nothing composited yet.
    last_composed: i64,
    /// The playhead value at the end of the previous `update()`, used to detect actual
    /// movement so we don't re-enter the (blocking) worker round-trip every frame on a
    /// stationary playhead — including after a failed compose, where `last_composed` is
    /// intentionally left unchanged.
    prev_playhead: i64,
    preview_inited: bool,
    status: String,
    shot_path: Option<String>,
    frames: u64,
    /// Snapshot-based undo/redo. `history.push(&project)` is called *before* every edit.
    history: History,
    /// P33 AUTO-SAVE: the `history.len()` (undo-stack depth) captured at the LAST auto-save. The
    /// periodic loop re-saves only when `history.len() != autosave_marker` (the project changed),
    /// so an idle/unedited project never re-writes the recovery sidecar. Init 0 (no save yet).
    autosave_marker: usize,
    /// P33 AUTO-SAVE: `self.frames` at the LAST auto-save. The loop waits
    /// `AUTOSAVE_INTERVAL_FRAMES` frames since this before considering another save, so the blocking
    /// `project_io::save` runs at most ~once per interval and never spams the UI thread. Init 0.
    autosave_frame: u64,
    /// Transport state. When true, `update()` advances the playhead in WALL-CLOCK time (Slice C):
    /// the playhead is derived from `play_anchor`/`play_anchor_frame` so the video track tracks
    /// real time and stays in approximate A/V sync with the real-time audio audition, looping at
    /// the program end. (Replaces the old per-update `cur_T += 1` advance, which ran at the egui
    /// repaint rate rather than at FPS.)
    playing: bool,
    /// `self.playing` at the end of the previous `update()`, used to detect the transport
    /// START edge (false->true) so audio playback fires exactly ONCE per Play, and the STOP edge
    /// (true->false) so we kill the audio audition exactly once (mirrors MojoMedia's
    /// `space and not prev_space` edge-detect for the play toggle).
    prev_playing: bool,
    /// Wall-clock anchor for wall-clock transport (Slice C). Set on the transport START edge to
    /// `Instant::now()`; while playing, the playhead = `play_anchor_frame + elapsed*FPS`. Re-set
    /// on a loop wrap so the next cycle starts timing from the wrap moment. `None` while paused.
    play_anchor: Option<std::time::Instant>,
    /// The playhead frame captured at the START edge (or the loop-wrap frame, 0). The wall-clock
    /// advance is measured RELATIVE to this so playing from a scrubbed-to frame is exact.
    play_anchor_frame: i64,
    /// Horizontal timeline scroll OFFSET in FRAMES (wave P1). The first frame drawn flush at the
    /// clip-area left edge. Mutated inside `timeline_ui` (mouse-wheel pan + keep-playhead-visible
    /// clamp) and read back each frame; the app only owns the storage. Starts at 0 (project start).
    x_scroll: f32,
    /// Snapping enabled (wave P1). Toggled by Ctrl+P (Shotcut's "Snap" toggle). When false,
    /// `timeline_ui` skips edge/marker snapping on clip moves/trims/drops. Default true.
    snap: bool,
    /// JKL shuttle rate (wave P1). 0 = no shuttle. Nonzero drives a SILENT wall-clock shuttle of
    /// the playhead at `shuttle * FPS` frames/sec (sign = direction): L steps it +1,+2,+4…; J steps
    /// it -1,-2,-4…; K (or Space, or any seek key) zeroes it. Mutually exclusive with `playing`
    /// (audio play): starting a shuttle stops audio play and vice-versa, mirroring Shotcut where
    /// Space is audio playback and J/L are silent fast rewind/forward. Reuses the wall-clock anchor.
    shuttle: i32,
    /// One-shot zoom-to-fit request (wave P1). The `0` key sets this; it is passed to `timeline_ui`
    /// (which knows the on-screen clip-area width) for ONE frame and then cleared. Computing the fit
    /// inside timeline_ui keeps the fit width exact (the app shell does not know the lane width).
    zoom_fit_pending: bool,
    /// 3-POINT EDIT in-point (P4): the timeline frame set by `I` (Shotcut "Set In", playerSetInAction).
    /// `None` = no in-point marked. With an out-point it defines a timeline TARGET RANGE
    /// [mark_in, mark_out) that the timeline draws as a shaded band; `B` (overwrite) drops the source
    /// clip at `mark_in` when set (else the playhead). Cleared after an edit that consumes it, and
    /// re-clamped when in > out (setting an in-point past the out-point clears the out-point, mirroring
    /// Shotcut's Player::setIn which resets the out when it would invert). Never auto-set — a project
    /// with no in/out gesture has `None`/`None` and renders byte-identically to P3.
    mark_in: Option<i64>,
    /// 3-POINT EDIT out-point (P4): the timeline frame set by `O` (Shotcut "Set Out",
    /// playerSetOutAction). `None` = no out-point. See `mark_in` for the target-range semantics.
    mark_out: Option<i64>,

    // ----- P18 SOURCE MONITOR (second preview of an opened pool/source clip) -------------------
    /// Which clip the CENTER preview shows: `Program` (composited timeline, default) or `Source`
    /// (a raw frame of the opened pool clip). Set by the preview_pane "Project"/"Source" tabs and
    /// by the pool "Open in Source" affordance / `GENESIS_SOURCE` hook.
    monitor: MonitorMode,
    /// Which tab the right dock shows (Properties / Scopes / Audio). Phase-2 scope-dock split.
    right_tab: RightTab,
    /// The `project.media` index opened in the Source monitor, or `None` when nothing is open.
    /// Set by "Open in Source" (and the `GENESIS_SOURCE` env hook); read by `compose_source` and
    /// `source_clip`. Bounds-checked against `project.media.len()` before every use.
    src_media: Option<usize>,
    /// SOURCE-LOCAL playhead (frame within the opened clip, NOT a timeline frame). Default 0,
    /// driven by the Source scrubber slider in `preview_pane`. Clamped >= 0 before decoding.
    src_playhead: i64,
    /// Source in-point (source-local frame) set by `I` while `monitor == Source`. `None` = unset.
    /// Used by `source_clip` for the 3-point source range. Independent of the timeline `mark_in`.
    src_in: Option<i64>,
    /// Source out-point (source-local frame) set by `O` while `monitor == Source`. `None` = unset.
    src_out: Option<i64>,
    /// The Source preview texture (a raw `worker::thumbnail` frame of `src_media` @ `src_playhead`),
    /// or `None` until the first source compose. Drawn centered (same as the program preview) when
    /// `monitor == Source`.
    src_preview: Option<egui::TextureHandle>,
    /// The `src_playhead` value `src_preview` was composed at — the source-monitor analogue of
    /// `last_composed`. `-1` = nothing composed yet. The update() recompose guard re-composes the
    /// source whenever `src_playhead != src_last` (a scrub) so we don't re-decode every frame.
    src_last: i64,
    /// The `src_playhead` value at the end of the previous `update()` — the source-monitor analogue
    /// of `prev_playhead`. Used so the source recompose guard fires only on an ACTUAL scrub (or a
    /// forced recompose, `src_last == -1`), NOT every frame after a failed decode (which leaves
    /// `src_last` unchanged). Without this, scrubbing past the clip end — where `thumbnail` returns
    /// `None` by design — would re-enter the worker round-trip every frame on a stationary slider.
    prev_src_playhead: i64,

    /// T2 RECENT FILES: most-recent-FIRST, de-duplicated list of project paths the user has opened
    /// or saved. APP state only (NOT part of the serialized `Project`, so old .json projects are
    /// untouched) — persisted to a tiny JSON sidecar (`recent_file_path()`, default
    /// `/tmp/genesis_recent.json`). Loaded once in `Genesis::new`; pushed-to + re-written on every
    /// real Open/Save (the toolbar buttons funnel their load through `open_project_path`, which
    /// records the path). The "Recent" toolbar menu lists these and re-opens a clicked entry via
    /// the SAME `open_project_path` load path. A missing/empty/corrupt sidecar loads to `[]` (no
    /// panic). Capped at `RECENT_CAP` so the list (and the sidecar) stays small.
    recents: Vec<String>,
}

/// Max absolute JKL shuttle speed multiplier (Shotcut caps repeated J/L presses; we cap at 8x).
const MAX_SHUTTLE: i32 = 8;

/// Timeline zoom (pixels-per-frame) bounds for the `=`/`-`/`0` zoom keys (wave P1). ~0.25..40 per
/// the slice spec: 0.25 ppf shows the whole of a long project; 40 ppf is a deep frame-level zoom.
const MIN_PPF: f32 = 0.25;
const MAX_PPF: f32 = 40.0;
/// Multiplicative step for the `=`/`+` (in) and `-` (out) zoom keys — one keypress scales ppf by
/// this factor (a comfortable ~1.25x per press).
const ZOOM_STEP: f32 = 1.25;

/// P33 AUTO-SAVE & CRASH RECOVERY — fixed sidecar path the app periodically writes the live project
/// to (Shotcut auto-saves + offers recovery; Genesis had none). PURE SIDE-EFFECT: this file is the
/// ONLY thing the feature touches — it never mutates the project, the render, or the user's real
/// project files. Kept a simple constant so it is trivially gateable in tests. Recovery on launch
/// reads it; the periodic loop + the GENESIS_AUTOSAVE gate write to it / a given path.
pub const RECOVERY_PATH: &str = "/tmp/genesis_recovery.json";

/// Auto-save cadence in `update()` frames. At ~30–60 fps this is roughly every 10–20 s — frequent
/// enough to bound data loss, rare enough that the (blocking) `project_io::save` never spams the UI
/// thread. Only fires when the project has actually changed since the last auto-save (see update()).
const AUTOSAVE_INTERVAL_FRAMES: u64 = 600;

/// T2 RECENT FILES — max entries kept in the recents list (and the sidecar). Shotcut keeps ~10.
const RECENT_CAP: usize = 10;

/// T2 RECENT FILES — path of the recents sidecar JSON. Overridable via `GENESIS_RECENT` (used by
/// tests / headless runs to point at a scratch file); defaults to `/tmp/genesis_recent.json`.
fn recent_file_path() -> String {
    std::env::var("GENESIS_RECENT").unwrap_or_else(|_| "/tmp/genesis_recent.json".to_string())
}

/// T2 RECENT FILES — PURE core (no I/O): push `path` to the FRONT of `list`, most-recent-first,
/// de-duplicated, capped at `cap`. Any existing equal entry is removed first (so re-opening a file
/// promotes it to the front rather than duplicating it), then the path is inserted at index 0 and
/// the list is truncated to `cap`. A `cap` of 0 clears the list (defensive; never used in practice).
/// This is the unit-tested heart of the feature — it touches no disk and no app state.
fn push_recent(list: &mut Vec<String>, path: &str, cap: usize) {
    list.retain(|p| p != path);
    list.insert(0, path.to_string());
    if list.len() > cap {
        list.truncate(cap);
    }
}

/// T2 RECENT FILES — load the recents sidecar (a JSON array of strings). Returns `[]` when the file
/// is missing, empty, or not valid JSON — NEVER panics. The list is de-duplicated + capped on load
/// (via `push_recent` applied oldest→newest) so a hand-edited / stale sidecar can't seed an
/// over-long or duplicated list.
fn load_recents() -> Vec<String> {
    let raw = match std::fs::read(recent_file_path()) {
        Ok(bytes) => bytes,
        Err(_) => return Vec::new(),
    };
    let parsed: Vec<String> = serde_json::from_slice(&raw).unwrap_or_default();
    // Re-normalize (dedup + cap), preserving most-recent-first order: fold from the OLDEST entry so
    // each `push_recent` promotes it to the front, leaving the original (newest-first) order intact.
    let mut out: Vec<String> = Vec::new();
    for p in parsed.into_iter().rev() {
        push_recent(&mut out, &p, RECENT_CAP);
    }
    out
}

/// T2 RECENT FILES — write `list` to the recents sidecar as pretty JSON. Best-effort: any I/O error
/// is swallowed (a flaky /tmp must never take down the editor — same policy as the auto-save loop).
fn save_recents(list: &[String]) {
    if let Ok(json) = serde_json::to_string_pretty(list) {
        let _ = std::fs::write(recent_file_path(), json);
    }
}

/// Decoded keyboard state for one `update()` frame — every shortcut Genesis reads, snapshotted in a
/// single `ctx.input()` borrow so the rest of `handle_keys` is borrow-free. Field = true iff the
/// corresponding (possibly modified) key was pressed this frame.
#[derive(Default)]
struct Keys {
    split: bool,
    razor_all: bool, // Shift+S — split EVERY clip on EVERY track at the playhead (Split All Tracks)
    freeze: bool,    // Shift+F — insert a freeze-frame still at the playhead inside the selected clip
    nudge_left: bool,  // , — nudge the selected clip -1 frame on its track (keyboard frame nudge)
    nudge_right: bool, // . — nudge the selected clip +1 frame on its track
    lift: bool,
    undo: bool,
    redo: bool,
    left: bool,
    right: bool,
    space: bool,
    marker: bool,
    prev_marker: bool,
    next_marker: bool,
    zoom_in: bool,
    zoom_out: bool,
    zoom_fit: bool,
    prev_edit: bool,
    next_edit: bool,
    home: bool,
    end: bool,
    j: bool,
    kk: bool,
    l: bool,
    snap_toggle: bool,
    // ----- P3 editing -----
    ripple_delete: bool, // X / Shift+Delete / Shift+Backspace — ripple delete (close the gap)
    copy: bool,          // Ctrl+C — copy the selection set to the clipboard
    cut: bool,           // Ctrl+X — copy then ripple-delete the selection set
    paste: bool,         // Ctrl+V — paste the clipboard at the playhead (offset-preserving)
    select_all: bool,    // Ctrl+A — select every clip
    // ----- P4 3-point editing (Shotcut player/timeline keys) -----
    mark_in: bool,    // I  — set the in-point at the playhead (Shotcut playerSetInAction)
    mark_out: bool,   // O  — set the out-point at the playhead (Shotcut playerSetOutAction)
    append: bool,     // A  — append the source clip to the track end (Shotcut timelineAppendAction)
    overwrite: bool,  // B  — overwrite at mark_in (or playhead) (Shotcut timelineOverwriteAction)
    insert: bool,     // V  — ripple-insert at the playhead (Shotcut Insert / default-ripple paste)
}

impl Genesis {
    pub fn new(cc: &eframe::CreationContext<'_>, project: Project) -> Self {
        theme::apply(&cc.egui_ctx);
        let shot_path = std::env::var("GENESIS_SHOT").ok();

        // P18 GENESIS_SOURCE headless hook (for the gate): if GENESIS_SOURCE parses to a usize
        // media index, open the Source monitor on it so the frame-2 compose + GENESIS_SHOT
        // screenshot capture the raw source clip (not the program). The actual source compose runs
        // in `ensure_preview` (the frame-2 path now composes whichever monitor is active). The
        // index is range-checked at compose time, so a stale/out-of-range value degrades to the
        // "No source" hint rather than panicking. With GENESIS_SOURCE unset, monitor stays Program.
        let (monitor, src_media) = match std::env::var("GENESIS_SOURCE").ok().and_then(|s| s.trim().parse::<usize>().ok()) {
            Some(idx) => (MonitorMode::Source, Some(idx)),
            None => (MonitorMode::Program, None),
        };

        // P33 CRASH RECOVERY (launch): `main.rs` sets `GENESIS_RECOVERED=1` when it restored the
        // /tmp recovery sidecar instead of the demo (only possible with GENESIS_OPEN unset/empty —
        // the gates never trigger this). Surface that as the startup status so the user knows the
        // session was recovered and how to start fresh. The status string lives here in app.rs.
        let status: String = if std::env::var_os("GENESIS_RECOVERED").is_some() {
            format!("Recovered unsaved project (delete {} to start fresh)", RECOVERY_PATH)
        } else {
            "compositing\u{2026}".into()
        };

        Genesis {
            preview: None,
            project,
            selected: 0,
            selection: Vec::new(),
            clipboard: Vec::new(),
            filter_clip: None,
            ppf: 6.0,
            playhead: 0,
            last_composed: -1,
            prev_playhead: 0,
            preview_inited: false,
            status,
            shot_path,
            frames: 0,
            history: History::new(),
            autosave_marker: 0,
            autosave_frame: 0,
            playing: false,
            prev_playing: false,
            play_anchor: None,
            play_anchor_frame: 0,
            x_scroll: 0.0,
            snap: true,
            shuttle: 0,
            zoom_fit_pending: false,
            mark_in: None,
            mark_out: None,
            // P18 source monitor.
            monitor,
            right_tab: RightTab::default(),
            src_media,
            src_playhead: 0,
            src_in: None,
            src_out: None,
            src_preview: None,
            src_last: -1,
            prev_src_playhead: 0,
            // T2: load the recents sidecar once at startup (missing/corrupt -> empty, no panic).
            recents: load_recents(),
        }
    }

    /// Frames per second of the program timeline. Matches `worker::RENDER_FPS` (the OPEN default)
    /// and MojoMedia's render config, so the wall-clock playhead advances at the rate the audio
    /// audition was assembled at — keeping the (approximate) A/V sync. Kept as a const (rather than
    /// reaching into the worker) because worker.rs is owned by another slice and exposes no FPS.
    const FPS: f64 = 30.0;

    /// Re-anchor the wall-clock transport to "now" at `frame`. Called on the START edge (anchor at
    /// the current playhead) and on a loop wrap (anchor at 0) so timing always restarts cleanly.
    fn anchor_transport(&mut self, frame: i64) {
        self.play_anchor = Some(std::time::Instant::now());
        self.play_anchor_frame = frame;
    }

    /// Clamp `self.selected` into `0..clips.len()` (collapsing to 0 when empty). Called after
    /// any edit that can shrink the clip list (delete / undo / redo / project replace).
    fn clamp_selected(&mut self) {
        let n = self.project.clips.len();
        if n == 0 {
            self.selected = 0;
        } else if self.selected >= n {
            self.selected = n - 1;
        }
    }

    /// Drop any stale clip indices from `selection` (after a delete / undo / redo / project
    /// replace shrank the clip list) and dedup. The PRIMARY `selected` is clamped separately by
    /// `clamp_selected`; this keeps the multi-select set in range so the timeline never highlights
    /// — or copy/cut never touches — a clip index that no longer exists.
    fn clamp_selection(&mut self) {
        let n = self.project.clips.len();
        self.selection.retain(|&i| i < n);
        self.selection.sort_unstable();
        self.selection.dedup();
    }

    /// The clip indices a copy/cut acts on: the multi-select `selection` if it has any members,
    /// else a single-element fallback to the primary `selected` (when there is a clip to act on).
    /// Mirrors Shotcut's copy/removeSelection, which falls back to the clip under the playhead /
    /// the current clip when nothing is multi-selected. Returns an empty Vec on an empty timeline.
    fn effective_selection(&self) -> Vec<usize> {
        if !self.selection.is_empty() {
            let mut v: Vec<usize> = self.selection.iter().copied().filter(|&i| i < self.project.clips.len()).collect();
            v.sort_unstable();
            v.dedup();
            v
        } else if self.selected < self.project.clips.len() {
            vec![self.selected]
        } else {
            Vec::new()
        }
    }

    /// COPY (Ctrl+C): snapshot the effective selection into the clipboard, offset-preserving +
    /// playhead-rebased (`Project::copy_clips`). No history push (copy does not mutate the project).
    fn copy_selection(&mut self) {
        let sel = self.effective_selection();
        if sel.is_empty() {
            return;
        }
        self.clipboard = self.project.copy_clips(&sel);
        self.status = format!("copied {} clip(s)", self.clipboard.len());
    }

    /// PASTE (Ctrl+V): drop the clipboard at the playhead (offset-preserving). Pushes history BEFORE
    /// the mutation; selects the first pasted clip and resets the multi-select to just it. No-op
    /// (no dead undo step) when the clipboard is empty or every clip lands on a locked track.
    fn paste_clipboard(&mut self) {
        if self.clipboard.is_empty() {
            return;
        }
        // Pre-check landability so a fully-blocked paste pushes NO history (avoids both a dead undo
        // step AND the spurious redo entry an undo-to-unwind would leave). At least one clipboard
        // clip must target a non-locked track.
        let any_landable = self.clipboard.iter().any(|c| !self.project.is_locked(c.track));
        if !any_landable {
            self.status = "paste blocked (locked track)".into();
            return;
        }
        self.history.push(&self.project);
        let clips = self.clipboard.clone();
        if let Some(first) = self.project.paste_clips(&clips, self.playhead) {
            self.selected = first;
            self.selection = vec![first];
            self.status = format!("pasted {} clip(s) at f{}", clips.len(), self.playhead);
        }
    }

    /// RIPPLE DELETE (X / Shift+Delete) the effective selection: remove every selected clip and close
    /// the gap on each affected track via `Project::ripple_delete_many` — a single ATOMIC batch that
    /// is correct regardless of how the clips' Vec-index order relates to their t0 order (skeptic #1).
    /// (Looping the single-clip `ripple_delete` in descending-index order double-shifts survivors when
    /// two same-track clips have the lower index at the higher t0, because each per-clip gap-close is
    /// t0-based and the cutoffs compound.) Skips clips on locked tracks (advisory enforcement, matching
    /// split/lift). Pushes history once before the batch. `cut` first copies to the clipboard. No-op
    /// (no history) when nothing deletable is selected.
    fn ripple_delete_selection(&mut self, cut: bool) {
        let sel = self.effective_selection();
        // Keep only clips on unlocked tracks (a locked clip is refused, like split/lift).
        let idxs: Vec<usize> = sel
            .into_iter()
            .filter(|&i| {
                self.project
                    .clips
                    .get(i)
                    .map(|c| !self.project.is_locked(c.track))
                    .unwrap_or(false)
            })
            .collect();
        if idxs.is_empty() {
            return;
        }
        if cut {
            // Copy BEFORE deleting (offset-preserving snapshot of exactly the clips we remove).
            self.clipboard = self.project.copy_clips(&idxs);
        }
        self.history.push(&self.project);
        // Atomic multi-clip ripple: snapshot positions, remove all, then close gaps per track using
        // the pre-delete layout (order-independent — see Project::ripple_delete_many).
        self.project.ripple_delete_many(&idxs);
        self.selection.clear();
        self.clamp_selected();
        self.clamp_playhead();
        self.status = if cut {
            format!("cut {} clip(s)", idxs.len())
        } else {
            format!("ripple-deleted {} clip(s)", idxs.len())
        };
    }

    /// SELECT ALL (Ctrl+A): put every clip index into the multi-select set. The primary `selected`
    /// is left as-is (it stays the panel target / split-lift focus). No history push (selection is
    /// not project state).
    fn select_all(&mut self) {
        self.selection = (0..self.project.clips.len()).collect();
        self.status = format!("selected all ({})", self.selection.len());
    }

    // ----- P4 3-POINT EDITING (in/out marks + append/overwrite/insert) -------------------------
    // The SOURCE for append/overwrite/insert is the currently-selected primary clip, cloned (keeping
    // its src_in/len/look/grade/transform/audio_fx/chroma — `set_in`/`set_out` source trimming is a
    // future slice; here a 3-point edit uses the whole selected clip as the source, mirroring
    // Shotcut's "open the clip, then append/overwrite/insert" once a source is loaded). The TARGET
    // TRACK is the source clip's own track (so an append puts a V2 clip after the last V2 clip).
    // Each op pushes history BEFORE mutating, selects the placed clip, and surfaces a status line.
    // No op fires without a key gesture, so the no-gesture render is byte-identical to P3.

    /// I — set the in-point at the playhead (Shotcut "Set In"). If the new in lands at/after a stored
    /// out-point, the out-point is cleared (an inverted range is meaningless), mirroring
    /// Player::setIn. Marks are app state, not project state, so NO history push.
    fn set_mark_in(&mut self) {
        let p = self.playhead;
        self.mark_in = Some(p);
        if let Some(o) = self.mark_out {
            if p >= o {
                self.mark_out = None; // setting in past out resets out (Shotcut parity)
            }
        }
        self.status = format!("in @ f{}", p);
    }

    /// O — set the out-point at the playhead (Shotcut "Set Out"). If the new out lands at/before a
    /// stored in-point, the in-point is cleared, mirroring Player::setOut. NO history push.
    fn set_mark_out(&mut self) {
        let p = self.playhead;
        self.mark_out = Some(p);
        if let Some(i) = self.mark_in {
            if p <= i {
                self.mark_in = None; // setting out before in resets in (Shotcut parity)
            }
        }
        self.status = format!("out @ f{}", p);
    }

    /// P18: I in SOURCE mode — set the SOURCE in-point at the source-local `src_playhead` (Shotcut
    /// source-monitor "Set In"). Same invert-resets-out rule as the timeline mark. App state, NO
    /// history push. Distinct from `set_mark_in` (timeline), branched in `handle_keys` on monitor.
    fn set_src_in(&mut self) {
        let p = self.src_playhead.max(0);
        self.src_in = Some(p);
        if let Some(o) = self.src_out {
            if p >= o {
                self.src_out = None;
            }
        }
        self.status = format!("source in @ f{}", p);
    }

    /// P18: O in SOURCE mode — set the SOURCE out-point at `src_playhead` (Shotcut source-monitor
    /// "Set Out"). Same invert-resets-in rule. App state, NO history push.
    fn set_src_out(&mut self) {
        let p = self.src_playhead.max(0);
        self.src_out = Some(p);
        if let Some(i) = self.src_in {
            if p <= i {
                self.src_in = None;
            }
        }
        self.status = format!("source out @ f{}", p);
    }

    /// P18: open a pool clip (media index `idx`) in the Source monitor — switch to Source mode,
    /// reset the source playhead + in/out marks, and trigger a recompose (next update() guard sees
    /// `src_playhead (0) != src_last`). Called from the pool "Open in Source" affordance. Bounds
    /// are re-checked at compose time, so an out-of-range index degrades to the "No source" hint.
    fn open_source(&mut self, idx: usize) {
        self.monitor = MonitorMode::Source;
        self.src_media = Some(idx);
        self.src_playhead = 0;
        self.src_in = None;
        self.src_out = None;
        // Force a source recompose on the next frame (src_last != 0 unless it was already 0).
        self.src_last = -1;
        self.status = format!("opened source \u{2022} media {}", idx);
    }

    /// The SOURCE clip for a 3-point edit (append/overwrite/insert). P18: when a pool clip is OPEN
    /// in the Source monitor (`src_media`), build a fresh `Clip::video` from it whose source range
    /// is the Source in/out marks — `src_in` from `self.src_in` (or 0) and `len` from
    /// `(src_out - src_in)` when BOTH marks are set (clamped >= 1), else a sensible default. The
    /// new clip targets V1 (track 0), matching the pool's "Add as clip → V1" — the placement ops
    /// read `src.track` for the target track and set `t0` from their own args, so the source's t0
    /// is ignored. When NO source is open, fall back to the legacy behavior: clone the primary
    /// `selected` timeline clip (so Program-mode 3-point editing is byte-identical to P4).
    ///
    /// `Clip::video` hardcodes `src_in: 0`, so we set the field after construction.
    fn source_clip(&self) -> Option<Clip> {
        const DEFAULT_SRC_LEN: i64 = 150;
        if let Some(m) = self.src_media {
            if m < self.project.media.len() {
                let s_in = self.src_in.unwrap_or(0).max(0);
                let len = match (self.src_in, self.src_out) {
                    (Some(a), Some(b)) => (b - a).max(1),
                    _ => DEFAULT_SRC_LEN,
                };
                let mut clip = Clip::video(m, 0, len, 0, "");
                clip.src_in = s_in;
                return Some(clip);
            }
        }
        self.project.clips.get(self.selected).cloned()
    }

    /// Display name for a `Clip.track` (0=V1, 1=V2, 2=A1) for status messages.
    fn track_name(track: u8) -> &'static str {
        match track {
            0 => "V1",
            1 => "V2",
            _ => "A1",
        }
    }

    /// A — APPEND the source clip to the end of its own track (Shotcut "Append", `A`). Pushes history
    /// before the op; selects the placed clip. No-op (no dead undo step) when there is no source clip
    /// or the target track is locked.
    fn append_source(&mut self) {
        let src = match self.source_clip() {
            Some(c) => c,
            None => return,
        };
        let track = src.track;
        if self.project.is_locked(track) {
            self.status = "append blocked (locked track)".into();
            return;
        }
        self.history.push(&self.project);
        if let Some(new_i) = self.project.append_clip(track, src) {
            self.selected = new_i;
            self.selection = vec![new_i];
            self.status = format!("appended to {} @ end", Self::track_name(track));
        }
    }

    /// B — OVERWRITE the source clip onto its own track at `mark_in` (if set) else the playhead
    /// (Shotcut "Overwrite", `B`; position -1 => playhead). Replaces the covered range (no ripple).
    /// Pushes history before the op; selects the placed clip; clears the in/out marks (the range was
    /// consumed). No-op when there is no source clip or the track is locked.
    fn overwrite_source(&mut self) {
        let src = match self.source_clip() {
            Some(c) => c,
            None => return,
        };
        let track = src.track;
        if self.project.is_locked(track) {
            self.status = "overwrite blocked (locked track)".into();
            return;
        }
        let at = self.mark_in.unwrap_or(self.playhead);
        self.history.push(&self.project);
        if let Some(new_i) = self.project.overwrite_clip(track, at, src) {
            self.selected = new_i;
            self.selection = vec![new_i];
            // Consumed the target range — clear the marks so a stale band doesn't linger.
            self.mark_in = None;
            self.mark_out = None;
            self.status = format!("overwrote {} @ f{}", Self::track_name(track), at);
        }
    }

    /// V — INSERT (ripple) the source clip onto its own track at `mark_in` (if set) else the playhead
    /// (Shotcut Insert / default-ripple paste). Opens a hole of the source's length and shifts
    /// downstream same-track clips right. Pushes history before the op; selects the placed clip;
    /// clears the marks. No-op when there is no source clip or the track is locked.
    fn insert_source(&mut self) {
        let src = match self.source_clip() {
            Some(c) => c,
            None => return,
        };
        let track = src.track;
        if self.project.is_locked(track) {
            self.status = "insert blocked (locked track)".into();
            return;
        }
        let at = self.mark_in.unwrap_or(self.playhead);
        self.history.push(&self.project);
        if let Some(new_i) = self.project.insert_clip(track, at, src) {
            self.selected = new_i;
            self.selection = vec![new_i];
            self.mark_in = None;
            self.mark_out = None;
            self.status = format!("inserted into {} @ f{}", Self::track_name(track), at);
        }
    }

    /// Clamp `self.playhead` into `0..total_frames` (always >= 0). Called after edits / seeks.
    fn clamp_playhead(&mut self) {
        let last = self.project.total_frames() - 1;
        if self.playhead > last {
            self.playhead = last;
        }
        if self.playhead < 0 {
            self.playhead = 0;
        }
    }

    /// Seek the playhead to the nearest clip EDIT point strictly after the current frame, or the
    /// last frame if none (Shotcut "Skip Next", Alt+Right). An edit point is any clip's `t0` or
    /// `end()` across ALL clips/tracks (the union of cut points). Cancels any JKL shuttle.
    fn seek_next_edit(&mut self) {
        let here = self.playhead;
        let mut best: Option<i64> = None;
        for c in &self.project.clips {
            for &edge in &[c.t0, c.end()] {
                if edge > here {
                    best = Some(best.map_or(edge, |b| b.min(edge)));
                }
            }
        }
        self.shuttle = 0;
        match best {
            Some(f) => self.playhead = f,
            None => self.playhead = (self.project.total_frames() - 1).max(0),
        }
        self.clamp_playhead();
    }

    /// Seek the playhead to the nearest clip EDIT point strictly before the current frame, or 0 if
    /// none (Shotcut "Skip Previous", Alt+Left). Cancels any JKL shuttle.
    fn seek_prev_edit(&mut self) {
        let here = self.playhead;
        let mut best: Option<i64> = None;
        for c in &self.project.clips {
            for &edge in &[c.t0, c.end()] {
                if edge < here {
                    best = Some(best.map_or(edge, |b| b.max(edge)));
                }
            }
        }
        self.shuttle = 0;
        self.playhead = best.unwrap_or(0);
        self.clamp_playhead();
    }

    /// Seek the playhead to the nearest MARKER strictly after the current frame (Shotcut "Next
    /// Marker", `>`). No-op if there is no later marker. Cancels any JKL shuttle.
    fn seek_next_marker(&mut self) {
        let here = self.playhead;
        if let Some(f) = self.project.markers.iter().copied().filter(|&m| m > here).min() {
            self.shuttle = 0;
            self.playhead = f;
            self.clamp_playhead();
        }
    }

    /// Seek the playhead to the nearest MARKER strictly before the current frame (Shotcut "Previous
    /// Marker", `<`). No-op if there is no earlier marker. Cancels any JKL shuttle.
    fn seek_prev_marker(&mut self) {
        let here = self.playhead;
        if let Some(f) = self.project.markers.iter().copied().filter(|&m| m < here).max() {
            self.shuttle = 0;
            self.playhead = f;
            self.clamp_playhead();
        }
    }

    /// FAST-FORWARD one shuttle step (the player's `⏩` button = the `L` key). Mirrors the JKL
    /// handler exactly: cancel audio play, step the FORWARD shuttle speed up (1,2,4,8 capped at
    /// MAX_SHUTTLE), and re-anchor the wall-clock transport at the current frame so update()'s
    /// shuttle advance measures from here. Repeated clicks accelerate, just like repeated L.
    fn shuttle_forward(&mut self) {
        self.playing = false;
        self.shuttle = if self.shuttle >= 1 { (self.shuttle * 2).min(MAX_SHUTTLE) } else { 1 };
        self.anchor_transport(self.playhead);
    }

    /// REWIND one shuttle step (the player's `⏪` button = the `J` key). Mirror of `shuttle_forward`
    /// in the reverse direction (-1,-2,-4,-8 capped at -MAX_SHUTTLE).
    fn shuttle_reverse(&mut self) {
        self.playing = false;
        self.shuttle = if self.shuttle <= -1 { (self.shuttle * 2).max(-MAX_SHUTTLE) } else { -1 };
        self.anchor_transport(self.playhead);
    }

    /// Add a marker at the current playhead (Shotcut "Create Marker", M), keeping `markers` sorted
    /// and deduped. Pushes undo BEFORE the mutation. `markers` is `Vec<i64>` on the model (no model
    /// method needed — we sort/dedup inline here per the slice spec). A marker already at the
    /// playhead is a no-op (no dead undo step, no duplicate).
    fn add_marker(&mut self) {
        let ph = self.playhead;
        if self.project.markers.contains(&ph) {
            return;
        }
        self.history.push(&self.project);
        self.project.markers.push(ph);
        self.project.markers.sort_unstable();
        self.project.markers.dedup();
    }

    /// Split the selected clip at the playhead (Shotcut "Split At Playhead", S). Gated exactly like
    /// the keyboard path: only when the playhead is STRICTLY inside the selected clip body and the
    /// track is unlocked, so a no-op leaves no dead undo step. Shared by `handle_keys` (S key) and
    /// the timeline toolbar split button so there is ONE source of truth for the gesture.
    fn do_split(&mut self) {
        let split_ok = self.project.clips.get(self.selected).map(|c| {
            let off = self.playhead - c.t0;
            (off > 0 && off < c.len, c.track)
        });
        if let Some((in_body, track)) = split_ok {
            if in_body && !self.project.is_locked(track) {
                self.history.push(&self.project);
                let _ = self.project.split_clip(self.selected, self.playhead);
            }
        }
    }

    /// Split EVERY clip on EVERY track that strictly spans the playhead (Shotcut "Split All Tracks",
    /// Shift+S). One undo per gesture, gated on at least one strictly-spanning clip so it leaves no
    /// dead undo step. Shared by `handle_keys`, the top toolbar "Razor all tracks", and the timeline
    /// toolbar razor button.
    fn do_razor_all(&mut self) {
        let t = self.playhead;
        let any_span = self.project.clips.iter().any(|c| c.t0 < t && t < c.end());
        if any_span {
            self.history.push(&self.project);
            let _ = self.project.split_all_at(t);
            self.clamp_selected();
            self.clamp_selection();
        }
    }

    /// LIFT the selected clip (Shotcut "Lift", Z): remove it leaving a GAP (does not ripple). Gated
    /// on the clip being in range and its track unlocked. Shared by `handle_keys` (Z) + the toolbar.
    fn do_lift(&mut self) {
        if self.project.clips.is_empty() || self.selected >= self.project.clips.len() {
            return;
        }
        let track = self.project.clips.get(self.selected).map(|c| c.track);
        let locked = track.map(|t| self.project.is_locked(t)).unwrap_or(false);
        if !locked {
            self.history.push(&self.project);
            self.project.delete_clip(self.selected);
            self.clamp_selected();
            self.clamp_selection();
        }
    }

    /// T2 RECENT FILES — record `path` as the most-recently-used project: promote it to the front of
    /// the in-memory recents list (deduped + capped via the pure `push_recent`) and persist the list
    /// to the sidecar. Called from `open_project_path` (after a successful load) and from the Save
    /// button (after a successful save) so both the Open and Save gestures keep the recents fresh.
    fn record_recent(&mut self, path: &str) {
        push_recent(&mut self.recents, path, RECENT_CAP);
        save_recents(&self.recents);
    }

    /// T2 RECENT FILES — the CANONICAL project-open path, shared by the toolbar "Open" button and the
    /// "Recent" menu so a recent click opens EXACTLY like a manual Open. Loads `path` via
    /// `project_io::load`, and on success swaps in the new project, resets every view/edit/source/
    /// transport latch, resets undo history, forces a frame-0 re-composite, and records the path in
    /// the recents list. On a load failure it sets a status line and leaves the current project
    /// untouched (a stale recents entry pointing at a deleted/renamed file degrades gracefully —
    /// see the greyed/skip handling in the Recent menu). Mirrors the reset block the Open button has
    /// always performed; factored here so both entry points share ONE code path.
    fn open_project_path(&mut self, path: &str) {
        match project_io::load(path) {
            Some(p) => {
                self.project = p;
                self.selected = 0;
                // A new project invalidates the multi-select set + the clipboard (a clipboard clip
                // from the old project could reference media indices that don't exist in the new one).
                self.selection.clear();
                self.clipboard.clear();
                // ... and the FILTER clipboard (a copied filter stack from the old project could
                // reference media/look indices that don't exist in the loaded one).
                self.filter_clip = None;
                // A new project invalidates the 3-point in/out marks: a stale [mark_in, mark_out)
                // band from the previous project would otherwise linger and a subsequent B/V would
                // drop the source at a frame in the OLD project's coordinate space.
                self.mark_in = None;
                self.mark_out = None;
                // A new project invalidates the Source monitor — `src_media` indexes the OLD
                // project's media list. Reset to Program with no source open.
                self.monitor = MonitorMode::Program;
                self.src_media = None;
                self.src_playhead = 0;
                self.src_in = None;
                self.src_out = None;
                self.src_preview = None;
                self.src_last = -1;
                self.prev_src_playhead = 0;
                self.playhead = 0;
                self.playing = false;
                // Keep the transport-edge tracker in lock-step with `playing` so the open does not
                // read as a false->true Start edge on the next update().
                self.prev_playing = false;
                self.play_anchor = None;
                self.play_anchor_frame = 0;
                // A new project invalidates undo history and the composed preview.
                self.history = History::new();
                self.last_composed = -1; // force a re-composite of frame 0
                self.status = format!("opened {}", path);
                // T2: a successful open promotes the path to the front of the recents list.
                self.record_recent(path);
            }
            None => self.status = format!("open failed: {}", path),
        }
    }

    /// Composite the current playhead into the preview texture. Marks `last_composed`.
    fn compose(&mut self, ctx: &egui::Context) {
        match worker::request_frame(&self.project, self.playhead) {
            Some(b) => {
                self.preview = Some(worker::rgba_to_texture(ctx, &b));
                self.status = format!("composite (gcompose worker) \u{2022} f{}", self.playhead);
                self.last_composed = self.playhead;
            }
            None => self.status = "worker failed — no preview".into(),
        }
    }

    /// P18: composite the RAW source frame (`src_media` @ `src_playhead`) into `src_preview`.
    /// Uses `worker::thumbnail` (decode ONE frame, no effects) at PVW×PVH — exactly a raw source
    /// frame — then `rgba_to_texture` (which expects PVW×PVH). No-op (leaves `src_preview` as-is)
    /// when no source is open or the index is out of range; marks `src_last = src_playhead` on a
    /// successful decode so the update() guard won't re-decode a stationary source. Mirrors
    /// `compose`/`last_composed`. Takes `&mut self` + `ctx`; holds no `&self.src_preview` borrow
    /// across the texture build.
    fn compose_source(&mut self, ctx: &egui::Context) {
        let m = match self.src_media {
            Some(m) if m < self.project.media.len() => m,
            _ => return,
        };
        let frame = self.src_playhead.max(0);
        // Clone the path out of the immutable borrow before the (no-borrow) worker round-trip.
        let media_path = self.project.media[m].clone();
        match worker::thumbnail(&media_path, frame, worker::PVW, worker::PVH) {
            Some(b) => {
                self.src_preview = Some(worker::rgba_to_texture(ctx, &b));
                self.src_last = self.src_playhead;
                self.status = format!("source \u{2022} media {} \u{2022} f{}", m, frame);
            }
            None => self.status = format!("source decode failed (media {})", m),
        }
    }

    /// First-frame preview gate (kept on the frame-2 boundary for the screenshot path). Composes
    /// whichever monitor is ACTIVE: in `Source` mode (e.g. set by `GENESIS_SOURCE`) it composes the
    /// raw source clip so the GENESIS_SHOT screenshot captures the source; otherwise the program.
    fn ensure_preview(&mut self, ctx: &egui::Context) {
        if self.preview_inited {
            return;
        }
        self.preview_inited = true;
        match self.monitor {
            MonitorMode::Source => self.compose_source(ctx),
            MonitorMode::Program => self.compose(ctx),
        }
    }

    /// Keyboard editing/transport shortcuts. Bindings mirror Shotcut (verified against
    /// src/docks/timelinedock.cpp + src/player.cpp + src/mainwindow.cpp):
    ///   S                     split selected clip at playhead (Split At Playhead)
    ///   Shift+S               split ALL tracks at playhead (Split All Tracks / razor all)
    ///   , / .                 nudge selected clip -1 / +1 frame on its track (frame nudge)
    ///   Delete / Z / Backspace lift selected clip (Shotcut "Lift" = Z, Backspace; Delete kept)
    ///   Ctrl+Z                undo   |  Ctrl+Shift+Z / Ctrl+Y  redo
    ///   M                     add a marker at the playhead (Create Marker)
    ///   < (Shift+Comma)       seek previous marker  |  > (Shift+Period) seek next marker
    ///   = / +                 zoom in   |  -  zoom out   |  0  zoom to fit (Zoom Timeline In/Out/Fit)
    ///   Alt+Left / Alt+Right  seek previous / next EDIT point (Skip Previous / Skip Next)
    ///   Home / End            playhead to first / last frame (Seek Start / Seek End)
    ///   J / K / L             shuttle reverse / pause / forward (Rewind / Pause / Fast Forward)
    ///   Ctrl+P                toggle snapping
    ///   X / Shift+Delete      ripple delete the selection (close the gap)  [Shotcut Ripple Delete]
    ///   Ctrl+C / Ctrl+X       copy / cut the selection to the clipboard    [Shotcut Copy / Cut]
    ///   Ctrl+V                paste the clipboard at the playhead (offset-preserving) [Shotcut Paste]
    ///   Ctrl+A                select all clips                              [Shotcut Select All]
    ///   I / O                 set in / out point at the playhead            [Shotcut Set In / Set Out]
    ///   A                     append the selected clip to its track end     [Shotcut Append]
    ///   B                     overwrite the selected clip at mark-in/playhead [Shotcut Overwrite]
    ///   V                     ripple-insert the selected clip at mark-in/playhead [Shotcut Insert]
    ///   Left / Right          step playhead -/+1 (clamped)  |  Space  toggle audio play/pause
    ///
    /// Focus guard: `ctx.wants_keyboard_input()` is true whenever ANY focusable widget holds
    /// focus, not just a `TextEdit`. In egui 0.31 it is `memory.focused().is_some()`, so it also
    /// covers the toolbar `Button`s — shortcuts are intentionally suppressed while a button or text
    /// field has focus (so typing in a future rename field can't razor a clip, and the focus guard
    /// makes a plain Backspace/Z lift safe: a focused numeric DragValue swallows them first).
    /// The toolbar Play/Pause button surrenders focus after creation so Space stays single-source.
    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }

        // Snapshot every key/modifier we care about in one input() borrow.
        let k = ctx.input(|i| {
            let m = &i.modifiers;
            // `command` is Ctrl on Linux/Windows and Cmd on macOS (egui-normalized).
            let cmd = m.command || m.ctrl;
            let z = i.key_pressed(egui::Key::Z);
            let y = i.key_pressed(egui::Key::Y);
            Keys {
                // S with no modifier → split the selected clip. Shift+S → razor ALL tracks at the
                // playhead (Shotcut "Split All Tracks"). The two are mutually exclusive on the shift
                // bit so one keypress can only fire one of them.
                split: i.key_pressed(egui::Key::S) && !cmd && !m.shift && !m.alt,
                razor_all: i.key_pressed(egui::Key::S) && m.shift && !cmd && !m.alt,
                freeze: i.key_pressed(egui::Key::F) && m.shift && !cmd && !m.alt, // Shift+F — freeze frame

                // Bare , / . nudge the selected clip -1 / +1 frame on its track. The MARKER-seek
                // pair is Shift+Comma / Shift+Period ('<' / '>'), so the nudge guards !shift (and
                // !cmd/!alt) to stay disjoint from those and from any Ctrl/Alt combos.
                nudge_left: i.key_pressed(egui::Key::Comma) && !m.shift && !cmd && !m.alt,
                nudge_right: i.key_pressed(egui::Key::Period) && !m.shift && !cmd && !m.alt,
                // Lift (Shotcut Z / Backspace / Delete): leaves a gap. NOW shift-guarded on Delete
                // too — Shift+Delete / Shift+Backspace are RIPPLE delete (below), so the plain-lift
                // path must not also fire for them (that would double-edit on one keypress). Z with
                // no Ctrl/Shift (Ctrl+Z stays undo); Delete/Backspace with no Shift and no Ctrl.
                lift: (i.key_pressed(egui::Key::Delete) && !m.shift && !cmd)
                    || (z && !cmd && !m.shift)
                    || (i.key_pressed(egui::Key::Backspace) && !m.shift && !cmd),
                // Ripple delete (Shotcut X / Shift+Delete / Shift+Backspace): close the gap. X with
                // no Ctrl (Ctrl+X is cut, below). The shift variants are the explicit ripple keys.
                ripple_delete: (i.key_pressed(egui::Key::X) && !cmd)
                    || (i.key_pressed(egui::Key::Delete) && m.shift && !cmd)
                    || (i.key_pressed(egui::Key::Backspace) && m.shift && !cmd),
                // Clipboard (Shotcut Ctrl+C / Ctrl+X / Ctrl+V). Require the command/ctrl modifier so
                // the bare C/V/X Shotcut aliases don't collide with our X = ripple delete above.
                copy: cmd && i.key_pressed(egui::Key::C) && !m.shift && !m.alt,
                cut: cmd && i.key_pressed(egui::Key::X) && !m.shift && !m.alt,
                paste: cmd && i.key_pressed(egui::Key::V) && !m.shift && !m.alt,
                select_all: cmd && i.key_pressed(egui::Key::A) && !m.shift && !m.alt, // Ctrl+A
                // P4 3-point editing. Bare (no Ctrl/Shift/Alt) so they don't collide with the
                // clipboard combos above: A (append) vs Ctrl+A (select all); V (insert) vs Ctrl+V
                // (paste). I/O set the in/out marks; B overwrites. Matching Shotcut's player/timeline
                // single-key shortcuts (I, O, A, B, V).
                mark_in: i.key_pressed(egui::Key::I) && !cmd && !m.shift && !m.alt,
                mark_out: i.key_pressed(egui::Key::O) && !cmd && !m.shift && !m.alt,
                append: i.key_pressed(egui::Key::A) && !cmd && !m.shift && !m.alt,
                overwrite: i.key_pressed(egui::Key::B) && !cmd && !m.shift && !m.alt,
                insert: i.key_pressed(egui::Key::V) && !cmd && !m.shift && !m.alt,
                undo: cmd && z && !m.shift,         // Ctrl+Z (no shift) → undo
                redo: cmd && ((z && m.shift) || y), // Ctrl+Shift+Z OR Ctrl+Y → redo
                // Step keys: arrows with NO Alt (Alt+arrow is skip-edit, handled separately).
                left: i.key_pressed(egui::Key::ArrowLeft) && !m.alt,
                right: i.key_pressed(egui::Key::ArrowRight) && !m.alt,
                space: i.key_pressed(egui::Key::Space),
                marker: i.key_pressed(egui::Key::M) && !cmd && !m.shift && !m.alt, // M = create marker
                // '<' = Shift+Comma / '>' = Shift+Period; exclude Ctrl so Ctrl+Shift+Comma/Period
                // doesn't fire marker-seek (consistency with marker/zoom_* which all guard !cmd).
                prev_marker: i.key_pressed(egui::Key::Comma) && m.shift && !cmd,
                next_marker: i.key_pressed(egui::Key::Period) && m.shift && !cmd,
                // Zoom (no Ctrl/Alt): '='/'+' in, '-' out, '0' fit. egui maps the `=`/`+` physical
                // key to Equals (unshifted) or Plus (shifted); accept both.
                zoom_in: (i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals)) && !cmd && !m.alt,
                zoom_out: i.key_pressed(egui::Key::Minus) && !cmd && !m.alt,
                zoom_fit: i.key_pressed(egui::Key::Num0) && !cmd && !m.alt,
                prev_edit: i.key_pressed(egui::Key::ArrowLeft) && m.alt,   // Alt+Left = Skip Previous
                next_edit: i.key_pressed(egui::Key::ArrowRight) && m.alt,  // Alt+Right = Skip Next
                home: i.key_pressed(egui::Key::Home),
                end: i.key_pressed(egui::Key::End),
                j: i.key_pressed(egui::Key::J) && !cmd && !m.alt,
                kk: i.key_pressed(egui::Key::K) && !cmd && !m.alt,
                l: i.key_pressed(egui::Key::L) && !cmd && !m.alt,
                snap_toggle: cmd && i.key_pressed(egui::Key::P), // Ctrl+P
            }
        });

        // --- structural edits: snapshot BEFORE each mutation so undo restores pre-edit state.
        // Only push history when the edit will actually mutate, so a no-op keypress (e.g. S
        // with the playhead outside the clip) doesn't create a dead undo step. We mirror
        // split_clip's own precondition (0 < off < len) to decide whether the split lands.
        // LOCK ENFORCEMENT (wave P1): a split/lift on a clip whose track is_locked() is refused
        // (same rule the timeline drag cascade applies to trim/move).
        if k.split {
            self.do_split();
        }

        // --- RAZOR ALL TRACKS (Shift+S): split every clip on every track that STRICTLY spans the
        // playhead. ONE history snapshot per gesture, gated so a playhead that lands on no clip body
        // (only edges / empty) leaves no dead undo step: we pre-check for at least one strictly
        // spanning clip before pushing. The model op (`split_all_at`) does the cuts via `split_clip`
        // (src continuity + PiP keyframe remap), so this UI path stays a thin wrapper.
        if k.razor_all {
            self.do_razor_all();
        }

        // --- T4 FREEZE FRAME (Shift+F): at the playhead INSIDE the selected clip, split it and insert
        // a 1-second (FPS-frame) silent held still that freezes the source frame under the playhead,
        // rippling the right part + later same-track clips right by the hold length. ONE history
        // snapshot per gesture, gated on the playhead being STRICTLY inside the selected clip body
        // (mirroring freeze_frame's own `t0 < t < end()` guard) and the track being unlocked, so a
        // no-op keypress (playhead on an edge / outside / locked track) leaves no dead undo step.
        if k.freeze {
            let t = self.playhead;
            let probe = self.project.clips.get(self.selected).map(|c| {
                (c.t0 < t && t < c.end(), c.track)
            });
            if let Some((in_body, track)) = probe {
                if in_body && !self.project.is_locked(track) {
                    let dur = Self::FPS as i64; // default 1-second hold
                    self.history.push(&self.project);
                    let _ = self.project.freeze_frame(self.selected, t, dur);
                    // The split + insert renumbers indices around the selected clip; keep it in range.
                    self.clamp_selected();
                    self.clamp_selection();
                }
            }
        }

        // --- FRAME NUDGE (, / .): move the selected clip -1 / +1 frame on its track (free move,
        // overlaps allowed, t0 clamped >= 0). ONE history snapshot per keypress, gated on the clip
        // being in range and on its track being unlocked (a locked clip refuses moves, like the
        // trim/move cascade). `. ` and `,` fire at most one per frame via the disjoint key bindings.
        if k.nudge_left || k.nudge_right {
            let delta = if k.nudge_right { 1 } else { -1 };
            let movable = self
                .project
                .clips
                .get(self.selected)
                .map(|c| !self.project.is_locked(c.track))
                .unwrap_or(false);
            if movable {
                self.history.push(&self.project);
                let _ = self.project.nudge_clip(self.selected, delta);
            }
        }

        if k.lift {
            self.do_lift();
        }

        // --- ripple delete (X / Shift+Delete) and cut (Ctrl+X). Both close the gap on the track;
        // cut copies to the clipboard first. The clipboard keys are handled together here so the
        // shift/ctrl disambiguation (above) cleanly routes X→ripple, Ctrl+X→cut, Ctrl+C→copy,
        // Ctrl+V→paste, Ctrl+A→select-all. Ripple/cut/paste push history themselves (guarded so a
        // no-op never leaves a dead undo step); copy/select-all do not mutate the project.
        if k.cut {
            self.ripple_delete_selection(true);
        } else if k.ripple_delete {
            self.ripple_delete_selection(false);
        }
        if k.copy {
            self.copy_selection();
        }
        if k.paste {
            self.paste_clipboard();
        }
        if k.select_all {
            self.select_all();
        }

        // --- P4 3-point editing (I/O marks, A append, B overwrite, V insert). The marks (I/O) are
        // app state and push no history; the placement ops (A/B/V) push history themselves (guarded
        // so a no-op never leaves a dead undo step). Append/overwrite/insert use the selected clip as
        // the source and its own track as the target, so they are inert on an empty timeline.
        // P18: I/O set the SOURCE in/out (source-local) when the Source monitor is active, else the
        // timeline marks (Program-mode behavior unchanged — byte-identical to P4).
        if k.mark_in {
            match self.monitor {
                MonitorMode::Source => self.set_src_in(),
                MonitorMode::Program => self.set_mark_in(),
            }
        }
        if k.mark_out {
            match self.monitor {
                MonitorMode::Source => self.set_src_out(),
                MonitorMode::Program => self.set_mark_out(),
            }
        }
        if k.append {
            self.append_source();
        }
        if k.overwrite {
            self.overwrite_source();
        }
        if k.insert {
            self.insert_source();
        }

        // --- undo / redo: redo wins if both somehow fire (shift state disambiguates above).
        if k.redo {
            self.history.redo(&mut self.project);
            self.clamp_selected();
            self.clamp_selection();
            self.clamp_playhead();
        } else if k.undo {
            self.history.undo(&mut self.project);
            self.clamp_selected();
            self.clamp_selection();
            self.clamp_playhead();
        }

        // --- markers (M / < / >).
        if k.marker {
            self.add_marker();
        }
        if k.prev_marker {
            self.seek_prev_marker();
        }
        if k.next_marker {
            self.seek_next_marker();
        }

        // --- zoom (= / + / - / 0). Multiplicative steps, clamped to the ppf range; fit is deferred
        // to timeline_ui (which knows the lane width) via the one-shot zoom_fit_pending flag.
        if k.zoom_in {
            self.ppf = (self.ppf * ZOOM_STEP).clamp(MIN_PPF, MAX_PPF);
        }
        if k.zoom_out {
            self.ppf = (self.ppf / ZOOM_STEP).clamp(MIN_PPF, MAX_PPF);
        }
        if k.zoom_fit {
            self.zoom_fit_pending = true;
        }

        // --- edit-point seek (Alt+Left / Alt+Right).
        if k.prev_edit {
            self.seek_prev_edit();
        }
        if k.next_edit {
            self.seek_next_edit();
        }

        // --- Home / End.
        if k.home {
            self.shuttle = 0;
            self.playhead = 0;
            self.clamp_playhead();
        }
        if k.end {
            self.shuttle = 0;
            self.playhead = (self.project.total_frames() - 1).max(0);
            self.clamp_playhead();
        }

        // --- snap toggle (Ctrl+P).
        if k.snap_toggle {
            self.snap = !self.snap;
            self.status = if self.snap { "snap on".into() } else { "snap off".into() };
        }

        // --- JKL shuttle. K pauses everything (audio play + shuttle). L steps the FORWARD shuttle
        // speed up (1,2,4,8); J steps the REVERSE shuttle speed up (-1,-2,-4,-8). Starting a shuttle
        // cancels audio play (Space) and re-anchors the wall-clock transport at the current frame
        // so the shuttle advance below is measured from here. Mirrors Shotcut: J/L are silent fast
        // rewind/forward, distinct from Space (audio playback).
        if k.kk {
            self.shuttle = 0;
            self.playing = false;
        }
        if k.l {
            self.playing = false;
            self.shuttle = if self.shuttle >= 1 {
                (self.shuttle * 2).min(MAX_SHUTTLE)
            } else {
                1 // was paused or reversing → start forward at 1x
            };
            self.anchor_transport(self.playhead);
        }
        if k.j {
            self.playing = false;
            self.shuttle = if self.shuttle <= -1 {
                (self.shuttle * 2).max(-MAX_SHUTTLE)
            } else {
                -1 // was paused or forwarding → start reverse at 1x
            };
            self.anchor_transport(self.playhead);
        }

        // --- transport / scrub. Space toggles AUDIO play; starting it cancels any JKL shuttle.
        // A single-frame step (Left/Right) also cancels the shuttle (matches Shotcut where a frame
        // step drops out of fast play).
        if k.space {
            self.playing = !self.playing;
            self.shuttle = 0;
        }
        if k.left {
            self.shuttle = 0;
            self.playhead -= 1;
            self.clamp_playhead();
        }
        if k.right {
            self.shuttle = 0;
            self.playhead += 1;
            self.clamp_playhead();
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add_space(6.0);
            ui.label(egui::RichText::new("Genesis").color(Color32::from_rgb(230, 214, 128)).size(15.0));
            ui.separator();

            // Add: import a media file into the pool (same outcome as the pool panel's
            // "+ Add media"). We inline the import here rather than call into pool.rs because
            // pool.rs is owned by another slice this wave and exposes only `pool_ui`. The
            // whitespace guard mirrors pool::pick_file: the gcompose serve protocol is a single
            // space-delimited line, so a path with spaces would inflate the field count and the
            // worker would reject it — we refuse it at import time and surface why in the status.
            if tb_button(ui, "add", "Add") {
                if let Some(path) = zenity(&["--file-selection", "--title=Add media"]) {
                    if path.contains(char::is_whitespace) {
                        self.status = format!("can't add '{}': path has whitespace", path);
                    } else {
                        let name = path.rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or(&path).to_string();
                        self.project.media.push(path);
                        self.project.names.push(name);
                        self.status = "media added".into();
                    }
                }
            }

            // Open: native picker → load JSON → replace the whole project, resetting view state.
            // T2: routed through `open_project_path` (the SAME canonical loader the Recent menu uses)
            // so a manual Open and a Recent click reset state identically AND both record the recents.
            if tb_button(ui, "open", "Open") {
                if let Some(path) = pick_file_open() {
                    self.open_project_path(&path);
                }
            }

            // Save: native save picker → serialize current project to JSON.
            if tb_button(ui, "save", "Save") {
                if let Some(path) = pick_file_save("project.gnp") {
                    match project_io::save(&self.project, &path) {
                        Ok(()) => {
                            self.status = format!("saved {}", path);
                            // T2: a successful save promotes the saved path to the recents list too,
                            // so a freshly-saved project is immediately re-openable from "Recent".
                            self.record_recent(&path);
                        }
                        Err(e) => self.status = format!("save failed: {}", e),
                    }
                }
            }

            // T2 RECENT FILES — a small "Recent" menu in the existing top chrome listing recently
            // opened/saved projects, most-recent-first. Clicking an entry re-opens it via the SAME
            // `open_project_path` loader used by Open (so state resets identically). Existing files
            // are clickable; missing files (deleted/renamed/moved) are shown GREYED and disabled so a
            // stale entry can't silently fail. ADDED here (after Save) — no existing button changed.
            // `recent_click` defers the actual open until AFTER the menu closure so the
            // `open_project_path(&mut self)` borrow doesn't overlap the closure's `&self.recents`.
            let mut recent_click: Option<String> = None;
            ui.menu_button("Recent", |ui| {
                if self.recents.is_empty() {
                    ui.add_enabled(false, egui::Button::new("(no recent projects)"));
                } else {
                    for path in &self.recents {
                        let exists = std::path::Path::new(path).exists();
                        // Show just the file name as the label, full path as hover text.
                        let name = path.rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or(path);
                        if ui
                            .add_enabled(exists, egui::Button::new(name))
                            .on_hover_text(path)
                            .on_disabled_hover_text(format!("missing: {}", path))
                            .clicked()
                        {
                            recent_click = Some(path.clone());
                            ui.close_menu();
                        }
                    }
                }
            });
            if let Some(path) = recent_click {
                self.open_project_path(&path);
            }

            // Render: native save picker (default out.mp4) → BLOCKING full-program encode.
            // worker::render_program() composites + encodes every frame on the UI thread and
            // does not return until the mp4 is finished — the window is frozen for the duration.
            // Accepted for this wave (mirrors MojoMedia's synchronous render loop); a background
            // render + progress bar is a follow-up.
            if tb_button(ui, "export", "Render") {
                if let Some(path) = pick_file_save("out.mp4") {
                    self.status = format!("rendering \u{2192} {} \u{2026}", path);
                    // NOTE: this only *schedules* a future repaint; we are mid-frame inside this
                    // closure and render_program() blocks the UI thread below, so the "rendering"
                    // status is NOT painted before the block — it jumps straight to rendered/
                    // failed. Kept (harmless) so a future background-render refactor already has
                    // the repaint nudge in place; it is effectively a no-op this wave.
                    ui.ctx().request_repaint();
                    let ok = worker::render_program(&self.project, &path);
                    self.status = if ok {
                        format!("rendered {}", path)
                    } else {
                        format!("render FAILED \u{2192} {}", path)
                    };
                }
            }

            ui.separator();

            // Undo / Redo buttons mirror the keyboard shortcuts. Disabled when the stack is empty.
            if tb_button_enabled(ui, self.history.can_undo(), "undo", "Undo") {
                self.history.undo(&mut self.project);
                self.clamp_selected();
                self.clamp_selection();
                self.clamp_playhead();
            }
            if tb_button_enabled(ui, self.history.can_redo(), "redo", "Redo") {
                self.history.redo(&mut self.project);
                self.clamp_selected();
                self.clamp_selection();
                self.clamp_playhead();
            }

            ui.separator();

            // Razor all tracks: split every clip on every track at the current playhead (mirrors the
            // Shift+S shortcut). Gated identically to the key path — push history only when at least
            // one clip strictly spans the playhead, so a no-op click leaves no dead undo step. There
            // is no dedicated "razor-all" icon in the baked blob; `split` reuses the single-razor
            // glyph (falls back to the text label when the icon is missing).
            if tb_button(ui, "split", "Razor all tracks") {
                let t = self.playhead;
                let any_span = self.project.clips.iter().any(|c| c.t0 < t && t < c.end());
                if any_span {
                    self.history.push(&self.project);
                    let _ = self.project.split_all_at(t);
                    self.clamp_selected();
                    self.clamp_selection();
                    self.status = "razor all tracks".into();
                }
            }

            ui.separator();

            // Play/Pause transport toggle. Space is the canonical toggle (handled in
            // handle_keys); this button is a click-only mirror. `tb_button` surrenders the
            // button's focus so egui's Space-activates-focused-button behaviour can never fire a
            // synthetic click here and double-toggle transport against the handle_keys Space path.
            let (play_icon, play_label) = if self.playing { ("pause", "Pause") } else { ("play", "Play") };
            if tb_button(ui, play_icon, play_label) {
                self.playing = !self.playing;
            }

            ui.separator();
            // Reload re-composites the current frame. There is no dedicated "reload" icon in the
            // baked blob; `loop` is the closest semantic match (and falls back to text if missing).
            if tb_button(ui, "loop", "Reload") {
                // Force a re-composite of the current frame. Do NOT clear `preview_inited`:
                // the frame-2 init gate only fires once, so clearing it would permanently
                // disable the playhead-moved re-composite path in update() (gated on preview_inited).
                // Setting last_composed = -1 (which differs from any valid playhead >= 0)
                // makes the next update() re-composite via that path.
                self.last_composed = -1;
            }
            ui.separator();
            ui.label(egui::RichText::new(&self.status).color(theme::ACCENT).size(11.0));
        });
    }

    /// P18: source scrubber upper bound. The clip length isn't known here without a decode, so we
    /// bound the slider at a generous fixed span (10s @ 30fps) — enough to scrub into any short
    /// pool clip. Decodes past the clip end fail gracefully (thumbnail -> None -> status note).
    const SRC_SCRUB_MAX: i64 = 600;

    /// Shotcut-style TIMELINE TOOLBAR: a row of icon buttons that fire the SAME edit gestures as the
    /// keyboard, so there is exactly one code path per op (the shared `do_*`/`*_selection`/`*_source`
    /// methods). Drawn at the top of the timeline panel, above `timeline_ui`. Every op method is
    /// internally guarded (no selection / locked track / out-of-body playhead => no-op + no dead undo
    /// step), so every button is always safe to click. `tb_button` surrenders focus so a click never
    /// suppresses the keyboard shortcuts on the next frame. Real baked Shotcut-dark icons with a word
    /// fallback. Wrapped so it degrades gracefully if the timeline panel is narrow.
    fn timeline_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if tb_button(ui, "split", "Split") {
                self.do_split();
            }
            if tb_button(ui, "slice", "Razor all") {
                self.do_razor_all();
            }
            ui.separator();
            if tb_button(ui, "lift", "Lift") {
                self.do_lift();
            }
            if tb_button(ui, "ripple", "Ripple") {
                self.ripple_delete_selection(false);
            }
            ui.separator();
            if tb_button(ui, "cut", "Cut") {
                self.ripple_delete_selection(true);
            }
            if tb_button(ui, "copy", "Copy") {
                self.copy_selection();
            }
            if tb_button(ui, "paste", "Paste") {
                self.paste_clipboard();
            }
            ui.separator();
            if tb_button(ui, "add", "Append") {
                self.append_source();
            }
            if tb_button(ui, "overwrite", "Overwrite") {
                self.overwrite_source();
            }
            if tb_button(ui, "", "Insert") {
                self.insert_source();
            }
            ui.separator();
            if tb_button(ui, "marker", "Marker") {
                self.add_marker();
            }
            // Snap is a toggle: show a check when on. The label doubles as the text fallback.
            let snap_label = if self.snap { "Snap \u{2713}" } else { "Snap" };
            if tb_button(ui, "snap", snap_label) {
                self.snap = !self.snap;
            }
            ui.separator();
            // Zoom mirrors the =/-/0 keys exactly (same ZOOM_STEP + clamp; fit defers to timeline_ui).
            if tb_button(ui, "zoom_out", "\u{2212}") {
                self.ppf = (self.ppf / ZOOM_STEP).clamp(MIN_PPF, MAX_PPF);
            }
            if tb_button(ui, "zoom_fit", "Fit") {
                self.zoom_fit_pending = true;
            }
            if tb_button(ui, "zoom_in", "+") {
                self.ppf = (self.ppf * ZOOM_STEP).clamp(MIN_PPF, MAX_PPF);
            }
        });
        ui.add_space(2.0);
    }

    fn preview_pane(&mut self, ui: &mut egui::Ui) {
        ui.painter().rect_filled(ui.max_rect(), egui::CornerRadius::ZERO, Color32::from_rgb(10, 10, 12));

        // --- TOP ROW: Project / Source monitor tabs (Shotcut Project vs Source). selectable_label
        // sets `self.monitor`; switching to Source with a clip already open re-uses its texture.
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            if ui
                .selectable_label(self.monitor == MonitorMode::Program, "Project")
                .clicked()
            {
                self.monitor = MonitorMode::Program;
            }
            if ui
                .selectable_label(self.monitor == MonitorMode::Source, "Source")
                .clicked()
            {
                self.monitor = MonitorMode::Source;
            }
        });
        ui.separator();

        match self.monitor {
            // PROGRAM: the composited-timeline preview + a transport/scrub strip beneath it.
            // Layout mirrors the Source monitor (bottom_up): the transport row sits at the very
            // bottom, the scrub slider just above it, and the program image fills the remainder.
            // We only MUTATE self.playhead/self.playing here — update()'s recompose guard (playhead
            // moved off last_composed) and its transport START/STOP edges pick the change up next
            // frame, so this method stays ctx-free and borrow-clean (no compose call from here).
            MonitorMode::Program => {
                let total = self.project.total_frames();
                let max_f = (total - 1).max(0);
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                    // Transport row (bottom-most). Real baked icons (skip_back/play/pause/stop/
                    // skip_fwd) with a word label that also serves as the text fallback when the
                    // icon blob is unavailable. tb_button surrenders focus so Space never double-
                    // toggles transport against the handle_keys Space path.
                    // Shotcut player transport order: skip-to-start, rewind, play/pause, fast-forward,
                    // skip-to-end, then the SMPTE timecode readout (current / total). Real baked icons
                    // (media-skip/seek/playback from the Shotcut-dark blob); the word labels double as
                    // the text fallback if the icon blob is unavailable. Rewind/FF reuse the JKL shuttle.
                    ui.horizontal(|ui| {
                        if tb_button(ui, "skip_back", "Start") {
                            self.playing = false;
                            self.shuttle = 0;
                            self.playhead = 0;
                        }
                        if tb_button(ui, "seek_back", "Rew") {
                            self.shuttle_reverse();
                        }
                        let (play_icon, play_label) =
                            if self.playing { ("pause", "Pause") } else { ("play", "Play") };
                        if tb_button(ui, play_icon, play_label) {
                            self.playing = !self.playing;
                        }
                        if tb_button(ui, "seek_fwd", "FF") {
                            self.shuttle_forward();
                        }
                        if tb_button(ui, "skip_fwd", "End") {
                            self.playing = false;
                            self.shuttle = 0;
                            self.playhead = max_f;
                        }
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!(
                                "{} / {}",
                                timecode(self.playhead, Self::FPS),
                                timecode(max_f, Self::FPS)
                            ))
                            .color(theme::ACCENT)
                            .monospace()
                            .size(13.0),
                        );
                    });
                    // Scrub slider just above the transport row. Dragging it PAUSES playback —
                    // otherwise the wall-clock advance in update() would immediately overwrite the
                    // scrubbed frame from the play anchor. Surrender focus so the slider never
                    // captures Space/arrows (which would suppress the global shortcuts in handle_keys).
                    let resp = ui.add(
                        egui::Slider::new(&mut self.playhead, 0..=max_f).show_value(false).text("scrub"),
                    );
                    if resp.changed() {
                        self.playing = false;
                    }
                    resp.surrender_focus();
                    // Remaining space (above the strip) holds the centered program image (or status).
                    ui.centered_and_justified(|ui| {
                        if let Some(tex) = &self.preview {
                            let src = egui::load::SizedTexture::new(tex.id(), tex.size_vec2());
                            ui.add(egui::Image::new(src).maintain_aspect_ratio(true).max_size(ui.available_size()));
                        } else {
                            ui.label(self.status.clone());
                        }
                    });
                });
            }
            // SOURCE: a raw frame of the opened pool clip + a source scrubber + in/out readout.
            MonitorMode::Source => {
                if self.src_media.is_none() {
                    ui.centered_and_justified(|ui| {
                        ui.label("No source \u{2014} open a clip from the pool (\u{25B6} Source)");
                    });
                    return;
                }

                // Reserve the scrubber strip at the BOTTOM, draw the image centered in what's left.
                // We draw the scrubber first into a bottom-up layout so the image gets the remainder.
                ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                    // Source in/out readout (only when set).
                    if self.src_in.is_some() || self.src_out.is_some() {
                        let a = self.src_in.map(|v| v.to_string()).unwrap_or_else(|| "\u{2014}".into());
                        let b = self.src_out.map(|v| v.to_string()).unwrap_or_else(|| "\u{2014}".into());
                        ui.label(
                            egui::RichText::new(format!("in {} out {}", a, b))
                                .color(theme::ACCENT)
                                .size(11.0),
                        );
                    }
                    // Source scrubber. Mutating `src_playhead` here is picked up by update()'s
                    // recompose guard (a scrub flips its `src_moved`/`!= src_last` terms) — we
                    // deliberately do NOT call compose_source from here (no ctx, and to keep this
                    // method borrow-clean).
                    ui.add(
                        egui::Slider::new(&mut self.src_playhead, 0..=Self::SRC_SCRUB_MAX)
                            .text("source frame"),
                    );

                    // The remaining space (above the strip) holds the centered source image.
                    ui.centered_and_justified(|ui| {
                        if let Some(tex) = &self.src_preview {
                            let src = egui::load::SizedTexture::new(tex.id(), tex.size_vec2());
                            ui.add(egui::Image::new(src).maintain_aspect_ratio(true).max_size(ui.available_size()));
                        } else {
                            ui.label("decoding source\u{2026}");
                        }
                    });
                });
            }
        }
    }
}

/// Pixel size the 32×32 icon blob is drawn at in the toolbar (down-scaled to fit a ~40px bar).
const TB_ICON: f32 = 18.0;

/// Build an `egui::Button` that shows the real PNG icon (`icons::icon(icon_name)`) to the LEFT of
/// `label`, falling back to a text-only button when the icon can't be loaded. `enabled` gates the
/// button (used by Undo/Redo).
///
/// egui 0.31: `Button::image_and_text(ImageSource, impl Into<WidgetText>)` lays the image before
/// the text. We feed it an `egui::Image` built from a `SizedTexture` over the cached icon's
/// `TextureId`, capped to `TB_ICON` square. On `None` (unknown name / missing blob) we degrade to
/// `Button::new(label)` so the toolbar is always usable.
fn icon_button(ui: &mut egui::Ui, icon_name: &str, label: &str, enabled: bool) -> egui::Response {
    let button = match icons::icon(ui.ctx(), icon_name) {
        Some(id) => {
            let img = egui::Image::new(egui::load::SizedTexture::new(
                id,
                egui::vec2(TB_ICON, TB_ICON),
            ))
            .fit_to_exact_size(egui::vec2(TB_ICON, TB_ICON));
            egui::Button::image_and_text(img, label)
        }
        None => egui::Button::new(label),
    };
    ui.add_enabled(enabled, button)
}

/// Add a toolbar button (icon + label) that never *keeps* keyboard focus: it surrenders focus
/// immediately after creation and returns whether it was clicked this frame.
///
/// Why surrender focus: `handle_keys` is gated on `ctx.wants_keyboard_input()`, which in egui 0.31
/// is true whenever ANY focusable widget holds focus — including a just-clicked toolbar `Button`.
/// If a toolbar button retained focus, every editing/transport shortcut (S, Delete, arrows,
/// Ctrl+Z, Space) would be silently suppressed until the user clicked elsewhere. Surrendering the
/// button's focus right after it is shown keeps shortcuts live after any toolbar action and, for
/// the Play button specifically, prevents egui's Space-activates-focused-button behaviour from
/// double-toggling transport against the `handle_keys` Space path.
///
/// `icon_name` is a lowercase name from the baked icon blob (see `icons.rs`); a missing/unknown
/// icon degrades gracefully to a text-only button.
fn tb_button(ui: &mut egui::Ui, icon_name: &str, label: &str) -> bool {
    let resp = icon_button(ui, icon_name, label, true);
    let clicked = resp.clicked();
    resp.surrender_focus();
    clicked
}

/// Like `tb_button` but for an enabled/disabled `Button` (Undo/Redo). Same focus-surrender
/// rationale and icon-with-text-fallback behaviour as `tb_button`.
fn tb_button_enabled(ui: &mut egui::Ui, enabled: bool, icon_name: &str, label: &str) -> bool {
    let resp = icon_button(ui, icon_name, label, enabled);
    let clicked = resp.clicked();
    resp.surrender_focus();
    clicked
}

/// A thin labeled dock-header bar drawn atop a panel.
fn dock_header(ui: &mut egui::Ui, label: &str) {
    let full = ui.available_rect_before_wrap();
    let h = 20.0;
    let bar = egui::Rect::from_min_size(full.min, egui::Vec2::new(full.width(), h));
    let painter = ui.painter();
    painter.rect_filled(bar, egui::CornerRadius::ZERO, theme::ALT_BASE);
    painter.text(
        bar.min + egui::Vec2::new(8.0, h * 0.5),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(11.0),
        Color32::from_rgb(160, 160, 170),
    );
    ui.allocate_rect(bar, egui::Sense::hover());
    ui.add_space(2.0);
}

/// Format a frame index as Shotcut-style SMPTE timecode `HH:MM:SS:FF` at `fps`. The frame field
/// (`FF`) counts 0..fps-1 within the current second; the rest is wall-clock from frame 0. `fps` is
/// rounded to an integer frame base (Genesis runs an integer 30 fps program, matching the OPEN
/// default) and floored at 1 so a degenerate fps never divides by zero. Negative frames clamp to 0.
fn timecode(frame: i64, fps: f64) -> String {
    let base = (fps.round() as i64).max(1);
    let f = frame.max(0);
    let ff = f % base;
    let secs = f / base;
    let (h, m, s) = (secs / 3600, (secs / 60) % 60, secs % 60);
    format!("{:02}:{:02}:{:02}:{:02}", h, m, s, ff)
}

impl eframe::App for Genesis {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.frames += 1;
        if self.frames == 2 {
            self.ensure_preview(ctx);
            // Headless render gate (GENESIS_RENDER=<out.mp4>): exercise the real UI->worker
            // render_program path, then exit. Mirrors the GENESIS_SHOT gate.
            if let Ok(out) = std::env::var("GENESIS_RENDER") {
                let ok = worker::render_program(&self.project, &out);
                eprintln!("GENESIS_RENDER {} -> {}", out, ok);
                worker::shutdown();
                std::process::exit(if ok { 0 } else { 1 });
            }
            // P33 headless auto-save gate (GENESIS_AUTOSAVE=<out.json>): exercise the real
            // project_io::save path the periodic auto-save uses, then exit. INDEPENDENT of
            // GENESIS_RENDER (own `if`); writes to the GIVEN path (not RECOVERY_PATH) so the gate is
            // self-contained. Read-only w.r.t. the project — it serializes `self.project` unchanged.
            if let Ok(out) = std::env::var("GENESIS_AUTOSAVE") {
                let ok = project_io::save(&self.project, &out).is_ok();
                eprintln!("GENESIS_AUTOSAVE {} -> {}", out, ok);
                worker::shutdown();
                std::process::exit(if ok { 0 } else { 1 });
            }
            // Headless audio-spectrum gate (GENESIS_SPECTRUM=<out.f32>): exercise the real
            // UI->worker program_spectrum path (MEAS / AUDIO* / SPECTRUM), write the returned bins
            // as little-endian f32, then exit. INDEPENDENT of GENESIS_RENDER (own `if`, runs whether
            // or not the render hook above fired — though render exits first if set). Read-only: it
            // changes nothing in the render/mix/LEVELS path. On None (nothing to measure / worker
            // flake) we write an EMPTY file so the harness still finds the artifact.
            if let Ok(out) = std::env::var("GENESIS_SPECTRUM") {
                let bins = worker::program_spectrum(&self.project, self.playhead);
                let nbins = bins.as_ref().map(|b| b.len()).unwrap_or(0);
                let mut bytes: Vec<u8> = Vec::with_capacity(nbins * 4);
                if let Some(bins) = &bins {
                    for &m in bins {
                        bytes.extend_from_slice(&m.to_le_bytes());
                    }
                }
                let ok = std::fs::write(&out, &bytes).is_ok();
                eprintln!("GENESIS_SPECTRUM {} -> {}", out, nbins);
                worker::shutdown();
                std::process::exit(if ok { 0 } else { 1 });
            }
            // P40 headless audio-waveform gate (GENESIS_SAMPLES=<out.f32>): exercise the real
            // UI->worker program_samples path (MEAS / AUDIO* / SAMPLES), write the returned raw
            // time-domain samples as little-endian f32, then exit. INDEPENDENT of GENESIS_SPECTRUM /
            // GENESIS_RENDER (own `if`, runs whether or not the hooks above fired — though they exit
            // first if set). Read-only: it changes nothing in the render/mix/LEVELS/SPECTRUM path. On
            // None (nothing to measure / worker flake) we write an EMPTY file so the harness still
            // finds the artifact.
            if let Ok(out) = std::env::var("GENESIS_SAMPLES") {
                let samples = worker::program_samples(&self.project, self.playhead);
                let n = samples.as_ref().map(|s| s.len()).unwrap_or(0);
                let mut bytes: Vec<u8> = Vec::with_capacity(n * 4);
                if let Some(samples) = &samples {
                    for &s in samples {
                        bytes.extend_from_slice(&s.to_le_bytes());
                    }
                }
                let ok = std::fs::write(&out, &bytes).is_ok();
                eprintln!("GENESIS_SAMPLES {} -> {}", out, n);
                worker::shutdown();
                std::process::exit(if ok { 0 } else { 1 });
            }

            // P46 headless audio-align gate (GENESIS_ALIGN=<out.txt>): cross-correlate clip 0 (ref)
            // and clip 1 (mov) via the real CLIPAUD decode + model::cross_correlation_offset, write the
            // recovered t0-DELTA (frames) to <out>, then exit. INDEPENDENT of the hooks above. None ->
            // "none".
            if let Ok(out) = std::env::var("GENESIS_ALIGN") {
                let delta = worker::align_audio_offset_frames(&self.project, 0, 1);
                let txt = delta.map(|d| d.to_string()).unwrap_or_else(|| "none".into());
                let ok = std::fs::write(&out, txt.as_bytes()).is_ok();
                eprintln!("GENESIS_ALIGN {} -> {}", out, txt);
                worker::shutdown();
                std::process::exit(if ok { 0 } else { 1 });
            }
        }

        // P33 PERIODIC AUTO-SAVE (crash-recovery sidecar). Placed AFTER the frames==2 headless-hook
        // block above so every GENESIS_* gate keeps its byte-for-byte exit path — those `if`s call
        // `process::exit` before control ever reaches here. PURE SIDE-EFFECT: it writes the live
        // project to RECOVERY_PATH and nothing else (no project/render mutation). It fires only when
        // BOTH (a) at least `AUTOSAVE_INTERVAL_FRAMES` frames have elapsed since the last save, and
        // (b) the project actually changed — `history.len()` (undo-stack depth) differs from the
        // marker captured at the last save. Best-effort: errors are ignored (a flaky /tmp must never
        // take down the editor), and it never blocks/log-spams (interval-gated, no eprintln).
        if self.frames.saturating_sub(self.autosave_frame) >= AUTOSAVE_INTERVAL_FRAMES
            && self.history.len() != self.autosave_marker
        {
            let _ = project_io::save(&self.project, RECOVERY_PATH);
            self.autosave_marker = self.history.len();
            self.autosave_frame = self.frames;
            self.status = "Auto-saved recovery".to_string();
        }

        if !self.preview_inited {
            ctx.request_repaint();
        }

        // Editing/transport keyboard shortcuts (guarded against text-field focus inside).
        // Skipped while a screenshot run is pending so the CI/screenshot path stays headless
        // and deterministic (no stray key events from the harness perturb the frame).
        if self.shot_path.is_none() {
            self.handle_keys(ctx);
        }

        // Transport START edge (Slice C): when `self.playing` flips false->true this frame (via
        // Space in handle_keys or the toolbar Play button), (a) ANCHOR the wall-clock transport at
        // the CURRENT playhead, then (b) audition the timeline audio from that frame. Both fire
        // from the edge (not every frame), detected against the `prev_playing` value stored at the
        // end of the previous update() — mirroring MojoMedia's `space and not prev_space`.
        //
        // `anchor_transport(self.playhead)` records `Instant::now()` so the wall-clock advance
        // below measures real elapsed time from the frame the user pressed Play on. The audio
        // (worker::play_program) writes a WAV + spawns a detached system player from the SAME
        // frame, so the real-time video advance (FPS-paced) and the real-time audio stay in
        // approximate A/V sync. play_program returns immediately (background thread); it is
        // best-effort (a missing player / worker flake is logged on its thread, not surfaced).
        let transport_started = self.playing && !self.prev_playing;
        // Transport STOP edge (Slice C): on true->false, kill any in-flight audio audition so the
        // sound stops with the picture (replaces the old "player keeps going to its end" behavior).
        let transport_stopped = !self.playing && self.prev_playing;
        if transport_started {
            self.anchor_transport(self.playhead);
            worker::play_program(&self.project, self.playhead);
        }
        if transport_stopped {
            worker::stop_playback();
            // Drop the anchor: while paused the user may scrub (arrows/timeline), so the next Play
            // must re-anchor at wherever the playhead then is, not at this stale anchor frame.
            self.play_anchor = None;
        }

        // Transport: while playing, advance the playhead in WALL-CLOCK time so the video track runs
        // at FPS regardless of the egui repaint rate (which is what keeps it matched to the
        // real-time audio). playhead = anchor_frame + floor(elapsed_seconds * FPS), looping at the
        // program end. On a loop wrap we RE-ANCHOR at 0 and restart the audio audition so the next
        // cycle is sample-fresh (the previous audition has run its tail; stop it first to be tidy).
        // We request a repaint every frame while playing so egui (otherwise reactive) keeps ticking
        // and the playhead advances smoothly. Mirrors MojoMedia's wall-clock playhead intent but
        // replaces its `cur_T += 1` per-update advance with true real-time pacing.
        //
        // Guard (skeptic #1): `total_frames()` floors at 1 (model.rs `.max(1)`), so on an empty /
        // 1-frame timeline `next` would cross the `>= total` wrap boundary every ~33 ms and fire
        // `stop_playback()` + `play_program(0)` + `pkill` several times per second — a stop/start
        // audio + pkill storm while "playing" nothing. We only run the wall-clock advance/wrap when
        // there is a REAL program to play (more than one frame). With a 0/1-frame timeline the
        // playhead simply stays put while `playing` is true (audio was already a no-op: an empty
        // program produces no audition), avoiding the thrash. We still request a repaint so the
        // transport toggle/UI stays responsive.
        if self.playing && self.project.total_frames() > 1 {
            // A defensive anchor: if we somehow entered the playing branch without one (e.g. a
            // future code path sets `playing` directly), anchor at the current playhead now.
            if self.play_anchor.is_none() {
                self.anchor_transport(self.playhead);
            }
            let total = self.project.total_frames();
            if let Some(anchor) = self.play_anchor {
                let elapsed = anchor.elapsed().as_secs_f64();
                let advanced = (elapsed * Self::FPS) as i64;
                let mut next = self.play_anchor_frame + advanced;
                if next >= total {
                    // Loop: wrap to 0, re-anchor timing at the wrap, and restart the audition.
                    // NOTE (skeptic #6 — cross-slice, worker-side fix required): on a loop wrap the
                    // prior cycle's audition background thread may still hold `PLAYBACK_IN_FLIGHT`
                    // (worker.rs clears it only when that thread finishes), so this restarting
                    // `play_program(0)` can be SILENTLY DROPPED by its compare_exchange guard while
                    // `stop_playback()` already killed the player child — leaving audio silent after
                    // the first loop. The integrator/Team A must clear `PLAYBACK_IN_FLIGHT` inside
                    // `worker::stop_playback()` (or otherwise allow an immediate restart) for loop
                    // audio to survive. This call is correct UI-side and needs no further change here.
                    next = 0;
                    self.anchor_transport(0);
                    worker::stop_playback();
                    worker::play_program(&self.project, 0);
                }
                self.playhead = next;
            }
            ctx.request_repaint();
        } else if self.playing {
            // Empty / 1-frame timeline: nothing to advance, but keep ticking so the toggle is live.
            ctx.request_repaint();
        }

        // --- JKL shuttle (wave P1): a SILENT wall-clock fast-forward / rewind, separate from the
        // audio-synced `playing` path above (they are mutually exclusive — see handle_keys). The
        // playhead advances `shuttle * FPS` frames/sec from the anchor captured when the shuttle
        // started/changed speed. It does NOT loop: it CLAMPS at the program ends and, on hitting an
        // end, stops the shuttle (matches Shotcut, where fast play halts at the boundary). Guarded
        // to a >1-frame program so an empty/1-frame timeline never thrashes. No audio audition.
        if self.shuttle != 0 && self.project.total_frames() > 1 {
            if self.play_anchor.is_none() {
                self.anchor_transport(self.playhead);
            }
            let total = self.project.total_frames();
            if let Some(anchor) = self.play_anchor {
                let elapsed = anchor.elapsed().as_secs_f64();
                let advanced = (elapsed * Self::FPS * self.shuttle as f64) as i64;
                let next = self.play_anchor_frame + advanced;
                if next <= 0 {
                    self.playhead = 0;
                    self.shuttle = 0; // hit the start: stop rewinding
                    self.play_anchor = None;
                } else if next >= total - 1 {
                    self.playhead = total - 1;
                    self.shuttle = 0; // hit the end: stop fast-forwarding
                    self.play_anchor = None;
                } else {
                    self.playhead = next;
                }
            }
            self.clamp_playhead();
            ctx.request_repaint();
        }

        // Re-composite when the playhead has moved off the frame we last composited.
        // (Skip until the initial gate has fired so the screenshot path stays deterministic.)
        //
        // `compose()` is a synchronous, blocking subprocess round-trip (spawn/IO/file-read).
        // It is run on the UI thread, which is acceptable only because the persistent worker
        // is fast. To avoid hammering it every frame, we only attempt a re-composite when:
        //   - the playhead actually moved since the previous frame (scrub/seek), or
        //   - a forced re-composite is pending: `last_composed == -1` (set by Reload), which
        //     re-runs exactly once because a successful compose sets last_composed = playhead.
        // Note: on a failed compose, `last_composed` is left unchanged; if the playhead is
        // also stationary we will NOT retry every frame (no busy subprocess loop).
        let playhead_moved = self.playhead != self.prev_playhead;
        let force_recomposite = self.last_composed == -1;
        if self.preview_inited && self.playhead != self.last_composed && (playhead_moved || force_recomposite) {
            self.compose(ctx);
        }
        self.prev_playhead = self.playhead;

        // P18: re-compose the SOURCE preview when the source clip or its (source-local) playhead
        // changed — mirrors the program recompose guard above EXACTLY, including its anti-busy-loop
        // movement term. Set by the Source scrubber, "Open in Source" (`open_source` sets
        // `src_last = -1`), and the GENESIS_SOURCE hook (ensure_preview already did the first source
        // compose at frame 2). Like the program path, we attempt a re-decode only when:
        //   - the source playhead actually moved since the previous frame (a scrub), or
        //   - a forced recompose is pending: `src_last == -1` (set by `open_source`), which runs
        //     exactly once because a successful `compose_source` sets `src_last = src_playhead`.
        // CRITICAL: on a FAILED decode `compose_source` leaves `src_last` unchanged, so without the
        // `src_moved` term a stationary slider parked past the clip end (where `thumbnail` returns
        // `None` by design — see SRC_SCRUB_MAX) would hammer the worker round-trip every frame on
        // the UI thread. The movement term gates that, exactly as `playhead_moved` does for the
        // program path. Gated on `preview_inited` (deterministic frame-2 screenshot) + a source
        // being open; `compose_source` re-checks the media index range.
        let src_moved = self.src_playhead != self.prev_src_playhead;
        let src_force = self.src_last == -1;
        if self.preview_inited
            && self.src_media.is_some()
            && self.src_playhead != self.src_last
            && (src_moved || src_force)
        {
            self.compose_source(ctx);
        }
        self.prev_src_playhead = self.src_playhead;
        // Latch transport state for next frame's START-edge detection (see `transport_started`).
        self.prev_playing = self.playing;

        if let Some(path) = self.shot_path.clone() {
            ctx.request_repaint();
            if self.frames == 6 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            }
            let shot = ctx.input(|i| {
                i.events.iter().rev().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(img) = shot {
                save_ppm(&img, &path);
                // `process::exit` does NOT run destructors, so WorkerProc::Drop (which kills +
                // reaps the gcompose child) would never run — leaking the worker subprocess on
                // every screenshot/CI run. Tear it down explicitly before exiting.
                worker::shutdown();
                std::process::exit(0);
            }
        }

        egui::TopBottomPanel::top("toolbar").exact_height(40.0).show(ctx, |ui| self.toolbar(ui));

        // P18: the pool reports an "Open in Source" click via this out-param; we act on it AFTER
        // the panel closure so `open_source(&mut self)` doesn't overlap the `&mut self.project`
        // borrow held inside `pool_ui`.
        let mut open_source_req: Option<usize> = None;
        egui::SidePanel::left("pool").default_width(220.0).show(ctx, |ui| {
            dock_header(ui, "MEDIA");
            pool::pool_ui(ui, &mut self.project, &mut self.history, &mut open_source_req);
        });
        if let Some(idx) = open_source_req {
            self.open_source(idx);
        }

        // Phase-2 scope docks: the right panel is now RESIZABLE + wider, and TABBED into Properties /
        // Scopes / Audio so each gets the full panel instead of being stacked + crammed. An outer
        // ScrollArea lets long content (the deep properties stack) scroll instead of clipping.
        egui::SidePanel::right("props").resizable(true).default_width(340.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.right_tab, RightTab::Properties, "Properties");
                ui.selectable_value(&mut self.right_tab, RightTab::Scopes, "Scopes");
                ui.selectable_value(&mut self.right_tab, RightTab::Audio, "Audio");
            });
            ui.separator();
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match self.right_tab {
                RightTab::Properties => {
                    panels::properties_ui(
                        ui,
                        &mut self.project,
                        self.selected,
                        &self.selection,
                        &mut self.history,
                        self.playhead,
                        &mut self.filter_clip,
                    );
                }
                // Video scopes: live histogram / luma waveform / vectorscope / RGB parade of the
                // composited program frame at the playhead.
                RightTab::Scopes => panels::scopes_ui(ui, &self.project, self.playhead),
                // Program-audio scopes: peak/RMS meters + FFT spectrum + time-domain oscilloscope.
                RightTab::Audio => panels::audio_meters_ui(ui, &self.project, self.playhead),
            });
        });

        egui::TopBottomPanel::bottom("timeline")
            .resizable(true)
            .min_height(210.0)
            .default_height(250.0)
            .show(ctx, |ui| {
                dock_header(ui, "TIMELINE");
                self.timeline_toolbar(ui);
                timeline::timeline_ui(
                    ui,
                    &mut self.project,
                    &mut self.selected,
                    &mut self.selection,
                    &mut self.playhead,
                    &mut self.history,
                    &mut self.ppf,
                    &mut self.x_scroll,
                    self.snap,
                    self.zoom_fit_pending,
                    // P4 3-point edit marks, READ-ONLY for the target-range band (the timeline draws
                    // [mark_in, mark_out) as a shaded band; it never mutates the marks — those are set
                    // only by the I/O keys in handle_keys).
                    self.mark_in,
                    self.mark_out,
                );
                // The one-shot zoom-to-fit is consumed by timeline_ui this frame; clear it so it
                // doesn't re-fit (and stomp the user's wheel/keyboard zoom) on every later frame.
                self.zoom_fit_pending = false;
            });

        egui::CentralPanel::default().show(ctx, |ui| self.preview_pane(ui));
    }
}

/// Write an egui ColorImage as a binary PPM (P6) — for the screenshot gate.
fn save_ppm(img: &egui::ColorImage, path: &str) {
    let [w, h] = img.size;
    let mut data = Vec::with_capacity(w * h * 3 + 32);
    data.extend_from_slice(format!("P6\n{} {}\n255\n", w, h).as_bytes());
    for px in &img.pixels {
        data.push(px.r());
        data.push(px.g());
        data.push(px.b());
    }
    let _ = std::fs::write(path, data);
}

/// Run zenity and return the chosen path (trimmed), or `None` on cancel / missing zenity /
/// empty selection. Shared spine for the Open and Save pickers below; mirrors pool.rs's
/// `pick_file`. Project JSON paths are user-chosen and are NOT pushed over the worker's
/// space-delimited protocol, so (unlike media paths) we do not reject whitespace here.
fn zenity(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("zenity").args(args).output().ok()?;
    if !out.status.success() {
        return None; // user cancelled
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

/// Native "open file" picker for loading a project. Returns the chosen path, if any.
fn pick_file_open() -> Option<String> {
    zenity(&["--file-selection", "--title=Open project"])
}

/// Native "save file" picker. `default_name` seeds the filename (e.g. "out.mp4",
/// "project.gnp"). Returns the chosen path, if any.
fn pick_file_save(default_name: &str) -> Option<String> {
    let fname = format!("--filename={}", default_name);
    zenity(&["--file-selection", "--save", "--confirm-overwrite", "--title=Save", &fname])
}

#[cfg(test)]
mod tests {
    use super::{load_recents, push_recent, save_recents};

    // T2 RECENT FILES — the pure core. Most-recent-FIRST, de-duplicated, capped.
    //   - pushing a, then b, then a (with cap 3) collapses to ["a","b"]: the second `a` REMOVES the
    //     stale `a` and re-inserts it at the front (no duplicate, promoted to most-recent).
    //   - pushing past the cap drops the OLDEST entry (the tail), keeping exactly `cap` newest.
    #[test]
    fn push_recent_dedups_promotes_and_caps() {
        // a, b, a  with cap 3  ->  ["a", "b"]  (a promoted to front, deduped; b second).
        let mut list: Vec<String> = Vec::new();
        push_recent(&mut list, "a", 3);
        push_recent(&mut list, "b", 3);
        push_recent(&mut list, "a", 3);
        assert_eq!(list, vec!["a".to_string(), "b".to_string()], "dedup + promote-to-front");

        // Pushing past the cap drops the oldest. Fill exactly to cap, then one more.
        let mut full: Vec<String> = Vec::new();
        push_recent(&mut full, "1", 3);
        push_recent(&mut full, "2", 3);
        push_recent(&mut full, "3", 3); // -> ["3","2","1"], at cap
        assert_eq!(full, vec!["3".to_string(), "2".to_string(), "1".to_string()]);
        push_recent(&mut full, "4", 3); // -> ["4","3","2"], "1" (oldest) dropped
        assert_eq!(
            full,
            vec!["4".to_string(), "3".to_string(), "2".to_string()],
            "past cap drops the oldest"
        );
        assert!(!full.contains(&"1".to_string()), "oldest entry was evicted");
    }

    // T2 — the I/O wrappers: a save→load round-trip preserves order, and a missing file loads to [].
    // Points GENESIS_RECENT at a unique scratch file so it never touches the real sidecar and
    // parallel test runs don't collide. Cleans up after itself; no panic on the missing-file path.
    #[test]
    fn recents_sidecar_roundtrip_and_missing_is_empty() {
        let uniq = format!(
            "genesis_recent_test_{}_{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let scratch = std::env::temp_dir().join(uniq);
        let scratch_str = scratch.to_string_lossy().into_owned();
        // Point the recents sidecar at our scratch file (edition-2021 safe; main.rs does the same).
        std::env::set_var("GENESIS_RECENT", &scratch_str);

        // Missing file -> empty, no panic.
        let _ = std::fs::remove_file(&scratch);
        assert!(load_recents().is_empty(), "missing sidecar loads to []");

        // Round-trip: write a most-recent-first list, read it back unchanged.
        let list = vec!["/proj/c.gnp".to_string(), "/proj/b.gnp".to_string(), "/proj/a.gnp".to_string()];
        save_recents(&list);
        let back = load_recents();
        assert_eq!(back, list, "save -> load preserves most-recent-first order");

        // A corrupt sidecar loads to [] (no panic).
        std::fs::write(&scratch, b"{ not valid json").expect("write corrupt sidecar");
        assert!(load_recents().is_empty(), "corrupt sidecar loads to []");

        let _ = std::fs::remove_file(&scratch);
        std::env::remove_var("GENESIS_RECENT");
    }
}
