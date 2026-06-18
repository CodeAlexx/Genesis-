//! App shell — eframe::App, toolbar, preview pane, panel/timeline layout, screenshot gate.
//!
//! Owned by the layout team. A Shotcut-style 3-column layout built from plain SidePanels
//! (left = media pool, right = properties + scopes, bottom = timeline, center = preview),
//! each topped by a thin labeled dock-header bar. Wires together model + worker + timeline
//! + pool + panels. The preview re-composites whenever the playhead moves off the last
//! frame we composited (`last_composed`), in addition to the initial frame-2 gate.

use crate::model::{History, Project};
use crate::{icons, panels, pool, project_io, theme, timeline, worker};
use eframe::egui::{self, Color32};

pub struct Genesis {
    preview: Option<egui::TextureHandle>,
    project: Project,
    selected: usize,
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
}

impl Genesis {
    pub fn new(cc: &eframe::CreationContext<'_>, project: Project) -> Self {
        theme::apply(&cc.egui_ctx);
        let shot_path = std::env::var("GENESIS_SHOT").ok();
        Genesis {
            preview: None,
            project,
            selected: 0,
            ppf: 6.0,
            playhead: 0,
            last_composed: -1,
            prev_playhead: 0,
            preview_inited: false,
            status: "compositing\u{2026}".into(),
            shot_path,
            frames: 0,
            history: History::new(),
            playing: false,
            prev_playing: false,
            play_anchor: None,
            play_anchor_frame: 0,
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

    /// First-frame preview gate (kept on the frame-2 boundary for the screenshot path).
    fn ensure_preview(&mut self, ctx: &egui::Context) {
        if self.preview_inited {
            return;
        }
        self.preview_inited = true;
        self.compose(ctx);
    }

    /// Keyboard editing shortcuts (mirrors MojoMedia main_editor.mojo key bindings):
    ///   S                 split selected clip at playhead
    ///   Delete            delete selected clip (clamp selection)
    ///   Ctrl+Z            undo
    ///   Ctrl+Shift+Z / Ctrl+Y  redo
    ///   Left / Right      step playhead -/+1 (clamped 0..total-1)
    ///   Space             toggle transport (play/pause) — the canonical transport source
    ///
    /// Focus guard: `ctx.wants_keyboard_input()` is true whenever ANY focusable widget holds
    /// focus, not just a `TextEdit`. In egui 0.31 it is `memory.focused().is_some()`, so it also
    /// covers the toolbar `Button`s. That means keyboard shortcuts (S, Delete, arrows, Ctrl+Z,
    /// Space) are intentionally suppressed while a button or text field has focus — typing an
    /// "s" in a future rename field can't razor a clip, but neither can shortcuts fire while a
    /// toolbar button is the focused widget (the user must click the preview/timeline first).
    /// The toolbar Play/Pause button surrenders its focus after creation (see `toolbar`) so it
    /// never swallows Space, keeping Space a single-source toggle here (no double-toggle).
    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }

        // Snapshot every key/modifier we care about in one input() borrow.
        let (split, del, undo, redo, left, right, space) = ctx.input(|i| {
            let m = &i.modifiers;
            // `command` is Ctrl on Linux/Windows and Cmd on macOS (egui-normalized).
            let cmd = m.command || m.ctrl;
            let z = i.key_pressed(egui::Key::Z);
            let y = i.key_pressed(egui::Key::Y);
            (
                // S with no modifier → split.
                i.key_pressed(egui::Key::S) && !cmd && !m.shift && !m.alt,
                // Delete only — Backspace dropped: it collides with DragValue/text editing
                // (a focused numeric field in panels.rs captures Backspace) and the focus guard
                // above does not always cover that case. Delete is the unambiguous razor key.
                i.key_pressed(egui::Key::Delete),
                cmd && z && !m.shift,            // Ctrl+Z (no shift) → undo
                cmd && ((z && m.shift) || y),    // Ctrl+Shift+Z OR Ctrl+Y → redo
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
                i.key_pressed(egui::Key::Space),
            )
        });

        // --- structural edits: snapshot BEFORE each mutation so undo restores pre-edit state.
        // Only push history when the edit will actually mutate, so a no-op keypress (e.g. S
        // with the playhead outside the clip) doesn't create a dead undo step. We mirror
        // split_clip's own precondition (0 < off < len) to decide whether the split lands.
        if split {
            if let Some(c) = self.project.clips.get(self.selected) {
                let off = self.playhead - c.t0;
                if off > 0 && off < c.len {
                    self.history.push(&self.project);
                    // Keep the left half selected (matches MojoMedia `sel_clip = sp`).
                    let _ = self.project.split_clip(self.selected, self.playhead);
                }
            }
        }

        if del && !self.project.clips.is_empty() && self.selected < self.project.clips.len() {
            self.history.push(&self.project);
            self.project.delete_clip(self.selected);
            self.clamp_selected();
        }

        // --- undo / redo: redo wins if both somehow fire (shift state disambiguates above).
        if redo {
            self.history.redo(&mut self.project);
            self.clamp_selected();
            self.clamp_playhead();
        } else if undo {
            self.history.undo(&mut self.project);
            self.clamp_selected();
            self.clamp_playhead();
        }

        // --- transport / scrub.
        if space {
            self.playing = !self.playing;
        }
        if left {
            self.playhead -= 1;
            self.clamp_playhead();
        }
        if right {
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
            if tb_button(ui, "open", "Open") {
                if let Some(path) = pick_file_open() {
                    match project_io::load(&path) {
                        Some(p) => {
                            self.project = p;
                            self.selected = 0;
                            self.playhead = 0;
                            self.playing = false;
                            // Keep the transport-edge tracker in lock-step with `playing` so the
                            // Open does not read as a false->true Start edge on the next update().
                            self.prev_playing = false;
                            // Drop any wall-clock anchor so the loaded project starts paused and a
                            // subsequent Play re-anchors cleanly at its (reset) playhead 0.
                            self.play_anchor = None;
                            self.play_anchor_frame = 0;
                            // A new project invalidates undo history and the composed preview.
                            self.history = History::new();
                            self.last_composed = -1; // force a re-composite of frame 0
                            self.status = format!("opened {}", path);
                        }
                        None => self.status = format!("open failed: {}", path),
                    }
                }
            }

            // Save: native save picker → serialize current project to JSON.
            if tb_button(ui, "save", "Save") {
                if let Some(path) = pick_file_save("project.gnp") {
                    match project_io::save(&self.project, &path) {
                        Ok(()) => self.status = format!("saved {}", path),
                        Err(e) => self.status = format!("save failed: {}", e),
                    }
                }
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
                self.clamp_playhead();
            }
            if tb_button_enabled(ui, self.history.can_redo(), "redo", "Redo") {
                self.history.redo(&mut self.project);
                self.clamp_selected();
                self.clamp_playhead();
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
                // disable the line-135 re-composite path (which is gated on preview_inited).
                // Setting last_composed = -1 (which differs from any valid playhead >= 0)
                // makes the next update() re-composite via that path.
                self.last_composed = -1;
            }
            ui.separator();
            ui.label(egui::RichText::new(&self.status).color(theme::ACCENT).size(11.0));
        });
    }

    fn preview_pane(&mut self, ui: &mut egui::Ui) {
        ui.painter().rect_filled(ui.max_rect(), egui::CornerRadius::ZERO, Color32::from_rgb(10, 10, 12));
        if let Some(tex) = &self.preview {
            let src = egui::load::SizedTexture::new(tex.id(), tex.size_vec2());
            ui.centered_and_justified(|ui| {
                ui.add(egui::Image::new(src).maintain_aspect_ratio(true).max_size(ui.available_size()));
            });
        } else {
            let s = self.status.clone();
            ui.centered_and_justified(|ui| {
                ui.label(s);
            });
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

        egui::SidePanel::left("pool").default_width(220.0).show(ctx, |ui| {
            dock_header(ui, "MEDIA");
            pool::pool_ui(ui, &mut self.project);
        });

        egui::SidePanel::right("props").default_width(260.0).show(ctx, |ui| {
            dock_header(ui, "PROPERTIES \u{2022} SCOPES");
            panels::properties_ui(ui, &mut self.project, self.selected, self.playhead);
            ui.add_space(10.0);
            // Slice C: scopes_ui now takes the project + playhead so it can ask the worker for a
            // live histogram/waveform/vectorscope of the composited program frame at the playhead.
            panels::scopes_ui(ui, &self.project, self.playhead); // properties_ui renders TRACKS (wave 5)
        });

        egui::TopBottomPanel::bottom("timeline")
            .resizable(true)
            .min_height(210.0)
            .default_height(250.0)
            .show(ctx, |ui| {
                dock_header(ui, "TIMELINE");
                timeline::timeline_ui(ui, &mut self.project, &mut self.selected, &mut self.playhead, self.ppf);
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
