//! Side panels — properties / filters (right) + scopes.
//!
//! Owned by the layout/panels team. Mirrors MojoMedia's properties ribbon (Color + Comp
//! tabs): per-clip PiP rect (X/Y/W/H, fractions 0..1), fades, look index/mix, plus the
//! program-wide grade (brightness/contrast/saturation). The SCOPES section (Slice C) shows a
//! live histogram / luma waveform / vectorscope of the composited program frame at the playhead,
//! computed on the GPU by the `gcompose` worker (`worker::scope`) and blitted as a 256×256 image
//! (mirrors MojoMedia main_editor.mojo's Shotcut-style Hist/Wave/Vec scope selector).

use crate::icons;
use crate::model::Project;
use crate::theme;
use crate::worker;
use eframe::egui;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// A thin labeled section header inside a panel.
fn section(ui: &mut egui::Ui, label: &str) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new(label).color(egui::Color32::from_rgb(150, 150, 160)).size(11.0));
    ui.separator();
}

/// Display label for a `Clip.track` index (0 = V1, 1 = V2, 2 = A1). Out-of-range
/// tracks fall through to a numeric "T{n}" so the header never lies (audio clips on
/// track 2 must read "A1", not "V2").
fn track_label(track: u8) -> String {
    match track {
        0 => "V1".into(),
        1 => "V2".into(),
        2 => "A1".into(),
        n => format!("T{n}"),
    }
}

pub fn properties_ui(ui: &mut egui::Ui, project: &mut Project, selected: usize, playhead: i64) {
    section(ui, "PROPERTIES");

    // ---- Comp tab: the selected clip's picture-in-picture rect + fades + look ----
    if let Some(c) = project.clips.get(selected) {
        ui.label(
            egui::RichText::new(format!("clip {selected}  \u{2022}  track {}  \u{2022}  t0 {}  \u{2022}  len {}", track_label(c.track), c.t0, c.len))
                .color(theme::TEXT)
                .size(11.0),
        );
    } else {
        ui.weak("no clip selected");
    }

    // The clip's timeline start (captured before the mutable borrow below) so the PiP Key
    // button can compute the CLIP-LOCAL frame (playhead - t0) once the borrow has ended.
    let mut clip_t0: Option<i64> = None;
    if let Some(c) = project.clips.get_mut(selected) {
        clip_t0 = Some(c.t0);
        section(ui, "PiP (picture-in-picture)");
        ui.add(egui::Slider::new(&mut c.px, 0.0..=1.0).text("X"));
        ui.add(egui::Slider::new(&mut c.py, 0.0..=1.0).text("Y"));
        ui.add(egui::Slider::new(&mut c.pw, 0.0..=1.0).text("W"));
        ui.add(egui::Slider::new(&mut c.ph, 0.0..=1.0).text("H"));
        if ui.button("Reset PiP (full frame)").clicked() {
            c.px = 0.0;
            c.py = 0.0;
            c.pw = 1.0;
            c.ph = 1.0;
        }

        section(ui, "Fades (frames)");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut c.fade_in).speed(1.0).range(0..=600).prefix("in "));
            ui.add(egui::DragValue::new(&mut c.fade_out).speed(1.0).range(0..=600).prefix("out "));
        });

        section(ui, "Look");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut c.look).speed(1.0).range(0..=64).prefix("LUT "));
            ui.add(egui::Slider::new(&mut c.look_amt, 0.0..=1.0).text("Mix"));
        });
    }

    // ---- PiP keyframes (only meaningful when a clip is selected) ----
    // Snapshot the clip's current px/py/pw/ph at the CLIP-LOCAL playhead frame. The mutable
    // clip borrow has ended, so we can now take &mut project for add_pip_key / pip_key_count.
    if let Some(t0) = clip_t0 {
        let local = playhead - t0;
        let n_pip = project.pip_key_count(selected);
        ui.horizontal(|ui| {
            if ui.button("Key PiP @ playhead").clicked() {
                project.add_pip_key(selected, local);
            }
            ui.weak(format!("{n_pip} key{}", if n_pip == 1 { "" } else { "s" }));
        });
        ui.label(
            egui::RichText::new(format!("PiP local frame {local}"))
                .color(egui::Color32::from_rgb(150, 150, 160))
                .size(10.0),
        );
    }

    // ---- Color tab: program-wide grade ----
    section(ui, "Grade");
    ui.add(egui::Slider::new(&mut project.bright, -1.0..=1.0).text("Brightness"));
    ui.add(egui::Slider::new(&mut project.contrast, 0.0..=2.0).text("Contrast"));
    ui.add(egui::Slider::new(&mut project.sat, 0.0..=2.0).text("Saturation"));
    if ui.button("Reset grade").clicked() {
        project.bright = 0.0;
        project.contrast = 1.0;
        project.sat = 1.0;
    }

    // Drop a grade keyframe (bright+contrast+sat snapshot) at the playhead, plus a per-track
    // key count so the user can see the animation building up. Empty tracks read "0 keys" and
    // the worker falls back to the static slider values above.
    ui.horizontal(|ui| {
        if ui.button("Key grade @ playhead").clicked() {
            project.add_grade_key(playhead);
        }
        ui.weak(format!(
            "B {}  C {}  S {}",
            project.bright_kf.len(),
            project.contrast_kf.len(),
            project.sat_kf.len(),
        ));
    });

    // ---- per-track Hide / Mute / Lock state (folded in so app.rs need not change) ----
    tracks_ui(ui, project);
}

/// An icon-or-text toggle button. Tries `icons::icon(ctx, icon_name)` for the glyph and
/// falls back to `text` when the blob/icon is unavailable (mirrors the toolbar's discipline).
/// `on` selects the active visual: an active toggle gets the theme accent tint, inactive a
/// muted neutral. Returns true on click (caller flips the backing bool).
fn toggle_button(ui: &mut egui::Ui, icon_name: &str, text: &str, on: bool, tooltip: &str) -> bool {
    let size = egui::vec2(26.0, 22.0);
    let tint = if on { theme::TEXT } else { egui::Color32::from_rgb(120, 120, 130) };

    let resp = if let Some(tex) = icons::icon(ui.ctx(), icon_name) {
        // (TextureId, Vec2) -> SizedTexture (egui 0.31 load::SizedTexture::from), fed to
        // Image::from_texture; ImageButton::new takes `impl Into<Image>`.
        let image = egui::Image::from_texture((tex, egui::vec2(16.0, 16.0))).tint(tint);
        ui.add_sized(size, egui::ImageButton::new(image).frame(true))
    } else {
        // Text fallback: dim the label when the toggle is "off" so state still reads.
        let label = egui::RichText::new(text).size(11.0).color(tint);
        ui.add_sized(size, egui::Button::new(label))
    };

    resp.on_hover_text(tooltip).clicked()
}

/// TRACKS section: one row per track (V2 / V1 / A1, timeline top-to-bottom) with Hide /
/// Mute / Lock toggles. Rows are displayed in V2,V1,A1 order but each maps to its Clip.track
/// index (0 = V1, 1 = V2, 2 = A1) so toggles write the correct slot of the `[bool; 3]` arrays.
///
/// Hide uses the eye glyphs ("visible" / "hidden") and applies to video tracks (V1/V2). Mute
/// uses the speaker glyphs ("volume" / "muted"). Lock uses the padlock glyphs ("unlocked" /
/// "locked"). Worker.rs honors track_hide (video) + track_mute (audio); lock is advisory.
pub fn tracks_ui(ui: &mut egui::Ui, project: &mut Project) {
    section(ui, "TRACKS");

    // (display label, Clip.track index, is_video). Audio track (A1) shows Mute as the
    // primary control; video tracks (V1/V2) show Hide. All three expose Mute + Lock.
    const ROWS: [(&str, usize, bool); 3] = [("V2", 1, true), ("V1", 0, true), ("A1", 2, false)];

    for (label, t, is_video) in ROWS {
        ui.horizontal(|ui| {
            ui.add_sized(
                egui::vec2(26.0, 22.0),
                egui::Label::new(egui::RichText::new(label).color(theme::TEXT).size(11.0)),
            );

            // Hide (eye) — meaningful for video tracks; the worker skips a hidden video
            // track when resolving base/over. Shown disabled-ish for audio (no video to hide).
            if is_video {
                let hidden = project.track_hide[t];
                let (name, txt) = if hidden { ("hidden", "H\u{0335}") } else { ("visible", "H") };
                if toggle_button(ui, name, txt, !hidden, "Hide / show this video track") {
                    project.track_hide[t] = !hidden;
                }
            } else {
                // Keep column alignment for the audio row: an inert placeholder.
                ui.add_sized(egui::vec2(26.0, 22.0), egui::Label::new(egui::RichText::new("\u{2014}").weak()));
            }

            // Mute (speaker) — applies to every track; the worker drops muted audio.
            let muted = project.track_mute[t];
            let (mname, mtxt) = if muted { ("muted", "M\u{0335}") } else { ("volume", "M") };
            if toggle_button(ui, mname, mtxt, !muted, "Mute / unmute this track's audio") {
                project.track_mute[t] = !muted;
            }

            // Lock (padlock) — advisory this wave; the editor blocks edits to a locked track.
            let locked = project.track_lock[t];
            let (lname, ltxt) = if locked { ("locked", "L\u{0335}") } else { ("unlocked", "L") };
            if toggle_button(ui, lname, ltxt, locked, "Lock / unlock edits on this track") {
                project.track_lock[t] = !locked;
            }
        });
    }
}

// ===========================================================================================
//  SCOPES (Slice C) — live histogram / luma waveform / vectorscope of the program frame.
// ===========================================================================================
//
// The scope image is computed on the GPU by the persistent `gcompose` worker
// (`worker::scope(project, playhead, kind) -> Option<Vec<u8>>`, RGBA8 SW×SH) and uploaded to an
// egui texture for display. Because the right-panel UI is a stateless free function, the selected
// scope kind AND the cached texture (so we don't re-run the worker every repaint) live in a
// process-global `OnceLock<Mutex<ScopeCache>>` — exactly the pattern `thumbs.rs` uses for the
// timeline visual cache.
//
// Fetch discipline (so the single serial worker is not hammered every repaint): the worker is
// only re-queried when the inputs that change the image change — the playhead frame or the
// selected kind — or when the user clicks Refresh. A successful fetch stores (kind, frame) +
// the uploaded `TextureHandle`; a failed fetch caches `None` for that (kind, frame) so an
// undecodable/worker-down frame is not retried every repaint (it shows "scope unavailable" until
// the playhead/kind moves or the user clicks Refresh).
//
// PLAYBACK NOTE (skeptic #4 / #7): during wall-clock playback the playhead advances on (almost)
// every repaint. A naive "refetch whenever (kind, frame) changed" therefore hits the worker EVERY
// playing frame — and each `worker::scope` is NOT a cheap round-trip: the worker re-composites the
// program frame (PREVIEW) and then runs the scope kernel (SCOPE) under one mutex hold. Stacked on
// the preview `compose()` (which also re-composites) this puts ~2 full GPU composites per frame on
// the single UI thread, all serialized on the worker mutex, dragging playback FPS well below 30 and
// starving the wall-clock pacer (stutter) whenever the SCOPES panel is mounted.
//
// The clean fix (gate auto-refetch on a `playing` flag) is not available: `playing` is app.rs state
// and this is a PINNED-signature free fn (`scopes_ui(ui, project, playhead)`) — adding a param is a
// contract change. Instead we THROTTLE the auto-refetch by WALL-CLOCK time, entirely inside this
// module: when the playhead is moving fast (playback), we recompute the scope at most ~5 Hz rather
// than every repaint, which keeps the scopes panel live (it visibly tracks playback) while leaving
// the bulk of the per-frame worker budget to the preview composite. A genuine seek/scrub or a kind
// change still updates promptly because the throttle window is short, and the Refresh button always
// forces an immediate recompute (it bypasses the throttle). Stationary playhead → no refetch at all
// (the (kind, frame) key is unchanged), so a paused frame still costs exactly one fetch.
const SCOPE_REFETCH_MIN_INTERVAL: f64 = 0.20; // seconds → ~5 Hz auto-refresh ceiling during playback

/// The three scope kinds, in the worker's `kind` order (0=histogram, 1=luma-waveform,
/// 2=vectorscope). `as u8` yields the worker wire value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScopeKind {
    Histogram = 0,
    Waveform = 1,
    Vectorscope = 2,
}

impl ScopeKind {
    fn label(self) -> &'static str {
        match self {
            ScopeKind::Histogram => "Hist",
            ScopeKind::Waveform => "Wave",
            ScopeKind::Vectorscope => "Vec",
        }
    }
}

/// Process-global scope state: the selected kind, the (kind, frame) the cached texture was built
/// for, and the cached texture itself. `tex == None` with a matching key means "we tried this
/// (kind, frame) and the worker returned None" — the failure sentinel, so we don't refetch every
/// repaint.
struct ScopeCache {
    kind: ScopeKind,
    /// (kind-as-u8, playhead frame) the cached `tex` was computed for. `None` = nothing fetched.
    key: Option<(u8, i64)>,
    /// The uploaded scope texture, or `None` when the last fetch for `key` failed.
    tex: Option<egui::TextureHandle>,
    /// Wall-clock time of the last *worker* fetch (success or failure). Used to throttle the
    /// auto-refetch during playback (skeptic #4): a stale key only triggers a worker call once the
    /// `SCOPE_REFETCH_MIN_INTERVAL` window has elapsed, capping the playback-time scope recompute
    /// rate. A kind change still bypasses the throttle (see `scopes_ui`). `None` = never fetched.
    last_fetch: Option<Instant>,
}

impl ScopeCache {
    fn new() -> ScopeCache {
        ScopeCache { kind: ScopeKind::Histogram, key: None, tex: None, last_fetch: None }
    }
}

static SCOPE: OnceLock<Mutex<ScopeCache>> = OnceLock::new();

fn scope_slot() -> &'static Mutex<ScopeCache> {
    SCOPE.get_or_init(|| Mutex::new(ScopeCache::new()))
}

/// Upload an SW×SH RGBA8 scope buffer as an egui texture (NEAREST — scopes are crisp synthetic
/// images; we don't want them blurred when scaled to fill the panel). Mirrors
/// `worker::rgba_to_texture` but at scope dims and nearest filtering.
fn scope_to_texture(ctx: &egui::Context, buf: &[u8]) -> egui::TextureHandle {
    let img = egui::ColorImage::from_rgba_unmultiplied([worker::SW, worker::SH], buf);
    ctx.load_texture("scope", img, egui::TextureOptions::NEAREST)
}

/// SCOPES section: a Hist / Wave / Vec selector, a Refresh button, and the 256×256 scope image of
/// the composited program frame at `playhead`. Auto-refreshes when the playhead or selected kind
/// changes; "scope unavailable" when the worker can't produce one (e.g. an empty timeline / a
/// worker flake). State (selected kind + cached texture) lives in the process-global `SCOPE`
/// cache because this is a stateless free fn (Slice C; signature changed from `scopes_ui(ui)`).
pub fn scopes_ui(ui: &mut egui::Ui, project: &Project, playhead: i64) {
    section(ui, "SCOPES");

    // On a poisoned lock just skip scopes this frame (never panic the whole UI), matching the
    // defensive posture of thumbs.rs's `with_cache`.
    let mut guard = match scope_slot().lock() {
        Ok(g) => g,
        Err(_) => {
            ui.weak("scope unavailable");
            return;
        }
    };

    // ---- kind selector (Hist / Wave / Vec) — a small segmented row of selectable labels ----
    ui.horizontal(|ui| {
        for kind in [ScopeKind::Histogram, ScopeKind::Waveform, ScopeKind::Vectorscope] {
            let selected = guard.kind == kind;
            if ui.selectable_label(selected, kind.label()).clicked() {
                guard.kind = kind;
            }
        }
    });

    // ---- Refresh button (force a re-fetch even if the key is unchanged) ----
    let force = ui.button("Refresh").on_hover_text("Recompute the scope at the playhead").clicked();

    // Decide whether to (re)fetch: the playhead moved OR the kind changed (both fold into the key,
    // which embeds `kind as u8`), we have no cached entry for the current (kind, frame), or the
    // user clicked Refresh.
    let kind = guard.kind;
    let want_key = (kind as u8, playhead);
    // Split staleness into "the selected kind changed" vs "only the frame changed": a kind change
    // must update promptly (the user clicked a different scope), whereas a frame-only change during
    // playback is throttled (skeptic #4) so we don't issue a worker recomposite every repaint.
    let kind_changed = guard.key.map(|(k, _)| k) != Some(kind as u8);
    let frame_stale = guard.key != Some(want_key);
    // Throttle gate: a frame-only stale key only earns a worker call once SCOPE_REFETCH_MIN_INTERVAL
    // has elapsed since the last fetch. `force` (Refresh) and `kind_changed` bypass the throttle;
    // a first-ever fetch (last_fetch == None) also bypasses it so the scope appears immediately.
    let throttle_ok = match guard.last_fetch {
        None => true,
        Some(t) => t.elapsed().as_secs_f64() >= SCOPE_REFETCH_MIN_INTERVAL,
    };
    if force || kind_changed || (frame_stale && throttle_ok) {
        // Ask the worker for the scope of the program frame at the playhead. None => store the
        // failure sentinel (tex = None) under this key so we don't refetch every repaint.
        let tex = worker::scope(project, playhead, kind as u8)
            .filter(|buf| buf.len() == worker::SW * worker::SH * 4)
            .map(|buf| scope_to_texture(ui.ctx(), &buf));
        guard.tex = tex;
        guard.key = Some(want_key);
        guard.last_fetch = Some(Instant::now());
    }

    // ---- display ----
    match &guard.tex {
        Some(tex) => {
            // Draw the scope square, capped to the panel width so it never overflows the side
            // panel. The source is square (SW == SH), so an exact square fit keeps its aspect.
            // Floor at SCOPE_MIN_SIDE (skeptic #9): on a collapsed/just-laid-out panel
            // `available_width()` can be ~0, which would render a 0×0 (invisible) scope that reads
            // as "no scope" rather than a shrunk one. Clamp up so it always stays visible.
            const SCOPE_MIN_SIDE: f32 = 64.0;
            let side = ui.available_width().clamp(SCOPE_MIN_SIDE, worker::SW as f32);
            let src = egui::load::SizedTexture::new(tex.id(), egui::vec2(side, side));
            ui.add(egui::Image::new(src).fit_to_exact_size(egui::vec2(side, side)));
        }
        None => {
            ui.weak("scope unavailable");
        }
    }
}
