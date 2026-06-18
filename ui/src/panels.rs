//! Side panels — properties / filters (right) + scopes.
//!
//! Owned by the layout/panels team. Mirrors MojoMedia's properties ribbon (Color + Comp
//! tabs): per-clip PiP rect (X/Y/W/H, fractions 0..1), fades, look index/mix, plus the
//! program-wide grade (brightness/contrast/saturation). Scopes stays a stub until the
//! worker computes histogram/waveform/vectorscope.

use crate::icons;
use crate::model::Project;
use crate::theme;
use eframe::egui;

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

pub fn properties_ui(ui: &mut egui::Ui, project: &mut Project, selected: usize) {
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

    if let Some(c) = project.clips.get_mut(selected) {
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

pub fn scopes_ui(ui: &mut egui::Ui) {
    section(ui, "SCOPES");
    ui.weak("(histogram / waveform / vectorscope: todo)");
}
