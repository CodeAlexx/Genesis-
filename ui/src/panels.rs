//! Side panels — properties / filters (right) + scopes.
//!
//! Owned by the layout/panels team. Mirrors MojoMedia's properties ribbon (Color + Comp
//! tabs): per-clip PiP rect (X/Y/W/H, fractions 0..1), fades, look index/mix, plus the
//! program-wide grade (brightness/contrast/saturation). Scopes stays a stub until the
//! worker computes histogram/waveform/vectorscope.

use crate::model::Project;
use crate::theme;
use eframe::egui;

/// A thin labeled section header inside a panel.
fn section(ui: &mut egui::Ui, label: &str) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new(label).color(egui::Color32::from_rgb(150, 150, 160)).size(11.0));
    ui.separator();
}

pub fn properties_ui(ui: &mut egui::Ui, project: &mut Project, selected: usize) {
    section(ui, "PROPERTIES");

    // ---- Comp tab: the selected clip's picture-in-picture rect + fades + look ----
    if let Some(c) = project.clips.get(selected) {
        ui.label(
            egui::RichText::new(format!("clip {selected}  \u{2022}  track V{}  \u{2022}  t0 {}  \u{2022}  len {}", c.track, c.t0, c.len))
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
}

pub fn scopes_ui(ui: &mut egui::Ui) {
    section(ui, "SCOPES");
    ui.weak("(histogram / waveform / vectorscope: todo)");
}
