//! Side panels — properties / filters (left or right) + scopes.
//!
//! Owned by the layout/panels team. Stubs for now; to grow: per-clip properties (PiP rect,
//! grade, look, fades, keyframes), and scopes (histogram / waveform / vectorscope from the
//! worker).

use crate::model::Project;
use eframe::egui;

pub fn properties_ui(ui: &mut egui::Ui, project: &mut Project, selected: usize) {
    ui.label(egui::RichText::new("PROPERTIES").color(egui::Color32::from_rgb(150, 150, 160)).size(11.0));
    ui.separator();
    if let Some(c) = project.clips.get(selected) {
        ui.label(format!("clip {selected}  track V{}  t0 {}  len {}", c.track, c.t0, c.len));
        ui.label(format!("PiP  x{:.2} y{:.2} w{:.2} h{:.2}", c.px, c.py, c.pw, c.ph));
    } else {
        ui.weak("no clip selected");
    }
    ui.add_space(6.0);
    ui.add(egui::Slider::new(&mut project.bright, -1.0..=1.0).text("Bright"));
    ui.add(egui::Slider::new(&mut project.contrast, 0.0..=2.0).text("Contrast"));
    ui.add(egui::Slider::new(&mut project.sat, 0.0..=2.0).text("Saturation"));
}

pub fn scopes_ui(ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("SCOPES").color(egui::Color32::from_rgb(150, 150, 160)).size(11.0));
    ui.separator();
    ui.weak("(histogram / waveform / vectorscope: todo)");
}
