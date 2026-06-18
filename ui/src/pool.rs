//! Media pool panel — import + thumbnail grid + drag-to-timeline.
//!
//! Owned by the layout/pool team. Stub for now; to grow: file-picker import (rfd or zenity),
//! per-media thumbnails (decoded by the worker), and drag onto a timeline lane.

use crate::model::Project;
use eframe::egui;

pub fn pool_ui(ui: &mut egui::Ui, project: &Project) {
    ui.label(egui::RichText::new("MEDIA").color(egui::Color32::from_rgb(150, 150, 160)).size(11.0));
    ui.separator();
    for (i, name) in project.names.iter().enumerate() {
        ui.label(format!("{i}: {name}"));
    }
    ui.add_space(8.0);
    ui.weak("(import + thumbnails + drag-to-timeline: todo)");
}
