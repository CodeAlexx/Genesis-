//! Genesis UI — egui front-end over the isolated C engine worker (gcompose).
//! Module layout (team ownership in comments):
//!   theme    — shared palette (stable)
//!   model    — project data model (timeline/model team)
//!   worker   — gcompose client / frame protocol (engine team)
//!   timeline — timeline widget (timeline/model team)
//!   pool     — media pool panel (layout team)
//!   panels   — properties / scopes (layout team)
//!   app      — eframe shell + layout wiring (layout team)

mod app;
mod model;
mod panels;
mod pool;
mod theme;
mod timeline;
mod worker;

use model::Project;

fn main() -> eframe::Result<()> {
    let media = std::env::args().nth(1).unwrap_or_else(|| "/tmp/editor_clip.mp4".to_string());
    let project = Project::demo(media);
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Genesis"),
        ..Default::default()
    };
    eframe::run_native("Genesis", opts, Box::new(move |cc| Ok(Box::new(app::Genesis::new(cc, project)))))
}
