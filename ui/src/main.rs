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
mod icons;
mod model;
mod panels;
mod pool;
mod project_io;
mod theme;
mod thumbs;
mod timeline;
mod worker;

use model::Project;

fn main() -> eframe::Result<()> {
    // GENESIS_OPEN=<project.json> loads a saved project at launch (used by the headless render/
    // screenshot gates and "open on startup"); otherwise build the single-clip demo from argv[1].
    let project = match std::env::var("GENESIS_OPEN").ok().and_then(|p| project_io::load(&p)) {
        Some(p) => p,
        None => {
            let media = std::env::args().nth(1).unwrap_or_else(|| "/tmp/editor_clip.mp4".to_string());
            Project::demo(media)
        }
    };
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("Genesis"),
        ..Default::default()
    };
    eframe::run_native("Genesis", opts, Box::new(move |cc| Ok(Box::new(app::Genesis::new(cc, project)))))
}
