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
    let genesis_open = std::env::var("GENESIS_OPEN").ok().filter(|p| !p.is_empty());
    let project = match genesis_open.as_ref().and_then(|p| project_io::load(p)) {
        Some(p) => p,
        None => {
            // P33 CRASH RECOVERY (launch): with NO project explicitly opened (GENESIS_OPEN
            // unset/empty) AND a recovery sidecar present on disk, restore it instead of the demo so
            // an unsaved session survives a crash. Non-destructive: this reads the separate
            // /tmp sidecar only — it never overwrites the user's real project files. When
            // GENESIS_OPEN IS set we are in the gate/open-on-startup path and skip recovery
            // entirely, so the headless gates stay byte-unaffected. `GENESIS_RECOVERED=1` tells the
            // app constructor to surface the recovery status line (status text stays in app.rs).
            if genesis_open.is_none() {
                if let Some(p) = project_io::load(app::RECOVERY_PATH) {
                    std::env::set_var("GENESIS_RECOVERED", "1");
                    p
                } else {
                    let media = std::env::args().nth(1).unwrap_or_else(|| "/tmp/editor_clip.mp4".to_string());
                    Project::demo(media)
                }
            } else {
                let media = std::env::args().nth(1).unwrap_or_else(|| "/tmp/editor_clip.mp4".to_string());
                Project::demo(media)
            }
        }
    };
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            // Open MAXIMIZED so the editor uses the full display (this is a 4K-class workstation —
            // a 1280×800 floating window wastes most of the screen). The inner_size is only the
            // fallback the window manager falls back to if it ignores `maximized` (or when the user
            // un-maximizes): a generous 2560×1440 so even un-maximized we get a usable NLE layout.
            .with_maximized(true)
            .with_inner_size([2560.0, 1440.0])
            .with_title("Genesis"),
        ..Default::default()
    };
    eframe::run_native("Genesis", opts, Box::new(move |cc| Ok(Box::new(app::Genesis::new(cc, project)))))
}
