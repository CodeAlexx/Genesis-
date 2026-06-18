//! Media pool panel — import + media list + add-as-clip.
//!
//! Owned by the layout/pool team. Mirrors MojoMedia's MEDIA pool: "+ Add media" imports a
//! file into the pool only (native picker via zenity), never auto-placing it on the
//! timeline. "Add as clip" appends a Clip on V1 at the end of the program (drag-to-timeline
//! is a follow-up). Thumbnails (decoded by the worker) are a later pass.

use crate::model::{Clip, Project};
use eframe::egui;

/// Drag-and-drop payload: a media-pool index being dragged toward the timeline.
///
/// egui 0.31 requires a DnD payload to be `Any + Send + Sync + 'static` (it wraps the value in an
/// `Arc` internally; see `Ui::dnd_drag_source` / `Response::dnd_release_payload`). A plain
/// `usize` would satisfy those bounds, but a newtype makes the payload's *type* the contract:
/// `timeline.rs` matches on `DragMedia` specifically, so an unrelated future `usize` payload can
/// never be mistaken for "a media item dropped on a lane". `timeline.rs` imports this via
/// `use crate::pool::DragMedia`.
#[derive(Clone, Copy)]
pub struct DragMedia(pub usize);

/// Outcome of the native file picker.
enum PickResult {
    /// User chose a usable path.
    Path(String),
    /// User cancelled / zenity missing — say nothing.
    Cancelled,
    /// User chose a path the worker protocol can't carry (contains whitespace). The string is
    /// the offending path, surfaced to the user so the failure isn't silent.
    Rejected(String),
}

/// Open a native file picker (zenity) and return the chosen path, if any.
///
/// Rejects paths containing whitespace: the gcompose serve protocol is a single
/// space-separated request line of EXACTLY 13 fields with no quoting/escaping, so a path
/// with a space inflates the field count and the worker hard-rejects the line
/// (ERR -> 3 restart attempts -> no preview, with no user-visible reason). We catch it
/// here at import time rather than push a path the worker can never consume.
fn pick_file() -> PickResult {
    let out = match std::process::Command::new("zenity")
        .args(["--file-selection", "--title=Add media"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return PickResult::Cancelled, // zenity missing
    };
    if !out.status.success() {
        return PickResult::Cancelled; // user cancelled
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if path.is_empty() {
        return PickResult::Cancelled;
    }
    if path.contains(char::is_whitespace) {
        return PickResult::Rejected(path);
    }
    PickResult::Path(path)
}

/// Basename of a path (everything after the last '/'), falling back to the whole string.
fn basename(path: &str) -> String {
    path.rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or(path).to_string()
}

/// Default length (frames) for an "Add as clip" placement until trimming lands.
const DEFAULT_CLIP_LEN: i64 = 120;

pub fn pool_ui(ui: &mut egui::Ui, project: &mut Project) {
    // Persisted (across frames) import-rejection notice. Kept in egui temp memory so the
    // pinned `pool_ui(ui, &mut Project)` signature doesn't need a status-out parameter.
    let warn_id = egui::Id::new("pool_import_warning");

    ui.add_space(2.0);
    if ui.button("\u{2795} Add media").clicked() {
        match pick_file() {
            PickResult::Path(path) => {
                let name = basename(&path);
                project.media.push(path);
                project.names.push(name);
                ui.memory_mut(|m| m.data.remove::<String>(warn_id));
            }
            PickResult::Rejected(path) => {
                let msg = format!(
                    "\u{26A0} '{}' has whitespace in its path — the engine can't load it. \
                     Rename or move it to a path without spaces.",
                    basename(&path)
                );
                ui.memory_mut(|m| m.data.insert_temp(warn_id, msg));
            }
            PickResult::Cancelled => {}
        }
    }

    if let Some(msg) = ui.memory(|m| m.data.get_temp::<String>(warn_id)) {
        ui.label(egui::RichText::new(msg).color(egui::Color32::from_rgb(220, 120, 90)).size(10.0));
    }

    ui.separator();

    if project.media.is_empty() {
        ui.weak("No media — click \u{2795} Add media\u{2026}");
        return;
    }

    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        for i in 0..project.media.len() {
            let name = project.names.get(i).cloned().unwrap_or_else(|| format!("media {i}"));
            let path = project.media.get(i).cloned();
            ui.group(|ui| {
                // The label rows are the DRAG HANDLE: dragging them carries `DragMedia(i)` for
                // the timeline lane drop zone (slice B). The "Add as clip" button below stays a
                // normal click target (it is OUTSIDE the drag source, so its click is not eaten
                // by the drag `Sense`). The id must be globally unique → key on the media index.
                let src = ui.dnd_drag_source(egui::Id::new(("poolmedia", i)), DragMedia(i), |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new(format!("{i}")).color(egui::Color32::from_rgb(150, 150, 160)).monospace());
                        ui.label(egui::RichText::new(&name).color(crate::theme::TEXT));
                    });
                    if let Some(path) = &path {
                        ui.label(egui::RichText::new(path).color(egui::Color32::from_rgb(120, 120, 130)).size(9.0));
                    }
                });
                // Hint the drag affordance on hover (the source only sets the Grab cursor while
                // hovered; a quiet hint keeps the gesture discoverable without a tutorial).
                if src.response.hovered() {
                    ui.label(
                        egui::RichText::new("drag onto a timeline lane \u{2193}")
                            .color(egui::Color32::from_rgb(120, 120, 130))
                            .size(9.0),
                    );
                }
                if ui.button("Add as clip \u{2192} V1").clicked() {
                    // Append on V1 (track 0) at the end of the program.
                    // `Clip::video` discards its `name_hint` arg (model.rs: `let _ = name_hint;`),
                    // so don't pay for a per-click clone/borrow of project.names — pass "".
                    let t0 = project.total_frames();
                    project.clips.push(Clip::video(i, t0, DEFAULT_CLIP_LEN, 0, ""));
                }
            });
            ui.add_space(2.0);
        }
    });
}
