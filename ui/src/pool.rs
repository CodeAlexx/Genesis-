//! Media pool panel — import + media list + add-as-clip.
//!
//! Owned by the layout/pool team. Mirrors MojoMedia's MEDIA pool: "+ Add media" imports a
//! file into the pool only (native picker via zenity), never auto-placing it on the
//! timeline. "Add as clip" appends a Clip on V1 at the end of the program (drag-to-timeline
//! is a follow-up). Thumbnails (decoded by the worker) are a later pass.

use crate::model::{Clip, History, Project};
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
}

/// Open a native file picker (zenity) and return the chosen path, if any.
///
/// WHITESPACE IS NOW ALLOWED: the UI→gcompose wire percent-encodes every path token (worker.rs
/// `enc_path`) and the engine decodes it (gcompose `dec_path`), so a media path containing spaces
/// loads + composes like any other. The old import-time rejection of spaced paths is removed; the
/// REAL (unencoded) path is stored in `project.media` so saved projects still reference the true
/// file. Only genuinely-empty/cancelled picks are dropped.
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
    PickResult::Path(path)
}

/// Basename of a path (everything after the last '/'), falling back to the whole string.
fn basename(path: &str) -> String {
    path.rsplit('/').next().filter(|s| !s.is_empty()).unwrap_or(path).to_string()
}

/// T4 RELINK — a native file picker (zenity) for the new location of a MISSING media file. Mirrors
/// `pick_file` (the "+ Add media" picker) exactly — blocking modal, cancel / missing-zenity / empty
/// selection all fold to `None` so the UI never panics on the dialog. Separate from `pick_file` only
/// for the dialog title ("Relink media"); the stored path is the REAL (unencoded) path, like import.
fn pick_relink_file() -> Option<String> {
    let out = std::process::Command::new("zenity")
        .args(["--file-selection", "--title=Relink media"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None; // user cancelled, or zenity errored / missing
    }
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

/// T4 RELINK — is `path` a REAL on-disk media path that we should check for existence (and offer to
/// relink when it's gone)? A path is "checkable" only when it's a genuine file reference: non-empty,
/// not the wire `-` sentinel, and not a `RAW:`-prefixed in-memory raster. Those non-file sentinels
/// never live in `project.media` today (they are worker-side wire values), but guarding here keeps
/// the missing-tag / Relink affordance off anything that isn't an actual file on disk.
fn is_checkable_media(path: &str) -> bool {
    !path.is_empty() && path != "-" && !path.starts_with("RAW:")
}

/// T4 RELINK — true when `path` names a real media file that is currently MISSING (the file does not
/// exist on disk). Non-file sentinels (empty / `-` / `RAW:`) and present files both return false, so
/// only a genuine relink candidate gets the red "missing" tag + a Relink button. Pure except for the
/// single `Path::exists()` stat (same pattern app.rs uses for the Recent-projects menu).
fn media_is_missing(path: &str) -> bool {
    is_checkable_media(path) && !std::path::Path::new(path).exists()
}

/// Default length (frames) for an "Add as clip" placement until trimming lands.
const DEFAULT_CLIP_LEN: i64 = 120;

/// P18: render the media pool. `open_source` is an OUT-PARAM (minimal-churn): when a pool item's
/// "\u{25B6} Source" button is clicked, this sets `*open_source = Some(media_index)`. The caller
/// (app.rs) reads it after the call and opens that media in the Source monitor. `None` on entry;
/// left `None` when no Open-in-Source button was clicked this frame.
///
/// T3 MEDIA BINS: `history` is taken so the two bin GESTURES (create a bin via "+ Bin", move a
/// media into a bin via the per-item picker) can snapshot ONE undo entry before mutating, mirroring
/// the file-wide edit discipline (`history.push(project)` → mutate). Binning is pool-organization
/// ONLY: it groups the pool display by `Project.bin_names` / `Project.media_bin` and never touches
/// clips / the timeline / the render. Both bin fields are `#[serde(default ..)]` on the model, so
/// pre-T3 `.json` projects still load (as one flat "Media" bin).
pub fn pool_ui(
    ui: &mut egui::Ui,
    project: &mut Project,
    history: &mut History,
    open_source: &mut Option<usize>,
) {
    ui.add_space(2.0);
    if ui.button("\u{2795} Add media").clicked() {
        match pick_file() {
            PickResult::Path(path) => {
                // Store the REAL (unencoded) path; the wire layer percent-encodes it per request
                // (worker.rs enc_path) and the engine decodes it, so spaced paths load fine and the
                // saved project still references the true file. No whitespace rejection.
                let name = basename(&path);
                project.media.push(path);
                project.names.push(name);
            }
            PickResult::Cancelled => {}
        }
    }

    // T3 — "+ Bin": a persistent (egui-memory) text field + button that creates a new media bin via
    // `Project::add_bin` (a blank name is a no-op there → returns 0, adds nothing). The text buffer
    // lives in egui temp memory (this fn is stateless), keyed by a fixed Id. We snapshot history
    // BEFORE the create so one click = one undo entry. add_bin is the ONLY mutation here.
    let new_bin_id = egui::Id::new("pool_new_bin_name");
    let mut new_bin_name: String =
        ui.data_mut(|d| d.get_temp::<String>(new_bin_id).unwrap_or_default());
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("\u{1F5C0} Bin").color(crate::theme::TEXT).size(11.0));
        ui.add(
            egui::TextEdit::singleline(&mut new_bin_name)
                .hint_text("new bin\u{2026}")
                .desired_width(96.0),
        );
        // Only commit a NON-blank name (mirrors add_bin's own blank guard); on success clear the
        // field so the next bin starts fresh.
        if ui.button("+ Bin").clicked() && !new_bin_name.trim().is_empty() {
            history.push(project); // pre-edit snapshot (one gesture = one undo entry)
            project.add_bin(&new_bin_name);
            new_bin_name.clear();
        }
    });
    ui.data_mut(|d| d.insert_temp(new_bin_id, new_bin_name));

    ui.separator();

    if project.media.is_empty() {
        ui.weak("No media — click \u{2795} Add media\u{2026}");
        return;
    }

    // T3 — the bin set to GROUP by. An empty `bin_names` is treated as one implicit "Media" bin so
    // a pre-T3 / cleared project still renders every media under one header (never a panic / empty UI).
    let bin_labels: Vec<String> = if project.bin_names.is_empty() {
        vec!["Media".to_string()]
    } else {
        project.bin_names.clone()
    };
    let nbins = bin_labels.len();

    // A deferred bin MOVE collected from the per-item picker (combo borrows `project` immutably while
    // open, so we apply AFTER the scroll area where `history.push` + `set_media_bin` are free). One
    // move per frame, snapshotting history first — exactly one undo entry per user gesture.
    let mut pending_move: Option<(usize, u32)> = None;

    // T4 RELINK — a deferred RELINK collected from a missing item's "Relink\u{2026}" button. The
    // zenity picker is blocking, but we still defer the model mutation to AFTER the scroll area
    // (mirroring `pending_move`) so the per-item `project` borrows inside the loop stay immutable and
    // `history.push(project)` + `project.relink_media(..)` run once, free, per user gesture.
    let mut pending_relink: Option<(usize, String)> = None;

    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        // GROUP the pool by bin: for each bin, a header then every media whose `bin_of == bin`.
        for bin in 0..nbins {
            let bin_u = bin as u32;
            // The media in this bin (any media index whose stored bin clamps to `bin`; bin_of
            // returns 0 for unassigned / past-end so unbinned media fall under bin 0 = "Media").
            let members: Vec<usize> =
                (0..project.media.len()).filter(|&i| project.bin_of(i) == bin_u).collect();

            // Bin header — show the count so an EMPTY bin is still visible (and never panics).
            ui.label(
                egui::RichText::new(format!("\u{1F5C0} {}  ({})", bin_labels[bin], members.len()))
                    .color(crate::theme::TEXT)
                    .strong()
                    .size(12.0),
            );
            if members.is_empty() {
                ui.weak("   (empty)");
                ui.add_space(2.0);
                continue;
            }

            for &i in &members {
                let name = project.names.get(i).cloned().unwrap_or_else(|| format!("media {i}"));
                let path = project.media.get(i).cloned();
                let cur_bin = project.bin_of(i);
                // T4 RELINK — does this media's file currently exist on disk? A real file that is
                // gone (deleted/renamed/moved) gets a red "missing" tag + a Relink button below;
                // a present file (or a non-file sentinel) shows neither. Computed once per item.
                let missing = path.as_deref().map(media_is_missing).unwrap_or(false);
                ui.group(|ui| {
                    // The label rows are the DRAG HANDLE: dragging them carries `DragMedia(i)` for
                    // the timeline lane drop zone (slice B). The "Add as clip" button below stays a
                    // normal click target (it is OUTSIDE the drag source, so its click is not eaten
                    // by the drag `Sense`). The id must be globally unique → key on the media index.
                    let src =
                        ui.dnd_drag_source(egui::Id::new(("poolmedia", i)), DragMedia(i), |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!("{i}"))
                                        .color(egui::Color32::from_rgb(150, 150, 160))
                                        .monospace(),
                                );
                                ui.label(egui::RichText::new(&name).color(crate::theme::TEXT));
                            });
                            if let Some(path) = &path {
                                ui.label(
                                    egui::RichText::new(path)
                                        .color(egui::Color32::from_rgb(120, 120, 130))
                                        .size(9.0),
                                );
                            }
                            // T4 RELINK — flag a missing file inline (red), right under its path, so
                            // it reads as "this file is gone" at a glance (mirrors Shotcut's red
                            // "missing" badge). Only shown for a real-but-missing file.
                            if missing {
                                ui.label(
                                    egui::RichText::new("\u{26A0} missing")
                                        .color(egui::Color32::from_rgb(220, 80, 80))
                                        .strong()
                                        .size(9.0),
                                );
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
                    // T4 RELINK — a missing file gets a "Relink\u{2026}" button that opens the native
                    // picker (blocking); on a chosen path we record a DEFERRED relink (applied after
                    // the scroll area where `history.push` + `relink_media` are free). A present file
                    // shows NO relink affordance. Cancelling the dialog (None) is a quiet no-op.
                    if missing {
                        ui.horizontal(|ui| {
                            if ui
                                .button(
                                    egui::RichText::new("\u{1F517} Relink\u{2026}")
                                        .color(crate::theme::TEXT)
                                        .size(11.0),
                                )
                                .on_hover_text("Choose the file's new location")
                                .clicked()
                            {
                                if let Some(new_path) = pick_relink_file() {
                                    pending_relink = Some((i, new_path));
                                }
                            }
                        });
                    }
                    // T3 — per-item BIN PICKER: a combo over the bins; selecting a DIFFERENT bin
                    // records a deferred move (applied after the scroll area, where history.push is
                    // free). Only meaningful with >1 bin; with a single bin it just shows "Media".
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("Bin")
                                .color(egui::Color32::from_rgb(150, 150, 160))
                                .size(10.0),
                        );
                        let cur_label = bin_labels
                            .get(cur_bin as usize)
                            .cloned()
                            .unwrap_or_else(|| bin_labels[0].clone());
                        egui::ComboBox::from_id_salt(("pool_bin_pick", i))
                            .selected_text(cur_label)
                            .show_ui(ui, |ui| {
                                for (b, label) in bin_labels.iter().enumerate() {
                                    let b_u = b as u32;
                                    if ui.selectable_label(b_u == cur_bin, label).clicked()
                                        && b_u != cur_bin
                                    {
                                        pending_move = Some((i, b_u));
                                    }
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Add as clip \u{2192} V1").clicked() {
                            // Append on V1 (track 0) at the end of the program.
                            // `Clip::video` discards its `name_hint` arg (model.rs:
                            // `let _ = name_hint;`), so pass "" (no per-click name clone).
                            let t0 = project.total_frames();
                            project.clips.push(Clip::video(i, t0, DEFAULT_CLIP_LEN, 0, ""));
                        }
                        // P18: open THIS media in the Source monitor (a second preview that scrubs
                        // the raw clip with its own playhead + in/out). Reports the index via the
                        // out-param; app.rs switches `monitor` to Source and recomposes the source.
                        if ui.button("\u{25B6} Source").clicked() {
                            *open_source = Some(i);
                        }
                    });
                });
                ui.add_space(2.0);
            }
            ui.add_space(4.0);
        }
    });

    // Apply the deferred bin move (if any): snapshot history BEFORE the mutation so the picker is one
    // undo entry, then `set_media_bin` (which clamps the bin + no-ops an out-of-range media itself).
    if let Some((media_idx, bin)) = pending_move {
        history.push(project);
        project.set_media_bin(media_idx, bin);
    }

    // T4 RELINK — apply the deferred relink (if any): snapshot history BEFORE the swap so one Relink
    // gesture = one undo entry, then `relink_media` (which itself no-ops an out-of-range index or an
    // empty path and returns false). Every clip referencing this media keeps its index, so it now
    // decodes the new file. A cancelled dialog left `pending_relink` None → nothing happens here.
    if let Some((media_idx, new_path)) = pending_relink {
        history.push(project);
        project.relink_media(media_idx, &new_path);
    }
}
