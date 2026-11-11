//! [`AgentEdge`]: the `EdgeContent` for parent→agent edges.
//!
//! Wraps the built-in `StepEdge` routing and colors the edge green while the
//! target agent is running ("alive") — so liveness reads on the graph structure
//! at any zoom. The agent's tool detail lives in its chips + detail panel, not
//! on the edge (no label).

use rataflow::{
    EdgeContent, EdgePathContext, EdgeRenderContext, EdgeStyle, Path, Position, StepEdge,
};
use ratatui::buffer::Buffer;
use ratatui::style::Style;

use crate::state::graph::WRAP_V_GAP;

// (Running edges use `palette.success` at render time — green = alive, the
// same presence language as the node glyphs. Resolved in `render`, not a
// const, so theme switches carry through.)

/// Step-routed parent edge with a distinct running color (no label).
#[derive(Debug, Default, Clone)]
pub struct AgentEdge {
    inner: StepEdge,
    /// Mirrored from the target agent's status by the graph sync; switches the
    /// edge chars to the palette's success green ("alive").
    pub running: bool,
    /// World x of the vertical trunk that feeds a wrapped row, set by the
    /// wrapped layout for children below the first row of their parent's grid.
    /// `None` routes with the plain step path.
    pub trunk_x: Option<f64>,
}

impl EdgeContent for AgentEdge {
    fn compute_path(&self, ctx: &EdgePathContext) -> Path {
        // A plain step path to a wrapped row turns halfway down, which is inside
        // the rows above it. Instead drop into the gap under the parent, run
        // along it to the trunk in the column gap, down the trunk to the gap
        // above the target's row, and across into the target. The distance
        // guard keeps a stale trunk (after a manual drag) from bending an edge
        // that no longer spans a row.
        if let Some(trunk_x) = self.trunk_x
            && ctx.to.y - ctx.from.y > WRAP_V_GAP * 1.5
        {
            let below_parent = ctx.from.y + WRAP_V_GAP / 2.0;
            let above_row = ctx.to.y - WRAP_V_GAP / 2.0;
            return Path::new(
                vec![
                    ctx.from,
                    Position::new(ctx.from.x, below_parent),
                    Position::new(trunk_x, below_parent),
                    Position::new(trunk_x, above_row),
                    Position::new(ctx.to.x, above_row),
                    ctx.to,
                ],
                ctx.source_position,
                ctx.target_position,
            );
        }
        self.inner.compute_path(ctx)
    }

    fn render(&self, ctx: &EdgeRenderContext, buf: &mut Buffer) {
        // Running edges get their distinct green so liveness survives overlapping
        // paths. No selection guard: edges are built non-selectable
        // (`with_selectable(false)` in graph sync), so `ctx.selected` is never
        // true here. No label — tool detail lives in the child's chips + panel.
        let style = if self.running {
            let green = ctx.theme.palette().success;
            EdgeStyle::default().with_stroke_style(Style::default().fg(green))
        } else {
            EdgeStyle::default()
        };
        ctx.render_path(&style, None, buf);
    }
}
