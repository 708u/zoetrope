//! Incremental projection of [`SessionModel`] onto a rataflow `Flow`.
//!
//! Never rebuilds: per agent, either mutate the existing node content in place
//! via `node_content_mut`, or `add_node` + `add_edge` (duplicate-id `Err` is an
//! idempotent no-op). Nodes are added before their edges. A structural change
//! (node/edge added) marks layout dirty; at sync end we run Sugiyama. Selection
//! survives because node ids are stable and we never clear-and-re-add.

use std::collections::{HashMap, HashSet};

use rataflow::{
    Edge, Flow, Handle, HandlePosition, Node, Position, Reconnectable, Sugiyama, Theme,
};
use ratatui::style::Color;

use super::session::{AgentInfo, AgentKind, AgentStatus, SessionModel};
use crate::ui::edges::AgentEdge;
use crate::ui::nodes::{AgentNode, MAIN_NODE_DIMS, SUB_NODE_DIMS};

/// The concrete `Flow` type zoetrope uses: agent-card nodes, step-routed parent
/// edges (no labels — liveness reads from color alone).
pub type AgentFlow = Flow<AgentNode, AgentEdge>;

/// Build an empty, fully-configured `Flow` for zoetrope.
///
/// Config: `with_deselect_on_pane_click(false)`, `deselect_on_drag = false`
/// (detail panel persists), `with_min_zoom(0.1)` (Sugiyama trees outgrow the
/// default fit-view limit). Hidden source/target handles for a clean look.
pub fn new_flow() -> AgentFlow {
    // zoetrope identity palette: stock dark base, but `accent` becomes GOLD —
    // selection highlights, done medals, the REPLAY badge. Green stays
    // exclusively "alive" (status), red "failed". Every surface resolves from
    // flow.theme, so this one assignment brands the whole app.
    let mut palette = Theme::Dark.palette();
    palette.accent = Color::Indexed(178);
    let mut flow = Flow::new()
        .with_theme(Theme::Custom(palette))
        .with_deselect_on_pane_click(false)
        // We drive the camera on selection ourselves (a center-glide via
        // `pending_center`), so suppress the library's instant ensure-visible pan
        // — otherwise the two stack into a jump-then-glide on off-screen nodes.
        .with_selection_reveal(rataflow::SelectionReveal::None)
        .with_min_zoom(0.1);
    flow.deselect_on_drag = false;
    flow
}

/// Title line for a node: the agent type the provider recorded, else the
/// generic label for its kind. No provider name appears here; the root
/// agent's is stated by its provider like any other agent's.
fn node_title(info: &AgentInfo) -> String {
    info.agent_type
        .clone()
        .unwrap_or_else(|| info.kind.default_label().to_string())
}

/// Fixed card dimensions for a node kind.
fn node_dims(kind: AgentKind) -> (f64, f64) {
    match kind {
        AgentKind::Main | AgentKind::Group => MAIN_NODE_DIMS,
        AgentKind::Subagent => SUB_NODE_DIMS,
    }
}

/// Whether a node's content already mirrors the agent — allocation-free
/// comparison so unchanged agents skip [`build_content`]'s String clones on
/// every sync (the steady state for almost all agents on almost all ticks).
fn content_matches(info: &AgentInfo, node: &AgentNode) -> bool {
    let title_ok = node.title
        == info
            .agent_type
            .as_deref()
            .unwrap_or(info.kind.default_label());
    title_ok
        && node.description.as_deref() == info.description.as_deref()
        && node.status == info.status
        && node.tool_count == info.tool_calls.len()
        && node.last_tool.as_deref() == info.last_tool()
        && node.output_tokens == info.output_tokens
        && node.interactive == info.is_interactive()
}

/// Build the [`AgentNode`] content mirrored from an [`AgentInfo`].
fn build_content(info: &AgentInfo) -> AgentNode {
    AgentNode {
        title: node_title(info),
        description: info.description.clone(),
        status: info.status,
        tool_count: info.tool_calls.len(),
        last_tool: info.last_tool().map(str::to_string),
        output_tokens: info.output_tokens,
        interactive: info.is_interactive(),
    }
}

/// Horizontal gap between locally-placed siblings (world units).
const LOCAL_H_GAP: f64 = 4.0;
/// Vertical gap below a parent for locally-placed children (world units).
const LOCAL_V_GAP: f64 = 5.0;

/// Default column count of the wrapped layout; `0` selects Sugiyama.
pub const DEFAULT_WRAP_COLUMNS: usize = 4;
/// Gap between columns of a wrapped grid (world units). Even, so the trunk
/// down the middle of a gap lands on a whole cell.
pub const WRAP_H_GAP: f64 = 4.0;
/// Gap above each row of a wrapped grid (world units): room for the chip stack
/// under the cards above (up to three rows) plus the bus line feeding the row.
pub const WRAP_V_GAP: f64 = 6.0;

/// Incrementally sync `flow` to `model`.
///
/// For each agent in spawn order: mutate the existing node content in place, or
/// add the node (then its parent edge). Updates edge `animated` from target
/// status. New nodes get LOCAL placement (below their parent, offset past
/// siblings) so they land somewhere sensible even without a relayout.
///
/// `wrap` is the column count of the wrapped layout (`0` for Sugiyama). The
/// wrapped layout appends a new child after its siblings, so it moves existing
/// cards only when a row grows taller; any structural change therefore applies
/// it in full, and the local placement never shows.
///
/// When `relayout` is true, any structural change ends with a full layout pass
/// (which overwrites the local placements). When false and `wrap` is `0` —
/// Manual camera: the user owns the view — nothing existing moves; the caller
/// tracks dirtiness and relayouts when the camera re-engages.
/// Returns `true` if structure changed.
pub fn sync(flow: &mut AgentFlow, model: &SessionModel, relayout: bool, wrap: usize) -> bool {
    let mut structural = false;

    // First pass: nodes (must exist before their edges).
    for id in &model.spawn_order {
        let Some(info) = model.agent(id) else {
            continue;
        };
        if let Some(existing) = flow.node_content_mut(id) {
            // Steady state: only rebuild (String clones) when something
            // visible changed — the per-second status tick and per-batch
            // syncs walk every agent, and most are unchanged.
            if !content_matches(info, existing) {
                *existing = build_content(info);
            }
        } else {
            // Sibling index for local placement — computed only for the rare
            // new node; the no-new-nodes steady state skips it entirely.
            let siblings = info
                .parent
                .as_deref()
                .map(|p| {
                    model
                        .spawn_order
                        .iter()
                        .take_while(|x| *x != id)
                        .filter(|x| model.agent(x).and_then(|a| a.parent.as_deref()) == Some(p))
                        .count()
                })
                .unwrap_or(0);
            let content = build_content(info);
            let (w, h) = node_dims(info.kind);
            // Local placement: below the parent, fanned past prior siblings.
            // Overwritten by Sugiyama when `relayout` runs; kept verbatim in
            // Manual so existing nodes never move underneath the user.
            let pos = info
                .parent
                .as_deref()
                .and_then(|p| flow.node(p))
                .map(|parent| {
                    (
                        parent.position.x + siblings as f64 * (w + LOCAL_H_GAP),
                        parent.position.y + parent.height + LOCAL_V_GAP,
                    )
                })
                .unwrap_or((0.0, 0.0));
            // Read-only monitor: nodes are selectable (detail panel) and
            // draggable (manual arrangement) — but never deletable and never
            // connection sources. Enforced at the DTO level, not just the key
            // whitelist, so no input path can mutate the graph.
            let node = Node::new(id.clone(), pos, (w, h), content)
                .with_deletable(false)
                .with_connectable(false)
                .with_handles(vec![
                    Handle::source(HandlePosition::Bottom).with_hidden(true),
                    Handle::target(HandlePosition::Top).with_hidden(true),
                ]);
            // Duplicate-id is an idempotent no-op; a genuine add is structural.
            if flow.add_node(node).is_ok() {
                structural = true;
            }
        }
    }

    // Second pass: edges from each agent to its parent.
    for id in &model.spawn_order {
        let Some(info) = model.agent(id) else {
            continue;
        };
        let Some(parent) = &info.parent else {
            continue;
        };
        let animated = info.status == AgentStatus::Running;
        let edge_id = edge_id(id);
        // Edge already present (the steady state on every sync): just refresh
        // animation — probing via `edge_content_mut` first avoids building a
        // throwaway Edge (three String clones) per agent per sync only for
        // `add_edge` to reject it as a duplicate. Edges carry no selectable
        // meaning in zoetrope (no edge panel), and a stray edge click would pin
        // Follow mode while closing the node panel — a dead state. Fully inert:
        // not selectable, deletable, or reconnectable. Liveness shows as the
        // running color + marching ants, NOT a label — the current tool already
        // shows in the child's chips and detail panel.
        if let Some(content) = flow.edge_content_mut(&edge_id) {
            content.running = animated;
            flow.set_edge_animated(&edge_id, animated);
        } else {
            let edge = Edge::new(edge_id.clone(), parent.clone(), id.clone())
                .with_animated(animated)
                .with_selectable(false)
                .with_deletable(false)
                .with_reconnectable(Reconnectable::None);
            if flow.add_edge(edge).is_ok() {
                structural = true;
            }
            if let Some(content) = flow.edge_content_mut(&edge_id) {
                content.running = animated;
            }
        }
    }

    if structural && (relayout || wrap > 0) {
        self::relayout(flow, wrap);
    }
    structural
}

/// Stable id for the (single) parent edge of `child`.
///
/// Keyed by the child alone: every agent has exactly one parent edge, and
/// `sync` never removes edges — so the id must never change once created.
/// Keying on `spawned_by` or the parent would orphan a stale edge if
/// either field were filled in after the edge existed (latent today, armed by
/// any future meta re-emission).
fn edge_id(child: &str) -> String {
    format!("e-{child}")
}

/// Remove several agents from the canvas in ONE pass.
///
/// Bulk, not a loop of single removals: `Flow::remove_node` costs O(nodes plus
/// edges) every time — it shifts every lookup index and rebuilds the edge
/// lookup per call — so removing k of them is quadratic. `retain_nodes` makes a
/// single pass over each vec and drops the connected edges itself.
pub fn remove_agents(flow: &mut AgentFlow, agents: &[String]) {
    if agents.is_empty() {
        return;
    }
    let doomed: std::collections::HashSet<&str> = agents.iter().map(String::as_str).collect();
    flow.retain_nodes(|n| !doomed.contains(n.id.as_str()));
}

/// Lay out `flow`: the wrapped grid with `wrap` columns, or the Sugiyama
/// vertical layout when `wrap` is `0`.
///
/// Split out so it can be called explicitly and unit-tested independently of
/// the per-agent diffing in [`sync`].
pub fn relayout(flow: &mut AgentFlow, wrap: usize) {
    if wrap == 0 {
        flow.apply_layout(Sugiyama::vertical());
        let ids: Vec<String> = flow.edges().iter().map(|e| e.id.clone()).collect();
        for id in ids {
            if let Some(content) = flow.edge_content_mut(&id) {
                content.trunk_x = None;
            }
        }
        return;
    }
    let layout = WrappedLayout::compute(flow, wrap);
    flow.set_node_positions(layout.positions.iter().map(|(id, p)| (id.as_str(), *p)));
    let edges: Vec<(String, String)> = flow
        .edges()
        .iter()
        .map(|e| (e.id.clone(), e.target.clone()))
        .collect();
    for (edge, target) in edges {
        if let Some(content) = flow.edge_content_mut(&edge) {
            content.trunk_x = layout.trunks.get(&target).copied();
        }
    }
}

/// Positions from the wrapped layout.
///
/// Each card sits at the top-left of its subtree's block, and its children fill
/// a grid of at most `cols` columns below it in spawn order, so a new child
/// lands after its siblings and the cards before it keep their place. A child
/// with children of its own takes one grid cell sized to its whole subtree;
/// each column is as wide as its widest cell and each row as tall as its
/// tallest, which keeps the column gaps aligned from row to row so the trunks
/// running down them never cross a card.
#[derive(Debug, Default)]
struct WrappedLayout {
    /// Top-left corner of every card.
    positions: HashMap<String, Position>,
    /// Trunk x for each child below the first row of its parent's grid, keyed
    /// by the child: the middle of the gap left of the child's column.
    trunks: HashMap<String, f64>,
}

impl WrappedLayout {
    fn compute(flow: &AgentFlow, cols: usize) -> Self {
        let dims: HashMap<&str, (f64, f64)> = flow
            .nodes()
            .map(|n| (n.id.as_str(), (n.width, n.height)))
            .collect();
        // Edges are added in spawn order, so each parent's children list keeps
        // the order its agents were spawned in.
        let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut has_parent: HashSet<&str> = HashSet::new();
        for e in flow.edges() {
            let (source, target) = (e.source.as_str(), e.target.as_str());
            if dims.contains_key(source) && dims.contains_key(target) {
                children.entry(source).or_default().push(target);
                has_parent.insert(target);
            }
        }
        let mut builder = Builder {
            cols: cols.max(1),
            dims,
            children,
            sizes: HashMap::new(),
            out: Self::default(),
        };
        // Roots side by side, separated by a double gap so their trunks stay
        // apart.
        let roots: Vec<&str> = flow
            .nodes()
            .map(|n| n.id.as_str())
            .filter(|id| !has_parent.contains(id))
            .collect();
        let mut x = 0.0;
        for root in roots {
            let (w, _) = builder.measure(root);
            builder.place(root, x, 0.0);
            x += w + WRAP_H_GAP * 2.0;
        }
        builder.out
    }
}

struct Builder<'a> {
    cols: usize,
    dims: HashMap<&'a str, (f64, f64)>,
    children: HashMap<&'a str, Vec<&'a str>>,
    /// Memoized subtree block sizes.
    sizes: HashMap<&'a str, (f64, f64)>,
    out: WrappedLayout,
}

impl<'a> Builder<'a> {
    fn kids(&self, id: &str) -> Vec<&'a str> {
        self.children.get(id).cloned().unwrap_or_default()
    }

    /// Column widths and row heights of the grid holding `kids`.
    fn grid(&mut self, kids: &[&'a str]) -> (Vec<f64>, Vec<f64>) {
        let cols = self.cols.min(kids.len());
        let mut col_w = vec![0.0_f64; cols];
        let mut row_h = vec![0.0_f64; kids.len().div_ceil(cols)];
        for (i, kid) in kids.iter().enumerate() {
            let (w, h) = self.measure(kid);
            col_w[i % cols] = col_w[i % cols].max(w);
            row_h[i / cols] = row_h[i / cols].max(h);
        }
        (col_w, row_h)
    }

    /// Width and height of the block holding `id` and its subtree.
    fn measure(&mut self, id: &'a str) -> (f64, f64) {
        if let Some(&size) = self.sizes.get(id) {
            return size;
        }
        let (w, h) = self.dims[id];
        let kids = self.kids(id);
        let size = if kids.is_empty() {
            (w, h)
        } else {
            let (col_w, row_h) = self.grid(&kids);
            let grid_w = col_w.iter().sum::<f64>() + WRAP_H_GAP * (col_w.len() - 1) as f64;
            let grid_h = row_h.iter().map(|h| h + WRAP_V_GAP).sum::<f64>();
            (w.max(grid_w), h + grid_h)
        };
        self.sizes.insert(id, size);
        size
    }

    fn place(&mut self, id: &'a str, left: f64, top: f64) {
        self.out
            .positions
            .insert(id.to_string(), Position::new(left, top));
        let kids = self.kids(id);
        if kids.is_empty() {
            return;
        }
        let (_, h) = self.dims[id];
        let (col_w, row_h) = self.grid(&kids);
        let mut col_x = Vec::with_capacity(col_w.len());
        let mut x = left;
        for w in &col_w {
            col_x.push(x);
            x += w + WRAP_H_GAP;
        }
        let mut y = top + h + WRAP_V_GAP;
        for (r, row) in kids.chunks(col_w.len()).enumerate() {
            for (c, kid) in row.iter().enumerate() {
                if r > 0 {
                    self.out
                        .trunks
                        .insert(kid.to_string(), col_x[c] - WRAP_H_GAP / 2.0);
                }
                self.place(kid, col_x[c], y);
            }
            y += row_h[r] + WRAP_V_GAP;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::claude::wire::SubagentMeta;

    /// A model with main + one direct subagent (running).
    fn model_with_subagent() -> SessionModel {
        let mut m = SessionModel::new("s1".into());
        // The root's name is the provider's to state.
        m.apply_fact(&crate::fact::Fact {
            agent: Some(super::super::session::MAIN_ID.to_string()),
            ts: None,
            kind: crate::fact::FactKind::Agent {
                kind: AgentKind::Main,
                parent: None,
                agent_type: Some("claude".into()),
                description: None,
                spawned_by: None,
                interactive: true,
            },
        });
        let meta = SubagentMeta {
            agent_type: Some("guide".into()),
            description: Some("research".into()),
            tool_use_id: Some("ag1".into()),
            stopped_by_user: None,
        };
        m.apply_meta("abc123", None, &meta);
        m
    }

    #[test]
    fn sync_creates_nodes_and_edge() {
        let model = model_with_subagent();
        let mut flow = new_flow();
        let structural = sync(&mut flow, &model, true, 0);
        assert!(structural);
        assert!(flow.node_content_mut("main").is_some());
        assert!(flow.node_content_mut("abc123").is_some());
        // One edge main -> abc123.
        assert_eq!(flow.edges().len(), 1);
    }

    #[test]
    fn sync_idempotent() {
        let model = model_with_subagent();
        let mut flow = new_flow();
        let first = sync(&mut flow, &model, true, 0);
        assert!(first);
        let node_count = flow.nodes().count();
        let edge_count = flow.edges().len();

        // Applying the same model again adds nothing structural.
        let second = sync(&mut flow, &model, true, 0);
        assert!(!second);
        assert_eq!(flow.nodes().count(), node_count);
        assert_eq!(flow.edges().len(), edge_count);
    }

    #[test]
    fn sync_preserves_selection() {
        let model = model_with_subagent();
        let mut flow = new_flow();
        sync(&mut flow, &model, true, 0);
        flow.select_node("abc123");
        assert_eq!(
            flow.selected_nodes().next().map(|n| n.id.clone()),
            Some("abc123".to_string())
        );

        // Re-sync after a non-structural change (e.g. a tool call added).
        let mut model2 = model;
        if let Some(a) = model2.agents.get_mut("abc123") {
            a.output_tokens += 100;
        }
        sync(&mut flow, &model2, true, 0);
        assert_eq!(
            flow.selected_nodes().next().map(|n| n.id.clone()),
            Some("abc123".to_string())
        );
    }

    #[test]
    fn edge_animation_follows_status() {
        let mut model = model_with_subagent();
        let mut flow = new_flow();
        sync(&mut flow, &model, true, 0);
        // Running subagent -> animated edge.
        let edge_id = edge_id("abc123");
        let animated = flow
            .edges()
            .iter()
            .find(|e| e.id == edge_id)
            .map(|e| e.animated);
        assert_eq!(animated, Some(true));

        // The edge content mirrors running-ness (drives the distinct color).
        assert!(flow.edge_content_mut(&edge_id).unwrap().running);

        // Mark done, re-sync -> no longer animated, color back to default.
        if let Some(a) = model.agents.get_mut("abc123") {
            a.status = AgentStatus::Done;
        }
        sync(&mut flow, &model, true, 0);
        let animated = flow
            .edges()
            .iter()
            .find(|e| e.id == edge_id)
            .map(|e| e.animated);
        assert_eq!(animated, Some(false));
        assert!(!flow.edge_content_mut(&edge_id).unwrap().running);
    }

    #[test]
    fn graph_is_structurally_read_only() {
        use rataflow::Reconnectable;

        let model = model_with_subagent();
        let mut flow = new_flow();
        sync(&mut flow, &model, true, 0);

        for node in flow.nodes() {
            assert!(node.selectable, "nodes stay selectable (detail panel)");
            assert!(node.draggable, "nodes stay draggable (manual arranging)");
            assert!(!node.deletable, "nodes must not be deletable");
            assert!(!node.connectable, "nodes must not start connections");
        }
        for edge in flow.edges() {
            assert!(!edge.selectable, "edges carry no selectable meaning");
            assert!(!edge.deletable);
            assert_eq!(edge.reconnectable, Reconnectable::None);
        }
    }

    #[test]
    fn manual_mode_local_placement_moves_nothing_existing() {
        let mut model = model_with_subagent();
        let mut flow = new_flow();
        // Initial layout (camera engaged).
        sync(&mut flow, &model, true, 0);
        let main_pos = flow.node("main").unwrap().position;
        let first_sub = flow.node("abc123").unwrap().position;

        // Camera now Manual: a second subagent arrives, relayout deferred.
        let meta2 = SubagentMeta {
            agent_type: Some("guide".into()),
            description: None,
            tool_use_id: Some("ag2".into()),
            stopped_by_user: None,
        };
        model.apply_meta("def456", None, &meta2);
        let structural = sync(&mut flow, &model, false, 0);
        assert!(structural);

        // Nothing existing moved...
        assert_eq!(flow.node("main").unwrap().position, main_pos);
        assert_eq!(flow.node("abc123").unwrap().position, first_sub);
        // ...and the newcomer landed below its parent, not at the origin.
        let new_pos = flow.node("def456").unwrap().position;
        assert!(new_pos.y > main_pos.y, "child placed below parent");
        assert_ne!((new_pos.x, new_pos.y), (0.0, 0.0));
    }

    #[test]
    fn cards_render_into_buffer() {
        use ratatui::buffer::Buffer;
        use ratatui::layout::Rect;
        use ratatui::widgets::Widget;

        let model = model_with_subagent();
        let mut flow = new_flow();
        sync(&mut flow, &model, true, 0);
        flow.request_fit_view();

        let area = Rect::new(0, 0, 100, 30);
        let mut buf = Buffer::empty(area);
        (&mut flow).render(area, &mut buf);

        let mut text = String::new();
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                text.push_str(buf[(x, y)].symbol());
            }
            text.push('\n');
        }
        assert!(
            text.contains("claude"),
            "main card title missing from render:\n{text}"
        );
        assert!(
            text.contains("guide"),
            "subagent card title missing from render:\n{text}"
        );
    }

    /// `model_with_subagent` plus `extra` more direct subagents.
    fn model_with_subagents(extra: usize) -> (SessionModel, Vec<String>) {
        let mut model = model_with_subagent();
        let mut ids = vec!["abc123".to_string()];
        for i in 0..extra {
            let id = format!("sub{i}");
            let meta = SubagentMeta {
                agent_type: Some("guide".into()),
                description: None,
                tool_use_id: Some(format!("ag-extra{i}")),
                stopped_by_user: None,
            };
            model.apply_meta(&id, None, &meta);
            ids.push(id);
        }
        (model, ids)
    }

    fn rect(flow: &AgentFlow, id: &str) -> (f64, f64, f64, f64) {
        let n = flow.node(id).unwrap();
        (n.position.x, n.position.y, n.width, n.height)
    }

    #[test]
    fn wrapped_layout_breaks_rows_after_wrap_columns() {
        let (model, ids) = model_with_subagents(5);
        let mut flow = new_flow();
        sync(&mut flow, &model, false, 4);

        let first_row: Vec<_> = ids[..4].iter().map(|id| rect(&flow, id)).collect();
        let second_row: Vec<_> = ids[4..].iter().map(|id| rect(&flow, id)).collect();
        assert!(first_row.iter().all(|r| r.1 == first_row[0].1));
        assert!(second_row.iter().all(|r| r.1 == second_row[0].1));
        assert!(
            second_row[0].1 > first_row[0].1 + first_row[0].3,
            "the fifth child wraps below the first row"
        );
        // Columns line up from row to row.
        assert_eq!(second_row[0].0, first_row[0].0);
        assert_eq!(second_row[1].0, first_row[1].0);

        // Only the wrapped row is routed over a trunk.
        for (i, id) in ids.iter().enumerate() {
            let trunk = flow.edge_content_mut(&edge_id(id)).unwrap().trunk_x;
            assert_eq!(trunk.is_some(), i >= 4, "trunk for child {i}");
        }
    }

    #[test]
    fn wrapped_layout_cards_never_overlap() {
        let (model, ids) = model_with_subagents(9);
        let mut flow = new_flow();
        sync(&mut flow, &model, false, 3);
        let mut all = ids.clone();
        all.push("main".into());
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                let (ax, ay, aw, ah) = rect(&flow, a);
                let (bx, by, bw, bh) = rect(&flow, b);
                let apart = ax + aw <= bx || bx + bw <= ax || ay + ah <= by || by + bh <= ay;
                assert!(apart, "{a} overlaps {b}");
            }
        }
    }

    #[test]
    fn wrapped_layout_keeps_existing_cards_when_a_sibling_arrives() {
        let (model, ids) = model_with_subagents(4);
        let mut flow = new_flow();
        sync(&mut flow, &model, false, 4);
        let before: Vec<_> = ids.iter().map(|id| rect(&flow, id)).collect();

        let (model, ids) = model_with_subagents(6);
        sync(&mut flow, &model, false, 4);
        for (id, was) in ids.iter().zip(&before) {
            assert_eq!(rect(&flow, id), *was, "{id} moved");
        }
    }

    #[test]
    fn wrap_zero_keeps_sugiyama_and_plain_edges() {
        let (model, ids) = model_with_subagents(5);
        let mut flow = new_flow();
        sync(&mut flow, &model, true, 0);
        let y = rect(&flow, &ids[0]).1;
        assert!(ids.iter().all(|id| rect(&flow, id).1 == y), "one rank");
        for id in &ids {
            assert!(
                flow.edge_content_mut(&edge_id(id))
                    .unwrap()
                    .trunk_x
                    .is_none()
            );
        }
    }
}
