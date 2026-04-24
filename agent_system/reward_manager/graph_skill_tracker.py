import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

from .skill_graph import SkillGraph


@dataclass
class GraphNode:
    skill_id: str
    trigger_pattern: str
    next_nodes: List[Tuple[str, float]] = field(default_factory=list)


class GraphSkillTracker:
    """
    Per-episode state machine that tracks which skill graph node each env is on.

    When obs+action text matches a neighbor node's trigger_pattern, the tracker
    transitions to that neighbor and fires a reward of edge_weight * reward_per_transition.
    Multiple transitions can fire in a single step if several neighbors match.
    """

    def __init__(
        self,
        batch_size: int,
        entry_nodes_per_env: List[List[str]],
        graph: SkillGraph,
        reward_per_transition: float = 0.15,
    ):
        self.batch_size = batch_size
        self.graph = graph
        self.reward_per_transition = reward_per_transition

        # current_nodes[env_idx] = list of active skill_ids (multi-entry support)
        self.current_nodes: List[List[str]] = [
            list(nodes) for nodes in entry_nodes_per_env
        ]

    def check(
        self,
        obs_texts: List[str],
        action_texts: List[str],
        active_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Check graph transitions for each env and return per-env rewards.

        For each env:
          1. Gather all unique neighbor nodes across all current active nodes.
          2. Match each neighbor's trigger_pattern against combined obs+action text.
          3. For each match: reward += edge_weight * reward_per_transition,
             and advance current_nodes to include the matched neighbor.
        """
        rewards = np.zeros(self.batch_size, dtype=np.float32)

        for env_idx in range(self.batch_size):
            if not active_masks[env_idx]:
                continue

            obs = obs_texts[env_idx] if obs_texts and env_idx < len(obs_texts) else ""
            action = action_texts[env_idx] if action_texts and env_idx < len(action_texts) else ""
            combined = (obs + " " + action).lower()

            # Collect all outgoing edges from all current active nodes
            seen_neighbors: dict = {}  # neighbor_skill_id -> edge_weight
            for node_id in self.current_nodes[env_idx]:
                for edge in self.graph.get_neighbors(node_id):
                    nid = edge["skill_id"]
                    w = float(edge.get("weight", 1.0))
                    # keep max weight if same neighbor appears via multiple paths
                    if nid not in seen_neighbors or w > seen_neighbors[nid]:
                        seen_neighbors[nid] = w

            newly_active = []
            for nid, weight in seen_neighbors.items():
                pattern = self.graph.get_trigger_pattern(nid)
                if not pattern:
                    continue
                if re.search(pattern, combined, re.IGNORECASE):
                    rewards[env_idx] += weight * self.reward_per_transition
                    newly_active.append(nid)

            if newly_active:
                # Advance: replace current nodes with the newly transitioned nodes.
                # Retain current nodes that have outgoing edges not yet traversed,
                # but cap list size to avoid unbounded growth.
                self.current_nodes[env_idx] = list(dict.fromkeys(
                    newly_active + self.current_nodes[env_idx]
                ))[:4]

        return rewards


def build_graph_skill_tracker(
    envs: Any,
    config: Any,
    skill_graph: SkillGraph,
) -> Optional[GraphSkillTracker]:
    """Factory: build a GraphSkillTracker from env + config, or None if disabled."""
    gsr_cfg = config.env.get("graph_skill_reward", {})
    if not gsr_cfg.get("enable", False):
        return None

    reward_per_transition = float(gsr_cfg.get("reward_per_transition", 0.15))

    tasks = getattr(envs, "tasks", None)
    if not tasks:
        return None

    entry_nodes_per_env = [
        skill_graph.get_entry_nodes_for_task(task) for task in tasks
    ]

    return GraphSkillTracker(
        batch_size=len(tasks),
        entry_nodes_per_env=entry_nodes_per_env,
        graph=skill_graph,
        reward_per_transition=reward_per_transition,
    )
