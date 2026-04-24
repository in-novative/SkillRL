import re
from typing import Dict, List, Optional, Tuple


class SkillGraph:
    """Directed graph over skills: nodes are skill dicts, edges are next_skills links."""

    # Maps ALFWorld task type keyword to the entry skill_id chain
    _TASK_ENTRY_NODES: Dict[str, List[str]] = {
        "pick_and_place": ["gen_010", "pic_001"],
        "pick_heat_then_place": ["gen_010", "hea_001"],
        "pick_cool_then_place": ["gen_010", "coo_001"],
        "pick_clean_then_place": ["gen_010", "cle_001"],
        "look_at_obj_in_light": ["gen_010", "loo_001"],
        "examine": ["gen_010", "exa_001"],
        "pick_two_obj_and_place": ["gen_010", "pic_001"],
    }

    def __init__(self, skills_data: dict):
        # skill_id -> full skill dict
        self._skill_map: Dict[str, dict] = {}
        # skill_id -> [{skill_id, weight}, ...]
        self._adj: Dict[str, List[Dict]] = {}

        self._load(skills_data)

    def _load(self, skills_data: dict):
        all_skills: List[dict] = []

        general = skills_data.get("general_skills", [])
        all_skills.extend(general)

        task_specific = skills_data.get("task_specific_skills", {})
        for skill_list in task_specific.values():
            all_skills.extend(skill_list)

        for skill in all_skills:
            sid = skill["skill_id"]
            self._skill_map[sid] = skill
            self._adj[sid] = skill.get("next_skills", [])

    def get_neighbors(self, skill_id: str) -> List[Dict]:
        """Return [{skill_id, weight}, ...] for outgoing edges from skill_id."""
        return self._adj.get(skill_id, [])

    def get_neighbor_skill_dicts(self, skill_id: str) -> List[dict]:
        """Return full skill dicts for each neighbor of skill_id."""
        result = []
        for edge in self._adj.get(skill_id, []):
            neighbor = self._skill_map.get(edge["skill_id"])
            if neighbor is not None:
                result.append(neighbor)
        return result

    def get_trigger_pattern(self, skill_id: str) -> Optional[str]:
        """Return the trigger_pattern regex for skill_id, or None if absent."""
        skill = self._skill_map.get(skill_id)
        if skill is None:
            return None
        return skill.get("trigger_pattern") or None

    def get_skill(self, skill_id: str) -> Optional[dict]:
        return self._skill_map.get(skill_id)

    def get_entry_nodes(self, task_type: str) -> List[str]:
        """Return ordered list of entry skill_ids for a given ALFWorld task type."""
        return list(self._TASK_ENTRY_NODES.get(task_type, ["gen_010"]))

    @classmethod
    def _infer_task_type(cls, task_desc: str) -> str:
        t = task_desc.lower()
        if "two" in t and ("put" in t or "place" in t):
            return "pick_two_obj_and_place"
        if "examine" in t or "look at" in t:
            return "look_at_obj_in_light"
        if "heat" in t:
            return "pick_heat_then_place"
        if "cool" in t:
            return "pick_cool_then_place"
        if "clean" in t:
            return "pick_clean_then_place"
        return "pick_and_place"

    def get_entry_nodes_for_task(self, task_desc: str) -> List[str]:
        task_type = self._infer_task_type(task_desc)
        return self.get_entry_nodes(task_type)

    @classmethod
    def from_skills_json(cls, skills_data: dict) -> "SkillGraph":
        return cls(skills_data)
