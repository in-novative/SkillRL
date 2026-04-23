import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SubgoalDef:
    id: str
    desc: str
    trigger_patterns: List[str]  # regex patterns matched against obs+action text
    reward: float = 0.1


class SubgoalTracker:
    """Per-episode state machine tracking subgoal completion across a batch of environments."""

    def __init__(self, batch_size: int, subgoal_defs_per_env: List[List[SubgoalDef]]):
        self.batch_size = batch_size
        self.defs = subgoal_defs_per_env
        # completed[env_idx][subgoal_idx] = bool
        self.completed: List[List[bool]] = [
            [False] * len(defs) for defs in subgoal_defs_per_env
        ]

    def check(
        self,
        obs_texts: List[str],
        action_texts: List[str],
        active_masks: np.ndarray,
    ) -> np.ndarray:
        """Check subgoal completion for each env and return per-env rewards."""
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        for env_idx in range(self.batch_size):
            if not active_masks[env_idx]:
                continue
            obs = obs_texts[env_idx] if obs_texts and env_idx < len(obs_texts) else ""
            action = action_texts[env_idx] if action_texts and env_idx < len(action_texts) else ""
            combined = (obs + " " + action).lower()
            for sg_idx, sg in enumerate(self.defs[env_idx]):
                if self.completed[env_idx][sg_idx]:
                    continue
                for pattern in sg.trigger_patterns:
                    if re.search(pattern, combined, re.IGNORECASE):
                        self.completed[env_idx][sg_idx] = True
                        rewards[env_idx] += sg.reward
                        break
        return rewards


class SubgoalExtractor:
    """Extracts SubgoalDef lists from task descriptions per environment type."""

    # ALFWorld subgoal templates keyed by task type keyword
    _ALFWORLD_TEMPLATES: Dict[str, List[Dict]] = {
        "pick_and_place": [
            {"id": "find_obj",   "desc": "navigate to object",      "patterns": [r"go to .+"]},
            {"id": "pick_obj",   "desc": "pick up object",           "patterns": [r"take .+", r"pick up .+"]},
            {"id": "find_recep", "desc": "navigate to receptacle",   "patterns": [r"go to .+"]},
            {"id": "place_obj",  "desc": "place object",             "patterns": [r"put .+", r"place .+"]},
        ],
        "pick_heat_then_place": [
            {"id": "find_obj",        "desc": "navigate to object",      "patterns": [r"go to .+"]},
            {"id": "pick_obj",        "desc": "pick up object",           "patterns": [r"take .+", r"pick up .+"]},
            {"id": "find_microwave",  "desc": "navigate to microwave",    "patterns": [r"go to microwave"]},
            {"id": "heat_obj",        "desc": "heat object",              "patterns": [r"heat .+", r"cool .+in microwave"]},
            {"id": "find_recep",      "desc": "navigate to receptacle",   "patterns": [r"go to .+"]},
            {"id": "place_obj",       "desc": "place object",             "patterns": [r"put .+", r"place .+"]},
        ],
        "pick_cool_then_place": [
            {"id": "find_obj",    "desc": "navigate to object",    "patterns": [r"go to .+"]},
            {"id": "pick_obj",    "desc": "pick up object",         "patterns": [r"take .+", r"pick up .+"]},
            {"id": "find_fridge", "desc": "navigate to fridge",     "patterns": [r"go to fridge"]},
            {"id": "cool_obj",    "desc": "cool object",            "patterns": [r"cool .+"]},
            {"id": "find_recep",  "desc": "navigate to receptacle", "patterns": [r"go to .+"]},
            {"id": "place_obj",   "desc": "place object",           "patterns": [r"put .+", r"place .+"]},
        ],
        "pick_clean_then_place": [
            {"id": "find_obj",   "desc": "navigate to object",    "patterns": [r"go to .+"]},
            {"id": "pick_obj",   "desc": "pick up object",         "patterns": [r"take .+", r"pick up .+"]},
            {"id": "find_sink",  "desc": "navigate to sink",       "patterns": [r"go to sink|go to sinkbasin"]},
            {"id": "clean_obj",  "desc": "clean object",           "patterns": [r"clean .+"]},
            {"id": "find_recep", "desc": "navigate to receptacle", "patterns": [r"go to .+"]},
            {"id": "place_obj",  "desc": "place object",           "patterns": [r"put .+", r"place .+"]},
        ],
        "look_at_obj_in_light": [
            {"id": "find_obj",  "desc": "navigate to object",  "patterns": [r"go to .+"]},
            {"id": "pick_obj",  "desc": "pick up object",       "patterns": [r"take .+", r"pick up .+"]},
            {"id": "find_lamp", "desc": "navigate to lamp",     "patterns": [r"go to .*(lamp|desklamp|floorlamp)"]},
            {"id": "examine",   "desc": "examine under light",  "patterns": [r"examine .+", r"use .+lamp.+with .+"]},
        ],
        "pick_two_obj_and_place": [
            {"id": "find_obj1",   "desc": "navigate to first object",      "patterns": [r"go to .+"]},
            {"id": "pick_obj1",   "desc": "pick up first object",           "patterns": [r"take .+", r"pick up .+"]},
            {"id": "find_recep1", "desc": "navigate to receptacle (first)", "patterns": [r"go to .+"]},
            {"id": "place_obj1",  "desc": "place first object",             "patterns": [r"put .+", r"place .+"]},
            {"id": "find_obj2",   "desc": "navigate to second object",      "patterns": [r"go to .+"]},
            {"id": "pick_obj2",   "desc": "pick up second object",           "patterns": [r"take .+", r"pick up .+"]},
            {"id": "place_obj2",  "desc": "place second object",             "patterns": [r"put .+", r"place .+"]},
        ],
    }

    # WebShop fixed subgoal stages
    _WEBSHOP_TEMPLATES: List[Dict] = [
        {"id": "search",        "desc": "search for product",    "patterns": [r"search\["]},
        {"id": "view_product",  "desc": "view product details",  "patterns": [r"click\[.*\]"]},
        {"id": "select_option", "desc": "select size/color",     "patterns": [r"click\[.*(size|color|option).*\]"]},
        {"id": "add_to_cart",   "desc": "add to cart / buy",     "patterns": [r"click\[buy now\]", r"click\[add to cart\]"]},
    ]

    @classmethod
    def _alfworld_task_type(cls, task_desc: str) -> str:
        task_lower = task_desc.lower()
        if "two" in task_lower and ("put" in task_lower or "place" in task_lower):
            return "pick_two_obj_and_place"
        if "examine" in task_lower or "look at" in task_lower:
            return "look_at_obj_in_light"
        if "heat" in task_lower:
            return "pick_heat_then_place"
        if "cool" in task_lower:
            return "pick_cool_then_place"
        if "clean" in task_lower:
            return "pick_clean_then_place"
        return "pick_and_place"

    @classmethod
    def extract_alfworld(cls, tasks: List[str], reward_per_subgoal: float) -> List[List[SubgoalDef]]:
        result = []
        for task in tasks:
            task_type = cls._alfworld_task_type(task)
            templates = cls._ALFWORLD_TEMPLATES.get(task_type, cls._ALFWORLD_TEMPLATES["pick_and_place"])
            defs = [
                SubgoalDef(
                    id=f"{task_type}_{t['id']}",
                    desc=t["desc"],
                    trigger_patterns=t["patterns"],
                    reward=reward_per_subgoal,
                )
                for t in templates
            ]
            result.append(defs)
        return result

    @classmethod
    def extract_webshop(cls, batch_size: int, reward_per_subgoal: float) -> List[List[SubgoalDef]]:
        defs = [
            SubgoalDef(
                id=t["id"],
                desc=t["desc"],
                trigger_patterns=t["patterns"],
                reward=reward_per_subgoal,
            )
            for t in cls._WEBSHOP_TEMPLATES
        ]
        return [list(defs) for _ in range(batch_size)]


def build_subgoal_tracker(envs: Any, config: Any) -> Optional[SubgoalTracker]:
    """Factory: build a SubgoalTracker from env + config, or return None if disabled."""
    sg_cfg = config.env.get("subgoal_reward", {})
    if not sg_cfg.get("enable", False):
        return None

    reward_per_subgoal = float(sg_cfg.get("reward_per_subgoal", 0.1))
    env_name = config.env.env_name.lower()
    tasks = getattr(envs, "tasks", None)

    if "alfworld" in env_name:
        if not tasks:
            return None
        defs = SubgoalExtractor.extract_alfworld(tasks, reward_per_subgoal)
    elif "webshop" in env_name:
        batch_size = len(tasks) if tasks else config.data.train_batch_size
        defs = SubgoalExtractor.extract_webshop(batch_size, reward_per_subgoal)
    else:
        return None

    return SubgoalTracker(batch_size=len(defs), subgoal_defs_per_env=defs)
