import math
import json
import numpy as np
from typing import Dict, List, Any

class DynamicSkillCurriculum:
    """
    Implements the Dynamic Curriculum from Skill0 (Lu et al., 2026).
    Evaluates skill helpfulness, ranks them, and selects a subset
    within a linearly decaying budget to drive skill internalization.
    """
    def __init__(self, 
                 total_stages: int = 3, 
                 initial_budget: int = 6,
                 validation_interval: int = 10):
        self.total_stages = total_stages
        self.current_stage = 1
        self.initial_budget = initial_budget
        self.validation_interval = validation_interval
        
        # Track helpfulness per skill file/category
        self.skill_helpfulness: Dict[str, float] = {}
        
    def get_current_budget(self) -> int:
        """Calculate M(s) = floor(N * (N_S - s) / (N_S - 1))"""
        if self.total_stages <= 1:
            return 0
        if self.current_stage >= self.total_stages:
            return 0
            
        ratio = (self.total_stages - self.current_stage) / (self.total_stages - 1)
        return math.floor(self.initial_budget * ratio)
        
    def update_stage(self, current_step: int, total_steps: int):
        """Update curriculum stage based on training progress"""
        steps_per_stage = max(1, total_steps // self.total_stages)
        new_stage = min(self.total_stages, (current_step // steps_per_stage) + 1)
        
        if new_stage > self.current_stage:
            print(f"[SkillCurriculum] Advancing to Stage {new_stage}/{self.total_stages}. "
                  f"Budget decaying from {self.get_current_budget()} to ...")
            self.current_stage = new_stage
            print(f"[SkillCurriculum] New budget: {self.get_current_budget()}")

    def update_helpfulness(self, category: str, acc_with_skill: float, acc_without_skill: float):
        """Update Delta_k = Acc_{w/ skill} - Acc_{w/o skill}"""
        delta = acc_with_skill - acc_without_skill
        self.skill_helpfulness[category] = delta
        
    def select_active_skills(self, skills_dict: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Filter, Rank, and Select top-M skills based on helpfulness"""
        budget = self.get_current_budget()
        
        if budget <= 0:
            return {} # Zero skills at inference / late stages!
            
        # 1. Filter out unhelpful skills (Delta <= 0)
        helpful_categories = [
            cat for cat, delta in self.skill_helpfulness.items() 
            if delta > 0 and cat in skills_dict
        ]
        
        # If we haven't evaluated helpfulness yet, default to all available up to budget
        if not self.skill_helpfulness:
            helpful_categories = list(skills_dict.keys())
            
        # 2. Rank by Delta (descending)
        ranked_categories = sorted(
            helpful_categories, 
            key=lambda c: self.skill_helpfulness.get(c, 0.0), 
            reverse=True
        )
        
        # 3. Select top-M categories
        selected_categories = ranked_categories[:budget]
        
        # Construct the active skill dictionary
        active_skills = {
            cat: skills_dict[cat] 
            for cat in selected_categories
        }
        
        return active_skills
