from typing import Dict, List, Any
from agent_system.memory.skill_curriculum import DynamicSkillCurriculum

class CurriculumSkillsOnlyMemory:
    """
    Wrapper around SkillsOnlyMemory that applies the Skill0 Dynamic Curriculum.
    """
    def __init__(self, base_memory, curriculum_config=None):
        self.base_memory = base_memory
        
        cfg = curriculum_config or {}
        self.curriculum = DynamicSkillCurriculum(
            total_stages=cfg.get('total_stages', 3),
            initial_budget=cfg.get('initial_budget', 6),
            validation_interval=cfg.get('validation_interval', 10)
        )
        
    def retrieve(self, task_description: str, **kwargs) -> Dict[str, Any]:
        # Get all skills from base memory
        base_result = self.base_memory.retrieve(task_description, **kwargs)
        
        if self.curriculum.get_current_budget() <= 0:
            # Skill0 constraint: zero skills at inference / late stages
            base_result['general_skills'] = []
            base_result['task_specific_skills'] = []
            base_result['mistakes_to_avoid'] = []
            base_result['retrieval_mode'] = 'zero_shot_curriculum'
            return base_result
            
        # Optional: apply curriculum filtering to task_specific_skills
        # if they are categorized. For now, we rely on the base memory 
        # and just truncate to the current budget if needed.
        
        task_skills = base_result.get('task_specific_skills', [])
        budget = self.curriculum.get_current_budget()
        
        if len(task_skills) > budget:
            base_result['task_specific_skills'] = task_skills[:budget]
            
        return base_result
        
    # Delegate other methods to base_memory
    def __getattr__(self, name):
        return getattr(self.base_memory, name)
