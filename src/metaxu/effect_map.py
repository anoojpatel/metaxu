"""Effect mapping system for Metaxu"""

class EffectMapping:
    def __init__(self):
        self.effect_maps = {}  # Maps effect names to C effect types
        
    def register_effect(self, metaxu_effect: str, c_effect: str):
        """Register a mapping from Metaxu effect to C effect"""
        self.effect_maps[metaxu_effect] = c_effect
        
    def get_c_effect(self, metaxu_effect: str) -> str:
        """Get the C effect type for a Metaxu effect"""
        return self.effect_maps.get(metaxu_effect)
        
    def has_mapping(self, metaxu_effect: str) -> bool:
        """Check if a Metaxu effect has a C mapping"""
        return metaxu_effect in self.effect_maps

class EffectParser:
    def __init__(self):
        self.effect_mapping = EffectMapping()
        
    def parse_effect_decl(self, effect_decl: str) -> tuple:
        """Parse an effect declaration with 'with' clause
        
        Example:
            fn spawn[T](f: fn() -> T) -> Thread[T] with EFFECT_SPAWN
            Returns: (effect_name, c_effect_type)
        """
        parts = effect_decl.split("with")
        if len(parts) != 2:
            return None, None
            
        effect_sig = parts[0].strip()
        c_effect = parts[1].strip()
        
        # Extract effect name from signature
        # This is simplified - would need proper parsing in real impl
        effect_name = effect_sig.split("(")[0].split()[-1]
        
        return effect_name, c_effect
        
    def register_effect_mapping(self, effect_decl: str):
        """Register effect mapping from declaration"""
        effect_name, c_effect = self.parse_effect_decl(effect_decl)
        if effect_name and c_effect:
            self.effect_mapping.register_effect(effect_name, c_effect)
            
    def get_c_effect(self, metaxu_effect: str) -> str:
        """Get C effect type for Metaxu effect"""
        return self.effect_mapping.get_c_effect(metaxu_effect)
