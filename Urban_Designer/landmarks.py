from ai_economist.foundation.entities.landmarks import Landmark, landmark_registry

@landmark_registry.add
class BaseZone(Landmark):
    ownable = True
    name = 'BaseZone'

@landmark_registry.add
class BaseBuilding(Landmark):
    ownable = True
    name = 'BaseBuilding'

    construction_cost = 0
    max_residental_population = 0
    max_commercial_population = 0