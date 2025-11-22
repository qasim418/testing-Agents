import numpy as np
import pybullet as p


class ObjectFactory:
    def __init__(self, client_id: int):
        self.cid = client_id

    def building(self, pos, size, color):
        half = [s / 2.0 for s in size]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.cid
        )
        base_pos = [pos[0], pos[1], size[2] / 2.0]
        return p.createMultiBody(0, col, vis, base_pos, physicsClientId=self.cid)

    def tree(self, pos, h=2.5, r=2.0):
        trunk_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=0.3, height=h, physicsClientId=self.cid
        )
        trunk_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.3, length=h, rgbaColor=[0.4, 0.25, 0.1, 1], physicsClientId=self.cid
        )
        trunk = p.createMultiBody(
            0, trunk_col, trunk_vis, [pos[0], pos[1], h / 2.0], physicsClientId=self.cid
        )
        canopy_col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=r, physicsClientId=self.cid
        )
        canopy_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=r, rgbaColor=[0.1, 0.5, 0.1, 1], physicsClientId=self.cid
        )
        canopy = p.createMultiBody(
            0, canopy_col, canopy_vis, [pos[0], pos[1], h + r * 0.7], physicsClientId=self.cid
        )
        return trunk, canopy

    def road(self, start, end, width=4.0):
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        length = float(np.linalg.norm(end[:2] - start[:2]))
        angle = float(np.arctan2(end[1] - start[1], end[0] - start[0]))
        center = [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0, 0.02]
        half = [length / 2.0, width / 2.0, 0.01]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=[0.2, 0.2, 0.2, 1], physicsClientId=self.cid
        )
        orn = p.getQuaternionFromEuler([0.0, 0.0, angle])
        return p.createMultiBody(0, col, vis, center, orn, physicsClientId=self.cid)

    def debris(self, pos, rng: np.random.RandomState):
        size = rng.uniform(0.4, 1.2, 3).astype(float)
        half = (size / 2.0).tolist()
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.cid)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=[0.4, 0.35, 0.3, 1], physicsClientId=self.cid
        )
        return p.createMultiBody(10.0, col, vis, pos, physicsClientId=self.cid)


class CityGenerator:
    def __init__(self, client_id: int, seed=None):
        self.rng = np.random.RandomState(seed)
        self.f = ObjectFactory(client_id)

    def generate(self, size=50, num_obstacles=None):
        bodies = []
        bodies += self._roads(size)
        bodies += self._buildings(size, num_obstacles)
        bodies += self._trees(size, num_obstacles)
        bodies += self._debris(size, num_obstacles)
        return bodies

    def _roads(self, s):
        bodies = []
        bodies.append(self.f.road([-s, 0, 0], [s, 0, 0], 4.0))
        bodies.append(self.f.road([0, -s, 0], [0, s, 0], 4.0))
        for i in range(-s // 15, s // 15 + 1):
            if i == 0:
                continue
            o = i * 15
            bodies.append(self.f.road([-s, o, 0], [s, o, 0], 4.0))
            bodies.append(self.f.road([o, -s, 0], [o, s, 0], 4.0))
        return bodies

    def _buildings(self, s, density=None):
        bodies = []
        colors = [[0.7, 0.4, 0.3, 1], [0.6, 0.6, 0.6, 1], [0.5, 0.6, 0.7, 1]]
        
        # Default full density if None
        if density is None:
            density = 1.0
            
        # Calculate skip factor based on density (lower density = higher skip)
        # density 1.0 -> skip 1 (every cell)
        # density 0.1 -> skip 10 (every 10th cell)
        skip = max(1, int(1.0 / max(0.01, density)))
        
        for bx in range(-s // 15, s // 15, skip):
            for by in range(-s // 15, s // 15, skip):
                if abs(bx) <= 0 and abs(by) <= 0:
                    continue
                cx, cy = bx * 15 + 7.5, by * 15 + 7.5
                for _ in range(self.rng.randint(1, 3)):
                    x = cx + float(self.rng.uniform(-3, 3))
                    y = cy + float(self.rng.uniform(-3, 3))
                    w = float(self.rng.uniform(4, 8))
                    d = float(self.rng.uniform(4, 8))
                    h = float(self.rng.uniform(6, 20))
                    color = colors[self.rng.randint(0, len(colors))]
                    bodies.append(self.f.building([x, y, 0.0], [w, d, h], color))
        return bodies

    def _trees(self, s, density=None):
        bodies = []
        if density is None:
            density = 1.0
        
        # Scale number of trees by density
        step = max(4, int(4 / max(0.01, density)))
        
        for i in range(-s // 3, s // 3, step):
            if abs(i) < 3:
                continue
            bodies += list(self.f.tree([i, 3, 0], float(self.rng.uniform(2, 3)), float(self.rng.uniform(1.5, 2.5))))
            bodies += list(self.f.tree([i, -3, 0], float(self.rng.uniform(2, 3)), float(self.rng.uniform(1.5, 2.5))))
            bodies += list(self.f.tree([3, i, 0], float(self.rng.uniform(2, 3)), float(self.rng.uniform(1.5, 2.5))))
            bodies += list(self.f.tree([-3, i, 0], float(self.rng.uniform(2, 3)), float(self.rng.uniform(1.5, 2.5))))
            
        num_random_trees = int(15 * density)
        for _ in range(num_random_trees):
            x, y = self.rng.uniform(-s / 2.0, s / 2.0, 2)
            if abs(x % 15) > 3 and abs(y % 15) > 3:
                bodies += list(self.f.tree([float(x), float(y), 0.0], float(self.rng.uniform(2, 4)), float(self.rng.uniform(1.5, 3))))
        return bodies

    def _debris(self, s, density=None):
        bodies = []
        if density is None:
            density = 1.0
            
        num_debris = int(25 * density)
        for _ in range(num_debris):
            x, y = self.rng.uniform(-s / 2.0, s / 2.0, 2)
            if abs(x % 15) > 3 and abs(y % 15) > 3:
                for _ in range(self.rng.randint(2, 5)):
                    dx, dy = self.rng.uniform(-1, 1, 2)
                    bodies.append(self.f.debris([float(x + dx), float(y + dy), float(self.rng.uniform(0.3, 1.5))], self.rng))
        return bodies
