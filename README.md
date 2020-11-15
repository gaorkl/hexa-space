# Welcome to Hexaworld

Toy environment with hexagonal layout (6 directions).
Goal: simplify environment for learning about:
- space and displacement
- object permanence
- working memory

Single agent:
- can move forward by one step and rotate by +- 1 angle of direction.
- observes in a 120 field of view in front of it. Range can be set.
- observations are a list of floats

Objects:
- Obstacles: non-traversable. 
- Movables: can be pushed. 
- Moving objects: move forward, and rotate in one direction (set randomly at initialization) 
when encountering a non-movable obstacle.

Each object has an appearance which is drawn from separate random gaussians (with different
means for each object type).
So, an agent can't know for certain that an object is of a particular type.
And, all objects have a different appearance.

Todo:
- documentation (of course)
- nice visualization with skimage
- different game modes with terminations and rewards
- edibles / poisons
- possibility to have all the object of a same type have the same appearance.
- goal: waypoints in order A then B then C then D
