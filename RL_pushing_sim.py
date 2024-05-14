__docformat__ = "reStructeredText"

import random
import pygame
import pymunk
import pymunk.pygame_util
from numpy import pi

class BouncyBalls(object):
    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((600, 600))
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # The agent to be controlled
        self._agent = self._create_dynamic_bodies(vertices=((0,0), (60,0), (30,60)), mass=100, position=[500, 500], elasticity=0.5, friction=(1,1))

        # The box to be pushed
        self._box = self._create_dynamic_bodies(vertices=((0,0), (0,50), (50,50), (50,0)), mass=10, position=[350, 500], elasticity=0.5, friction=(.2,1))

        # Execution control and time unitl the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10

    def run(self) -> None:
        """
        The mail loop of the game
        :return: None
        """
        # Main Loop
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)
            
            self._process_events()
            self._apply_friction()
            self._clear_screen()
            self._draw_objects()
            
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
    
    def _add_static_scenery(self) -> None:
        """
        Create the static bodies
        :return: None
        """
        static_body = self._space.static_body
        static_obstacles = [
            pymunk.Poly(static_body, ((0,150), (325,150), (325, 175), (0,175))),
            pymunk.Poly(static_body, ((600, 350), (275, 350), (275, 375), (600, 375)))
        ]
        for obstacle in static_obstacles:
            obstacle.elasticity = 0.95
            obstacle.friction = 0.9
        self._space.add(*static_obstacles)

        static_lines = [
            pymunk.Segment(static_body, (0,0), (0,600), 1),
            pymunk.Segment(static_body, (0,0), (600,0), 1),
            pymunk.Segment(static_body, (599,0), (599,599), 1),
            pymunk.Segment(static_body, (0,599), (599,599), 1),
        ]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.9
        self._space.add(*static_lines)

    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False

            elif event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                    if keys[pygame.K_RIGHT]:
                        self._actions('rot_cw')
                    elif keys[pygame.K_LEFT]:
                        self._actions('rot_ccw')
                else:
                    if keys[pygame.K_UP]:
                        self._actions('up')
                    elif keys[pygame.K_DOWN]:
                        self._actions('down')
                    elif keys[pygame.K_LEFT]:
                        self._actions('left')
                    elif keys[pygame.K_RIGHT]:
                        self._actions('right')


            elif event.type == pygame.KEYUP:
                self._actions('')
        

    def _apply_friction(self) -> None:
        for object in [self._agent, self._box]:
            object[1].angle = object[0].body.angle
    
    def _create_dynamic_bodies(self, vertices: list[tuple[int, int]], mass: float, position: list[int] = [0,0], elasticity: float = 1, friction: tuple[float,float] = (1,1)) -> pymunk.Poly:
        """
        Create the dynamic bodies
        :return: Pymunk Polygon
        """
        control_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        control_body.position = position
        self._space.add(control_body)
        
        body = pymunk.Body()
        body.position = position
        shape = pymunk.Poly(body, vertices)
        shape.mass = mass
        shape.elasticity = elasticity
        shape.friction = friction[0]
        self._space.add(body, shape)
        
        pivot = pymunk.PivotJoint(control_body, body, shape.center_of_gravity, shape.center_of_gravity)
        self._space.add(pivot)
        pivot.max_bias = 0 # disable joint correction
        pivot.max_force = 10000*friction[0] # emulate linear friction

        gear = pymunk.GearJoint(control_body, body, 0.0, 1.0)
        self._space.add(gear)
        gear.error_bias = 0 # attempt to fully correct the joint each step
        gear.max_bias = 1.2 # but limit it's angular correction rate
        gear.max_force = 50000*friction[1] # emulate angular friction

        return shape, control_body
    
    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)
    
    def _actions(self, action) -> None:
        """
        action: 'up', 'down', 'left', 'right', 'rot_cw', 'rot_ccw'
        :return: None
        """
        if action == "up":
            self._agent[0].body.velocity += (0,-50)
        elif action == "down":
            self._agent[0].body.velocity += (0,50)
        elif action == "left":
            self._agent[0].body.velocity += (-50,0)
        elif action == "right":
            self._agent[0].body.velocity += (50,0)
        elif action == "rot_cw":
            self._agent[0].body.angle += 0.2
        elif action == "rot_ccw":
            self._agent[0].body.angle += -0.2
        '''else:
            self._agent.body.velocity = (0,0)'''
        
def main():
    game = BouncyBalls()
    game.run()

if __name__ == "__main__":
    main()