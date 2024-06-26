__docformat__ = "reStructeredText"

import random
import pygame
import pymunk
import pymunk.pygame_util
from numpy import pi

class Maze_Sim(object):
    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)

        # Physics
        # Time step
        self._dt = 1.0 / 120.0
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
        self._agent = self._create_dynamic_bodies(vertices=((0,0), (0,50), (50,50), (50,0)), mass=10, position=[350, 500])
        self._agent.reward = 0

        # Collision Handling
        # self.handlers = [self._space.add_collision_handler(0,i) for i in range(len(self._space.shapes)-1)]
        # for handler in self.handlers:
            # handler.begin = self._space.shapes[1].col_hand
        self.handler = self._space.add_collision_handler(0,1)
        self.handler.begin = self.collision_rewards

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
            self._clear_screen()
            self._draw_objects()
            
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(110)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
    
    def run_controlled(self):
        # Progress time forward
        for x in range(self._physics_steps_per_frame):
            self._space.step(self._dt)
        
        self._process_events()
        self._clear_screen()
        self._draw_objects()
        
        pygame.display.flip()
        # Delay fixed time between frames
        self._clock.tick(110)
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
            obstacle.elasticity = 0
            obstacle.friction = 1
            obstacle.collision_type = 1
            obstacle.reward = -5
        self._space.add(*static_obstacles)

        static_border = [
            pymunk.Segment(static_body, (0,0), (0,600), 1),
            pymunk.Segment(static_body, (0,0), (600,0), 1),
            pymunk.Segment(static_body, (599,0), (599,599), 1),
            pymunk.Segment(static_body, (0,599), (599,599), 1),
        ]
        for line in static_border:
            line.elasticity = 0.95
            line.friction = 0.9
            line.collision_type = 1
            line.reward = -5
        self._space.add(*static_border)

        static_goal = [
            pymunk.Poly(static_body, ((0,0), (50,0), (50,150), (0, 150))),
            pymunk.Poly(static_body, ((275, 350), (375, 350), (375, 250), (275, 250)))
        ]
        
        for goal in static_goal:
            goal.collision_type = 1
            pass

        static_goal[0].color = (0, 255, 0, 255)
        static_goal[0].reward = 100
        static_goal[1].color = (255, 0, 0, 255)
        static_goal[1].reward = -50

        self._space.add(*static_goal)

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


            '''elif event.type == pygame.KEYUP:
                self._actions('')'''
    
    def _create_dynamic_bodies(self, vertices: list[tuple[int, int]], mass: float, position: list[int] = [0,0], elasticity: float = 0, friction: float = 1) -> pymunk.Poly:
        """
        Create the dynamic bodies
        :return: Pymunk Polygon
        """
        body = pymunk.Body()
        body.position = position
        body.center = lambda : body.position - body.center_of_gravity
        shape = pymunk.Poly(body, vertices)
        shape.mass = mass
        shape.elasticity = elasticity
        shape.friction = friction
        shape.collision_type = 0
        self._space.add(body, shape)
        return shape
    
    def collision_rewards(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        shapes[0].reward += shapes[1].reward
        # If collides with goal (add labels)
        if shapes[1].reward == 100:
            self._running = False
        # print(shapes[0].reward, end='\r')
        return True

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
        action: 'up', 'down', 'left', 'right'
        :return: None
        """
        # Negative Reward for taking action
        self._agent.reward -= 1
        # print(self._agent.reward, end='\r')

        pos = self._agent.body.position

        if action == "up":
            self._agent.body.velocity += (0,-50)
        elif action == "down":
            self._agent.body.velocity += (0,50)
        elif action == "left":
            self._agent.body.velocity += (-50,0)
        elif action == "right":
            self._agent.body.velocity += (50,0)
        '''else:
            self._agent.body.velocity = (0,0)'''
        if self._agent.body.position.x <= 0 or self._agent.body.position.x >= 600:
            self._agent.body.position = pos
            self._agent.reward -= 5
        
        if self._agent.body.position.y <= 0 or self._agent.body.position.y >= 600:
            self._agent.body.position = pos
            self._agent.reward -= 5
        
        
def main():
    game = Maze_Sim()
    game.run()

if __name__ == "__main__":
    main()