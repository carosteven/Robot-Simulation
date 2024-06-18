__docformat__ = "reStructeredText"

import random
import pygame
import pymunk
import pymunk.pygame_util

def limit_velocity(body, gravity, damping, dt):
        max_velocity = 100
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        l = body.velocity.length
        if l > max_velocity:
            body.score -= 0.25
            scale = max_velocity / l
            body.velocity = body.velocity * scale
        elif l < 2:
            body.score -= 0.25

class Wheeled_Robot_Sim(object):
    def __init__(self) -> None:
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)

        # Physics
        # Time step
        self._dt = 1.0 / 240.0
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
        self._agent = self._create_agent(vertices=((-25,-25), (-25,25), (25,25), (25,-25)), mass=10, position=(450, 500))
        for key in self._agent:
            self._agent[key].score = 0

        # Available actions
        self.available_act = ['w2_forward', 'w2_backward', 'w1_backward', 'w1_forward', 'nothing']

        self.left_sensor_data = None
        self.right_sensor_data = None
        self.distance_to_goal = [600, 600]

        # Collision Handling
        self.collision_occuring = False
        self.handler = self._space.add_collision_handler(0,1)
        self.handler.begin = self.collision_rewards
        self.handler.separate = self.collision_stop_rewards
        self.goal_handler = self._space.add_collision_handler(0,2)
        self.goal_handler.begin = self.collision_goal

        # Execution control and time unitl the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 10

    def _update(self):
        self._agent['robot'].score += (self._agent['wheel_1'].score + self._agent['wheel_2'].score)
        self._agent['wheel_1'].score = 0
        self._agent['wheel_2'].score = 0
        # print(pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS()), pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b10), end='\r')
        self.left_sensor_data = self._space.point_query_nearest(point=self._agent['robot'].local_to_world((-25, -25)), max_distance=30, shape_filter=pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b101))
        self.right_sensor_data = self._space.point_query_nearest(point=self._agent['robot'].local_to_world((25, -25)), max_distance=30, shape_filter=pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b101))
        self.distance_to_goal[0] = self._space.point_query_nearest(point=self._agent['robot'].local_to_world((0, 0)), max_distance=1200, shape_filter=pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS() ^ 0b11))[2]
        if self.collision_occuring:
            self._agent['robot'].score -= 0.25

        if self.distance_to_goal[0] - self.distance_to_goal[1] < 0:
            self._agent['robot'].score += 0.25
            # print('closer ', end='\r')
        else:
            self._agent['robot'].score -= 0.25
            # print('further', end='\r')
        self.distance_to_goal[1] = self.distance_to_goal[0]

        for sensor in [self.left_sensor_data, self.right_sensor_data]:
            if sensor is not None:
                # self._agent['robot'].score -= round((30-sensor[2])/60, 2)
                # print(round((30-sensor[2])/60, 2), end='\r')
                pass
        
        # add reward for velocity in same direction as robot
        if self.left_sensor_data is None and self.right_sensor_data is None:
            # self._agent['robot'].score += abs(round(self._agent['robot'].velocity.get_angle_degrees_between(self._agent['robot']. rotation_vector.perpendicular())/720, 2)) # 180 is straight ahead
            pass
                        
            '''
            if self._agent['robot'].velocity.length > 15:
                self._agent['robot'].score += 1
            else:
                self._agent['robot'].score += 0
            '''
        if self._agent['wheel_1'].latch:
            if self._agent['wheel_1'].forward:
                self._agent['wheel_1'].velocity = -abs(self._agent['wheel_1'].velocity) * self._agent['wheel_1'].rotation_vector.perpendicular()
            else:
                self._agent['wheel_1'].velocity = abs(self._agent['wheel_1'].velocity) * self._agent['wheel_1'].rotation_vector.perpendicular()

        if self._agent['wheel_2'].latch:
            if self._agent['wheel_2'].forward:
                self._agent['wheel_2'].velocity = -abs(self._agent['wheel_2'].velocity) * self._agent['wheel_2'].rotation_vector.perpendicular()
            else:
                self._agent['wheel_2'].velocity = abs(self._agent['wheel_2'].velocity) * self._agent['wheel_2'].rotation_vector.perpendicular()

        for key in self._agent:
            if key != 'robot':
                if self._agent[key].position.x > 600:
                    self._agent[key].position = (590, self._agent[key].position.y)
                
                elif self._agent[key].position.x < 0:
                    self._agent[key].position = (10, self._agent[key].position.y)

                if self._agent[key].position.y > 600:
                    self._agent[key].position = (self._agent[key].position.x, 590)
                
                elif self._agent[key].position.y < 0:
                    self._agent[key].position = (self._agent[key].position.x, 10)

                if (self._agent[key].position - self._agent['robot'].position).length > 34:
                    if key == 'wheel_1':
                        self._agent[key].position = self._agent['robot'].local_to_world((-30, 15))
                    else:
                        self._agent[key].position = self._agent['robot'].local_to_world((30, 15))


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
            # print(self._agent['robot'].score, end='\r')
            self._process_events()
            self._update()
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

        '''for constraint in self._space._constraints:
            constraint._set_distance = 0'''
        self._process_events()
        self._update()
        self._clear_screen()
        self._draw_objects()
        
        pygame.display.flip()
        # Delay fixed time between frames
        self._clock.tick(230)
        pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
    
    def _add_static_scenery(self) -> None:
        """
        Create the static bodies
        :return: None
        """
        static_body = self._space.static_body
        static_obstacles = [
            pymunk.Poly(static_body, ((275,150), (325,150), (325, 200), (275,200))),
            pymunk.Poly(static_body, ((300, 350), (200, 350), (200, 400), (300, 400)))
        ]
        for obstacle in static_obstacles:
            obstacle.elasticity = 0
            obstacle.friction = 1
            obstacle.collision_type = 1
            obstacle.filter = pymunk.ShapeFilter(categories=0b10)
            obstacle.reward = -0.25
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
            line.reward = -0.25
            line.filter = pymunk.ShapeFilter(categories=0b10)
        self._space.add(*static_border)

        static_goal = [
            pymunk.Poly(static_body, ((0,0), (50,0), (50,150), (0, 150))),
        ]
        


        static_goal[0].color = (0, 255, 0, 255)
        static_goal[0].reward = 1
        static_goal[0].collision_type = 2
        static_goal[0].filter = pymunk.ShapeFilter(categories=0b101)
        # static_goal[0].filter = pymunk.ShapeFilter(categories=0b1)

        self._space.add(*static_goal)
    
    def _create_agent(self, vertices: list[tuple[int, int]], mass: float, position: tuple[int] = (0,0), elasticity: float = 0, friction: float = 1) -> pymunk.Poly:
        """
        Create the agent
        :return: Pymunk Polygon
        """
        robot_body = pymunk.Body()
        robot_body.position = position
        robot_body.center = lambda : robot_body.local_to_world(robot.center_of_gravity)
        robot_body.direction = lambda : (robot_body.local_to_world(robot.center_of_gravity) - robot_body.local_to_world((robot.center_of_gravity.x, 0))) / (25*1.6)
        robot_body.damping = 0

        robot = pymunk.Poly(robot_body, vertices)
        robot.label = 'robot'
        robot.mass = mass
        robot.elasticity = elasticity
        robot.friction = friction
        robot.collision_type = 0
        robot.filter = pymunk.ShapeFilter(categories=0b1)
        self._space.add(robot_body, robot)

        wheel_1_body = pymunk.Body()
        wheel_1_body.position = pymunk.vec2d.Vec2d(position[0], position[1]) - pymunk.vec2d.Vec2d(30, -15)
        wheel_2_body = pymunk.Body()
        wheel_2_body.position = pymunk.vec2d.Vec2d(position[0], position[1]) + pymunk.vec2d.Vec2d(30, 15)

        wheel_1 = pymunk.Poly(wheel_1_body, ([-5, -10], [5, -10], [5, 10], [-5, 10]))
        wheel_2 = pymunk.Poly(wheel_2_body, ([-5, -10], [5, -10], [5, 10], [-5, 10]))
        for i, wheel in enumerate([wheel_1, wheel_2]):
            wheel.body.direction = lambda : (wheel.body.local_to_world(wheel.center_of_gravity) - wheel.body.local_to_world((wheel.center_of_gravity.x, 0))) / (25*1.6)
            wheel.body.damping = 0
            wheel.body.velocity_func = limit_velocity
            wheel.body.latch = False
            wheel.body.forward = True

            wheel.label = f'wheel_{i+1}'
            wheel.mass = 2
            wheel.elasticity = 0
            wheel.friction = 100000
            wheel.collision_type = 0
            wheel.color = (0, 0, 0, 255)
            wheel.filter = pymunk.ShapeFilter(categories=0b1)
            self._space.add(wheel.body, wheel)

        joint_1 = pymunk.constraints.PinJoint(robot.body, wheel_1.body, (-25,25), (5,10))
        joint_2 = pymunk.constraints.PinJoint(robot.body, wheel_1.body, (-25,5), (5,-10))
        joint_5 = pymunk.constraints.PinJoint(wheel_1.body, wheel_2.body, (-5,0), (5,0))
        joint_3 = pymunk.constraints.PinJoint(robot.body, wheel_2.body, (25,25), (-5,10))
        joint_4 = pymunk.constraints.PinJoint(robot.body, wheel_2.body, (25,5), (-5,-10))
        joint_6 = pymunk.constraints.PinJoint(wheel_1.body, wheel_2.body, (-5,0), (5,0))
        for joint in [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]:
            # joint.error_bias = pow(1.0 - .2, 60)
            # joint.max_bias = 100
            # joint.max_force = 100000
            # joint.collide_bodies = False
            pass
        self._space.add(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)
        robot_body.angle = round(random.randrange(314)/100, 2)
        return {'robot': robot_body, 'wheel_1': wheel_1_body, 'wheel_2': wheel_2_body}

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
                        self._actions('w2_forward')
                    elif keys[pygame.K_DOWN]:
                        self._actions('w2_backward')
                    elif keys[pygame.K_LEFT]:
                        self._actions('w1_backward')
                    elif keys[pygame.K_RIGHT]:
                        self._actions('w1_forward')


            '''elif event.type == pygame.KEYUP:
                self._actions('')'''
    
    def collision_rewards(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        # self._agent['robot'].score += shapes[1].reward
        self.collision_occuring = True
        self._agent['wheel_1'].latch = False
        self._agent['wheel_2'].latch = False
        return True
    
    def collision_stop_rewards(self, arbiter, space, dummy):
        self.collision_occuring = False
        return True
    
    def collision_goal(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        self._agent['robot'].score += shapes[1].reward
        self._running = False
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
        action: 'w2_forward', 'w2_backward', 'w1_backward', 'w1_forward', 'nothing'
        :return: None
        """
        # Negative Reward for taking action
        self._agent['robot'].score -= 0.25
        '''
        if action != 'nothing':
            self._agent['robot'].score -= 2
        else:
            self._agent['robot'].score -= 1
        '''
        '''
        if action == 'w2_forward':
            self._agent['wheel_2'].apply_force_at_world_point(self._agent['wheel_2'].rotation_vector.perpendicular() * -10000, self._agent['wheel_2'].position)
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = True
        elif action == 'w2_backward':
            self._agent['wheel_2'].apply_force_at_world_point(self._agent['wheel_2'].rotation_vector.perpendicular() * 10000, self._agent['wheel_2'].position)
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = False
        elif action == 'w1_backward':
            self._agent['wheel_1'].apply_force_at_world_point(self._agent['wheel_1'].rotation_vector.perpendicular() * 10000, self._agent['wheel_1'].position)
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = False
        elif action == 'w1_forward':
            self._agent['wheel_1'].apply_force_at_world_point(self._agent['wheel_1'].rotation_vector.perpendicular() * -10000, self._agent['wheel_1'].position)
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = True
        '''
        
        if action == 'w2_forward':
            self._agent['wheel_2'].velocity += self._agent['wheel_2'].rotation_vector.perpendicular() * -50
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = True
        elif action == 'w2_backward':
            self._agent['wheel_2'].velocity += self._agent['wheel_2'].rotation_vector.perpendicular() * 50
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = False
        elif action == 'w1_backward':
            self._agent['wheel_1'].velocity += self._agent['wheel_1'].rotation_vector.perpendicular() * 50
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = False
        elif action == 'w1_forward':
            self._agent['wheel_1'].velocity += self._agent['wheel_1'].rotation_vector.perpendicular() * -50
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = True

        # print(self._agent['wheel_1'].force, self._agent['wheel_2'].force, end='\r')
        
        
def main():
    game = Wheeled_Robot_Sim()
    game.run()

if __name__ == "__main__":
    main()
