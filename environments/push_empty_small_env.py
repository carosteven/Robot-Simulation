__docformat__ = "reStructeredText"

import random
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from PIL import Image
from skimage.transform import resize, rescale

def limit_velocity(body, gravity, damping, dt):
        max_velocity = 100
        pymunk.Body.update_velocity(body, gravity, body.damping, dt)
        l = body.velocity.length
        if l > max_velocity:
            scale = max_velocity / l
            body.velocity = body.velocity * scale

def custom_damping(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, body.damping, dt)

def distance(pos1, pos2):
    return abs(pos1 - pos2)

class Push_Empty_Small_Env(object):
    def __init__(self, config=None) -> None:
        self.state_type = 'vision'
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)

        # Physics
        # Time step
        self._dt = 1.0 / 240.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.display.init()
        pygame.font.init()
        self.screen_size = (304,304)
        self._screen = pygame.display.set_mode(self.screen_size)
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)
        self.pxarr = pygame.PixelArray(self._draw_options.surface)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()
        
        # The object to be pushed
        self._object = self._create_object(radius=15, mass=5, position=tuple([c/2 for c in self.screen_size]), damping=.99)

        # The agent to be controlled
        y_pos = random.randint(50,self.screen_size[1]-50)
        self._agent = self._create_agent(vertices=((-25,-25), (-25,25), (25,25), (25,-25)), mass=10, position=(self.screen_size[0]*0.75, y_pos), damping=0.99)

        self.state = np.zeros((1, self.screen_size[0], self.screen_size[1])).astype(int)
        self.get_state()

        self.take_action = None

        # Agent cumulative rewards
        self.reward = 0
        self.reward_from_last_action = 0

        # Rewards
        self.collision_penalty      = config['collision_penalty'] if config is not None else 10
        self.action_penalty         = config['action_penalty'] if config is not None else 1
        self.push_reward            = config['push_reward'] if config is not None else 1
        self.obj_to_goal_reward     = config['obj_to_goal_reward'] if config is not None else 1000
        self.exploration_reward     = config['exploration_reward'] if config is not None else 0.1
        self.partial_rewards_scale  = config['partial_rewards_scale'] if config is not None else 1

        # Available actions
        self.available_actions = ['forward', 'backward', 'turn_cw', 'turn_ccw']
        self.action_completed = False

        self.goal_position = (25,75)
        self.initial_object_dist = distance(self._object.position, self.goal_position)

        # Collision Handling
        # Robot: 0, Obstacles: 1, Goal: 2, Object: 3
        self.collision_occuring = False
        self.obj_coll_obst = False
        self.is_pushing = False
        self.handler = self._space.add_collision_handler(0,1)
        self.handler.begin = self.collision_begin
        self.handler.separate = self.collision_end
        self.obj_goal_handler = self._space.add_collision_handler(3,2)
        self.obj_goal_handler.begin = self.collision_obj_goal
        self.obj_obst_handler = self._space.add_collision_handler(3,1)
        self.obj_obst_handler.begin = self.collision_obj_obst_begin
        self.obj_obst_handler.separate = self.collision_obj_obst_end
        self.robo_goal_handler = self._space.add_collision_handler(0,2)
        self.robo_goal_handler.begin = self.collision_begin
        self.robo_goal_handler.separate = self.collision_end
        self.robo_obj_handler = self._space.add_collision_handler(0,3)
        self.robo_obj_handler.begin = self.collision_robo_obj_begin
        self.robo_obj_handler.separate = self.collision_robo_obj_end

        # Execution control
        self._running = True
        self._done = False

    def _add_static_scenery(self) -> None:
        """
        Create the static bodies
        :return: None
        """
        static_body = self._space.static_body

        static_border = [
            pymunk.Segment(static_body, (0,0), (0,self.screen_size[1]), 1),
            pymunk.Segment(static_body, (0,0), (self.screen_size[0],0), 1),
            pymunk.Segment(static_body, (self.screen_size[0]-1,0), (self.screen_size[0]-1,self.screen_size[1]-1), 1),
            pymunk.Segment(static_body, (0,self.screen_size[1]-1), (self.screen_size[0]-1,self.screen_size[1]-1), 1),

            pymunk.Segment(static_body, (self.screen_size[0]*0.9,0), (self.screen_size[0],self.screen_size[1]*0.1), 1),
            pymunk.Segment(static_body, (self.screen_size[0]*0.9,self.screen_size[1]), (self.screen_size[0],self.screen_size[1]*0.9), 1),
            pymunk.Segment(static_body, (0,self.screen_size[1]*0.9), (self.screen_size[0]*0.1,self.screen_size[1]), 1),
        ]
        for line in static_border:
            line.elasticity = 0.95
            line.friction = 0.9
            line.collision_type = 1
            line.filter = pymunk.ShapeFilter(categories=0b10)
        self._space.add(*static_border)

        static_goal = [
            pymunk.Poly(static_body, ((0,0), (50,0), (50,150), (0, 150))),
        ]
        
        static_goal[0].color = (0, 255, 0, 255)
        static_goal[0].collision_type = 2
        static_goal[0].filter = pymunk.ShapeFilter(categories=0b101)

        self._space.add(*static_goal)

    def _create_object(self, radius: float, mass: float, position: tuple[int] = (0,0), elasticity: float = 0, friction: float = 1.0, damping: float = 0.0) -> pymunk.Poly:
        """
        Create the object to be pushed
        :return: Pymunk Polygon
        """
        object_body = pymunk.Body()
        object_body.position = position
        object_body.damping = damping
        object_body.velocity_func = custom_damping
        
        object = pymunk.Poly(object_body, ((-radius, -radius), (-radius, radius), (radius, radius), (radius, -radius)))
        object.label = 'object'
        object.mass = mass
        object.elasticity = elasticity
        object.friction = friction
        object.collision_type = 3
        # object.filter = pymunk.ShapeFilter(categories=0b1)
        self._space.add(object_body, object)

        return object_body
    
    def _create_agent(self, vertices: list[tuple[int, int]], mass: float, position: tuple[int] = (0,0), elasticity: float = 0, friction: float = 1.0, damping: float = 1.0) -> pymunk.Poly:
        """
        Create the agent
        :return: Pymunk Polygon
        """
        robot_body = pymunk.Body()
        robot_body.position = position
        robot_body.damping = damping

        robot = pymunk.Poly(robot_body, vertices)
        robot.label = 'robot'
        robot.mass = mass
        robot.elasticity = elasticity
        robot.friction = friction
        robot.collision_type = 0
        robot.filter = pymunk.ShapeFilter(categories=0b1)
        self._space.add(robot_body, robot)
        robot_body.angle = round(random.randrange(314)/100, 2)

        wheel_1_body = pymunk.Body()
        wheel_1_body.position = robot_body.local_to_world((-30, 15))
        wheel_1_body.angle = robot_body.angle
        wheel_2_body = pymunk.Body()
        wheel_2_body.position = robot_body.local_to_world((30, 15))
        wheel_2_body.angle = robot_body.angle

        wheel_1 = pymunk.Poly(wheel_1_body, ([-5, -10], [5, -10], [5, 10], [-5, 10]))
        wheel_2 = pymunk.Poly(wheel_2_body, ([-5, -10], [5, -10], [5, 10], [-5, 10]))
        for i, wheel in enumerate([wheel_1, wheel_2]):
            wheel.body.damping = damping
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

        self._space.add(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)
        return {'robot': robot_body, 'wheel_1': wheel_1_body, 'wheel_2': wheel_2_body}

    def run(self) -> None:
        """
        The main loop of the game
        :return: None
        """
        # Main Loop
        reached_loc = False
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            action = self._process_events()
            self._actions(action)
            coord = (200,75)
            if not reached_loc:
                reached_loc = self.straight_line_navigation(coord)
            self._update()
            self._clear_screen()
            # make a dot where the coord is
            pygame.draw.circle(self._screen, (0,0,0), coord, 5)
            self._draw_objects()
            self.get_state()
            
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(110)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

            if action is not None:
                self.reward += self.reward_from_last_action
                self.reward_from_last_action = 0

            # Calculate reward
            self.reward_from_last_action += self.get_reward(True if action is not None else False)
            # self.reward = self.get_reward(True if action is not None else False)

            # Calculate reward
            # robot_reward = self.get_reward()
            # self.reward += robot_reward
            # print(f'{self.reward} \t\t\t\t {self.reward_from_last_action}', end='\r')
            
            if self._done:
                self.reset()
    
    def step(self, action):
        """
        Progress the simulation one time step
        inputs: action
        outputs: state, reward, done, info
        """
        # Progress time forward
        for x in range(self._physics_steps_per_frame):
            self._space.step(self._dt)

        self._process_events()
        # self._actions(action)
        self.action_completed = self.take_action(action)
        self._update()
        self._clear_screen()
        self._draw_objects()
        self.get_state()
        
        pygame.display.flip()
        
        # Delay fixed time between frames
        self._clock.tick(230)
        pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

        if action is not None:
            self.reward += self.reward_from_last_action
            self.reward_from_last_action = 0

        # Calculate reward
        self.reward_from_last_action += self.get_reward(True if action is not None else False)

        # Items to return
        state = self.state
        reward = self.reward_from_last_action
        done = self._done
        info = {
            'inactivity': None,
            'cumulative_cubes': 0,
            'cumulative_distance': 0,
            'cumulative_reward': self.reward
        }

        return state, reward, done, info
    
    def _process_events(self) -> str:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        action = None
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
                        action = 'forward'
                    elif keys[pygame.K_DOWN]:
                        action = 'backward'
                    elif keys[pygame.K_LEFT]:
                        action = 'turn_ccw'
                    elif keys[pygame.K_RIGHT]:
                        action = 'turn_cw'
        return action
    
    def _update(self) -> None:
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

    def _actions(self, action) -> bool:
        """
        action: 'forward', 'backward', 'turn_cw', 'turn_ccw'
        :return: action_completed
        """
        if action == 'forward':
            self._agent['wheel_1'].velocity += self._agent['wheel_1'].rotation_vector.perpendicular() * -50
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = True
            self._agent['wheel_2'].velocity += self._agent['wheel_2'].rotation_vector.perpendicular() * -50
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = True

        elif action == 'backward':
            self._agent['wheel_1'].velocity += self._agent['wheel_1'].rotation_vector.perpendicular() * 50
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = False
            self._agent['wheel_2'].velocity += self._agent['wheel_2'].rotation_vector.perpendicular() * 50
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = False

        elif action == 'turn_cw':
            self._agent['wheel_1'].velocity += self._agent['wheel_1'].rotation_vector.perpendicular() * -50
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = True
            self._agent['wheel_2'].velocity += self._agent['wheel_2'].rotation_vector.perpendicular() * 50
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = False

        elif action == 'turn_ccw':
            self._agent['wheel_1'].velocity += self._agent['wheel_1'].rotation_vector.perpendicular() * 50
            self._agent['wheel_1'].latch = True
            self._agent['wheel_1'].forward = False
            self._agent['wheel_2'].velocity += self._agent['wheel_2'].rotation_vector.perpendicular() * -50
            self._agent['wheel_2'].latch = True
            self._agent['wheel_2'].forward = True
        
        return True

    def straight_line_navigation(self, coords) -> bool:
        if self.collision_occuring or (self.obj_coll_obst and self.is_pushing) or self._done:
            return True
        # Get the heading of the robot
        angle = (self._agent['robot'].angle - (np.pi/2)) % (2*np.pi)

        # Get the angle between the front of the robot and the coordinates
        front_x = self._agent['robot'].local_to_world((0, -25)).x
        front_y = self._agent['robot'].local_to_world((0, -25)).y
        angle_to_coords = np.arctan2(coords[1] - front_y, coords[0] - front_x) % (2*np.pi)

        # Get the distance between the front of the robot and the coordinates
        dist = distance(pymunk.vec2d.Vec2d(front_x, front_y), coords)
        
        # If the robot is close enough to the coordinates, stop
        if dist < 5:
            return True
        
        # If the robot heading is close enough to the coordinates, move forward to minimize the distance
        elif abs(angle - angle_to_coords) < 0.1:
            if abs(dist) > 5:
                self._actions('forward')

        elif abs(angle + np.pi - angle_to_coords) < 0.1: # robot gets stuck if over coord, so push it backward a bit
            self._actions('backward')
        
        # Adjust heading of robot using actions to minimize the angle between the robot and the coordinates
        elif angle_to_coords > angle:
            if angle_to_coords - angle > np.pi:
                self._actions('turn_ccw')
            else:
                self._actions('turn_cw')
        else:
            if angle - angle_to_coords > np.pi:
                self._actions('turn_cw')
            else:
                self._actions('turn_ccw')
        return False
        


    def collision_begin(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        self.collision_occuring = True
        self._agent['wheel_1'].latch = False
        self._agent['wheel_2'].latch = False
        return True
    
    def collision_end(self, arbiter, space, dummy):
        self.collision_occuring = False
        return True
    
    def collision_obj_obst_begin(self, arbiter, space, dummy):
        self.obj_coll_obst = True
        return True
    
    def collision_obj_obst_end(self, arbiter, space, dummy):
        self.obj_coll_obst = False
        return True
    
    def collision_robo_obj_begin(self, arbiter, space, dummy):
        self.is_pushing = True
        if self.obj_coll_obst:
            self._agent['wheel_1'].latch = False
            self._agent['wheel_2'].latch = False
        return True
    
    def collision_robo_obj_end(self, arbiter, space, dummy):
        self.is_pushing = False
        return True
    
    def collision_obj_goal(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        self._done = True
        return True

    def get_state(self):
        """
        Gets integer pixel values from screen
        """
        self.state = (np.array(self.pxarr).astype(np.float64)/2e32).transpose()
        self.state = np.resize(rescale(self.state, 0.5)*2e32, (1,int(self.screen_size[0]/2),int(self.screen_size[1]/2)))

    def get_reward(self, action_taken: bool):
        """
        Penalty for collision with walls
        Penalty for taking an action
        Reward for pushing object into goal
        Partial reward/penalty for pushing object closer to / further from goal
        """
        reward = 0
        reward_tracker = ""
        if action_taken:
            reward -= self.action_penalty
            reward_tracker += ":action penalty: "

        if self.collision_occuring:
            reward -= self.collision_penalty
            reward_tracker += ":collision penalty: "
        
        '''
        if self.is_pushing:
            reward += self.push_reward
            reward_tracker += ":push: "
        '''
        if self.is_pushing:
            reward += self.push_reward
            reward_tracker += ":no push: "

        if self._done:
            reward += self.obj_to_goal_reward
        
        '''
        if not self.is_pushing and not self.collision_occuring and action_taken:
            reward += self.exploration_reward # Small reward for diverse actions
            reward_tracker += ":exploration:"
        '''
        # print(reward_tracker, self.reward_from_last_action)

        dist = distance(self._object.position, self.goal_position)
        dist_moved = self.initial_object_dist - dist
        self.initial_object_dist = dist
        reward += self.partial_rewards_scale*dist_moved

        return reward
    
    def reset(self):
        cumulative_reward = self.reward
        reward_from_last_action = self.reward_from_last_action
        action_function = self.take_action
        self.__init__({'collision_penalty': self.collision_penalty, 'action_penalty': self.action_penalty, 'push_reward': self.push_reward, 'obj_to_goal_reward': self.obj_to_goal_reward, 'exploration_reward': self.exploration_reward, 'partial_rewards_scale': self.partial_rewards_scale})
        self.reward = cumulative_reward
        self.reward_from_last_action = reward_from_last_action
        self.take_action = action_function
        return self.state
        
    def close(self):
        self._running = False

def main():
    game = Push_Empty_Small_Env()
    game.run()
    # img = Image.fromarray(game.state2)
    # img.show()

if __name__ == "__main__":
    main()
