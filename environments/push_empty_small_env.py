__docformat__ = "reStructeredText"

import random
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from PIL import Image
from skimage.transform import resize, rescale
import yaml

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
        self.config = config

        self.state_type = config['state_type'] if config is not None else 'vision'
        self.state_info =  config['state_info'] if config is not None else 'colour'
        self.MINISTEP_SIZE = config['ministep_size']            # Scaling for distance moved by agent

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

        # For curriculum learning
        self.num_boxes = config['num_boxes'] if config is not None else 5
        self.training_step = config['training_step'] if config is not None else 0

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()
        
        self.goal_position = (35,35)
        # self.initial_box_dists = [distance(box.position, self.goal_position) for box in self._boxes]
        # self.initial_object_dist = distance(self._object.position, self.goal_position)

        # The objects to be pushed
        self.boxes_remaining = config['num_boxes'] if config is not None else 1
        self._boxes = {}
        for i in range(self.boxes_remaining):
            self._boxes[f'{i}'] = {}
            self._boxes[f'{i}']['body'] = self._create_object(id=i, radius=15, mass=5, position=(random.randint(65,200), random.randint(150,250)), damping=.99)
            self._boxes[f'{i}']['initial_dist'] = distance(self._boxes[f'{i}']['body'].position, self.goal_position)
            self._boxes[f'{i}']['collision_occuring'] = False
            self._boxes[f'{i}']['push_occuring'] = False

        # self._object = self._create_object(radius=15, mass=5, position=tuple([c/2 for c in self.screen_size]), damping=.99)

        # The agent to be controlled
        y_pos = random.randint(50,self.screen_size[1]-50)
        self._agent = self._create_agent(vertices=((-25,-25), (-25,25), (25,25), (25,-25)), mass=10, position=(self.screen_size[0]*0.8, y_pos), damping=0.99)
        self.initial_agent_pos = self._agent['robot'].position
        self.agent_distance = 0
        self.front_x = self._agent['robot'].local_to_world((0, -25)).x
        self.front_y = self._agent['robot'].local_to_world((0, -25)).y
        self.agent_last_pos = (self.front_x, self.front_y)

        self.state = np.zeros((1, self.screen_size[0], self.screen_size[1])).astype(int)
        self.get_state()
        '''
        if config['action_type'] == 'straight-line-navigation':
            self.take_action = self.straight_line_navigation
        elif config['action_type'] == 'action-control':
            self.take_action = self._actions
        else:
            self.take_action = None
        '''

        # Agent cumulative rewards
        self.reward = 0
        self.reward_from_last_action = 0

        self.nonmovement_tracker = [1]*5

        # Rewards
        self.collision_penalty      = config['collision_penalty'] if config is not None else 10
        self.action_penalty         = config['action_penalty'] if config is not None else 1
        self.no_movement_penalty    = config['no_movement_penalty'] if config is not None else 0
        self.push_reward            = config['push_reward'] if config is not None else 1
        self.obj_to_goal_reward     = config['obj_to_goal_reward'] if config is not None else 1000
        self.exploration_reward     = config['exploration_reward'] if config is not None else 0.1
        self.partial_rewards_scale  = config['partial_rewards_scale'] if config is not None else 1

        # Available actions
        self.available_actions = ['forward', 'backward', 'turn_cw', 'turn_ccw']
        self.action_completed = False


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
            pymunk.Poly(static_body, ((0,0), (70,0), (70,70), (0, 70))),
        ]
        
        static_goal[0].color = (0, 255, 0, 255)
        static_goal[0].collision_type = 2
        static_goal[0].filter = pymunk.ShapeFilter(categories=0b101)
        self._space.add(*static_goal)

    def _create_object(self, id: int, radius: float, mass: float, position: tuple[int] = (0,0), elasticity: float = 0, friction: float = 1.0, damping: float = 0.0) -> pymunk.Poly:
        """
        Create the object to be pushed
        :return: Pymunk Polygon
        """
        object_body = pymunk.Body()
        object_body.position = position
        object_body.damping = damping
        object_body.label = 'box_'+str(id)
        object_body.velocity_func = custom_damping
        
        object = pymunk.Poly(object_body, ((-radius, -radius), (-radius, radius), (radius, radius), (radius, -radius)))
        object.color = (255, 0, 0, 255)
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
            self.agent_last_pos = (self.front_x, self.front_y)

            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            action = self._process_events()
            self._actions(action)
            coord = (200,0)
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
                print(self.reward_from_last_action)
                self.reward_from_last_action = 0

            # Calculate reward
            self.reward_from_last_action += self.get_reward(True if action is not None else False)
            # if action is not None:
            #     print(self.reward_from_last_action)
            # self.reward = self.get_reward(True if action is not None else False)

            # Calculate reward
            # robot_reward = self.get_reward()
            # self.reward += robot_reward
            # print(self.reward_from_last_action)
            
            if self._done:
                self.reset()
    
    def step(self, action, primitive=False, test=False) -> tuple:
        """
        Progress the simulation one time step
        inputs: action
        outputs: state, reward, done, info
        """
        self.agent_last_pos = (self.front_x, self.front_y)

        # Progress time forward
        for x in range(self._physics_steps_per_frame):
            self._space.step(self._dt)

        self._process_events()
        # self._actions(action)
        # self.action_completed = self.take_action(action)
        if not primitive:
            self.action_completed = self.straight_line_navigation(action)
        else:
            self._actions(action)

        if self.action_completed:
            self.agent_distance = distance(self._agent['robot'].position, self.initial_agent_pos)
            self.initial_agent_pos = self._agent['robot'].position
            self.nonmovement_tracker = [1]*5
            
        self._update()
        self._clear_screen()
        if test:
            pygame.draw.circle(self._screen, (0,0,0), action, 5)
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
        self.reward_from_last_action += self.get_reward(True if action is not None else False, primitive)

        # Items to return
        state = self.state
        reward = self.reward_from_last_action
        done = self._done
        ministeps = self.agent_distance / self.MINISTEP_SIZE
        info = {
            'ministeps': ministeps,
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
                self._done = True
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
        # Get the heading of the robot
        angle = (self._agent['robot'].angle - (np.pi/2)) % (2*np.pi)

        # Get the angle between the front of the robot and the coordinates
        self.front_x = self._agent['robot'].local_to_world((0, -25)).x
        self.front_y = self._agent['robot'].local_to_world((0, -25)).y
        angle_to_coords = np.arctan2(coords[1] - self.front_y, coords[0] - self.front_x) % (2*np.pi)

        # Get the distance between the front of the robot and the coordinates
        dist = distance(pymunk.vec2d.Vec2d(self.front_x, self.front_y), coords)
        
        # If the robot is close enough to the coordinates, stop
        if dist < 5:
            return True
        
        # If the robot heading is close enough to the coordinates, move forward to minimize the distance
        elif abs(angle - angle_to_coords) < 0.1:
            if abs(dist) > 5:
                self._actions('forward')

        elif abs((angle + np.pi) % (2*np.pi) - angle_to_coords) < 0.1 and dist < 50: # robot gets stuck if over coord, so push it backward a bit
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

        self.nonmovement_tracker.append(distance(pymunk.vec2d.Vec2d(self.front_x, self.front_y), self.agent_last_pos))
        self.nonmovement_tracker.pop(0)
        dist_travelled = 0
        for dist in self.nonmovement_tracker:
            dist_travelled += dist

        if dist_travelled < 0.45:
            return True
        
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
        shapes = arbiter.shapes
        for box in self._boxes.values():
            if box['body'].label == shapes[0].body.label:
                box['collision_occuring'] = True
                break
        return True
    
    def collision_obj_obst_end(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        for box in self._boxes.values():
            if box['body'].label == shapes[0].body.label:
                box['collision_occuring'] = False
                break
        return True
    
    def collision_robo_obj_begin(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        for box in self._boxes.values():
            if box['body'].label == shapes[1].body.label:
                box['push_occuring'] = True
                break
        
        for box in self._boxes.values():
            if box['push_occuring'] and box['collision_occuring']:
                self._agent['wheel_1'].latch = False
                self._agent['wheel_2'].latch = False

        self.is_pushing = True
        return True
    
    def collision_robo_obj_end(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        for box in self._boxes.values():
            if box['body'].label == shapes[1].body.label:
                box['push_occuring'] = False
                break

        self.is_pushing = False
        return True
    
    def collision_obj_goal(self, arbiter, space, dummy):
        shapes = arbiter.shapes
        for box in self._boxes:
            if self._boxes[box]['body'].label == shapes[0].body.label:
                self._boxes.pop(box)
                self._space.remove(shapes[0], shapes[0].body)
                break
        if len(self._boxes) == 0:
            self._done = True
        return True

    def get_state(self):
        """
        Gets integer pixel values from screen
        """
        state = (np.array(self.pxarr).astype(np.float64)/2e32).transpose()
        state = np.resize(rescale(state, 0.5)*2e32, (1,int(self.screen_size[0]/2),int(self.screen_size[1]/2)))
        
        if self.state_info == 'colour':
            self.state = state
        
        elif self.state_info == 'multiinfo':
            self.state = []
            self.state.append(state)
            # append agent position
            self.state.append(self._agent['robot'].position)

    def get_reward(self, action_taken: bool, primitive=False):
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
            # print("Collision")
            reward_tracker += ":collision penalty: "

        if self.is_pushing:
            reward += self.push_reward
            reward_tracker += ":push: "

        self.boxes_in_goal = self.boxes_remaining-len(self._boxes)
        if self.boxes_in_goal > 0:
            reward += (self.boxes_in_goal)*self.obj_to_goal_reward
        self.boxes_remaining = len(self._boxes)
        
        '''
        if not self.is_pushing and not self.collision_occuring and action_taken:
            reward += self.exploration_reward # Small reward for diverse actions
            reward_tracker += ":exploration:"
        '''
        # print(reward_tracker, self.reward_from_last_action)

        # Calculate reward for pushing object closer to goal
        dist_tracker = 0
        for box in self._boxes.values():
            dist = distance(box['body'].position, self.goal_position)
            dist_moved = box['initial_dist'] - dist
            box['initial_dist'] = dist
            reward += self.partial_rewards_scale*dist_moved
            dist_tracker += dist_moved*self.partial_rewards_scale
        reward_tracker += f":dist moved: {dist_tracker} "

        # dist = distance(self._object.position, self.goal_position)
        # dist_moved = self.initial_object_dist - dist
        # self.initial_object_dist = dist
        # reward += self.partial_rewards_scale*dist_moved
        if self.action_completed and self.agent_distance < (3 if primitive else 25):
            reward -= self.no_movement_penalty
        return reward
    
    def reset(self):
        cumulative_reward = self.reward
        self.__init__(self.config)
        self.reward = cumulative_reward
        return self.state
        
    def close(self):
        self._running = False

def main():
    # config_path = 'configurations/config_train_complex.yml'
    config_path = 'configurations/config_cmplx_test.yml'
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    game = Push_Empty_Small_Env(config)
    game.run()

if __name__ == "__main__":
    main()
