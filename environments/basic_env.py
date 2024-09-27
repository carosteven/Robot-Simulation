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

class Basic_Env(object):
    def __init__(self, config=None) -> None:
        self.config = config

        self.state_type = config['state_type'] if config is not None else 'vision'
        self.state_info =  config['state_info'] if config is not None else 'colour'

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

        # Environment
        self.grid_size = config['grid_size'] if config is not None else 10
        self.grid_world = np.full((self.grid_size, self.grid_size), '', dtype=object)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        self.goal_position = (30,60)
        # self.initial_box_dists = [distance(box.position, self.goal_position) for box in self._boxes]
        # self.initial_object_dist = distance(self._object.position, self.goal_position)
        
        # The objects to be pushed
        self.box_uncertainty = config['box_uncertainty'] if config is not None else 0.1
        self.boxes_remaining = config['num_boxes'] if config is not None else 1
        self.boxes_in_goal = config['num_boxes'] - self.boxes_remaining
        self._boxes = {}
        for i in range(self.boxes_remaining):
            pos = (random.randint(1,7), random.randint(3,6))
            occupied = True
            while occupied:
                occupied = False
                for box in self._boxes.values():
                    if pos == self.env_to_grid(box['body'].position):
                        occupied = True
                        pos = (random.randint(1,7), random.randint(3,6))
            # self._boxes.append(self._create_object(id=i, radius=14, mass=5, position=(4,4), damping=.99))
            self._boxes[f'{i}'] = {}
            self._boxes[f'{i}']['body'] = self._create_object(id=i, radius=14, mass=5, position=pos, damping=.99)
            self._boxes[f'{i}']['initial_dist'] = distance(self._boxes[f'{i}']['body'].position, self.goal_position)
            self._boxes[f'{i}']['collision_occuring'] = False
            self._boxes[f'{i}']['push_occuring'] = False

        # The agent to be controlled
        self._agent = self._create_agent(vertices=((-14,-14), (-14,14), (14,14), (14,-14)), mass=10, position=(random.randint(1,8), 8), damping=0.99)
        # self._agent = self._create_agent(vertices=((-14,-14), (-14,14), (14,14), (14,-14)), mass=10, position=(4, 5), damping=0.99)
        self.initial_agent_pos = self._agent['robot'].position

        self.state = np.zeros((1, self.screen_size[0], self.screen_size[1])).astype(int)
        self.get_state()

        if config['action_type'] == 'action-control':
            self.take_action = self._actions
        else:
            self.take_action = None

        self.is_pushing = False
        self.collision_occuring = False

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
        self.corner_penalty         = config['corner_penalty'] if config is not None else 10

        # Available actions
        self.available_actions = ['N', 'E', 'S', 'W', 'NE', 'SE', 'SW', 'NW']
        self.action_completed = False

        # Execution control
        self._running = True
        self._done = False

    def grid_to_env(self, grid_pos):
        grid_scale = self.screen_size[0] // self.grid_size
        return (grid_pos[1]*grid_scale+grid_scale//2, grid_pos[0]*grid_scale+grid_scale//2)
    
    def env_to_grid(self, env_pos):
        grid_scale = self.screen_size[0] // self.grid_size
        return (int(env_pos[1]//grid_scale), int(env_pos[0]//grid_scale))

    def _add_static_scenery(self) -> None:
        """
        Create the static bodies
        :return: None
        """
        grid_scale = self.screen_size[0] // self.grid_size
        static_body = self._space.static_body

        static_goal = [
            pymunk.Poly(static_body, ((0,0), (60,0), (60,120), (0, 120))),
        ]
        
        static_goal[0].color = (0, 255, 0, 255)
        static_goal[0].collision_type = 2
        static_goal[0].filter = pymunk.ShapeFilter(categories=0b101)
        self._space.add(*static_goal)

        for i in range(4):
            for j in range(2):
                self.grid_world[i][j] = 'g'

        # Put gray square in each corner of the screen
        static_border = []
        for coord in [(9,0), (9,9), (0,9)]:
            self.grid_world[coord] = 'w'
            c = self.grid_to_env(coord)
            static_border.append(pymunk.Poly(static_body, ((c[1]-14, c[0]-14), (c[1]-14, c[0]+14), (c[1]+14, c[0]+14), (c[1]+14, c[0]-14))))
        
        # Label new corners
        for coord in [(8,0), (9,1), (8,9), (9,8), (0,8), (1,9)]:
            self.grid_world[coord] = 'c'

        # Create the border walls
        for i in range(self.grid_size +1):
            static_border.append(pymunk.Segment(static_body, (i*grid_scale,0), (i*grid_scale,self.screen_size[1]), 1))
            static_border.append(pymunk.Segment(static_body, (0,i*grid_scale), (self.screen_size[0],i*grid_scale), 1))

        for line in static_border:
            line.elasticity = 0.95
            line.friction = 0.9
            line.collision_type = 1
            line.filter = pymunk.ShapeFilter(categories=0b10)
            line.color = (0, 0, 0, 255)
        self._space.add(*static_border)

    def _create_object(self, id: int, radius: float, mass: float, position: tuple[int] = (0,0), elasticity: float = 0, friction: float = 1.0, damping: float = 0.0) -> pymunk.Poly:
        """
        Create the object to be pushed
        :return: Pymunk Polygon
        """
        self.grid_world[position] = 'b'
        object_body = pymunk.Body()
        object_body.position = self.grid_to_env(position)
        object_body.damping = damping
        object_body.label = 'box_'+ str(id)
        object_body.velocity_func = custom_damping
        object_body.in_corner = False
        
        object = pymunk.Poly(object_body, ((-radius, -radius), (-radius, radius), (radius, radius), (radius, -radius)))
        object_body.object = object
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
        self.grid_world[position] = 'r'
        robot_body = pymunk.Body()
        robot_body.position = self.grid_to_env(position)
        robot_body.damping = damping

        robot = pymunk.Poly(robot_body, vertices)
        robot.label = 'robot'
        robot.mass = mass
        robot.elasticity = elasticity
        robot.friction = friction
        robot.collision_type = 0
        robot.filter = pymunk.ShapeFilter(categories=0b1)
        self._space.add(robot_body, robot)

        return {'robot': robot_body}

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
            self._update()
            self._clear_screen()
            self._draw_objects()
            # pygame.draw.circle(self._screen, (0,0,0), (self.goal_position), 5)
            self.get_state()
            
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(110)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

            if action is not None:
                # print(self.reward_from_last_action)
                self.reward += self.reward_from_last_action
                self.reward_from_last_action = 0

            # Calculate reward
            self.reward_from_last_action += self.get_reward(True if action is not None else False)
            if action is not None:
                print(self.reward_from_last_action)

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
        # Progress time forward
        for x in range(self._physics_steps_per_frame):
            self._space.step(self._dt)

        self._process_events()
        # self._actions(action)
        # self.action_completed = self.take_action(action)
        if primitive:
            self._actions(action)
        self._update()
        self._clear_screen()
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

        robot_distance = distance(self._agent['robot'].position, self.initial_agent_pos)
        self.initial_agent_pos = self._agent['robot'].position

        # Items to return
        state = self.state
        reward = self.reward_from_last_action
        done = self._done
        info = {
            'distance': robot_distance,
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
                    if keys[pygame.K_UP]:
                        action = 'NW'
                    elif keys[pygame.K_DOWN]:
                        action = 'SE'
                    elif keys[pygame.K_LEFT]:
                        action = 'SW'
                    elif keys[pygame.K_RIGHT]:
                        action = 'NE'
                else:
                    if keys[pygame.K_UP]:
                        action = 'N'
                    elif keys[pygame.K_DOWN]:
                        action = 'S'
                    elif keys[pygame.K_LEFT]:
                        action = 'W'
                    elif keys[pygame.K_RIGHT]:
                        action = 'E'
                    elif keys[pygame.K_SPACE]:
                        print(self.grid_world)
        return action
    
    def _update(self) -> None:
        pass

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

    def move_box(self, grid_coords, action) -> tuple:
        """
        Move the object in the grid
        """
        prob = self.box_uncertainty
        random_action = np.random.choice([-1, 0, 1], p=[prob/2, 1-prob, prob/2])
        rand_1, rand_2 = 1, 1
        if random_action == -1:
            rand_1 = 0
        elif random_action == 1:
            rand_2 = 0

        if action == 'N':
            return (grid_coords[0]-1, grid_coords[1]+random_action)
        elif action == 'E':
            return (grid_coords[0]+random_action, grid_coords[1]+1)
        elif action == 'S':
            return (grid_coords[0]+1, grid_coords[1]+random_action)
        elif action == 'W':
            return (grid_coords[0]+random_action, grid_coords[1]-1)
        elif action == 'NE':
            return (grid_coords[0]- (1 * rand_1), grid_coords[1]+ (1 * rand_2))
        elif action == 'SE':
            return (grid_coords[0]+ (1 * rand_1), grid_coords[1]+ (1 * rand_2))
        elif action == 'SW':
            return (grid_coords[0]+ (1 * rand_1), grid_coords[1]- (1 * rand_2))
        elif action == 'NW':
            return (grid_coords[0]- (1 * rand_1), grid_coords[1]- (1 * rand_2)) 

    def get_box_index(self, grid_coords) -> int:
        """
        Get the index of the box in the list of boxes
        """
        for box in self._boxes.values():
            if grid_coords == self.env_to_grid(box['body'].position):
                return box['body'].label[-1]
        return -1       
    
    def check_collision(self, grid_coords, action, obj_type, box_idx=None) -> bool:
        """
        Check if the object will collide with the walls or other objects
        """
        collision_detected = False
        if grid_coords[0] < 0 or grid_coords[0] >= self.grid_size or grid_coords[1] < 0 or grid_coords[1] >= self.grid_size or self.grid_world[grid_coords] == 'w':
            collision_detected = True
            self.collision_occuring = True if obj_type == 'r' else False
            return collision_detected
        
        grid_label = self.grid_world[grid_coords]
        grid_label = grid_label[-1] if grid_label != '' else grid_label
        if grid_label == 'b':
            if obj_type == 'r':
                self.is_pushing = True
            new_box_coords = self.move_box(grid_coords, action)
            box_idx = self.get_box_index(grid_coords)
            num_boxes = len(self._boxes)
            if self.check_collision(new_box_coords, action, 'b', box_idx):
                collision_detected = True
            elif self.grid_world[new_box_coords] != 'g':
                self._boxes[box_idx]['body'].position = self.grid_to_env(new_box_coords)
                self.grid_world[grid_coords] = self.grid_world[grid_coords][:-1]
                self.grid_world[new_box_coords] += "b"
            elif self.grid_world[new_box_coords] == 'g':
                self.grid_world[grid_coords] = self.grid_world[grid_coords][:-1]
        
        elif grid_label == 'g':
            if obj_type == 'r':
                collision_detected = True
                self.collision_occuring = True
            elif obj_type == 'b':
                self._space.remove(self._boxes[box_idx]['body'], self._boxes[box_idx]['body'].object)
                self._boxes.pop(box_idx)
        
        elif grid_label == 'c':
            if obj_type == 'b':
                # box_idx = self.get_box_index(grid_coords)
                self._boxes[box_idx]['body'].in_corner = True
        
        if len(self._boxes) == 0:
            self._done = True

        return collision_detected

    def _actions(self, action) -> bool:
        """
        action: 'N', 'E', 'S', 'W', 'NE', 'SE', 'SW', 'NW'
        :return: action_completed
        """
        grid_coords = self.env_to_grid(self._agent['robot'].position)
        if action == 'N':
            new_coords = (grid_coords[0]-1, grid_coords[1])
        elif action == 'E':
            new_coords = (grid_coords[0], grid_coords[1]+1)
        elif action == 'S':
            new_coords = (grid_coords[0]+1, grid_coords[1])
        elif action == 'W':
            new_coords = (grid_coords[0], grid_coords[1]-1)
        elif action == 'NE':
            new_coords = (grid_coords[0]-1, grid_coords[1]+1)
        elif action == 'SE':
            new_coords = (grid_coords[0]+1, grid_coords[1]+1)
        elif action == 'SW':
            new_coords = (grid_coords[0]+1, grid_coords[1]-1)
        elif action == 'NW':
            new_coords = (grid_coords[0]-1, grid_coords[1]-1)
        else:
            return True

        if self.check_collision(new_coords, action, 'r'):
            return False
        self._agent['robot'].position = self.grid_to_env((new_coords))
        self.grid_world[grid_coords] = self.grid_world[grid_coords][:-1]
        self.grid_world[new_coords] += 'r'
        
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
            # reward_tracker += ":action penalty: "

        if self.collision_occuring:
            reward -= self.collision_penalty
            reward_tracker += f":collision penalty: {self.collision_penalty} "
        self.collision_occuring = False

        if self.is_pushing:
            reward += self.push_reward
            reward_tracker += f":push: {self.push_reward} "
        self.is_pushing = False

        for box in self._boxes.values():
            if box['body'].in_corner:
                reward -= self.corner_penalty
            box['body'].in_corner = False

        new_box_in_goal = self.boxes_remaining-len(self._boxes)
        if new_box_in_goal > 0:
            reward_tracker += f":goal: {(new_box_in_goal)*self.obj_to_goal_reward} "
        reward += (new_box_in_goal)*self.obj_to_goal_reward
        self.boxes_remaining = len(self._boxes)
        self.boxes_in_goal = self.config['num_boxes'] - self.boxes_remaining
        
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
        # if action_taken:
        #     print(reward_tracker)
        return reward
    
    def reset(self):
        cumulative_reward = self.reward
        self.__init__(self.config)
        self.reward = cumulative_reward
        return self.state
        
    def close(self):
        self._running = False

def main():
    config_path = 'configurations/config_basic_test.yml'
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    game = Basic_Env(config)
    game.run()

if __name__ == "__main__":
    main()
