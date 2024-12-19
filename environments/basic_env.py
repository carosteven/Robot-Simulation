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

        # For curriculum learning
        self.num_boxes = config['num_boxes'] if config is not None else 5
        self.training_step = config['training_step'] if config is not None else 0

        # Environment
        self.grid_size = config['grid_size'] if config is not None else 10
        self.gridscale = self.screen_size[0] // self.grid_size
        self.grid_world = np.full((self.grid_size, self.grid_size), '', dtype=object)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()
        
        # The agent to be controlled
        self._agent = self._create_agent(vertices=((-14,-14), (-14,14), (14,14), (14,-14)), mass=10, position=(random.randint(1,self.grid_size-2), random.randint(2,self.grid_size-2)), damping=0.99)
        self.initial_agent_pos = self._agent['robot'].position

        self.goal_position = self.grid_to_env((0,0))
        # self.initial_box_dists = [distance(box.position, self.goal_position) for box in self._boxes]
        # self.initial_object_dist = distance(self._object.position, self.goal_position)
        
        # The objects to be pushed
        self.box_uncertainty = config['box_uncertainty'] if config is not None else 0.1
        self.boxes_remaining = config['num_boxes'] if config is not None else 1
        self.boxes_in_goal = config['num_boxes'] - self.boxes_remaining
        self._boxes = {}
        for i in range(self.boxes_remaining):
            pos = (random.randint(1,self.grid_size-2), random.randint(2,self.grid_size-2))
            occupied = True
            while occupied:
                occupied = False
                if pos == self.env_to_grid(self._agent['robot'].position):
                    occupied = True

                for box in self._boxes.values():
                    if pos == self.env_to_grid(box['body'].position):
                        occupied = True
                
                if occupied:
                    pos = (random.randint(1,self.grid_size-2), random.randint(2,self.grid_size-2))

            # self._boxes.append(self._create_object(id=i, radius=14, mass=5, position=(4,4), damping=.99))
            self._boxes[f'{i}'] = {}
            self._boxes[f'{i}']['body'] = self._create_object(id=i, radius=14, mass=5, position=pos, damping=.99)
            self._boxes[f'{i}']['initial_dist'] = distance(self._boxes[f'{i}']['body'].position, self.goal_position)
            self._boxes[f'{i}']['collision_occuring'] = False
            self._boxes[f'{i}']['push_occuring'] = False

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
        return (grid_pos[1]*self.gridscale+self.gridscale//2, grid_pos[0]*self.gridscale+self.gridscale//2)
    
    def env_to_grid(self, env_pos):
        return (int(env_pos[1]//self.gridscale), int(env_pos[0]//self.gridscale))

    def _add_static_scenery(self) -> None:
        """
        Create the static bodies
        :return: None
        """
        static_body = self._space.static_body

        static_goal = [
            pymunk.Poly(static_body, ((0,0), (self.gridscale*2,0), (self.gridscale*2,self.gridscale*2), (0, self.gridscale*2))),
        ]
        
        static_goal[0].color = (0, 255, 0, 255)
        static_goal[0].collision_type = 2
        static_goal[0].filter = pymunk.ShapeFilter(categories=0b101)
        self._space.add(*static_goal)

        for i in range(2):
            for j in range(2):
                self.grid_world[i][j] = 'g'

        # Put gray square in each corner of the screen
        static_border = []
        edge = self.grid_size - 1
        for coord in [(edge,0), (edge,edge), (0,edge)]:
            self.grid_world[coord] = 'w'
            c = self.grid_to_env(coord)
            c0, c1, c2, c3 = c[0]+self.gridscale//2, c[1]+self.gridscale//2, c[0]-self.gridscale//2, c[1]-self.gridscale//2
            static_border.append(pymunk.Poly(static_body, ((c0, c1), (c0, c3), (c2, c3), (c2, c1))))
        
        # Label new corners
        for coord in [(edge-1,0), (edge,1), (edge-1,edge), (edge,edge-1), (0,edge-1), (1,edge)]:
            self.grid_world[coord] = 'c'

        # Create the border walls
        for i in range(self.grid_size +1):
            static_border.append(pymunk.Segment(static_body, (i*self.gridscale,0), (i*self.gridscale,self.screen_size[1]), 1))
            static_border.append(pymunk.Segment(static_body, (0,i*self.gridscale), (self.screen_size[0],i*self.gridscale), 1))

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
        
        radius = self.gridscale // 2 - 1
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

        radius = self.gridscale // 2 - 1
        robot = pymunk.Poly(robot_body, ((-radius, -radius), (-radius, radius), (radius, radius), (radius, -radius)))
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
                # self._actions(self.available_actions[random.randint(0,7)])
                self._done = True
                # self._running = False
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
                        grid_world = np.zeros_like(self.grid_world)
                        for i, row in enumerate(self.grid_world):
                            for j, cell in enumerate(row):
                                if cell != '':
                                    grid_world[i][j] = cell
                                else:
                                    grid_world[i][j] = ' '
                        print(grid_world)
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
            action = 'N' if random_action == 0 else 'NE' if random_action == 1 else 'NW'
            return (grid_coords[0]-1, grid_coords[1]+random_action), action
        elif action == 'E':
            action = 'E' if random_action == 0 else 'SE' if random_action == 1 else 'NE'
            return (grid_coords[0]+random_action, grid_coords[1]+1), action
        elif action == 'S':
            action = 'S' if random_action == 0 else 'SE' if random_action == 1 else 'SW'
            return (grid_coords[0]+1, grid_coords[1]+random_action), action
        elif action == 'W':
            action = 'W' if random_action == 0 else 'SW' if random_action == 1 else 'NW'
            return (grid_coords[0]+random_action, grid_coords[1]-1), action
        elif action == 'NE':
            action = 'NE' if random_action == 0 else 'N' if random_action == 1 else 'E'
            return (grid_coords[0] - rand_1, grid_coords[1] + rand_2), action
        elif action == 'SE':
            action = 'SE' if random_action == 0 else 'S' if random_action == 1 else 'E'
            return (grid_coords[0] + rand_1, grid_coords[1] + rand_2), action
        elif action == 'SW':
            action = 'SW' if random_action == 0 else 'S' if random_action == 1 else 'W'
            return (grid_coords[0] + rand_1, grid_coords[1] - rand_2), action
        elif action == 'NW':
            action = 'NW' if random_action == 0 else 'N' if random_action == 1 else 'W'
            return (grid_coords[0] - rand_1, grid_coords[1] - rand_2), action

    def get_box_index(self, grid_coords) -> int:
        """
        Get the index of the box in the list of boxes
        """
        for box in self._boxes.values():
            if grid_coords == self.env_to_grid(box['body'].position):
                return box['body'].label.split('_')[-1]
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
        second_label = self.grid_world[grid_coords][:-1] if grid_label != '' else None
        if grid_label == 'b':
            if obj_type == 'r':
                self.is_pushing = True
            new_box_coords, action = self.move_box(grid_coords, action)
            box_idx = self.get_box_index(grid_coords)
            num_boxes = len(self._boxes)
            if second_label == 'c'and obj_type == 'b':
                if action == 'NW':
                    if new_box_coords[0] == -1:
                        action = 'W'
                    else:
                        action = 'N'
                elif action == 'SE':
                    if new_box_coords[0] == 2:
                        action = 'S'
                    else:
                        action = 'E'
                elif action == 'NE':
                    action = 'N'
                elif action == 'SW':
                    action = 'W'
                new_box_coords, action = self.move_box(grid_coords, action)
                # self.check_collision(new_box_coords, action, 'b', box_idx)
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

        if self.config['curriculum']:
            if len(self._boxes) == self.num_boxes - (self.training_step+1):
                self._done = True
        else:
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
        state_half = np.resize(rescale(state, 0.5)*2e32, (1,int(self.screen_size[0]/2),int(self.screen_size[1]/2)))
        
        if self.state_info == 'colour':
            self.state = state_half
        
        elif self.state_info == 'multiinfo':
            self.state = []
            self.state.append(state_half)
            # append agent position
            self.state.append(self._agent['robot'].position)

        elif self.state_info == 'multicam':
            self.state = []
            self.state.append(np.resize(state, (1, self.screen_size[0], self.screen_size[1])))
            # append pixels around agent
            agent_pos = self.env_to_grid(self._agent['robot'].position)
            pixel_array = np.zeros((3, 3, 3), dtype=int)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    grid_pos = (agent_pos[0] + i, agent_pos[1] + j)
                    if 0 <= grid_pos[0] < self.grid_size and 0 <= grid_pos[1] < self.grid_size:
                        cell = self.grid_world[grid_pos]
                        if 'r' in cell:
                            pixel_array[i + 1, j + 1] = [0, 0, 255]  # Blue
                        elif 'b' in cell:
                            pixel_array[i + 1, j + 1] = [255, 0, 0]  # Red
                        elif 'w' in cell:
                            pixel_array[i + 1, j + 1] = [0, 0, 0]  # Black
                        else:
                            pixel_array[i + 1, j + 1] = [255, 255, 255]  # White
                    else:
                        pixel_array[i + 1, j + 1] = [0, 0, 0]  # Black for out of bounds
            self.state.append(pixel_array.transpose(2, 0, 1))

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
        training_step = self.training_step
        self.__init__(self.config)
        self.reward = cumulative_reward
        self.training_step = training_step
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
