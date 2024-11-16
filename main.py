import pygame
import numpy as np
import random
import os
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('my_model1 (30).h5')

interpreter = tf.lite.Interpreter(model_path='my_model1 (6).tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(inputs):
    # Flatten any nested arrays and ensure all elements are numeric
    flattened_inputs = []
    for item in inputs:
        if isinstance(item, np.ndarray):
            # Extract the value from the numpy array
            item = item.flatten()[0]
        flattened_inputs.append(float(item))  # Convert to float if necessary

    input_data = np.array(flattened_inputs, dtype=np.int8)
    print(f"input_data: {input_data}")

    # Retrieve scale and zero_point for the input tensor
    input_scale, input_zero_point = input_details[0]['quantization']
    print(input_scale, input_zero_point)

    # Normalize input data to quantized format
    input_data = np.clip(input_data / input_scale + input_zero_point, -128, 127).astype(np.int8)
    print(f"input_data: {input_data}")

    # Set the tensor for input
    interpreter.set_tensor(input_details[0]['index'], [input_data])

    # Run inference
    interpreter.invoke()

    # Get the result
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Retrieve scale and zero_point for the output tensor
    #output_scale, output_zero_point = output_details[0]['quantization']
    #print(output_scale, output_zero_point)

    # Dequantize output data if necessary
    #output_data = (output_scale * output_data)

    # Print output data and action
    print(f"output_data: {output_data}")
    print(f"output_action: {np.argmax(output_data)}")

    return np.argmax(output_data)


def predict(inputs):
    # Flatten any nested arrays and ensure all elements are numeric
    flattened_inputs = []
    for item in inputs:
        if isinstance(item, np.ndarray):
            # Extract the value from the numpy array
            item = item.flatten()[0]
        flattened_inputs.append(float(item))  # Convert to float if necessary

    input_data = np.array(flattened_inputs, dtype=np.float32)

    # Check the shape and type of input_data
    print(f"Input data shape before reshape: {input_data.shape}")

    # Reshape input data to match the model's expected input shape
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)  # Reshape to 2D if it's a 1D array

    # Check the shape after reshaping
    print(f"Input data shape after reshape: {input_data.shape}")

    # Run inference
    output_data = model.predict(input_data)
    print(f"output_data: {output_data}")
    print(f"output_action: {np.argmax(output_data)}")
    return np.argmax(output_data)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1000, 800))
clock = pygame.time.Clock()

# Define constants
NUM_SECTORS = 5
ALPHA = 0.05  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.05  # Exploration rate
T = 0.2

# Define actions
ACTIONS = [0, 1, 2]
ACTION_SIZE = len(ACTIONS)
action = 0

distance_ranges = [2, 2, 2, 2, 2]

# Initialize Q-table
STATE_SIZE = 3 ** NUM_SECTORS  # State size (discrete distance ranges for NUM_SECTORS sensors)
Q_TABLE_FILE = 'q_table.csv'


def load_q_table():
    """Load the Q-table from a CSV file."""
    if os.path.exists(Q_TABLE_FILE):
        df = pd.read_csv(Q_TABLE_FILE, header=None)
        return df.to_numpy()
    else:
        return np.zeros((STATE_SIZE, ACTION_SIZE))


def save_q_table():
    """Save the Q-table to a CSV file."""
    df = pd.DataFrame(Q_TABLE)
    df.to_csv(Q_TABLE_FILE, index=False, header=False)


Q_TABLE = load_q_table()

# Define Robot
robot_color = (0, 0, 255)
robot_size = 15
robot_pos = [500, 400]
robot_angle = random.randint(0, 359)
sensor_range = 75

# Define Obstacles and Boundaries
obstacles = [
    pygame.Rect(700, 750, 70, 30),
    pygame.Rect(300, 400, 30, 80),
    pygame.Rect(20, 200, 200, 20),
    pygame.Rect(490, 600, 100, 30),
    pygame.Rect(600, 250, 30, 100),
    pygame.Rect(800, 350, 200, 50),
    pygame.Rect(200, 200, 30, 100),
    pygame.Rect(300, 600, 30, 30),
    pygame.Rect(300, 600, 30, 100),
]

brick_size = 10


# Generate bricks for each boundary
def generate_bricks(start_x, start_y, end_x, end_y):
    bricks = []
    x = start_x
    y = start_y
    while x <= end_x:
        while y <= end_y:
            bricks.append(pygame.Rect(x, y, brick_size, brick_size))
            y += brick_size
        y = start_y
        x += brick_size
    return bricks


# Define boundary bricks
top_boundary_bricks = generate_bricks(0, 0, 1000 - brick_size, brick_size)
left_boundary_bricks = generate_bricks(0, brick_size, brick_size, 800 - brick_size)
bottom_boundary_bricks = generate_bricks(0, 790 - brick_size, 1000 - brick_size, 790)
right_boundary_bricks = generate_bricks(990 - brick_size, brick_size, 1000, 800 - brick_size)

# Combine all boundary bricks
boundaries = top_boundary_bricks + left_boundary_bricks + bottom_boundary_bricks + right_boundary_bricks


def draw_obstacles():
    """Draw obstacles and boundaries on the screen."""
    for obstacle in obstacles:
        pygame.draw.rect(screen, (255, 0, 0), obstacle)
    for boundary in boundaries:
        pygame.draw.rect(screen, (0, 0, 0), boundary)  # Black color for boundaries


def draw_robot():
    """Draw the robot and its sensors on the screen."""
    pygame.draw.circle(screen, robot_color, (int(robot_pos[0]), int(robot_pos[1])), robot_size)
    for angle in range(-60, 61, 30):
        x_end = robot_pos[0] + sensor_range * np.cos(np.radians(robot_angle + angle))
        y_end = robot_pos[1] + sensor_range * np.sin(np.radians(robot_angle + angle))
        pygame.draw.line(screen, (0, 255, 0), (int(robot_pos[0]), int(robot_pos[1])), (int(x_end), int(y_end)), 2)


def move_robot(action):
    """Move the robot based on the chosen action."""
    global robot_angle
    dx, dy = 0, 0
    if action == 0:  # Move forward
        dx = 5 * np.cos(np.radians(robot_angle))
        dy = 5 * np.sin(np.radians(robot_angle))
    elif action == 1:  # Turn left
        robot_angle -= 5
    elif action == 2:  # Turn right
        robot_angle += 5

    new_pos = [robot_pos[0] + dx, robot_pos[1] + dy]

    if not check_collision(new_pos):
        robot_pos[:] = new_pos
    return new_pos


def check_collision(new_pos):
    """Check for collision with obstacles or boundaries."""
    robot_rect = pygame.Rect(new_pos[0] - robot_size, new_pos[1] - robot_size, robot_size * 2, robot_size * 2)

    # Check boundary collisions
    if new_pos[0] < robot_size or new_pos[0] > 1000 - robot_size or new_pos[1] < robot_size or new_pos[
        1] > 800 - robot_size:
        return True

    # Check obstacle and boundary collisions
    for obstacle in obstacles:
        if robot_rect.colliderect(obstacle):
            return True
    for boundary in boundaries:
        if robot_rect.colliderect(boundary):
            return True

    return False


def get_range(distance):
    """Determine distance range."""
    if 0 <= distance <= 40:
        return 0
    elif 40 < distance <= 70:
        return 1
    else:
        return 2


def get_distance_ranges():
    """Get distance ranges from sensors."""
    distances = [get_range(simulate_sensor(angle)) for angle in range(-60, 61, 30)]
    return distances


def simulate_sensor(angle):
    """Simulate sensor reading with obstacles and boundaries."""
    radians = np.radians(angle + robot_angle)
    dx = np.cos(radians) * sensor_range
    dy = np.sin(radians) * sensor_range
    end_pos = [robot_pos[0] + dx, robot_pos[1] + dy]

    sensor_line = (robot_pos[0], robot_pos[1], end_pos[0], end_pos[1])

    min_distance = sensor_range  # Start with max range

    # Check collision with obstacles
    for obstacle in obstacles:
        distance = line_intersection_distance(sensor_line, obstacle)
        if distance is not None:
            min_distance = min(min_distance, distance)

    # Check collision with boundaries
    for boundary in boundaries:
        distance = line_intersection_distance(sensor_line, boundary)
        if distance is not None:
            min_distance = min(min_distance, distance)

    return min_distance


def line_intersection_distance(sensor_line, rect):
    """Calculate the distance from the sensor to the intersection with the rectangle's edges."""
    x1, y1, x2, y2 = sensor_line
    rect_edges = [
        (rect.topleft, rect.topright),
        (rect.topright, rect.bottomright),
        (rect.bottomright, rect.bottomleft),
        (rect.bottomleft, rect.topleft)
    ]

    closest_distance = None

    for edge_start, edge_end in rect_edges:
        intersection = line_intersection((x1, y1), (x2, y2), edge_start, edge_end)
        if intersection:
            distance = calculate_distance(robot_pos, intersection)
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance

    return closest_distance


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def line_intersection(p1, p2, p3, p4):
    """Find the intersection point of two line segments."""

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B, C, D = p1, p2, p3, p4
    if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
        denom = (B[1] - A[1]) * (D[0] - C[0]) - (B[0] - A[0]) * (D[1] - C[1])
        if denom == 0:
            return None  # Parallel lines

        ua = ((C[0] - A[0]) * (D[1] - C[1]) - (C[1] - A[1]) * (D[0] - C[0])) / denom
        intersection_x = A[0] + ua * (B[0] - A[0])
        intersection_y = A[1] + ua * (B[1] - A[1])
        return (intersection_x, intersection_y)

    return None


def get_state(distance_ranges):
    """Convert distance ranges to state index."""
    state = sum(range_value * (3 ** i) for i, range_value in enumerate(distance_ranges))
    return min(state, STATE_SIZE - 1)


def choose_action(state):
    """Choose an action using epsilon-greedy strategy."""
    if random.uniform(0, 1) < EPSILON:
        return random.choice(range(ACTION_SIZE))  # Explore
    return np.argmax(Q_TABLE[state])  # Exploit


def update_q_table(state, action, reward, next_state):
    """Update Q-table using the Q-Learning formula."""
    best_next_action = np.max(Q_TABLE[next_state])
    Q_TABLE[state, action] += ALPHA * (reward + GAMMA * best_next_action - Q_TABLE[state, action])


def get_reward(distance_ranges, prev_distance_ranges, action, prev_action, new_pos):
    """Calculate reward based on distance ranges and actions."""

    # Define action rewards
    action_rewards = {0: 0.2, 1: -0.1, 2: -0.1}
    r1 = action_rewards.get(action, 0)

    # Reward for maintaining or increasing distance from obstacles
    distance_change = np.array(distance_ranges) - np.array(prev_distance_ranges)
    r2 = 0.2 if distance_change.sum() >= 0 else -0.2

    # Penalize for transitioning between actions that might lead to undesirable states
    if (prev_action == 1 and action == 2) or (prev_action == 2 and action == 1):
        r3 = -0.8
    else:
        r3 = 0

    # Check for collisions
    if check_collision(new_pos):
        reward = -100  # Significant penalty for collision
    else:
        reward = r1 + r2 + r3

    return reward


def soft_max_selection(Q, state, T):
    random_value = np.random.rand()

    # Boltzmann distribution
    P = np.exp(Q[state, :] / T) / np.sum(np.exp(Q[state, :] / T))

    if np.isnan(P).any():
        a = get_best_action(Q, state)
    else:
        if random_value < P[0]:
            a = 0
        elif P[0] <= random_value < np.sum(P[0:2]):
            a = 1
        else:
            a = 2

    return a


def get_best_action(Q, state):
    # Select the best action a in state s

    status = np.any(Q[state, :])  # Check if there are non-zero values in Q[state,:]

    if not status:
        # Randomly select an action if Q[state,:] contains only zeroes
        a = np.random.randint(0, 3)
    else:
        # Get the action corresponding to the maximum Q-value
        a = np.argmax(Q[state, :])  # +1 to match 1-based indexing like in MATLAB

    return a


def log_data(distance_ranges, prev_distance_ranges, prev_action, action):
    """Log the distance ranges and action to a CSV file with each distance range in a separate column."""
    # Prepare the data with each distance range in a separate column
    data = {
        'Distance_Range_1': [distance_ranges[0]],
        'Distance_Range_2': [distance_ranges[1]],
        'Distance_Range_3': [distance_ranges[2]],
        'Distance_Range_4': [distance_ranges[3]],
        'Distance_Range_5': [distance_ranges[4]],
        'prev_action': [prev_action],
        'prev_distance_range_1': [prev_distance_ranges[0]],
        'prev_distance_range_2': [prev_distance_ranges[1]],
        'prev_distance_range_3': [prev_distance_ranges[2]],
        'prev_distance_range_4': [prev_distance_ranges[3]],
        'prev_distance_range_5': [prev_distance_ranges[4]],
        'Action': [action]
    }
    df = pd.DataFrame(data)
    # Append the data to the CSV file or create a new file if it doesn't exist
    if os.path.exists('log_data.csv'):
        df.to_csv('log_data.csv', mode='a', header=False, index=False)
    else:
        df.to_csv('log_data.csv', index=False)


def main():
    global Q_TABLE, robot_pos, robot_angle, action, distance_ranges
    step = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((125, 125, 125))
        draw_obstacles()
        draw_robot()
        prev_distance_ranges = distance_ranges
        distance_ranges = get_distance_ranges()
        # state = get_state(distance_ranges)
        prev_action = action
        inputs = []
        for _ in distance_ranges:
            inputs.append(_)
        # Append the previous action as a scalar, not an array
        inputs.append(prev_action)  # Extract the scalar value from the array
        #for _ in prev_distance_ranges:
            #inputs.append(_)

        #action = predict(inputs)
        action = predict_tflite(inputs)

        print(inputs)
        move_robot(action)
        # reward = get_reward(distance_ranges, prev_distance_ranges, action, prev_action)
        # next_distance_ranges = get_distance_ranges()
        # next_state = get_state(next_distance_ranges)
        # update_q_table(state, action, reward, next_state)
        # log_data(distance_ranges, prev_distance_ranges, prev_action, action)
        # step += 1

        # Debug information
        # print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        # print(f"Q-Value Update: {Q_TABLE[state, action]}")

        pygame.display.flip()
        clock.tick(30)
        # if reward == -1000:
        # save_q_table()
        # robot_pos = [500, 400]

    # save_q_table()  # Save Q-table to CSV file
    pygame.quit()


if __name__ == '__main__':
    main()
