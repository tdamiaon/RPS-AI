import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import get_custom_objects, register_keras_serializable
from tensorflow.keras.preprocessing.sequence import pad_sequences 

@register_keras_serializable()
class CustomMasking(keras.layers.Layer):
    def __init__(self, mask_value=0.0, **kwargs):
        super(CustomMasking, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs, **kwargs):
        mask = tf.math.not_equal(inputs, self.mask_value)
        return inputs, mask

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, self.mask_value)

    def get_config(self):
        config = super(CustomMasking, self).get_config()
        config.update({'mask_value': self.mask_value})
        return config

@register_keras_serializable()
class CustomNotEqual(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomNotEqual, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        a, b = inputs
        return tf.math.not_equal(a, b)

    def get_config(self):
        return super(CustomNotEqual, self).get_config()

# Mapping for moves: 's' for scissors, 'p' for paper, 'x' for rock
move_to_n = {
    's': 1,
    'p': 2,
    'x': 3
}

# Convert raw data into a list of games. Each game is a list of rounds,
# where each round is represented as a numpy array [player1_move, player2_move].
def convert_data(file_obj):
    result = []
    game = []
    for row in file_obj:
        row = row.strip()
        if row == '-':
            if len(game) > 2:
                # Store game if it has more than 2 rounds
                result.append(game)
            game = []
            continue
        moves = []
        for move in row:
            moves.append(move_to_n[move])
        if len(moves) > 1:
            game.append(np.array(moves))
    return result  # Return list of games

with open("data.txt", 'r') as f:
    data = convert_data(f)

# Create a mirrored version of the games for player2 by flipping each round's moves
data2 = [[np.flip(move_arr) for move_arr in game] for game in data]
# Concatenate the two datasets: data2 (player2 perspective) + data (player1 perspective)
data = data2 + data

# Calculate the outcome of a round from player2's perspective using match-case.
def calculate_winner(x):
    """
    Returns:
      1 if player2 wins,
      0 if it's a tie,
     -1 if player2 loses.
    Rules:
      - Rock ('x' or 3) beats Scissors ('s' or 1)
      - Scissors ('s' or 1) beats Paper ('p' or 2)
      - Paper ('p' or 2) beats Rock ('x' or 3)
    """
    match (x[0], x[1]):
        case (1, 3):  # player1: scissors, player2: rock
            return 1
        case (2, 1):  # player1: paper, player2: scissors
            return 1
        case (3, 2):  # player1: rock, player2: paper
            return 1
        case _:
            if x[0] == x[1]:
                return 0  # tie
            else:
                return -1  # player2 loses

# Create X and y datasets from the game data.
def create_xy_winner(data):
    """
    For each game, X consists of rounds (excluding the last round)
    and y consists of:
      - the next round's player2 move (to be predicted as one-hot),
      - the accumulated score until that round.
    """
    result_x = []
    result_y = []
    for game in data:
        game_y = []
        score = 0
        for i, moves in enumerate(game):
            if i + 1 == len(game):
                result_x.append(game[:-1])  # X: all rounds except the last one
                result_y.append(game_y)       # y: list of next moves and accumulated scores
                break
            score += calculate_winner(game[i])
            # Use player2's move from the next round (index 1) as the target.
            game_y.append([game[i+1][1], score])
    return result_x, result_y

X, y = create_xy_winner(data)

# Prepare data for training
def prepare_training_data(X, y):
    X_list = []
    y_move_list = []
    y_score_list = []
    
    for game_x, game_y in zip(X, y):
        # Convert moves to one-hot representation
        X_game = keras.utils.to_categorical(game_x, num_classes=4).reshape(-1, 8)
        X_list.append(X_game)
        
        # Prepare move predictions (one-hot encoded)
        y_move_game = keras.utils.to_categorical([move[0] for move in game_y], num_classes=4)
        y_move_list.append(y_move_game)
        
        # Prepare score predictions
        y_score_game = np.array([move[1] for move in game_y]).reshape(-1, 1)
        y_score_list.append(y_score_game)
    
    # Determine the maximum number of timesteps (rounds)
    max_timesteps = max([seq.shape[0] for seq in X_list]) if X_list else 0
    
    # Pad sequences to max_timesteps
    X_padded = pad_sequences(
        X_list, 
        maxlen=max_timesteps,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    
    y_move_padded = pad_sequences(
        y_move_list,
        maxlen=max_timesteps,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    
    y_score_padded = pad_sequences(
        y_score_list,
        maxlen=max_timesteps,
        dtype='float32',
        padding='post',
        truncating='post',
        value=0.0
    )
    
    return X_padded, [y_move_padded, y_score_padded]

# Prepare the training data
X_train, y_train = prepare_training_data(X, y)

# Model architecture
model = keras.Sequential([
    keras.layers.Input(shape=(None, 8)),
    keras.layers.Masking(mask_value=0.0),
    keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='softmax', name='move_output'),
    keras.layers.Dense(1, activation='linear', name='score_output')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'move_output': 'categorical_crossentropy', 
        'score_output': 'mse'
    },
    loss_weights={'move_output': 1.0, 'score_output': 0.2},
    metrics={
        'move_output': 'accuracy', 
        'score_output': 'mse'
    }
)

# Train the model
model.fit(
    X_train, 
    {
        'move_output': y_train[0], 
        'score_output': y_train[1]
    }, 
    epochs=10,  # Increased epochs for better learning
    batch_size=32,  # Increased batch size
    verbose=1
)

# Save the model
model.save("RPS_AI.keras")