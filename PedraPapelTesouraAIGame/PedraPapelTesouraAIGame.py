import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import random

@register_keras_serializable()
class CustomMasking(keras.layers.Layer):
    def __init__(self, mask_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mask_value = mask_value
        self.supports_masking = True

    def call(self, inputs):
        return inputs

    def compute_mask(self, inputs, mask=None):
        mask = keras.backend.not_equal(inputs, self.mask_value)
        mask = keras.backend.any(mask, axis=-1) 
        return mask

    def get_config(self):
        return {'mask_value': self.mask_value, **super().get_config()}

move_to_n = {'s': 0, 'p': 1, 'r': 2}
n_to_move = {0: 's', 1: 'p', 2: 'r'}
move_names = {0: 'Scissors', 1: 'Paper', 2: 'Rock'}

def convert_data(file_obj):
    result = []
    game = []
    for row in file_obj:
        row = row.strip()
        if row == '-':
            if len(game) >= 3:
                result.append(game)
            game = []
            continue
        moves = [move_to_n[c] for c in row if c in move_to_n]
        if len(moves) == 2:
            game.append(np.array(moves))
    return result

# Calculate winner and create_xy_winner remain unchanged
def calculate_winner(x):
    p1, p2 = x[0], x[1]
    if (p1 == 0 and p2 == 2) or (p1 == 1 and p2 == 0) or (p1 == 2 and p2 == 1):
        return -1
    elif p1 == p2:
        return 0
    else:
        return 1

def create_xy_winner(data):
    result_x, result_y = [], []
    for game in data:
        game_y = []
        score = 0
        for i in range(len(game)-1):
            score += calculate_winner(game[i])
            next_move = game[i+1][1]
            game_y.append([next_move, score])
        result_x.append(game[:-1])
        result_y.append(game_y)
    return result_x, result_y

def prepare_training_data(X, y):
    X_list, y_move, y_score = [], [], []
    for game_x, game_y in zip(X, y):
        X_game = np.array([
            np.concatenate([tf.one_hot(round[0], 3), tf.one_hot(round[1], 3)])
            for round in game_x
        ])
        X_list.append(X_game)
        y_move.append(tf.one_hot([m[0] for m in game_y], 3))
        y_score.append(np.array([m[1] for m in game_y])[:, np.newaxis])
    max_len = max(len(seq) for seq in X_list)
    X_pad = pad_sequences(X_list, maxlen=max_len, dtype='float32', padding='post', value=0.0)
    y_move_pad = pad_sequences(y_move, maxlen=max_len, dtype='float32', padding='post', value=0.0)
    y_score_pad = pad_sequences(y_score, maxlen=max_len, dtype='float32', padding='post', value=0.0)
    return X_pad, (y_move_pad, y_score_pad)
def save_game(game_history):
    with open("data.txt", "a") as f:
        for round in game_history:
            f.write(f"{n_to_move[round[0]]}{n_to_move[round[1]]}\n")  
        f.write("-\n")  
# Define masked MSE loss
def masked_mse(y_true, y_pred):
    # Create a mask where y_true is not 0
    mask = tf.cast(tf.math.not_equal(y_true, 0.0), y_true.dtype)
    # Apply the mask to both true and predicted values
    masked_y_true = y_true * mask
    masked_y_pred = y_pred * mask
    return tf.reduce_mean(tf.square(masked_y_pred - masked_y_true))

def create_and_train_model():
    print("Model not found. Creating and training a new model...")
    with open("data.txt", 'r') as f:
        data = convert_data(f)
    data2 = [[np.flip(round) for round in game] for game in data]
    data = data + data2
    X, y = create_xy_winner(data)
    X_train, y_train = prepare_training_data(X, y)
    inputs = keras.Input(shape=(None, 6))
    x = keras.layers.Masking(mask_value=0.0)(inputs)
    lstm_out = keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    dense = keras.layers.Dense(32, activation='relu')(lstm_out) #Relu funcao ativacao que se for negativa passa a 0 se positiva mantem. Masi eficianete que sigmoid
    move_out = keras.layers.Dense(3, activation='softmax', name='move')(dense)
    score_out = keras.layers.Dense(1, name='score')(lstm_out)

    model = keras.Model(inputs, [move_out, score_out])

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss={'move': 'categorical_crossentropy', 'score': masked_mse},
        loss_weights=[1.0, 0.2],
        metrics={'move': 'accuracy'}
    )

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    model.save("RPS_AI.keras")
    return model

def play_game():
    points=[0,0]
    model_path = "RPS_AI.keras"
    
    if not os.path.exists(model_path):
        model = create_and_train_model()
    else:
        print("Loading existing model...")
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'CustomMasking': CustomMasking,
                'masked_mse': masked_mse
            }
        )
    game_history = []
    score = 0
    
    print("Enter 's' for Scissors, 'p' for Paper, 'r' for Rock, or 'q' to quit")
    
    while True:
        player_move_str = input("\nYour move: ").lower()
        if player_move_str == 'q':
            break
        if player_move_str not in move_to_n:
            print("Invalid move! Please enter 's', 'p', 'r', or 'q'")
            continue
        
        player_move = move_to_n[player_move_str]
        if len(game_history) > 0:
            X_game = np.array([
                np.concatenate([tf.one_hot(round[0], 3), tf.one_hot(round[1], 3)]) #onehot encoding (1,2)=[0,1];(2,6)=[0,0,1,0,0,0]
                for round in game_history
            ])
            X_game = X_game.reshape(1, len(game_history), 6) 
            move_pred, score_pred = model.predict(X_game, verbose=0)
            ai_move_probs = move_pred[0, -1]
            ai_move = np.argmax(ai_move_probs)
            if random.random() < 0.1: 
                ai_move = random.randint(0, 2)
        else:
            ai_move = random.randint(0, 2)
        winning_moves = {0: 2, 1: 0, 2: 1} 
        ai_move = winning_moves[ai_move]

        round_result = calculate_winner([player_move, ai_move])
        score += round_result
        print(f"You played: {move_names[player_move]}")
        print(f"AI played: {move_names[ai_move]}")
        
        if round_result == 1:
            points[1]+=1
            print("You win this round!")
        elif round_result == 0:
            print("It's a tie!")
        else:
            points[1]+=1
            points[0]+=1
            print("AI wins this round!")
        
        percentage_ai_wins = (points[0] / points[1] * 100) if points[1] > 0 else 0
        print(f"Current score: {score}")
        print(f"Percentage AI Wins: {percentage_ai_wins:.2f}%")
        game_history.append([player_move, ai_move])
    
    print("\nFinal score:", score)
    if score > 0:
        print("Congratulations! You won the game!")
    elif score < 0:
        print("AI won the game")
    else:
        print("The game ended in a tie!")
    save_game(game_history)

if __name__ == "__main__":
    play_game()