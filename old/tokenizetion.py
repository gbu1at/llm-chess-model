import chess
import torch


def final_board_position(moves_string: str):
    board = chess.Board()

    moves = moves_string.split()

    for move in moves:
        board.push_san(move)

    return "".join(board.__str__().split())


def get_token_by_moves(moves_string: str):
    board = final_board_position(moves_string)

    tensor_list = [0] * 64

    for i in range(64):
        if board[i] == ".":
            tensor_list[i] = 0
        elif board[i].isupper():
            tensor_list[i] = 1
        else:
            tensor_list[i] = 2

    tensor = torch.tensor(tensor_list)

    return tensor


def custom_tokenizer(moves):
    board_tensor = get_token_by_moves(moves)
    return torch.tensor(board_tensor, dtype=torch.long)


def tokenize_board_function(examples):
    inputs = [custom_tokenizer(move) for move in examples["input"]]
    targets = [custom_tokenizer(move) for move in examples["target"]]

    inputs_tensor = torch.stack(inputs)
    targets_tensor = torch.stack(targets)

    result = {
        "input_ids": inputs_tensor,
        "attention_mask": torch.ones_like(inputs_tensor),
        "labels": targets_tensor,
    }

    return result


if __name__ == "__main__":
    examples = {
        "input": ["e4 e6 d4 d5 Nd2"],
        "target": ["e4 e6 d4 d5 Nd2 Nc6"]
    }

    tokenized_output = tokenize_board_function(examples)
    print(tokenized_output)