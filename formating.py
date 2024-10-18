import sys
import re
from pprint import pprint


def formating_delete_spaces(text:str):
    text_without_space = re.sub(r'[ \t]+', '', text)
    return text_without_space

def formating_extract_chess_games(text:str):
    game_pattern = re.compile(r'1\..*?(?:0-1|1-0|1/2-1/2)', re.DOTALL)
    games = game_pattern.findall(text)
    cleaned_games = [game.replace('\n', '').strip() for game in games]
    cleaned_games = "\n".join(cleaned_games)
    return cleaned_games


def formating_delete_info(text:str):
    pattens = [  
                "(\[Event.*\])",
                "(\[Site.*\])",
                "(\[Date.*\])",
                "(\[Round.*\])",
                "(\[White.*\])",
                "(\[Black.*\])",
                "(\[Result.*\])",
                "(\[ECO.*\])"
            ]
    pattern = "|".join(pattens)
    new_text = re.sub(pattern, '', text)
    return new_text


def formating_delete_result(text:str):
    pattens = [  
                "(0-1)",
                "(1/2-1/2)",
                "(1-0)"
            ]
    pattern = "|".join(pattens)
    new_text = re.sub(pattern, '', text)
    return new_text


def formating_delete_number_of_move(text:str):
    pattern_more_one = "([2-9]|\d{2,})\."
    new_text = re.sub(pattern_more_one, ' ', text)
    new_text = re.sub("1\.", '', new_text)
    return new_text.strip()

def formating_delete_double_spaces(text:str):
    new_text = re.sub("  ", ' ', text)
    return new_text.strip()

def write_new_base(text:str, v:int):
    assert(v < 100 and v > -1)
    version = str(v) if v >= 10 else f"0{v}"
    file_path_out = "/Users/apple/Desktop/cource/project/BASE/" + base_name + version
    with open(file_path_out, "w") as f:
        f.write(text)


if __name__ == "__main__":
    _, base_name = sys.argv
    base_name = base_name.upper()
    file_path_in = "/Users/apple/Desktop/cource/project/BASE/" + base_name

    data_base = []

    with open(file_path_in, "r") as f:
        text = f.read()

    # text_0 = formating_delete_spaces(text)
    text_1 = formating_delete_info(text)
    text_2 = formating_extract_chess_games(text_1)
    text_3 = formating_delete_result(text_2)
    text_4 = formating_delete_number_of_move(text_3)
    text_5 = formating_delete_double_spaces(text_4)

    write_new_base(text_1, 1)
    write_new_base(text_2, 2)
    write_new_base(text_3, 3)
    write_new_base(text_4, 4)
    write_new_base(text_5, 5)
