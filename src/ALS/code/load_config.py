from pathlib import Path
import json


class AttrDict:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __getattr__(self, attr):
        if attr in self.dictionary:
            return self.dictionary[attr]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'"
            )

    def __repr__(self):
        pairs = [f"{key} : {value}" for key, value in self.dictionary.items()]
        return "{" + ", ".join(pairs) + "}"


def load_config() -> AttrDict:
    config_path = Path.cwd() / "code" / "config.json"

    with open(config_path) as f:
        json_file = json.load(f)

    ## 추후 config 덮어 쓰는 동작을 위해.
    args = AttrDict(json_file)

    return args
