# %%
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import json
from hydamo_validation import validator
import os
import shutil

import hydamo_validation

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

API_VERSION = "0.0"
SCHEMAS_PATH = Path(hydamo_validation.__file__).parent / "schemas"


@dataclass
class User:
    name: str = "test_user"


@dataclass
class Task:
    id: int
    format: str = "geopackage,csv,geojson"


@dataclass
class TaskHelper:
    data_folder: str = DATA_DIR.as_posix()


TaskHelper = TaskHelper()
SCHEMAS_TO_PATH = Path(f"{TaskHelper.data_folder}/schemas")


def execute_task_by_id_and_format(
    self, user: User, task: Task, api_version: str
) -> dict:
    result = {}
    try:
        data = {
            "api_version": api_version,
            "user_name": user.name,
            "task_started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_finished": "",
        }

        directory = f"{TaskHelper.data_folder}/{task.id}"
        covarage = f"{TaskHelper.data_folder}/dtm"
        schemas = f"{TaskHelper.data_folder}/schemas"

        hydamo_validator = validator(
            output_types=task.format.split(","),
            coverages={"AHN": covarage},
            schemas_path=Path(schemas),
        )

        # start validate
        datamodel, layer_summary, result_summary = hydamo_validator(
            directory=directory, raise_error=False
        )

        result = result_summary.to_dict()

        # end time of validation
        data["task_finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # append api information to the validation result
        results_file = f"{directory}/results/validation_result.json"
        if os.path.exists(results_file):
            with open(results_file, "r+") as file:
                result_data = json.load(file)
                data.update(result_data)

            with open(results_file, "w") as file:
                json.dump(data, file, indent=4)

    except Exception as e:
        result = {"Exception": str(e)}

    with open(f"{directory}/results/result.json", "w") as f:
        f.write(str(result))

    return result


def test_run_productie():
    if not SCHEMAS_TO_PATH.exists():
        shutil.copytree(SCHEMAS_PATH, f"{TaskHelper.data_folder}/schemas")
    user = User()
    task = Task(1877)

    result = execute_task_by_id_and_format(
        self=None, user=user, task=task, api_version=API_VERSION
    )

    assert result["success"]
