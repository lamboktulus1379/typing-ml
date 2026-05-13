import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"


def _load_shared_generator_module():
    module_path = SRC_PATH / "generate_synthetic_data.py"
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

    spec = importlib.util.spec_from_file_location("seed_shared_generator", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load shared generator module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_shared_generator = _load_shared_generator_module()
build_seed_telemetry_dataframe = _shared_generator.build_seed_telemetry_dataframe
save_dataframe_to_csv = _shared_generator.save_dataframe_to_csv


USER_ID = "9f55a4ee-7be6-4c54-a5c6-bf173ea2ad74"
ROW_COUNT = 50
RNG_SEED = 42


def main() -> None:
    dataframe = build_seed_telemetry_dataframe(
        user_id=USER_ID,
        row_count=ROW_COUNT,
        seed=RNG_SEED,
    )
    output_path = (PROJECT_ROOT / "../Data/seed_telemetry.csv").resolve()
    save_dataframe_to_csv(dataframe, str(output_path))
    print(f"Generated {len(dataframe)} rows at {output_path}")


if __name__ == "__main__":
    main()
