
# hydamo_validation/__main__.py
"""
Entry point voor running the HyDAMO-validation as module:
    python -m hydamo_validation <directory> [--log-level INFO|DEBUG] [--output-types ...] [--raise-error]
"""

import argparse
import json
from html import parser
from .validator import validator

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m hydamo_validation",
        description="HyDAMO validatie-runner"
    )
    parser.add_argument(
        "directory",
        help="Pad naar de input directory met de te valideren data"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["INFO", "DEBUG"],
        help="Logniveau"
    )
    parser.add_argument(
        "--output-types",
        nargs="+",
        default=["geopackage"],
        help="Uitvoerformaten (bijv. geopackage, csv, geojson)"
    )

    parser.add_argument("--coverages", action="append",
                        help="coverage as KEY=VALUE (repeatable)")

    parser.add_argument(
        "--raise-error",
        action="store_true",
        help="Exceptions doorgeven"
    )

    args = parser.parse_args()

    coverages = {}
    if args.coverages:
        for item in args.coverages:
            try:
                k, v = item.split("=", 1)
            except ValueError:
                raise SystemExit(f"Invalid format for --coverages: {item}. Expected KEY=VALUE.")
            coverages[k] = v

    # Create a callable (partial) to _validator(...)
    run = validator(
        output_types=args.output_types,
        log_level=args.log_level,
        coverages=coverages,
    )

    # Execute the actual validation
    result = run(directory=args.directory, raise_error=args.raise_error)

    print(result)

if __name__ == "__main__":
    main()
