# hydamo_validation/__main__.py
"""
Start de HyDAMO-validatie met hydamo-validate of python -m hydamo_validation.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from hydamo_validation.validator import OUTPUT_TYPES, validator


def main() -> int:
    """Voer de validatie uit en retourneer de technische exitcode (0 of 1)."""
    parser = argparse.ArgumentParser(
        description="Valideer HyDAMO-data in een projectmap.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Structuur van `directory`:

directory/
├── datasets/
│   └── hydamo.gpkg
└── validationrules.json

Plaats een of meer GeoPackages in datasets/.
De validatieresultaten worden geschreven naar directory/results/.
""",
    )
    parser.add_argument(
        "directory",
        help="Projectmap met de submap datasets en het bestand validationrules.json",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["INFO", "DEBUG"],
        help="Detailniveau van de logging (standaard: INFO)",
    )
    parser.add_argument(
        "--output-types",
        nargs="+",
        default=["geopackage"],
        choices=OUTPUT_TYPES,
        help="Een of meer uitvoerformaten (standaard: geopackage)",
    )

    parser.add_argument(
        "--coverages",
        action="append",
        metavar="KEY=VALUE",
        help="Locatie van rasterdata, bijvoorbeeld AHN=C:/data/ahn (herhaalbaar)",
    )

    parser.add_argument(
        "--raise-error",
        action="store_true",
        help="Geef technische fouten door met een volledige traceback",
    )

    args = parser.parse_args()

    coverages = {}
    if args.coverages:
        for item in args.coverages:
            k, separator, v = item.partition("=")
            if not separator or not k.strip() or not v.strip():
                parser.error(
                    f"ongeldige --coverages {item!r}: verwacht KEY=VALUE "
                    "met een niet-lege sleutel en waarde"
                )
            coverages[k] = v

    directory = Path(args.directory).resolve()
    results_path = directory / "results"
    started = perf_counter()
    errors = []
    exception = None
    try:
        run = validator(
            output_types=args.output_types,
            log_level=args.log_level,
            coverages=coverages,
        )
        result = run(directory=args.directory, raise_error=args.raise_error)

        # De validator retourneert None bij een afgebroken run en registreert
        # technische fouten per regel ook wanneer de run verder kan gaan.
        if result is None:
            errors.append(
                "Validatie afgebroken; controleer de projectmap en de loguitvoer."
            )
            result_file = results_path / "validation_result.json"
            if result_file.is_file():
                summary = json.loads(result_file.read_text(encoding="utf-8"))
                errors.extend(summary.get("error", []))
                errors.extend(summary.get("errors") or [])
        else:
            summary = result[2]
            errors.extend(summary.error)
            errors.extend(summary.errors or [])
            if not summary.success and not errors:
                errors.append("Validatie niet technisch succesvol afgerond.")
    except Exception as error:
        exception = error
        errors.append(f"{type(error).__name__}: {error}")

    print(f"Projectmap: {directory}")
    print(
        f"Status: {'technisch mislukt' if errors else 'technisch succesvol afgerond'}"
    )
    print(f"Doorlooptijd: {perf_counter() - started:.2f} seconden")
    print(f"Resultaten: {results_path}")
    for error in errors:
        print(f"Fout: {error}", file=sys.stderr)
    if args.raise_error and exception is not None:
        raise exception
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
