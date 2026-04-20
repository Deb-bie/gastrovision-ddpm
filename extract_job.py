#!/usr/bin/env python3
"""
scripts/extract_job.py
Extracts a single named Job from a multi-document YAML file and prints it
to stdout so the Makefile can pipe it to kubectl create -f -.

Usage:
    python3 scripts/extract_job.py <job-name> <yaml-file>

Example:
    python3 scripts/extract_job.py gastro-diffusion jobs/nautilus_jobs.yaml \
        | kubectl create -f - -n YOUR_NAMESPACE
"""
import sys
import yaml   # PyYAML — comes with Python standard installs


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <job-name> <yaml-file>", file=sys.stderr)
        sys.exit(1)

    job_name = sys.argv[1]
    yaml_path = sys.argv[2]

    with open(yaml_path) as f:
        docs = list(yaml.safe_load_all(f))

    matched = [
        d for d in docs
        if d
        and d.get("kind") == "Job"
        and d.get("metadata", {}).get("name") == job_name
    ]

    if not matched:
        available = [
            d.get("metadata", {}).get("name")
            for d in docs
            if d and d.get("kind") == "Job"
        ]
        print(
            f"ERROR: job '{job_name}' not found in {yaml_path}.\n"
            f"Available jobs: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(yaml.dump(matched[0], default_flow_style=False))


if __name__ == "__main__":
    main()
