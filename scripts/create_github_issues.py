#!/usr/bin/env python3
"""Create GitHub issues for the cvol algorithm backlog.

Usage:
  python scripts/create_github_issues.py --repo owner/name --token $GITHUB_TOKEN
  python scripts/create_github_issues.py --repo owner/name --token $GITHUB_TOKEN --dry-run

If --repo or --token are omitted, the script falls back to GITHUB_REPOSITORY and
GITHUB_TOKEN environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ISSUES_FILE = pathlib.Path("docs/issues/github_issues.json")


def load_issues() -> list[dict[str, str]]:
    data = json.loads(ISSUES_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("github_issues.json must be a list")
    for item in data:
        if not isinstance(item, dict) or "title" not in item or "body" not in item:
            raise ValueError("Each issue must contain title and body")
    return data


def create_issue(repo: str, token: str, title: str, body: str, dry_run: bool) -> str:
    if dry_run:
        return "DRY_RUN"

    url = f"https://api.github.com/repos/{repo}/issues"
    payload = json.dumps({"title": title, "body": body}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        response = json.loads(resp.read().decode("utf-8"))
    return response["html_url"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    issues = load_issues()
    if not args.repo:
        print("error: --repo or GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2
    if not args.token and not args.dry_run:
        print("error: --token or GITHUB_TOKEN is required unless --dry-run is set", file=sys.stderr)
        return 2

    created_urls: list[str] = []
    for i, issue in enumerate(issues, start=1):
        try:
            url = create_issue(args.repo, args.token, issue["title"], issue["body"], args.dry_run)
        except urllib.error.HTTPError as exc:
            print(f"[{i}] FAILED {issue['title']}: {exc.code} {exc.reason}", file=sys.stderr)
            return 1
        print(f"[{i}] {issue['title']} -> {url}")
        created_urls.append(url)

    if args.dry_run:
        print(f"Dry run complete. Would create {len(created_urls)} issues.")
    else:
        print(f"Successfully created {len(created_urls)} issues.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
