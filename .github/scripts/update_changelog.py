#!/usr/bin/env python3
"""Update CHANGELOG.md after a push to main using a Cursor agent."""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import re
import subprocess
import sys

VERSION_RE = re.compile(r'^__version__ = ["\']([^"\']+)["\']', re.M)
INIT_PATH = pathlib.Path("icrn/__init__.py")
_TRUTHY = frozenset({"1", "true", "yes"})


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def _parse_version(text: str) -> str:
    match = VERSION_RE.search(text)
    if not match:
        raise ValueError("Could not parse __version__ from icrn/__init__.py")
    return match.group(1)


def _current_version() -> str:
    return _parse_version(INIT_PATH.read_text())


def _previous_version() -> str | None:
    try:
        text = _run("git", "show", "HEAD~1:icrn/__init__.py")
    except subprocess.CalledProcessError:
        return None
    return _parse_version(text)


def _write_github_output(is_release: bool, version: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"is_release={'true' if is_release else 'false'}\n")
        handle.write(f"version={version}\n")


def _dry_run_enabled(*, cli_flag: bool) -> bool:
    env = os.environ.get("DRY_RUN", "").strip().lower()
    return cli_flag or env in _TRUTHY


def _unreleased_prompt(sha: str, commit_message: str, diff_stat: str) -> str:
    return f"""Update CHANGELOG.md for the icrn Python package.

The file follows Keep a Changelog (https://keepachangelog.com/en/1.1.0/).
Ensure an [Unreleased] section exists after the introductory paragraph.
Add a concise bullet under the appropriate category (Added, Changed, Fixed,
Removed, Deprecated, Security) summarizing user-facing changes from the commit
below. Skip CI-only, formatting-only, or other non-user-facing changes; in that
case leave CHANGELOG.md unchanged.

Do not remove existing entries. Do not edit files other than CHANGELOG.md.

Commit: {sha}
Commit message:
{commit_message}

Diff stat:
{diff_stat}
"""


def _release_prompt(
    sha: str,
    commit_message: str,
    diff_stat: str,
    *,
    previous_version: str,
    current_version: str,
    release_date: str,
) -> str:
    return f"""Update CHANGELOG.md for the icrn Python package.

A new release was detected: __version__ bumped from {previous_version} to
{current_version} in icrn/__init__.py.

Perform the Keep a Changelog release ritual:
1. If the current commit includes user-facing changes not yet listed under
   [Unreleased], add concise bullets for them first under the appropriate
   Added/Changed/Fixed/Removed/Deprecated/Security subheadings.
2. If ## [{current_version}] already exists, do not create a duplicate version
   section. Ensure [Unreleased] exists and only add any missing bullets from
   this commit.
3. Otherwise, rename the [Unreleased] section heading to
   ## [{current_version}] - {release_date}.
4. Insert a new empty ## [Unreleased] section immediately after the introductory
   paragraph and before all dated version sections.

Preserve all existing dated version sections and their order (newest first after
[Unreleased]). Do not edit files other than CHANGELOG.md.

Commit: {sha}
Commit message:
{commit_message}

Diff stat:
{diff_stat}
"""


def _build_prompt(
    sha: str,
    commit_message: str,
    diff_stat: str,
    *,
    current_version: str,
    previous_version: str | None,
) -> tuple[str, bool]:
    is_release = (
        previous_version is not None and previous_version != current_version
    )
    if is_release:
        prompt = _release_prompt(
            sha,
            commit_message,
            diff_stat,
            previous_version=previous_version,
            current_version=current_version,
            release_date=datetime.date.today().isoformat(),
        )
    else:
        prompt = _unreleased_prompt(sha, commit_message, diff_stat)
    return prompt, is_release


def _print_dry_run(
    *,
    sha: str,
    commit_message: str,
    diff_stat: str,
    current_version: str,
    previous_version: str | None,
    is_release: bool,
    prompt: str,
) -> None:
    mode = "release" if is_release else "unreleased"
    print("=== changelog bot dry run ===", file=sys.stderr)
    print(f"commit: {sha}", file=sys.stderr)
    print(f"current_version: {current_version}", file=sys.stderr)
    print(f"previous_version: {previous_version}", file=sys.stderr)
    print(f"is_release: {str(is_release).lower()}", file=sys.stderr)
    print(f"mode: {mode}", file=sys.stderr)
    if is_release:
        print(
            f"release_date: {datetime.date.today().isoformat()}",
            file=sys.stderr,
        )
    print(file=sys.stderr)
    print("commit message:", file=sys.stderr)
    print(commit_message, file=sys.stderr)
    print(file=sys.stderr)
    print("diff stat:", file=sys.stderr)
    print(diff_stat or "(empty)", file=sys.stderr)
    print(file=sys.stderr)
    print("--- prompt ---")
    print(prompt)


def _changed_paths() -> list[str]:
    modified = [
        line for line in _run("git", "diff", "--name-only").splitlines() if line
    ]
    staged = [
        line
        for line in _run("git", "diff", "--cached", "--name-only").splitlines()
        if line
    ]
    untracked = [
        line
        for line in _run(
            "git", "ls-files", "--others", "--exclude-standard"
        ).splitlines()
        if line
    ]
    return sorted(set(modified + staged + untracked))


def _assert_changelog_only_changes() -> None:
    """Raise if the working tree changed anywhere except CHANGELOG.md."""
    unexpected = [path for path in _changed_paths() if path != "CHANGELOG.md"]
    if unexpected:
        raise RuntimeError(
            "Changelog bot modified files other than CHANGELOG.md: "
            + ", ".join(unexpected)
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update CHANGELOG.md using a Cursor agent.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print detection metadata and agent prompt without calling Cursor."
        ),
    )
    parser.add_argument(
        "--verify-changes-only",
        action="store_true",
        help="Exit non-zero if any file other than CHANGELOG.md changed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dry_run = _dry_run_enabled(cli_flag=args.dry_run)

    if args.verify_changes_only:
        try:
            _assert_changelog_only_changes()
        except RuntimeError as err:
            print(str(err), file=sys.stderr)
            return 3
        return 0

    sha = _run("git", "rev-parse", "HEAD")
    commit_message = _run("git", "log", "-1", "--format=%B")
    diff_stat = _run("git", "show", "--stat", "--format=", "HEAD")

    current_version = _current_version()
    previous_version = _previous_version()
    prompt, is_release = _build_prompt(
        sha,
        commit_message,
        diff_stat,
        current_version=current_version,
        previous_version=previous_version,
    )
    _write_github_output(is_release, current_version)

    if dry_run:
        _print_dry_run(
            sha=sha,
            commit_message=commit_message,
            diff_stat=diff_stat,
            current_version=current_version,
            previous_version=previous_version,
            is_release=is_release,
            prompt=prompt,
        )
        return 0

    api_key = os.environ.get("CURSOR_API_KEY")
    if not api_key:
        print("CURSOR_API_KEY is not set", file=sys.stderr)
        return 1

    from cursor_sdk import (
        Agent,
        AgentOptions,
        CursorAgentError,
        LocalAgentOptions,
    )

    if is_release:
        print(
            f"Release detected: {previous_version} -> {current_version}",
            file=sys.stderr,
        )

    try:
        result = Agent.prompt(
            prompt,
            AgentOptions(
                api_key=api_key,
                model="composer-2.5",
                local=LocalAgentOptions(
                    cwd=os.getcwd(),
                    setting_sources=["project"],
                ),
            ),
        )
    except CursorAgentError as err:
        print(f"Agent startup failed: {err.message}", file=sys.stderr)
        return 1

    if result.status == "error":
        print(f"Agent run failed: {result.id}", file=sys.stderr)
        return 2

    try:
        _assert_changelog_only_changes()
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 3

    print(result.result or "Agent finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
