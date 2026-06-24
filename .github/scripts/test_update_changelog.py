"""Tests for .github/scripts/update_changelog.py."""

from __future__ import annotations

import importlib.util
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = Path(__file__).with_name("update_changelog.py")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "update_changelog",
        _SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


update_changelog = _load_module()


class TestParseVersion(unittest.TestCase):
    def test_parses_single_quoted_version(self) -> None:
        text = '__version__ = "1.2.3"\n'
        self.assertEqual(update_changelog._parse_version(text), "1.2.3")

    def test_parses_single_quoted_version_char(self) -> None:
        text = "__version__ = '4.5.6'\n"
        self.assertEqual(update_changelog._parse_version(text), "4.5.6")

    def test_raises_when_version_missing(self) -> None:
        with self.assertRaises(ValueError):
            update_changelog._parse_version("# no version here\n")


class TestBuildPrompt(unittest.TestCase):
    def test_unreleased_mode_when_versions_match(self) -> None:
        prompt, is_release = update_changelog._build_prompt(
            "abc123",
            "feat: add solver option",
            "icrn/solver.py | 10 +++++",
            current_version="0.5.0",
            previous_version="0.5.0",
        )
        self.assertFalse(is_release)
        self.assertIn("Ensure an [Unreleased] section exists", prompt)
        self.assertIn("Commit: abc123", prompt)

    def test_release_mode_when_version_bumped(self) -> None:
        prompt, is_release = update_changelog._build_prompt(
            "def456",
            "chore: release 0.6.0",
            "icrn/__init__.py | 1 +-",
            current_version="0.6.0",
            previous_version="0.5.0",
        )
        self.assertTrue(is_release)
        self.assertIn("bumped from 0.5.0 to\n0.6.0", prompt)
        self.assertIn("## [0.6.0] -", prompt)

    def test_unreleased_mode_when_no_previous_version(self) -> None:
        _, is_release = update_changelog._build_prompt(
            "abc123",
            "initial",
            "",
            current_version="0.1.0",
            previous_version=None,
        )
        self.assertFalse(is_release)


class TestDryRunEnabled(unittest.TestCase):
    def test_cli_flag_enables_dry_run(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(update_changelog._dry_run_enabled(cli_flag=True))

    def test_env_truthy_values_enable_dry_run(self) -> None:
        for value in ("1", "true", "YES"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ, {"DRY_RUN": value}, clear=True
                ):
                    self.assertTrue(
                        update_changelog._dry_run_enabled(cli_flag=False)
                    )

    def test_disabled_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(update_changelog._dry_run_enabled(cli_flag=False))


class TestWriteGithubOutput(unittest.TestCase):
    def test_writes_release_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output"
            with mock.patch.dict(
                os.environ,
                {"GITHUB_OUTPUT": str(output_path)},
                clear=True,
            ):
                update_changelog._write_github_output(True, "0.6.0")
            content = output_path.read_text()
            self.assertIn("is_release=true\n", content)
            self.assertIn("version=0.6.0\n", content)

    def test_writes_unreleased_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output"
            with mock.patch.dict(
                os.environ,
                {"GITHUB_OUTPUT": str(output_path)},
                clear=True,
            ):
                update_changelog._write_github_output(False, "0.5.0")
            content = output_path.read_text()
            self.assertIn("is_release=false\n", content)
            self.assertIn("version=0.5.0\n", content)


class TestAssertChangelogOnlyChanges(unittest.TestCase):
    def test_passes_when_only_changelog_changed(self) -> None:
        with mock.patch.object(
            update_changelog,
            "_changed_paths",
            return_value=["CHANGELOG.md"],
        ):
            update_changelog._assert_changelog_only_changes()

    def test_passes_when_nothing_changed(self) -> None:
        with mock.patch.object(
            update_changelog, "_changed_paths", return_value=[]
        ):
            update_changelog._assert_changelog_only_changes()

    def test_raises_when_other_files_changed(self) -> None:
        with mock.patch.object(
            update_changelog,
            "_changed_paths",
            return_value=["CHANGELOG.md", "icrn/solver.py"],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "other than CHANGELOG.md",
            ):
                update_changelog._assert_changelog_only_changes()

    def test_verify_flag_exits_nonzero_on_unexpected_changes(self) -> None:
        with mock.patch.object(
            update_changelog,
            "_changed_paths",
            return_value=["README.md"],
        ):
            exit_code = update_changelog.main(["--verify-changes-only"])
        self.assertEqual(exit_code, 3)


class TestMainDryRun(unittest.TestCase):
    def setUp(self) -> None:
        self._old_cwd = Path.cwd()
        os.chdir(_REPO_ROOT)

    def tearDown(self) -> None:
        os.chdir(self._old_cwd)

    def test_dry_run_exits_zero_without_api_key(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        env = {k: v for k, v in os.environ.items() if k != "CURSOR_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = update_changelog.main(["--dry-run"])
        self.assertEqual(exit_code, 0)
        self.assertIn("=== changelog bot dry run ===", stderr.getvalue())
        self.assertIn("--- prompt ---", stdout.getvalue())
        self.assertIn("Update CHANGELOG.md", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
