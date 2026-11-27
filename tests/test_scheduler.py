"""Tests for the scheduler's command handling."""

from unittest.mock import MagicMock, patch

import pytest

from makim.scheduler import Config, _sanitize_command, run_makim_task


@pytest.mark.parametrize(
    'cmd',
    [
        ['makim', '--file', 'config.yaml', 'build'],
        ['makim', '--file', 'my config.yaml', 'build'],
        ['makim', '--task', "it's fine"],
        ['makim', '--message', 'a;b && echo pwned'],
    ],
)
def test_sanitize_command_preserves_argv(cmd):
    """Argv passed to subprocess.run must not be shell-quoted.

    shlex.quote only applies to shell command strings. When the result is
    handed to subprocess.run as a list, the quotes become literal characters
    in the argument and the invoked program receives a different value.
    """
    assert _sanitize_command(cmd) == cmd


def test_run_makim_task_passes_unquoted_args(tmp_path):
    """run_makim_task must forward the original argv to subprocess.run."""
    config_file = tmp_path / 'my config.yaml'
    config_file.write_text('tasks: {}')
    Config.config_file = str(config_file)
    Config.job_history_path = tmp_path / 'history.json'

    result = MagicMock()
    result.stdout = 'ok'

    with patch('makim.scheduler.subprocess.run', return_value=result) as run:
        run_makim_task('build', 'test-scheduler', {'msg': 'hello world'})

    assert run.call_args.args[0] == [
        'makim',
        '--file',
        str(config_file),
        'build',
        '--msg',
        'hello world',
    ]
