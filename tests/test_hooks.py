"""Regression tests for hook execution isolation and state preservation."""

import json
import subprocess
import sys

from pathlib import Path
from typing import Any

import makim
import pytest


def _write_config(tmp_path: Path, yaml_text: str) -> Path:
    """Write a temporary Makim YAML configuration file."""
    cfg_path = tmp_path / '.makim.yaml'
    cfg_path.write_text(yaml_text, encoding='utf-8')
    return cfg_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON event records from a test output file."""
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_sequential_pre_run_hooks_preserve_parent_and_hook_contexts(
    tmp_path: Path,
) -> None:
    """Sequential pre-run hooks keep own scope without corrupting later hooks.

    Validates ordering, parent vars/env preservation, hook-specific vars/env,
    hook arguments, and prevention of env leakage from Hook 1 to Hook 2+.
    """
    events_file = tmp_path / 'events.jsonl'
    cfg = _write_config(
        tmp_path,
        f"""
groups:
  helper:
    vars:
      env_name: "helper_dev"
      group_var: "helper_group_val"
    env:
      HELPER_GROUP_ENV: "helper_group_env"
    tasks:
      hook-one:
        vars:
          hook_var: "from_hook_one"
        env:
          HOOK_OWN_ENV: "hook1_own"
          SHARED_STAGE: "corrupted_from_hook1"
          LEAKED_ENV: "hook1_secret"
        backend: bash
        run: |
          python3 -c '
          import json, os
          rec = {{
              "step": "hook1",
              "parent_only_env": os.environ.get("PARENT_ONLY_ENV", ""),
              "helper_group_env": os.environ.get("HELPER_GROUP_ENV", ""),
              "shared_stage": os.environ.get("SHARED_STAGE", ""),
              "hook_own_env": os.environ.get("HOOK_OWN_ENV", ""),
              "group_var": "${{{{ vars.group_var }}}}",
              "hook_var": "${{{{ vars.hook_var }}}}",
          }}
          with open("{events_file}", "a") as f:
              f.write(json.dumps(rec) + "\\n")
          '
      hook-two:
        vars:
          hook_var: "from_hook_two"
        env:
          HOOK_OWN_ENV: "hook2_own"
        args:
          stage:
            type: string
            help: Parent shared stage passed via hook args
            default: "missing"
        backend: bash
        run: |
          python3 -c '
          import json, os
          rec = {{
              "step": "hook2",
              "parent_only_env": os.environ.get("PARENT_ONLY_ENV", ""),
              "helper_group_env": os.environ.get("HELPER_GROUP_ENV", ""),
              "shared_stage": os.environ.get("SHARED_STAGE", ""),
              "hook_own_env": os.environ.get("HOOK_OWN_ENV", ""),
              "leaked_env": os.environ.get("LEAKED_ENV", "clean"),
              "stage_arg": "${{{{ args.stage }}}}",
              "group_var": "${{{{ vars.group_var }}}}",
              "hook_var": "${{{{ vars.hook_var }}}}",
          }}
          with open("{events_file}", "a") as f:
              f.write(json.dumps(rec) + "\\n")
          '
      hook-three:
        args:
          stage:
            type: string
            help: Parent shared stage passed to third hook
            default: "missing"
        backend: bash
        run: |
          python3 -c '
          import json, os
          rec = {{
              "step": "hook3",
              "parent_only_env": os.environ.get("PARENT_ONLY_ENV", ""),
              "shared_stage": os.environ.get("SHARED_STAGE", ""),
              "stage_arg": "${{{{ args.stage }}}}",
              "leaked_env": os.environ.get("LEAKED_ENV", "clean"),
          }}
          with open("{events_file}", "a") as f:
              f.write(json.dumps(rec) + "\\n")
          '
  deploy:
    vars:
      env_name: "production"
    env:
      PARENT_ONLY_ENV: "prod_group_env"
    tasks:
      ship:
        vars:
          deploy_tier: "primary"
        env:
          SHARED_STAGE: "prod_task_stage"
        hooks:
          pre-run:
            - task: helper.hook-one
            - task: helper.hook-two
              args:
                stage: "${{{{ env.SHARED_STAGE }}}}"
            - task: helper.hook-three
              args:
                stage: "${{{{ env.SHARED_STAGE }}}}"
        backend: bash
        run: |
          python3 -c '
          import json, os
          rec = {{
              "step": "parent",
              "parent_only_env": os.environ.get("PARENT_ONLY_ENV", ""),
              "shared_stage": os.environ.get("SHARED_STAGE", ""),
              "leaked_env": os.environ.get("LEAKED_ENV", "clean"),
          }}
          with open("{events_file}", "a") as f:
              f.write(json.dumps(rec) + "\\n")
          '
""",
    )

    m = makim.Makim()
    m.load(file=str(cfg))
    assert m.run({'task': 'deploy.ship'}) is True

    records = _read_jsonl(events_file)
    assert [r['step'] for r in records] == [
        'hook1',
        'hook2',
        'hook3',
        'parent',
    ]

    hook1, hook2, hook3, parent = records
    assert hook1['parent_only_env'] == 'prod_group_env'
    assert hook1['helper_group_env'] == 'helper_group_env'
    assert hook1['shared_stage'] == 'corrupted_from_hook1'
    assert hook1['hook_own_env'] == 'hook1_own'
    assert hook1['group_var'] == 'helper_group_val'
    assert hook1['hook_var'] == 'from_hook_one'

    assert hook2['parent_only_env'] == 'prod_group_env'
    assert hook2['helper_group_env'] == 'helper_group_env'
    assert hook2['shared_stage'] == 'prod_task_stage'
    assert hook2['stage_arg'] == 'prod_task_stage'
    assert hook2['hook_own_env'] == 'hook2_own'
    assert hook2['leaked_env'] == 'clean'
    assert hook2['group_var'] == 'helper_group_val'
    assert hook2['hook_var'] == 'from_hook_two'

    assert hook3['parent_only_env'] == 'prod_group_env'
    assert hook3['shared_stage'] == 'prod_task_stage'
    assert hook3['stage_arg'] == 'prod_task_stage'
    assert hook3['leaked_env'] == 'clean'

    assert parent['parent_only_env'] == 'prod_group_env'
    assert parent['shared_stage'] == 'prod_task_stage'
    assert parent['leaked_env'] == 'clean'
    assert m.group_name == 'deploy'
    assert m.task_name == 'ship'


def test_conditional_hooks_evaluate_against_parent_context(
    tmp_path: Path,
) -> None:
    """Conditional hook `if:` checks must resolve parent vars and env."""
    events_file = tmp_path / 'cond_events.jsonl'
    cfg = _write_config(
        tmp_path,
        f"""
groups:
  helper:
    vars:
      env_name: "staging"
      helper_only: "yes"
    env:
      RELEASE_MODE: "staging"
    tasks:
      prep:
        vars:
          prep_only: "yes"
        backend: bash
        run: echo "prep" >> "{events_file}"
      should-skip:
        backend: bash
        run: echo "wrongly_ran" >> "{events_file}"
      should-run:
        args:
          mode:
            type: string
            help: release mode
        backend: bash
        run: echo "cond:${{{{ args.mode }}}}" >> "{events_file}"
  deploy:
    vars:
      env_name: "prod"
    env:
      RELEASE_MODE: "prod"
    tasks:
      ship:
        vars:
          ship_ready: "yes"
        hooks:
          pre-run:
            - task: helper.prep
            - task: helper.should-skip
              if: ${{{{ vars.env_name == "staging" }}}}
            - task: helper.should-run
              if: >-
                ${{{{
                  vars.env_name == "prod"
                  and vars.ship_ready == "yes"
                  and env.RELEASE_MODE == "prod"
                }}}}
              args:
                mode: "${{{{ env.RELEASE_MODE }}}}"
        backend: bash
        run: echo "main" >> "{events_file}"
""",
    )

    m = makim.Makim()
    m.load(file=str(cfg))
    assert m.run({'task': 'deploy.ship'}) is True

    lines = events_file.read_text(encoding='utf-8').splitlines()
    assert lines == ['prep', 'cond:prod', 'main']


def test_post_run_hooks_preserve_parent_context_across_multiple_hooks(
    tmp_path: Path,
) -> None:
    """Sequential post-run hooks must not mutate parent vars or env."""
    out_file = tmp_path / 'post_run.txt'
    cfg = _write_config(
        tmp_path,
        f"""
groups:
  helper:
    vars:
      scope_tag: "helper_scope"
    env:
      AUDIT_TARGET: "wrong_helper"
    tasks:
      notify-one:
        vars:
          step_var: "notify1"
        env:
          AUDIT_TARGET: "wrong_notify1"
        backend: bash
        run: echo "post1:${{{{ vars.step_var }}}}" >> "{out_file}"
      notify-two:
        args:
          audit:
            type: string
            help: Audit target from parent env
        backend: bash
        run: echo "post2:${{{{ args.audit }}}}" >> "{out_file}"
  deploy:
    vars:
      scope_tag: "deploy_scope"
    env:
      AUDIT_TARGET: "prod_audit"
    tasks:
      ship:
        vars:
          task_tag: "ship_tag"
        hooks:
          post-run:
            - task: helper.notify-one
            - task: helper.notify-two
              if: >-
                ${{{{
                  vars.scope_tag == "deploy_scope"
                  and vars.task_tag == "ship_tag"
                }}}}
              args:
                audit: "${{{{ env.AUDIT_TARGET }}}}"
        backend: bash
        run: echo "main" >> "{out_file}"
""",
    )

    m = makim.Makim()
    m.load(file=str(cfg))
    assert m.run({'task': 'deploy.ship'}) is True

    lines = out_file.read_text(encoding='utf-8').splitlines()
    assert lines == ['main', 'post1:notify1', 'post2:prod_audit']


@pytest.mark.parametrize('failure_origin', ['command', 'pre_run'])
def test_failure_hooks_preserve_parent_context_across_multiple_hooks(
    tmp_path: Path, failure_origin: str
) -> None:
    """Sequential failure hooks see parent vars/env on command/pre-run fail."""
    out_file = tmp_path / 'failure_hooks.txt'
    pre_run_block = (
        """          pre-run:
            - task: helper.bad-prerun
"""
        if failure_origin == 'pre_run'
        else ''
    )
    main_cmd = (
        'exit 1' if failure_origin == 'command' else 'echo "unreachable"'
    )
    cfg = _write_config(
        tmp_path,
        f"""
groups:
  helper:
    vars:
      alert_channel: "dev_slack"
    env:
      ALERT_ENV: "wrong_alert"
    tasks:
      bad-prerun:
        retry:
          count: 1
        backend: bash
        run: exit 1
      alert-one:
        vars:
          local_msg: "first_alert"
        env:
          ALERT_ENV: "corrupted_alert"
          LEAKED_ALERT: "should_not_leak"
        backend: bash
        run: echo "fail1:${{{{ vars.local_msg }}}}" >> "{out_file}"
      alert-two:
        args:
          channel-env:
            type: string
            help: Alert environment from failed parent task
        backend: bash
        run: |
          val="${{{{ args.channel_env }}}}:${{LEAKED_ALERT:-clean}}"
          echo "fail2:${{val}}" >> "{out_file}"
  deploy:
    vars:
      alert_channel: "prod_pager"
    env:
      ALERT_ENV: "prod_alert"
    tasks:
      ship:
        vars:
          critical: "yes"
        hooks:
{pre_run_block}          failure:
            - task: helper.alert-one
            - task: helper.alert-two
              if: >-
                ${{{{
                  vars.alert_channel == "prod_pager"
                  and vars.critical == "yes"
                }}}}
              args:
                channel-env: "${{{{ env.ALERT_ENV }}}}"
        backend: bash
        run: {main_cmd}
""",
    )

    m = makim.Makim()
    m.load(file=str(cfg))
    assert m.run({'task': 'deploy.ship'}) is False

    lines = out_file.read_text(encoding='utf-8').splitlines()
    assert lines == ['fail1:first_alert', 'fail2:prod_alert:clean']
    assert m.group_name == 'deploy'
    assert m.task_name == 'ship'


def test_hook_ignored_failure_preserves_next_hook_and_parent(
    tmp_path: Path,
) -> None:
    """Failing Hook 1 with ignore-errors leaves Hook 2 and parent intact."""
    out_file = tmp_path / 'exc_isolation.txt'
    cfg = _write_config(
        tmp_path,
        f"""
groups:
  helper:
    vars:
      mode: "helper"
    env:
      EXEC_MODE: "corrupted_helper"
    tasks:
      failing-hook:
        options:
          ignore-errors: true
        vars:
          mode: "corrupted_hook1"
        env:
          EXEC_MODE: "corrupted_env"
          POISON: "from_failing_hook"
        backend: bash
        run: exit 1
      verify-after:
        args:
          mode-arg:
            type: string
            help: Mode arg from parent env
        backend: bash
        run: |
          res="${{{{ args.mode_arg }}}}:${{POISON:-clean}}"
          echo "after:${{res}}" >> "{out_file}"
  deploy:
    vars:
      mode: "parent"
    env:
      EXEC_MODE: "prod_mode"
    tasks:
      ship:
        vars:
          parent_only: "yes"
        hooks:
          pre-run:
            - task: helper.failing-hook
            - task: helper.verify-after
              if: >-
                ${{{{
                  vars.mode == "parent"
                  and vars.parent_only == "yes"
                }}}}
              args:
                mode-arg: "${{{{ env.EXEC_MODE }}}}"
        backend: bash
        run: echo "parent:${{POISON:-clean}}" >> "{out_file}"
""",
    )

    m_cmd_fail = makim.Makim()
    m_cmd_fail.load(file=str(cfg))
    assert m_cmd_fail.run({'task': 'deploy.ship'}) is True
    assert out_file.read_text(encoding='utf-8').splitlines() == [
        'after:prod_mode:clean',
        'parent:clean',
    ]


@pytest.mark.parametrize('ignore_errors', [True, False])
def test_hook_raised_exception_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ignore_errors: bool
) -> None:
    """Unhandled exception inside Hook 1 preserves Hook 2 and parent state."""
    out_file = tmp_path / 'raised_exc.txt'
    ignore_block = (
        """        options:
          ignore-errors: true
"""
        if ignore_errors
        else ''
    )
    cfg = _write_config(
        tmp_path,
        f"""
groups:
  helper:
    vars:
      mode: "helper"
    env:
      EXEC_MODE: "corrupted_helper"
    tasks:
      failing-hook:
{ignore_block}        backend: bash
        run: echo "never"
      verify-after:
        args:
          mode-arg:
            type: string
            help: Mode arg from parent env
        backend: bash
        run: |
          res="${{{{ args.mode_arg }}}}:${{POISON:-clean}}"
          echo "after:${{res}}" >> "{out_file}"
  deploy:
    vars:
      mode: "parent"
    env:
      EXEC_MODE: "prod_mode"
    tasks:
      ship:
        vars:
          parent_only: "yes"
        hooks:
          pre-run:
            - task: helper.failing-hook
            - task: helper.verify-after
              if: >-
                ${{{{
                  vars.mode == "parent"
                  and vars.parent_only == "yes"
                }}}}
              args:
                mode-arg: "${{{{ env.EXEC_MODE }}}}"
        backend: bash
        run: echo "parent:${{POISON:-clean}}" >> "{out_file}"
""",
    )

    orig_run = makim.Makim.run

    def flaky_run(self: makim.Makim, args: dict[str, Any]) -> bool:
        if args.get('task') == 'helper.failing-hook':
            self._change_task('helper.failing-hook')
            self.env_scoped = {'POISON': 'from_exception'}
            raise RuntimeError('simulated hook failure')
        return orig_run(self, args)

    monkeypatch.setattr(makim.Makim, 'run', flaky_run)
    m = makim.Makim()
    m.load(file=str(cfg))
    assert m.run({'task': 'deploy.ship'}) is ignore_errors
    assert m.group_name == 'deploy'
    assert m.task_name == 'ship'
    if ignore_errors:
        assert out_file.read_text(encoding='utf-8').splitlines() == [
            'after:prod_mode:clean',
            'parent:clean',
        ]
    else:
        assert not out_file.exists()


def test_single_hook_compatibility(tmp_path: Path) -> None:
    """Single-hook pre-run and post-run preserve baseline behavior."""
    out_file = tmp_path / 'single_compat.txt'
    cfg = _write_config(
        tmp_path,
        f"""
groups:
  mid:
    tasks:
      single:
        args:
          tag:
            type: string
            help: tag from parent
        backend: bash
        run: echo "single:${{{{ args.tag }}}}" >> "{out_file}"
  top:
    vars:
      owner: "top_group"
    env:
      LAYER: "top_env"
    tasks:
      single-parent:
        hooks:
          pre-run:
            - task: mid.single
              if: ${{{{ vars.owner == "top_group" }}}}
              args:
                tag: "${{{{ env.LAYER }}}}"
          post-run:
            - task: mid.single
              if: ${{{{ vars.owner == "top_group" }}}}
              args:
                tag: "${{{{ env.LAYER }}}}"
        backend: bash
        run: echo "top-single" >> "{out_file}"
""",
    )

    m = makim.Makim()
    m.load(file=str(cfg))
    assert m.run({'task': 'top.single-parent'}) is True
    assert out_file.read_text(encoding='utf-8').splitlines() == [
        'single:top_env',
        'top-single',
        'single:top_env',
    ]


def test_nested_multi_hooks_isolation(tmp_path: Path) -> None:
    """Nested multi-hooks keep inner and outer scopes isolated."""
    out_file = tmp_path / 'nested_compat.txt'
    cfg = _write_config(
        tmp_path,
        f"""
groups:
  leaf:
    vars:
      owner: "leaf_group"
    env:
      LAYER: "leaf_env"
    tasks:
      step-a:
        env:
          LAYER: "corrupted_leaf_a"
        backend: bash
        run: echo "leaf-a" >> "{out_file}"
      step-b:
        args:
          layer:
            type: string
            help: layer env
        backend: bash
        run: echo "leaf-b:${{{{ args.layer }}}}" >> "{out_file}"
  mid:
    vars:
      owner: "mid_group"
    env:
      LAYER: "mid_env"
    tasks:
      parent-hook-1:
        vars:
          mid_tag: "mid_only"
        env:
          LAYER: "mid_task_env"
        hooks:
          pre-run:
            - task: leaf.step-a
            - task: leaf.step-b
              if: >-
                ${{{{
                  vars.owner == "mid_group" and vars.mid_tag == "mid_only"
                }}}}
              args:
                layer: "${{{{ env.LAYER }}}}"
        backend: bash
        run: echo "mid-1" >> "{out_file}"
      parent-hook-2:
        args:
          layer:
            type: string
            help: top layer env
        backend: bash
        run: echo "top-2:${{{{ args.layer }}}}" >> "{out_file}"
  top:
    vars:
      owner: "top_group"
    env:
      LAYER: "top_env"
    tasks:
      nested-parent:
        vars:
          top_tag: "top_only"
        hooks:
          pre-run:
            - task: mid.parent-hook-1
            - task: mid.parent-hook-2
              if: >-
                ${{{{
                  vars.owner == "top_group" and vars.top_tag == "top_only"
                }}}}
              args:
                layer: "${{{{ env.LAYER }}}}"
        backend: bash
        run: echo "top-main" >> "{out_file}"
""",
    )

    m = makim.Makim()
    m.load(file=str(cfg))
    assert m.run({'task': 'top.nested-parent'}) is True
    assert out_file.read_text(encoding='utf-8').splitlines() == [
        'leaf-a',
        'leaf-b:mid_task_env',
        'mid-1',
        'top-2:top_env',
        'top-main',
    ]


def test_cli_multi_hook_preserves_parent_context(tmp_path: Path) -> None:
    """Verify multi-hook parent context preservation end-to-end via CLI."""
    cfg = _write_config(
        tmp_path,
        """
groups:
  helper:
    tasks:
      h1:
        env:
          HOOK_ENV: "staging"
          TARGET_ENV: "staging"
        backend: bash
        run: echo "[HOOK 1] parent=${PARENT_ENV} hook=${HOOK_ENV}"
      h2:
        env:
          HOOK_ENV: "testing"
        args:
          target:
            type: string
            help: parent target env
        backend: bash
        run: |
          arg_val="${{ args.target }}"
          echo "[HOOK 2] parent=${PARENT_ENV} hook=${HOOK_ENV} arg=${arg_val}"
  deploy:
    vars:
      env_name: "production"
    env:
      PARENT_ENV: "production"
      TARGET_ENV: "production"
    tasks:
      run-it:
        hooks:
          pre-run:
            - task: helper.h1
            - task: helper.h2
              if: ${{ vars.env_name == "production" }}
              args:
                target: "${{ env.TARGET_ENV }}"
        backend: bash
        run: echo "[PARENT] completed"
""",
    )

    res = subprocess.run(
        [sys.executable, '-m', 'makim', '--file', str(cfg), 'deploy.run-it'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert '[HOOK 1] parent=production hook=staging' in res.stdout
    assert '[HOOK 2] parent=production hook=testing arg=production' in (
        res.stdout
    )
    assert '[PARENT] completed' in res.stdout
