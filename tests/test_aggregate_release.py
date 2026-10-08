"""Prevent concurrent fits from overwriting one another."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from cfm_project.aggregate_benchmarks import sha256, write_json
from cfm_project.benchmarks import aerosol

def runner():
    return vars(aerosol)

def test_identical_live_fit_is_joined_without_a_second_training_owner(tmp_path):
    api=runner();join=api['wait_for_fit_owner'];stamp=dict(method='gmi_lin',seed=3,config={'alpha':1.})
    command=[sys.executable,'-c',
             'import pathlib,sys,time; time.sleep(.15); pathlib.Path(sys.argv[1]).write_text(sys.argv[2])',
             str(tmp_path/'result.json'),json.dumps(stamp)]
    process=subprocess.Popen(command)
    try:
        write_json(tmp_path/'run.json',dict(**stamp,pid=process.pid,
                   core_sha256=sha256(Path(__file__).resolve().parents[1]/'src/cfm_project/aggregate_benchmarks.py')))
        join(tmp_path,stamp)
        assert json.loads((tmp_path/'result.json').read_text())==stamp
    finally:process.wait(timeout=5)


def test_live_fit_join_rejects_different_configuration(tmp_path):
    api=runner();join=api['wait_for_fit_owner']
    write_json(tmp_path/'run.json',dict(method='gmi_lin',seed=7,config={'alpha':1.},pid=12345))
    with pytest.raises(RuntimeError,match='different method/configuration/data'):
        join(tmp_path,dict(method='gmi_lin',seed=3,config={'alpha':1.}))


def test_live_fit_join_does_not_resume_a_dead_owner(tmp_path,monkeypatch):
    api=runner();join=api['wait_for_fit_owner'];stamp=dict(method='gmi_lin',seed=3,config={'alpha':1.})
    write_json(tmp_path/'run.json',dict(**stamp,pid=12345,
               core_sha256=sha256(Path(__file__).resolve().parents[1]/'src/cfm_project/aggregate_benchmarks.py')))
    def dead(*args):raise ProcessLookupError()
    monkeypatch.setattr(join.__globals__['os'],'kill',dead)
    with pytest.raises(RuntimeError,match='no live owner'):
        join(tmp_path,stamp)
    assert not (tmp_path/'result.json').exists()


def test_live_fit_join_accepts_result_published_during_owner_exit(tmp_path,monkeypatch):
    api=runner();join=api['wait_for_fit_owner'];stamp=dict(method='gmi_lin',seed=3,config={'alpha':1.})
    write_json(tmp_path/'run.json',dict(**stamp,pid=12345,
               core_sha256=sha256(Path(__file__).resolve().parents[1]/'src/cfm_project/aggregate_benchmarks.py')))
    def publish_then_exit(*args):
        write_json(tmp_path/'result.json',stamp)
        raise ProcessLookupError()
    monkeypatch.setattr(join.__globals__['os'],'kill',publish_then_exit)
    join(tmp_path,stamp)
    assert json.loads((tmp_path/'result.json').read_text())==stamp
