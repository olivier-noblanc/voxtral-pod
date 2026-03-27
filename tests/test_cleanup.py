import os
import pytest
import time
import json
import sqlite3
import subprocess

from unittest.mock import patch, MagicMock
from backend.state import get_db, add_job
from backend.cleanup import clean_old_jobs, clean_old_files, compress_old_wavs, run_cleanup

@pytest.fixture
def mock_dirs(tmp_path, monkeypatch):
    trans_dir = tmp_path / "transcriptions_terminees"
    temp_dir = tmp_path / "temp_batch"
    live_audio = trans_dir / "live_audio"
    batch_audio = trans_dir / "batch_audio"
    
    for d in [trans_dir, temp_dir, live_audio, batch_audio]:
        d.mkdir(parents=True, exist_ok=True)
        
    monkeypatch.setattr("backend.cleanup.TRANSCRIPTIONS_DIR", str(trans_dir))
    monkeypatch.setattr("backend.cleanup.TEMP_DIR", str(temp_dir))
    return {"trans_dir": trans_dir, "temp_dir": temp_dir}

def create_mock_file(path, age_days):
    """Creates a mock file with an artificially mutated modification time."""
    path.write_text("dummy")
    # set timestamps backward
    mtime = time.time() - (age_days * 86400)
    os.utime(str(path), (mtime, mtime))
    return path

def test_clean_old_jobs():
    # Insert jobs with old timestamp
    with get_db() as conn:
        conn.execute("DELETE FROM jobs") # start clean
        conn.execute("INSERT INTO jobs (job_id, data, created_at) VALUES ('job_old', '{}', datetime('now', '-95 days'))")
        conn.execute("INSERT INTO jobs (job_id, data, created_at) VALUES ('job_new', '{}', datetime('now', '-10 days'))")
        conn.commit()
    
    count = clean_old_jobs(90)
    assert count == 1
    
    with get_db() as conn:
        cursor = conn.execute("SELECT job_id FROM jobs")
        jobs = [r['job_id'] for r in cursor.fetchall()]
        assert "job_old" not in jobs
        assert "job_new" in jobs

def test_clean_old_files(mock_dirs):
    trans_dir = mock_dirs['trans_dir']
    
    f1 = create_mock_file(trans_dir / "live_audio" / "old.wav", 100)
    f2 = create_mock_file(trans_dir / "live_audio" / "new.wav", 10)
    f3 = create_mock_file(trans_dir / "old.md", 95)
    f4 = create_mock_file(trans_dir / "new.json", 2)
    
    # Run cleanup for 90 days
    count = clean_old_files(90)
    
    assert count == 2
    assert not f1.exists()
    assert f2.exists()
    assert not f3.exists()
    assert f4.exists()

@patch('subprocess.run')
def test_compress_old_wavs(mock_subprocess, mock_dirs):
    trans_dir = mock_dirs['trans_dir']
    live_audio = trans_dir / "live_audio"
    
    f1 = create_mock_file(live_audio / "old_audio.wav", 2) # older than 0.1 day (2.4h)
    f2 = create_mock_file(live_audio / "new_audio.wav", 0.01) # very new
    
    # Mock ffmpeg success
    def mock_run_side_effect(*args, **kwargs):
        # cmd is likely args[0]
        cmd = args[0]
        # find the output file (usually the last argument or after -b:a 64k)
        mp3_path = cmd[-1]
        with open(mp3_path, 'w') as f:
            f.write("mp3 data")
        m = MagicMock()
        m.returncode = 0
        return m
        
    mock_subprocess.side_effect = mock_run_side_effect
    
    count = compress_old_wavs(0.1)
    
    assert count == 1
    assert not f1.exists() # original wav deleted
    assert (live_audio / "old_audio.mp3").exists() # mp3 created
    assert f2.exists() # new wav left alone
    assert mock_subprocess.called

def test_run_cleanup(mock_dirs, monkeypatch):
    monkeypatch.setattr("backend.cleanup.clean_old_jobs", lambda d: 5)
    monkeypatch.setattr("backend.cleanup.clean_old_files", lambda d: 10)
    monkeypatch.setattr("backend.cleanup.compress_old_wavs", lambda d: 3)
    
    stats = run_cleanup(90)
    assert stats["jobs_deleted"] == 5
    assert stats["files_deleted"] == 10
    assert stats["compressed_files"] == 3
