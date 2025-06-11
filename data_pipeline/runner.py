import subprocess

def run_train_script(yolo_script_path, work_dir):
    return subprocess.run(
        ["python", yolo_script_path],
        cwd=work_dir,
        stdout=None,   # 터미널에만 출력
        stderr=None
    ).returncode