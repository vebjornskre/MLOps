import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "cookie_test"
PYTHON_VERSION = "3.11"

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context, lr=0.001, e=5):
    cmd = f"uv run src/{PROJECT_NAME}/train.py --lr {lr} --e {e}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)

@task
def evaluate(ctx: Context, f='models/trained_model.pth'):
    cmd = f'uv run src/{PROJECT_NAME}/evaluate.py {f}'
    ctx.run(cmd, echo=True, pty=not WINDOWS)

@task
def show_model(ctx: Context) -> None:
    cmd = f'uv run src/{PROJECT_NAME}/model.py'
    ctx.run(cmd, echo=True, pty=not WINDOWS)

@task
def visualize(ctx: Context, fname='models/trained_model.pth', embed='t') -> None:
    cmd = f'uv run src/{PROJECT_NAME}/visualize.py -f {fname} -e {embed}'
    ctx.run(cmd, echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
