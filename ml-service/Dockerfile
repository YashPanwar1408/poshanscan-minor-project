FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements-segmentation.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2 && pip install --no-cache-dir -r requirements-segmentation.txt

COPY app ./app

# Fail the image build if the checkpoint changed or the pinned runtime cannot load it.
RUN python -c "import hashlib; from pathlib import Path; from ultralytics import YOLO; p=Path('app/models/arm_segmentation/best.pt'); expected='dc42cf4f893051abeb6faca2bef0f0c56f44f8182b12f2ec904eec71a2a026b1'; assert p.is_file(), f'Missing checkpoint: {p}'; digest=hashlib.sha256(p.read_bytes()).hexdigest(); assert digest == expected, f'Checkpoint SHA-256 mismatch: {digest}'; m=YOLO(str(p)); names={int(k): str(v) for k,v in m.names.items()} if isinstance(m.names, dict) else {i: str(v) for i,v in enumerate(m.names)}; assert m.task == 'segment', f'Expected task=segment, got {m.task!r}'; assert names == {0: 'upper_arm'}, f'Unexpected class map: {names!r}'; print(f'Validated SHA-256={digest}, YOLO task={m.task}, names={names}')"

EXPOSE 8001

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
