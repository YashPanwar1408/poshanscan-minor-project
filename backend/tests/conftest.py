import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, get_db
from app.core.security import get_password_hash, create_access_token
from app.models.user import User
from app.models.child import Child
from app.main import app

from app.schemas.cv import CVInferResponse, CVQualityFlags
from app.routers import scans as scans_router
from app.routers import sync as sync_router

# In-memory SQLite for isolated test runs with StaticPool
TEST_DB_PATH = "test_poshanscan.db"
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)
SQLALCHEMY_DATABASE_URL = f"sqlite:///./{TEST_DB_PATH}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    
    # Create test worker
    worker = User(
        id="worker-uuid-001",
        name="Test Worker",
        phone="9876543210",
        password_hash=get_password_hash("worker123"),
        role="worker",
    )
    # Create test supervisor
    supervisor = User(
        id="super-uuid-001",
        name="Test Supervisor",
        phone="9000000000",
        password_hash=get_password_hash("super123"),
        role="supervisor",
    )
    # Create test child
    child = Child(
        id="child-uuid-001",
        name="Aarav Kumar",
        dob="2024-01-01",
        gender="M",
        guardian_name="Sunita Kumar",
        village="Rampur",
        registered_by="worker-uuid-001",
    )
    db.add(worker)
    db.add(supervisor)
    db.add(child)
    db.commit()
    db.close()
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def mock_cv_for_tests():
    async def fake_cv_inference(
        image_bytes,
        filename="scan.jpg",
        reference_type="aruco",
        reference_size_mm=50.0,
        age_category="child_6_59m",
    ):
        norm_age = "adult" if str(age_category).strip().lower() == "adult" else "child_6_59m"
        muac = 260.0 if norm_age == "adult" else 130.0
        risk_band = "Normal" if norm_age == "adult" else "Normal"
        if norm_age == "adult":
            risk_band = "Normal"
        return CVInferResponse(
            muac_estimate_mm=muac,
            risk_band=risk_band,
            confidence_score=0.9,
            quality_flags=CVQualityFlags(
                reference_detected=True,
                arm_detected=True,
                segmentation_ok=True,
            ),
            pipeline_version="test",
            age_category=norm_age,
        )

    original_scans = scans_router.run_cv_inference
    original_sync = sync_router.run_cv_inference
    scans_router.run_cv_inference = fake_cv_inference
    sync_router.run_cv_inference = fake_cv_inference
    yield
    scans_router.run_cv_inference = original_scans
    sync_router.run_cv_inference = original_sync


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def test_db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def worker_token():
    return create_access_token(
        data={"sub": "worker-uuid-001", "role": "worker", "phone": "9876543210"}
    )


@pytest.fixture
def supervisor_token():
    return create_access_token(
        data={"sub": "super-uuid-001", "role": "supervisor", "phone": "9000000000"}
    )
