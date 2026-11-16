from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class LoginRequest(BaseModel):
    phone: str = Field(..., description="Worker or supervisor phone number")
    password: str = Field(..., description="User password")


class RegisterRequest(BaseModel):
    phone: str = Field(..., description="Worker or supervisor phone number")
    password: str = Field(..., description="User password")
    role: str = Field(default="worker", description="User role ('worker', 'supervisor', 'admin')")
    name: Optional[str] = Field(None, description="Full name of the user")
    location_id: Optional[str] = Field(None, description="Optional location or village identifier")
    email: Optional[str] = Field(None, description="Worker email address")
    village: Optional[str] = Field(None, description="Worker village or field area")


class LoginResponse(BaseModel):
    access_token: str
    token: str          # For Dhruv's dashboard
    role: str
    name: str           # For Dhruv's dashboard
    user_id: str
    phone: str
    email: Optional[str] = None
    village: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    name: str
    phone: str
    role: str
    location_id: Optional[str] = None
    email: Optional[str] = None
    village: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
