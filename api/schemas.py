from pydantic import BaseModel, Field
from typing import List, Optional

class GenerateRequests(BaseModel):
    prompt : str = Field(..., description = "Text prompt to generate a 3D model", max_length = 100)
    
    base_file : Optional[str] = Field("generated_mesh", description = "Base Filename for output meshes")

    formats: List[str] = Field(default_factory = lambda: ["ply"] , description = "Mesh formats to export")

    resume_latents : bool = Field(False, description = "Resume from cached latents if available")

    render_latents : bool = Field(False, description = "Render latents for direct preview")



class GenerateResponse(BaseModel):
    status: str = Field(..., description="Status of the generation task, e.g. 'success' or 'failed'")
    prompt : str
    mesh_count : int
    saved_files : List[str]
    latents_path : Optional[str] = None
    output_dir: Optional[str] = None
    job_id: Optional[str] = None #for async stuff


class ErrorResponse(BaseModel):
    status: str = Field("error")
    message : str

    