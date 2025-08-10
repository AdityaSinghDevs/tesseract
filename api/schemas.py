from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class GenerateRequests(BaseModel):
    prompt : str = Field(..., description = "Text prompt to generate a 3D model", max_length = 100)

    batch_size : int = Field(3, description = "Nummber of outputs to be generated")
    
    base_file : Optional[str] = Field("generated_mesh", description = "Base Filename for output meshes")

    guidance_scale : Optional[float] = Field(12 , description = "Guidance scale for results")

    karras_steps : Optional[float] = Field(30, description= "Number of steps taken for generation")

    formats: Optional[List[str]] = Field(default_factory=lambda: ["ply"], description="Mesh formats to export")

    resume_latents : bool = Field(False, description = "Resume from cached latents if available")

    render_latents : bool = Field(False, description = "Render latents for direct preview")

@field_validator("formats", mode="before")
def ensure_list_and_default(cls, v):
       
        if not v:
            return ["ply"]
      
        if isinstance(v, str):
            return [v]
       
        return v




class GenerateResponse(BaseModel):
    status: str = Field(..., description="Status of the generation task, e.g. 'success' or 'failed'")
    prompt : str
    mesh_count : int
    saved_files : List[str]
    latents_path : Optional[str] = None
    output_dir: Optional[str] = None
    job_id: Optional[str] = None #for async stuff


# class ErrorResponse(BaseModel):
#     status: str = Field("error")
#     message : str

