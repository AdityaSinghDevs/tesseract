from typing import Any, List, Dict
import os
import numpy as np
import trimesh

import torch
from ..config.config import OUTPUT_DIR, DEFAULT_FORMATS
from ..loggers.logger import get_logger
from .shap_e.util.notebooks import decode_latent_mesh

logger = get_logger(__name__ , log_file="app.log")


def validate_latents_inputs(model : Any , latents : Any)->None:
   if model is None:
      logger.error("Model not provided")
      raise ValueError("Model not provided")
   
   if latents is None or len(latents) ==0:
      logger.error("Latents input is empty")
      raise ValueError("Latents input is empty")

   for i, latent in enumerate(latents):
      if latent is None :
         logger.error(f"Latent {i} is empty, Nothing to decode")
         raise ValueError(f"Latent {i} is None, nothing to decode")
      
      if isinstance(latent, torch.Tensor):
         if latent.numel() ==0:
            logger.error(f"Latent {i} is an empty tensor")
            raise ValueError(f"Latent {i} is an empty tensor")
         
      elif isinstance(latent, (list, tuple)):
         if len(latent) == 0:
            logger.error(f"Latent {i} is an empty list/tuple")
            raise ValueError(f"Latent {i} is empty")
            
      

def validate_decoded_mesh(mesh : List[Any] , output_dir : str,
                         formats : List[Any])->None:
   if not isinstance(mesh, list) or len(mesh)==0:
      logger.error("Decoded mesh must be a non empty list")
      raise
   if not isinstance(output_dir, str):
      logger.error("Output directory must be a non empty str")
      raise
   if not isinstance(formats, list) or len(formats)==0:
      logger.error("Format not provided")
      raise
   
def convert_to_glb(mesh : Any, output_path :str) ->str:
   
   if not mesh:
      logger.error("No mesh provided for GLB export.")
      raise ValueError("Mesh is None, can't export to GLB")
   
   try :
      vertices = np.array(mesh.verts, dtype=np.float32)
      faces = np.array(mesh.faces, dtype=np.int32)
      
      if vertices.size == 0 or faces.size == 0 :
         logger.error ("Mesh has no vertices or faces cannot export")
         raise ValueError("Empty mesh, cannot export to glb")
      
   except Exception as e:
      logger.error(f"Failed to extract vertices/ faces : {e}")
      raise
   
   try:
      tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
   except Exception as e:
      logger.error(f"Failed to create Trimesh object : {e}")
      raise RuntimeError("Trimesh conversion failed") from e
   
   try: 
      tri_mesh.export(output_path, file_type="glb")
      logger.info(f"Mesh successfully exported to GLB : {output_path}")
      return output_path
   except Exception as e:
      logger.error(f"FAILED to export mesh to glb : {e}" )
      raise RuntimeError(f"GLB export failed : {e}")
   

def decode_latents(model : Any, latents: Any)->List[Any]:

    output_meshes = []
   
    try:
        validate_latents_inputs(model, latents)
        logger.info("Inputs for decoding Validated Successfully")
    except Exception as e:
       logger.error(f"Inputs for decoding could not be verified : {e}")
       raise

    
    for i, latent in enumerate(latents):
      try: 
           mesh = decode_latent_mesh(model, latent).tri_mesh()
           output_meshes.append(mesh)
      except Exception as e:
        logger.error(f"Failed to decode latent {i}: {e}", exc_info=True)

      if not output_meshes:
        raise RuntimeError("All latents failed to decode into meshes")
      
      logger.info("Latents decoded successfully into meshes..")

    return output_meshes



def save_mesh(meshes : List[Any] , base_file : str, 
              output_dir: str = OUTPUT_DIR,
              formats : List[Any] = DEFAULT_FORMATS)->Dict[str, Any]:
   
   files = []
   failed_formats = []
   
   validate_decoded_mesh(meshes, output_dir, formats)
   logger.info("Inputs for saving mesh validated successfully")

   os.makedirs(output_dir, exist_ok=True)
   logger.info(f"Created/Found output directory at : {output_dir}")

   for mesh_id, single_mesh in enumerate(meshes):

      verts = getattr(single_mesh, 'verts', None)
      faces = getattr(single_mesh, 'faces', None)
      
      if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
            logger.warning(f"Mesh {mesh_id} is empty; skipping save.")
            continue
      for format in formats:
        output_path = os.path.join(output_dir, f"{base_file}_{mesh_id}.{format}")
        logger.info(f"Saving {base_file} to {output_path}")

        try:
            if format == "ply":
                with open(output_path, 'wb') as f:
                 single_mesh.write_ply(f)
                 files.append(output_path)
            elif format == "obj":
                with open(output_path, 'w') as f:
                 single_mesh.write_obj(f)
                 files.append(output_path)
            elif format == "glb":
                glb_path = convert_to_glb(single_mesh, output_path)
                files.append(glb_path)
            else:
                logger.error(f"Unsupported format : {format}")
                failed_formats.append(format)
                continue
            logger.info(f"Exported {mesh_id} successfully to {output_path}")
            
            
        except Exception as e:   
            logger.error(f"Failed to save mesh in {format} : {e}")
            

   return{ 'saved_files' : files,
            'failed_formats' : failed_formats,
            'count' : len(files),
            'output_dir': output_dir
       }