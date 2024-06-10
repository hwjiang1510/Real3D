import math
import os
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image

from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
    get_rays,
    get_ray_directions
)


class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int

        image_tokenizer_cls: str
        image_tokenizer: dict

        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

    cfg: Config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
            use_saved_ckpt = True
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )
            use_saved_ckpt = False

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu")
        if use_saved_ckpt:
            if "module" in list(ckpt["state_dict"].keys())[0]:
                ckpt = {key.replace('module.',''): item for key, item in ckpt["state_dict"].items()}
            else:
                ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt)
        return model

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None


    def forward(self, 
                inputs: torch.FloatTensor, 
                rays_o: torch.FloatTensor,
                rays_d: torch.FloatTensor,
                ):
        # input images in shape [b,1,c,h,w], value range [0,1]
        # rays_o and rays_d in shape [b,Nv,h,w,3]
        batch_size, n_views = rays_o.shape[:2]

        # get triplane
        input_image_tokens: torch.Tensor = self.image_tokenizer(inputs)         # [b,1,c,n]
        input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C')
        tokens: torch.Tensor = self.tokenizer(batch_size)                       # [b,ct,Np*Hp*Wp]
        tokens = self.backbone(tokens, encoder_hidden_states=input_image_tokens)# triplanes in [b,Np,Ct,Hp,Wp]
        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))    # triplanes in [b,Np,Ct',Hp',Wp']
        
        # replicate triplanes
        scene_codes = rearrange(scene_codes.unsqueeze(1).repeat(1,n_views,1,1,1,1),
                                'b Nv Np Ct Hp Wp -> (b Nv) Np Ct Hp Wp')

        # render
        rays_o = rearrange(rays_o, 'b Nv h w c -> (b Nv) h w c')
        rays_d = rearrange(rays_d, 'b Nv h w c -> (b Nv) h w c')
        render_images, render_masks = self.renderer(self.decoder, 
                                                    scene_codes, 
                                                    rays_o, rays_d, 
                                                    return_mask=True)  # [b*Nv,h,w,3], [b*Nv,h,w]
        render_images = rearrange(render_images, '(b Nv) h w c -> b Nv c h w', Nv=n_views)
        render_masks = rearrange(render_masks, '(b Nv) h w c -> b Nv c h w', Nv=n_views)
        
        return {'images_rgb': render_images, 
                'images_weight': render_masks}


    def get_latent_from_img(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(
            device
        )
        batch_size = rgb_cond.shape[0]

        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )

        tokens: torch.Tensor = self.tokenizer(batch_size)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes

    def render_360(
        self,
        scene_codes,
        n_views: int,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                images_.append(process_output(image))
            images.append(images_)
        return images

    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(self, scene_codes, resolution: int = 256, threshold: float = 25.0):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            with torch.no_grad():
                color = self.renderer.query_triplane(
                    self.decoder,
                    v_pos,
                    scene_code,
                )["color"]
            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
                vertex_colors=color.cpu().numpy(),
            )
            meshes.append(mesh)
        return meshes
