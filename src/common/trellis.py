from common.imports import *
from jaxtyping import jaxtyped, Float, Bool, Integer, Shaped

def attach_extra_models(pipeline: "TrellisImageTo3DPipeline"):
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis import models as _trellis_models
    device = pipeline.device
    extra_model_path = {
        # 'sparse_structure_decoder': 'ckpts/ss_dec_conv3d_16l8_fp16',
        'sparse_structure_encoder': 'ckpts/ss_enc_conv3d_16l8_fp16',
        # 'slat_decoder_gs': 'ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        # 'slat_decoder_rf': 'ckpts/slat_dec_rf_swin8_B_64l8r16_fp16',
        # 'slat_decoder_mesh': 'ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16',
        'slat_encoder': 'ckpts/slat_enc_swin8_B_64l8_fp16',
    }
    # extra_model_class = {
    #     'sparse_structure_encoder': SparseStructureEncoder,
    #     'slat_encoder': SLatEncoder,
    # }
    with device, torch.cuda.device(device.index):
        models = {
            k: _trellis_models.from_pretrained(f"JeffreyXiang/TRELLIS-image-large/{v}").to(device)
            for k, v in extra_model_path.items()
        }
        for model in models.values():
            model.eval()
            model.requires_grad_(False)
        pipeline.models.update(models)
    return pipeline


@torch.no_grad()
def run_trellis(
    self: "TrellisImageTo3DPipeline",
    image: Float[torch.Tensor, 'b c 518 518'],
    num_samples: int = 1,
    seed: int = 42,
    sparse_structure_sampler_params: dict = {},
    slat_sampler_params: dict = {},
    formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
) -> dict:
    with self.device, torch.cuda.device(self.device.index):
        cond = self.get_cond(image)
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

