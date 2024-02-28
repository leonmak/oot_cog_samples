from cog import BasePredictor, Input, Path
import tempfile
import pillow_avif

from oot_diffusion import OOTDiffusionModel


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = OOTDiffusionModel()
        self.model.load_pipe()

        return self.model

    # The arguments and types the model takes as input
    def predict(
        self,
        model_image: Path = Input(
            description="Clear picture of the model",
            default="https://raw.githubusercontent.com/viktorfa/oot_diffusion/main/oot_diffusion/assets/model_1.png",
        ),
        garment_image: Path = Input(
            description="Clear picture of upper body garment",
            default="https://raw.githubusercontent.com/viktorfa/oot_diffusion/main/oot_diffusion/assets/cloth_1.jpg",
        ),
        steps: int = Input(
            default=20, description="Inference steps", ge=1, le=40),
        guidance_scale: float = Input(
            default=2.0, description="Guidance scale", ge=1.0, le=5.0
        ),
        seed: int = Input(default=0, description="Seed",
                          ge=0, le=0xFFFFFFFFFFFFFFFF),
        garment_category: str = Input(
            default="upperbody", description="upperbody, bottom, dress")
    ) -> list[Path]:
        """Run a single prediction on the model"""
        generated_images, mask_image = self.model.generate(
            cloth_path=garment_image,
            model_path=model_image,
            steps=steps,
            cfg=guidance_scale,
            seed=seed,
            num_samples=4,
            garment_categor=garment_category,
        )

        result_paths: list[Path] = []

        for i, img in enumerate(generated_images):
            result_path = Path(tempfile.mktemp(suffix=".png"))
            img.save(result_path, "PNG")
            result_paths.append(result_path)

        return result_paths
