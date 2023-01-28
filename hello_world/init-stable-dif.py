import os
import io
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

def text_to_image(prompt):
    if isinstance(prompt, str):
        stability_api = initialize()
        answers = stability_api.generate(
        prompt=prompt,
        steps=30, # Amount of inference steps performed on image generation. Defaults to 30. 
        cfg_scale=8.0, # Defaults to 7.0 if not specified.
        width=512,
        height=512,
        samples=1, # Number of images to generate, defaults to 1 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
        )
        return answers
    else:
        print("error")
        return None

def initialize():
    stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-v1-5", 
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 
  # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
    )
    
    return stability_api

def print_answers(answers):
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                img.save(str(artifact.seed)+ ".png") # Save our generated images with their seed number as the filename.
                print("Image saved!")
    return

def img_to_img(prompt,img_path):
    if isinstance(prompt, str):
        with open(img_path, "rb") as f:
            binary_data = f.read()
        image = Image.open(io.BytesIO(binary_data))
        stability_api = initialize()
        answers = stability_api.generate(
        prompt=prompt,
        init_image=image,
        steps=30, # Amount of inference steps performed on image generation. Defaults to 30. 
        cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
        width=512,
        height=512,
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
    )
        return answers
    else : 
        print("error")
        return None


os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-yZy25gQpVcedzyd1l34DDL2G2keINp1g2m2khXZQCoR0sHDc'
#print_answers(text_to_image("A happy dog on the park,4K"))
print_answers(img_to_img("Add a cat next to the happy dog,4K","dog.png"))

