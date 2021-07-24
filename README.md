# vid2minecraft

Converts videos and images to Minecraft blocks.

## Getting Started

Install dependencies with:

    pip install -r "requirements.txt"

Download Minecraft and retrieve `[VERSION NUMBER].jar` (this will be under `%appdata%/.minecraft/versions/[VERSION NUMBER]` on Windows).

Open `[VERSION NUMBER].jar` in 7Zip and extract the `/assets/minecraft/textures/block` folder to the repo root as `./block`.

Convert video with:

    python vid2minecraft.py -i [INPUT IMAGE/VIDEO PATH] [OUTPUT IMAGE/VIDEO PATH]
