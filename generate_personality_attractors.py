import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from hashlib import sha256
from scipy.ndimage import gaussian_filter

# ----------------------------- CONFIG -----------------------------
COLLECTION_SIZE = 1000
IMAGE_SIZE = 3840
DPI = 300
POINTS = 200000
STEPS = 20000

OUTPUT_DIR = "output_cosmic_souls_daring_shapes"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")

try:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    print("Daring shapes folders ready!")
except Exception as e:
    print(f"Folder error: {e}")
    exit()

# ULTRA CONTRAST palettes
palette_colors = {
    "dreamy": ["#000066", "#0000ff", "#00ccff", "#00ffff", "#ffffff"],
    "energetic": ["#660000", "#ff0000", "#ff6600", "#ffff33", "#ffffff"],
    "turbulent": ["#330066", "#aa00ff", "#ff00ff", "#ff66ff", "#ffffff"],
    "harmonious": ["#006600", "#00ff00", "#66ff66", "#99ffcc", "#ffffff"],
    "elegant": ["#000000", "#444444", "#999999", "#dddddd", "#ffffff"],
}

custom_cmaps = {k: LinearSegmentedColormap.from_list(k, v, N=256) for k, v in palette_colors.items()}

# ----------------------------- HELPERS -----------------------------
def seeded_random(seed):
    return np.random.RandomState(seed % 2**32)

def generate_big_five(seed):
    rng = seeded_random(seed)
    return {
        "Openness": int(np.clip(rng.normal(50, 14), 0, 100)),
        "Conscientiousness": int(np.clip(rng.normal(50, 12), 0, 100)),
        "Extraversion": int(np.clip(rng.normal(50, 13), 0, 100)),
        "Agreeableness": int(np.clip(rng.normal(50, 11), 0, 100)),
        "Neuroticism": int(np.clip(rng.normal(50, 14), 0, 100)),
    }

def top_traits(traits, n=3):
    return sorted(traits.items(), key=lambda x: x[1], reverse=True)[:n]

def blend_cmaps(cmaps, weights):
    if len(cmaps) == 1: return cmaps[0]
    norm_weights = np.array(weights) ** 1.8
    norm_weights = norm_weights / norm_weights.sum()
    cols = [c(np.linspace(0, 1, 256)) for c in cmaps]
    blended = sum(w * c for w, c in zip(norm_weights, cols))
    blended = np.clip(blended, 0, 1)
    return LinearSegmentedColormap.from_list("blend", blended)

# ----------------------------- CORE (Daring Lorenz Shapes) -----------------------------
def map_traits_to_params(traits):
    O = traits["Openness"] / 100
    C = traits["Conscientiousness"] / 100
    E = traits["Extraversion"] / 100
    A = traits["Agreeableness"] / 100
    N = traits["Neuroticism"] / 100

    # Daring ranges for extreme shapes
    sigma = np.clip(8.0 + 22.0 * O, 8.0, 30.0)      # Massive wings for high Openness
    rho = np.clip(20.0 + 40.0 * N, 20.0, 60.0)      # Wild turbulence for high Neuroticism
    beta = np.clip((8/3) * (0.7 + 1.5 * C), 1.8, 3.5)  # Flat vs tall for Conscientiousness

    top = top_traits(traits, 3)
    map_dict = {"Openness":"dreamy", "Extraversion":"energetic", "Neuroticism":"turbulent", 
                "Agreeableness":"harmonious", "Conscientiousness":"elegant"}
    cmaps = [custom_cmaps[map_dict[k]] for k, _ in top]
    weights = [v for _, v in top]
    cmap = blend_cmaps(cmaps, weights)

    glow_strength = A + 0.5*O + 0.4*E

    return sigma, rho, beta, cmap, glow_strength

def lorenz_attractor(sigma, rho, beta, seed):
    rng = seeded_random(seed)
    x = rng.uniform(-0.1, 0.1)
    y = rng.uniform(-0.1, 0.1)
    z = rng.uniform(-0.1, 0.1)

    xs, ys, zs = np.zeros(POINTS), np.zeros(POINTS), np.zeros(POINTS)
    dt = 0.006

    for _ in range(STEPS):
        x += sigma * (y - x) * dt
        y += (x * (rho - z) - y) * dt
        z += (x * y - beta * z) * dt

    for i in range(POINTS):
        x += sigma * (y - x) * dt
        y += (x * (rho - z) - y) * dt
        z += (x * y - beta * z) * dt
        xs[i] = x
        ys[i] = y
        zs[i] = z

    return xs, ys, zs

def plot_attractor(ax, xs, ys, zs, cmap, glow_strength):
    # Normalize to fill frame perfectly
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-8) * 2 - 1
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-8) * 2 - 1

    z_min, z_max = zs.min(), zs.max()
    z_range = z_max - z_min + 1e-8
    norm_z = (zs - z_min) / z_range

    # Density alpha
    hist, _, _ = np.histogram2d(xs, ys, bins=200, range=[[-1,1], [-1,1]])
    density = gaussian_filter(hist, sigma=3)
    density_norm = density / (density.max() + 1e-8)

    x_idx = np.clip(((xs + 1) / 2 * 199).astype(int), 0, 199)
    y_idx = np.clip(((ys + 1) / 2 * 199).astype(int), 0, 199)
    density_alpha = 0.3 + 0.6 * density_norm[x_idx, y_idx]

    base_sizes = 0.05 + 0.11 * norm_z  # Slightly larger for impact

    colors = cmap(norm_z)
    colors[:, 3] = density_alpha

    ax.scatter(xs, ys, c=colors, s=base_sizes, lw=0)

    ax.scatter(xs, ys, c='white', s=base_sizes * 0.6, alpha=0.3, lw=0)

    if glow_strength > 0.55:
        glow_mask = np.random.choice([True, False], size=POINTS, p=[0.08, 0.92])
        ax.scatter(xs[glow_mask], ys[glow_mask], c='white', s=base_sizes[glow_mask]*7, alpha=0.35, lw=0)

    ax.axis('off')
    ax.set_facecolor('black')
    ax.set_aspect('equal')

# ----------------------------- GENERATION -----------------------------
for token_id in range(1, COLLECTION_SIZE + 1):
    if token_id % 50 == 0 or token_id <= 5:
        print(f"Generating daring shapes #{token_id}/{COLLECTION_SIZE}...")

    seed = int(sha256(f"cosmic_daring_shapes_{token_id}".encode()).hexdigest(), 16)
    traits = generate_big_five(seed)

    sigma, rho, beta, cmap, glow = map_traits_to_params(traits)
    xs, ys, zs = lorenz_attractor(sigma, rho, beta, seed)

    fig = plt.figure(figsize=(IMAGE_SIZE/DPI, IMAGE_SIZE/DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    plot_attractor(ax, xs, ys, zs, cmap, glow)
    plt.savefig(os.path.join(IMAGES_DIR, f"{token_id}.png"), facecolor='black')
    plt.close(fig)

    attributes = [
        {"trait_type": "Openness", "value": traits["Openness"]},
        {"trait_type": "Conscientiousness", "value": traits["Conscientiousness"]},
        {"trait_type": "Extraversion", "value": traits["Extraversion"]},
        {"trait_type": "Agreeableness", "value": traits["Agreeableness"]},
        {"trait_type": "Emotional Stability", "value": 100 - traits["Neuroticism"]},
        {"trait_type": "Sigma", "value": round(sigma, 1)},
        {"trait_type": "Rho", "value": round(rho, 1)},
        {"trait_type": "Beta", "value": round(beta, 1)},
    ]

    metadata = {
        "name": f"Cosmic Soul Dust #{token_id}",
        "description": "1,000 personality-driven dust clouds expressing the beauty and chaos of the human soul, creating an ethereal visual portrait of the mathematics hidden inside every individual",
        "image": f"ipfs://YOUR_CID/{token_id}.png",
        "attributes": attributes,
    }

    with open(os.path.join(METADATA_DIR, f"{token_id}.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

print("\n🎉 Daring shapes collection complete! Extreme variety, filled frames, vibrant colors. Launch it! 🌌✨")