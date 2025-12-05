import jax
from einops import rearrange

from equimo.experimental.text import Tokenizer
from equimo.io import load_image, load_model
from equimo.utils import PCAVisualizer, normalize, plot_image_and_feature_map

# Random demo inputs
key = jax.random.PRNGKey(42)
image = load_image("./demo.jpg", size=448)
text = [
    "A baby discovering happiness",
    "A computer",
]

# Loading pretrained models
image_encoder = load_model("vit", "tips_vits14_hr")
text_encoder = load_model("experimental.textencoder", "tips_vits14_hr_text")

# Encoding text and image
ids, paddings = Tokenizer(identifier="sentencepiece_tips").tokenize(text, max_len=64)

text_embedding = normalize(
    jax.vmap(text_encoder, in_axes=(0, 0, None))(ids, paddings, key)
)
image_embedding = jax.vmap(image_encoder.norm)(image_encoder.features(image, key))
cls_token = normalize(image_embedding[0])
spatial_features = rearrange(
    image_embedding[2:], "(h w) d -> h w d", h=int(448 / 14), w=int(448 / 14)
)

# Getting probabilities based on Cosine Similarity
cos_sim = jax.nn.softmax(
    ((cls_token[None, :] @ text_embedding.T) / text_encoder.temperature), axis=-1
)

# Plot the results
label_idxs = jax.numpy.argmax(cos_sim, axis=-1)
cos_sim_max = jax.numpy.max(cos_sim, axis=-1)
label_predicted = text[label_idxs[0]]
similarity = cos_sim_max[0]
pca_obj = PCAVisualizer(spatial_features)
image_pca = pca_obj(spatial_features)

plot_image_and_feature_map(
    image.transpose(1, 2, 0),
    image_pca,
    "./out.png",
    "Input Image",
    f"{label_predicted}, prob: {similarity * 100:.2f}%",
)