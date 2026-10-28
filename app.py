import streamlit as st
import tensorflow as tf
import numpy as np
import kagglehub

from pathlib import Path
from PIL import Image
from tensorflow.keras.layers import Layer, Dense, Multiply

IMG_SIZE = 260

LEVEL1_NAMES = ["Organic", "NonOrganic"]
LEVEL2_NAMES = ["Food", "Plastic", "Metal", "Paper", "Glass"]
LEVEL3_NAMES = [
    "PET", "HDPE", "Aluminum", "Steel",
    "Newspaper", "Cardboard", "ClearGlass", "BrownGlass"
]


# Custom attention layer used in the trained CNN
@tf.keras.utils.register_keras_serializable()
class SEBlock(Layer):
    def __init__(self, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = Dense(
            channels // self.reduction,
            activation="swish",
            name=f"{self.name}_squeeze"
        )
        self.excite = Dense(
            channels,
            activation="sigmoid",
            name=f"{self.name}_excite"
        )
        super().build(input_shape)

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.excite(x)
        return Multiply()([inputs, x])

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction})
        return config


@st.cache_resource
def load_model():
    folder = kagglehub.dataset_download(
        "kavyavars/waste-cnn-model"
    )

    files = list(
        Path(folder).rglob("best_waste_model_v3.keras")
    )

    if not files:
        raise FileNotFoundError(
            "best_waste_model_v3.keras was not found "
            "in the Kaggle dataset."
        )

    return tf.keras.models.load_model(
        str(files[0]),
        custom_objects={"SEBlock": SEBlock},
        compile=False
    )


def predict_waste(model, image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Match the notebook's 0-255 float32 input convention.
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    predictions = model.predict(array, verbose=0)

    if isinstance(predictions, dict):
        outputs = predictions
    else:
        outputs = dict(zip(model.output_names, predictions))

    label_maps = {
        "organic_output": LEVEL1_NAMES,
        "material_output": LEVEL2_NAMES,
        "subtype_output": LEVEL3_NAMES
    }

    results = {}

    for output_name, names in label_maps.items():
        probs = np.asarray(outputs[output_name])[0]
        index = int(np.argmax(probs))

        results[output_name] = {
            "label": names[index],
            "confidence": float(probs[index]) * 100,
            "probabilities": probs
        }

    return results


# ---------------- FRONTEND ----------------

st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ AI-Based Waste Classification")
st.write(
    "Upload a waste image to classify it using "
    "a multi-branch CNN with EfficientNetV2S."
)

st.info(
    "The model predicts the waste category, material, "
    "and fine-grained subtype."
)

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=350
    )

    if st.button("Classify Waste", type="primary"):
        try:
            with st.spinner("Loading model and analyzing image..."):
                model = load_model()
                results = predict_waste(model, image)

            st.subheader("Classification Results")

            columns = st.columns(3)

            cards = [
                ("organic_output", "Level 1", "Waste Category"),
                ("material_output", "Level 2", "Material"),
                ("subtype_output", "Level 3", "Subtype")
            ]

            for col, (key, level, title) in zip(columns, cards):
                result = results[key]

                with col:
                    st.markdown(f"**{level}: {title}**")
                    st.success(result["label"])
                    st.metric(
                        "Confidence",
                        f'{result["confidence"]:.2f}%'
                    )

            with st.expander("View class probabilities"):
                for key, level, title in cards:
                    st.markdown(f"**{level}: {title}**")

                    for name, probability in sorted(
                        zip(
                            {
                                "organic_output": LEVEL1_NAMES,
                                "material_output": LEVEL2_NAMES,
                                "subtype_output": LEVEL3_NAMES
                            }[key],
                            results[key]["probabilities"]
                        ),
                        key=lambda item: item[1],
                        reverse=True
                    ):
                        st.write(
                            f"{name}: {probability * 100:.2f}%"
                        )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.exception(error)

