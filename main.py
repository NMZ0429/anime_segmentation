import streamlit as st
import numpy
from PIL import Image
import paddlehub as hub
import tempfile

# layout setting
st.set_page_config(
    page_title="Made by Namazur",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_style = """
                        <style>
                        #MainMenu {visibility: hidden;}
                        footer {
                            
                            visibility: hidden;
                            
                            }
                        footer:after {
                            content:'Made for catfish'; 
                            visibility: visible;
                            display: block;
                            position: relative;
                            #background-color: red;
                            padding: 5px;
                            top: 2px;
                        }
                        </style>
                        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


@st.cache(allow_output_mutation=True)
def cache_model():
    model = hub.Module(name="U2Net")
    return model


u2net = cache_model()


@st.cache(ttl=60)
def segment(img, out_dir, model=u2net):
    result = model.Segmentation(
        # images=[cv2.imread(img.name)],
        images=[numpy.asarray(img)],
        paths=None,
        batch_size=1,
        input_size=320,
        output_dir=out_dir,
        visualization=True,
    )
    return result[0]["front"], result[0]["mask"]


# user interaction

with tempfile.TemporaryDirectory(prefix="tmp_", dir=".") as dirpath:
    uploaded_file = st.file_uploader("Choose a image file")

    if uploaded_file is not None:
        image_pil_array = Image.open(uploaded_file)
        st.image(image_pil_array, caption="uploaded image", use_column_width=True)

        front, mask = segment(img=image_pil_array, out_dir=dirpath)
        st.image(front)
        st.image(mask)
