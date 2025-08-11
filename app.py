
import streamlit as st
import numpy as np
import cv2, os
import tensorflow as tf

st.set_page_config(page_title='DR Detection', layout='wide')
IMG_SIZE = 256
CLASS_NAMES = ['0: No DR', '1: Mild', '2: Moderate', '3: Severe', '4: Proliferative DR']
ALPHA = [0.5, 1.2, 1.0, 1.3, 1.2]

def crop_foreground(img, tol=7):
    import cv2, numpy as np
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.any():
        coords = np.argwhere(mask)
        y0,x0 = coords.min(axis=0); y1,x1 = coords.max(axis=0)+1
        return img[y0:y1, x0:x1]
    return img

def ben_preprocess(img, radius_scale=0.9):
    import cv2, numpy as np
    h,w = img.shape[:2]
    s = min(h,w)
    y0=(h-s)//2; x0=(w-s)//2
    img = img[y0:y0+s, x0:x0+s]
    mask = np.zeros((s,s), np.uint8)
    cv2.circle(mask, (s//2,s//2), int(s*radius_scale/2), 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    blur = cv2.GaussianBlur(img, (0,0), 10)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8,8)).apply(l)
    img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    img = crop_foreground(img)
    return img

def pad_resize(img, size=IMG_SIZE):
    import cv2, numpy as np
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    img = cv2.resize(img, (nw, nh))
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    y0 = (size-nh)//2; x0 = (size-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = img
    return canvas

def categorical_focal_loss(gamma=2.0, alpha=None):
    import tensorflow as tf
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0-1e-7)
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
        focal = tf.pow(1-pt, gamma) * ce
        if alpha is not None:
            a = tf.constant(alpha, dtype=tf.float32)
            focal = focal * a
        return tf.reduce_sum(focal, axis=-1)
    return loss

@st.cache_resource
def load_dr_model():
    return tf.keras.models.load_model('models/best_model.keras',
        custom_objects={'loss': categorical_focal_loss(gamma=2.0, alpha=ALPHA)})

def find_last_conv_layer(model):
    import tensorflow as tf
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.SeparableConv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    try:
        return model.get_layer('top_conv').name
    except:
        return None

def make_gradcam_heatmap(img_array, model, pred_index=None, last_conv_layer_name=None):
    import tensorflow as tf, cv2, numpy as np
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    weights = tf.reduce_mean(grads, axis=(0,1,2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()

st.title('👁️ Diabetic Retinopathy Detection')
st.caption('APTOS 2019 • Ben’s preprocessing • Focal Loss • Grad-CAM')

uploaded = st.file_uploader('Upload fundus image (.jpg/.png)', type=['jpg','jpeg','png'])
if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb = ben_preprocess(rgb)
    rgb = pad_resize(rgb, IMG_SIZE)
    x = rgb.astype(np.float32)/255.0
    x = np.expand_dims(x, 0)

    model = load_dr_model()
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader('Pre-processed Image')
        st.image(rgb, use_container_width=True)

    with col2:
        st.subheader('Possibilities')
        st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(5)})
        st.metric('Prediction', CLASS_NAMES[pred_idx])

    heat = make_gradcam_heatmap(x, model)
    h = (heat*255).astype(np.uint8)
    h = cv2.resize(h, (IMG_SIZE, IMG_SIZE))
    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb.astype(np.uint8), 0.6, cv2.cvtColor(h, cv2.COLOR_BGR2RGB), 0.4, 0)

    st.subheader('Grad-CAM')
    st.image(overlay, use_container_width=True)
