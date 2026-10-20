import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
from PIL import Image

st.set_page_config(page_title = 'Heart Disease Risk', page_icon = '🫀', layout = 'wide')

art_dir = os.path.join(os.path.dirname(__file__), 'artifacts')


# ---------------- loading the models (once, cached) ----------------

@st.cache_resource
def load_tabular():
    model = joblib.load(os.path.join(art_dir, 'tabular_model.joblib'))
    config = json.load(open(os.path.join(art_dir, 'tabular_config.json')))
    return model, config


@st.cache_resource
def load_cxr():
    # the x-ray model is optional, the app works without it
    if not os.path.exists(os.path.join(art_dir, 'cxr_model.pt')):
        return None, None, None
    import torch
    import torchvision
    from torchvision import transforms

    config = json.load(open(os.path.join(art_dir, 'cxr_config.json')))
    model = torchvision.models.densenet121(weights = None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load(os.path.join(art_dir, 'cxr_model.pt'), map_location = 'cpu'))
    model.eval()

    # exactly the eval transform of the notebook
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels = 3),
        transforms.Resize(config['img_size']),
        transforms.CenterCrop(config['crop_size']),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std'])
    ])
    return model, config, tf


def grad_cam(model, x):
    # same function as in the notebook
    import torch
    store = {}

    def fwd_hook(module, inp, out):
        store['act'] = out
        out.register_hook(lambda g: store.update(grad = g))

    h = model.features.register_forward_hook(fwd_hook)
    logit = model(x).squeeze()
    model.zero_grad()
    logit.backward()
    h.remove()

    weights = store['grad'].mean(dim = (2, 3), keepdim = True)
    cam = torch.relu((weights * store['act']).sum(dim = 1)).squeeze(0)
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(cam[None, None], size = x.shape[-2:], mode = 'bilinear', align_corners = False)[0, 0]
    return torch.sigmoid(logit).item(), cam.detach().numpy()


def overlay(img, cam, config):
    # the image as the model saw it (resize + center crop) with the heat map on top
    s, c = config['img_size'], config['crop_size']
    w, h = img.size
    scale = s / min(w, h)
    img = img.convert('L').resize((round(w * scale), round(h * scale)))
    left, top = (img.size[0] - c) // 2, (img.size[1] - c) // 2
    gray = np.array(img.crop((left, top, left + c, top + c))).astype(float) / 255
    heat = matplotlib.colormaps['jet'](cam)[:, :, :3]
    mix = 0.6 * np.stack([gray] * 3, axis = -1) + 0.4 * heat
    return Image.fromarray((mix * 255).astype('uint8')).resize((448, 448))


tab_model, tab_cfg = load_tabular()
cxr_model, cxr_cfg, cxr_tf = load_cxr()


# ---------------- sidebar ----------------

with st.sidebar:
    st.header('About the models')
    m = tab_cfg['test_metrics']
    st.markdown(f"**Clinical model:** {tab_cfg['model_name']}")
    st.markdown(f"Trained on {tab_cfg['n_train']} UCI patients. Test set ({tab_cfg['n_test']}): "
                f"recall {m['recall']:.2f}, precision {m['precision']:.2f}, ROC AUC {m['roc_auc']:.2f}. "
                f"Flag threshold {tab_cfg['threshold']:.2f}, set for {tab_cfg['target_recall']:.0%} recall.")
    if cxr_cfg is not None:
        c = cxr_cfg['test_metrics']
        st.markdown('**X-ray model:** DenseNet-121, cardiomegaly')
        st.markdown(f"Trained on {cxr_cfg['n_train']} NIH chest X-rays. Test set ({cxr_cfg['n_test']}): "
                    f"recall {c['recall']:.2f}, precision {c['precision']:.2f}, ROC AUC {c['roc_auc']:.2f}.")
    else:
        st.info('X-ray model not found in artifacts/, only the clinical model is used.')
    st.divider()
    st.caption('Student project for learning. It is not a medical device and must not be used to make decisions about anyone\'s health.')


# ---------------- inputs ----------------

st.title('🫀 Heart Disease Risk')
st.write('Fill in the clinical values, optionally upload a frontal chest X-ray, and press **Predict**. '
         'Fields you do not know can be left as "Unknown", the model fills them with the typical training value.')

UNKNOWN = 'Unknown'
cp_map = {'Typical angina': 1, 'Atypical angina': 2, 'Non-anginal pain': 3, 'No chest pain (asymptomatic)': 4}
ecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Left ventricular hypertrophy': 2}
slope_map = {UNKNOWN: np.nan, 'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
thal_map = {UNKNOWN: np.nan, 'Normal': 3, 'Fixed defect': 6, 'Reversible defect': 7}
yn_map = {'No': 0, 'Yes': 1}
yn_unknown_map = {UNKNOWN: np.nan, 'No': 0, 'Yes': 1}

rng = tab_cfg['ranges']
features = tab_cfg['features']


def num_input(col, label, help_text, step = 1.0, optional = False):
    lo, hi, med = rng[col]
    if optional:
        known = st.checkbox(f'I know my {label.lower()}', value = True, key = col + '_known')
        if not known:
            return np.nan
    fmt = '%.1f' if step < 1 else '%.0f'
    return st.number_input(label, min_value = float(lo), max_value = float(hi), value = float(round(med)), step = step, format = fmt, help = help_text)


left, right = st.columns([3, 2])

with left:
    st.subheader('Clinical values')
    a, b = st.columns(2)
    with a:
        age = num_input('age', 'Age (years)', 'Age in years.')
        sex = st.radio('Sex', ['Male', 'Female'], horizontal = True)
        cp = st.selectbox('Chest pain type', list(cp_map), help = 'Typical angina: pressure in the chest during effort that goes away with rest. '
                          'Atypical: only some of these signs. Non-anginal: chest pain that does not look heart related.')
        trestbps = num_input('trestbps', 'Resting blood pressure (mm Hg)', 'Systolic (upper) number at rest.')
        chol = num_input('chol', 'Cholesterol (mg/dl)', 'Total serum cholesterol from a blood test.', optional = True)
        fbs = st.radio('Fasting blood sugar > 120 mg/dl', list(yn_unknown_map), index = 1, horizontal = True)
    with b:
        restecg = st.selectbox('Resting ECG result', list(ecg_map))
        st.markdown('**Exercise (treadmill) test**')
        thalach = num_input('thalach', 'Max heart rate reached', 'Highest heart rate during the exercise test.')
        exang = st.radio('Chest pain during exercise', list(yn_map), horizontal = True)
        oldpeak = num_input('oldpeak', 'ST depression (oldpeak)', 'How much the ST segment of the ECG dropped during exercise compared to rest, in mm.', step = 0.1)
        slope = st.selectbox('Slope of peak exercise ST segment', list(slope_map))
        if 'ca' in features:
            ca = st.selectbox('Major vessels coloured by fluoroscopy', [UNKNOWN, 0, 1, 2, 3])
        if 'thal' in features:
            thal = st.selectbox('Thallium stress test', list(thal_map))

with right:
    st.subheader('Chest X-ray (optional)')
    if cxr_model is None:
        st.write('The X-ray model is not available in this deployment.')
        upload = None
    else:
        st.caption('Frontal (PA or AP) chest X-ray as PNG or JPG. The model only looks for an enlarged heart (cardiomegaly). '
                   'It will give a number for any image, so only upload chest X-rays.')
        upload = st.file_uploader('Upload X-ray', type = ['png', 'jpg', 'jpeg'])
        if upload is not None:
            st.image(Image.open(upload), caption = 'Uploaded image', width = 300)

predict = st.button('Predict', type = 'primary', width = 'stretch')


# ---------------- prediction ----------------

if predict:
    row = {'age': age, 'sex': 1 if sex == 'Male' else 0, 'cp': cp_map[cp], 'trestbps': trestbps, 'chol': chol,
           'fbs': yn_unknown_map[fbs], 'restecg': ecg_map[restecg], 'thalach': thalach, 'exang': yn_map[exang],
           'oldpeak': oldpeak, 'slope': slope_map[slope]}
    if 'ca' in features:
        row['ca'] = np.nan if ca == UNKNOWN else ca
    if 'thal' in features:
        row['thal'] = thal_map[thal]
    X = pd.DataFrame([{c: row[c] for c in features}], dtype = float)

    tab_prob = float(tab_model.predict_proba(X)[0, 1])
    tab_flag = tab_prob >= tab_cfg['threshold']

    cxr_prob, cxr_flag, cam_img = None, False, None
    if upload is not None:
        img = Image.open(upload)
        x = cxr_tf(img).unsqueeze(0)
        cxr_prob, cam = grad_cam(cxr_model, x)
        cxr_flag = cxr_prob >= cxr_cfg['threshold']
        cam_img = overlay(img, cam, cxr_cfg)

    st.divider()
    st.subheader('Result')

    # three levels instead of "either model flags": both thresholds are set to catch ~90% of sick patients,
    # so each model alone also flags many healthy people. One flag is common, two agreeing flags are stronger evidence.
    # (no averaged score: the models were trained on different patients, see notebook part 2)
    if cxr_prob is None:
        # only the clinical model was used, so there is no second opinion
        if tab_flag:
            st.warning('**Flagged (clinical values only).** No X-ray was uploaded, so there is no second check.')
        else:
            st.success('**Not flagged.** The clinical model is below its threshold. No X-ray was uploaded.')
    elif tab_flag and cxr_flag:
        st.error('**Higher concern.** Both the clinical values and the chest X-ray are above their thresholds.')
    elif tab_flag or cxr_flag:
        if tab_flag:
            which = 'Only the clinical values are above their threshold, the chest X-ray is not.'
        else:
            which = 'Only the chest X-ray is above its threshold, the clinical values are not.'
        st.warning(f'**Moderate concern.** {which} '
                   'One flag alone is common, because each model is set to catch about 9 in 10 sick patients and so also flags many healthy ones.')
    else:
        st.success('**Low concern.** Neither model is above its threshold.')

    r1, r2 = st.columns(2)
    with r1:
        st.metric('Clinical model: probability of heart disease', f'{tab_prob:.0%}')
        st.progress(min(max(tab_prob, 0.0), 1.0))
        st.caption(f"Flag threshold {tab_cfg['threshold']:.0%}. The probability is calibrated: of training patients with a "
                   f"value like this, about this share had > 50% narrowing of a major heart artery.")
    with r2:
        if cxr_prob is None:
            st.metric('X-ray model: cardiomegaly score', '-')
            st.caption('No X-ray uploaded.')
        else:
            st.metric('X-ray model: cardiomegaly score', f'{cxr_prob:.0%}')
            st.progress(min(max(cxr_prob, 0.0), 1.0))
            st.caption(f"Flag threshold {cxr_cfg['threshold']:.0%}. This score is not a calibrated probability.")
            st.image(cam_img, caption = 'Grad-CAM: red areas pushed the score up. For a good prediction they should be on the heart border.', width = 300)

    st.info('The two models were trained on different patients (UCI for the clinical values, NIH for the X-rays), '
            'so their numbers are shown separately and not averaged.')
