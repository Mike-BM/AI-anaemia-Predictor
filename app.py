import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import os
import io
from geopy.geocoders import Nominatim

MODEL_PATH = 'anemia_model.pkl'

# --- Theme Switcher ---
# (CSS and theme switcher removed as requested)

# --- Welcome Message ---
st.balloons()
st.info('👋 **Welcome to the Anaemia Prediction App!**\n\nThis tool helps you predict anaemia status using blood image features and hemoglobin. Enter the details below and get instant results!', icon="💡")

# --- Sidebar ---
st.sidebar.title('🩸 Anaemia Prediction App')
st.sidebar.info('''
Enter the patient details in the main panel to predict anaemia status. 

**Features used:**
- Sex (M/F)
- %Red Pixel
- %Green pixel
- %Blue pixel
- Hemoglobin (Hb)
''')

# --- Image Upload ---
st.markdown('<div class="main-title">🩺 Anaemia Prediction</div>', unsafe_allow_html=True)
st.subheader('Predict if a patient is anaemic based on blood image features and hemoglobin')

uploaded_image = st.file_uploader('Upload a blood sample image (only blood-based images accepted)', type=['png', 'jpg', 'jpeg'])
image_is_blood = False
if uploaded_image:
    from PIL import Image
    import numpy as np
    image = Image.open(uploaded_image)
    arr = np.array(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        red_ratio = np.mean(arr[:,:,0]) / (np.mean(arr[:,:,1]) + np.mean(arr[:,:,2]) + 1e-5)
        if red_ratio > 1.2:  # Threshold can be adjusted for strictness
            image_is_blood = True
            st.image(uploaded_image, caption='Your Uploaded Image', use_container_width=True)
        else:
            st.error("Please upload a valid blood sample image (image does not appear to be blood-based).")
    else:
        st.error("Invalid image format. Please upload a color image of a blood sample.")
else:
    st.image('https://cdn.pixabay.com/photo/2017/01/31/13/14/anemia-2028244_1280.png', caption='Healthy Blood Cells', use_container_width=True)

# --- Input Fields in Columns ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-block">', unsafe_allow_html=True)
    sex = st.selectbox('Sex', ['M', 'F'], help='Select M for Male, F for Female')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-block">', unsafe_allow_html=True)
    red_pixel = st.number_input('%Red Pixel', min_value=0.0, max_value=100.0, value=45.0, help='Percentage of red pixels in image')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-block">', unsafe_allow_html=True)
    hb = st.number_input('Hemoglobin (Hb)', min_value=0.0, max_value=25.0, value=10.0, help='Hemoglobin level (g/dL)')
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-block">', unsafe_allow_html=True)
    green_pixel = st.number_input('%Green pixel', min_value=0.0, max_value=100.0, value=30.0, help='Percentage of green pixels in image')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-block">', unsafe_allow_html=True)
    blue_pixel = st.number_input('%Blue pixel', min_value=0.0, max_value=100.0, value=25.0, help='Percentage of blue pixels in image')
    st.markdown('</div>', unsafe_allow_html=True)

# --- Location Input ---
st.markdown('---')
st.markdown('### 📍 Patient Location')
location_mode = st.radio('How would you like to enter your location?', ['County/Subcounty', 'Latitude/Longitude'])
latitude = longitude = None
location_text = ''

if location_mode == 'County/Subcounty':
    with st.form("location_form"):
        county = st.text_input('Enter County (e.g., Kitui)')
        subcounty = st.text_input('Enter Subcounty/Town/Area (optional)')
        country = st.text_input('Enter Country', value='Kenya')
        location_parts = [part for part in [subcounty, county, country] if part.strip()]
        location_text = ', '.join(location_parts)
        find_location = st.form_submit_button("Find Location")
        if find_location and location_text:
            geolocator = Nominatim(user_agent="your_email_or_app_name_here")
            try:
                location = geolocator.geocode(location_text, timeout=10)
                if location:
                    latitude, longitude = location.latitude, location.longitude
                    st.success(f"Found location: {location_text} ({latitude}, {longitude})")
                    # Display map immediately after successful lookup
                    st.markdown('<div class="map-card">', unsafe_allow_html=True)
                    st.markdown('#### 🗺️ Patient Location Map')
                    st.markdown('<span style="color:#1e88e5;">This map shows the location you entered for the patient. You can use this to visualize where the prediction was made.</span>', unsafe_allow_html=True)
                    map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
                    st.map(map_df)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning('Could not find the location. Please check your input or try entering latitude/longitude manually.')
            except Exception as e:
                st.warning('Geocoding service is currently unavailable. Please try again later or enter latitude/longitude manually.')
else:
    latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=0.0)
    longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=0.0)

# --- Encode categorical variables ---
sex_encoded = 0 if sex == 'M' else 1

# --- Prepare input for prediction ---
input_data = pd.DataFrame({
    'Sex': [sex_encoded],
    '%Red Pixel': [red_pixel],
    '%Green pixel': [green_pixel],
    '%Blue pixel': [blue_pixel],
    'Hb': [hb]
})

# --- Data Summary Display ---
st.markdown('<div class="user-summary"><b>📝 Your Input Summary:</b><br>'
            f'Sex: <b>{sex}</b> | %Red Pixel: <b>{red_pixel}</b> | %Green pixel: <b>{green_pixel}</b> | %Blue pixel: <b>{blue_pixel}</b> | Hb: <b>{hb}</b></div>', unsafe_allow_html=True)
if latitude is not None and longitude is not None:
    if location_text:
        st.markdown(f'<b>Location:</b> {location_text} <br><b>Coordinates:</b> ({latitude}, {longitude})', unsafe_allow_html=True)
    else:
        st.markdown(f'<b>Coordinates:</b> ({latitude}, {longitude})', unsafe_allow_html=True)

# --- Predict Button ---
predict_btn = st.button('🔍 Predict Anaemia', use_container_width=True)

# --- Progress Bar/Loader ---
if predict_btn:
    with st.spinner('Predicting...'):
        clf = load(MODEL_PATH)  # reload in case theme switcher changed state
        prediction = clf.predict(input_data)[0]
        confidence = None
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(input_data)[0]
            confidence = np.max(proba)
        # --- Advanced Health Tips ---
        tips = []
        learn_more = ''
        if hb < 12:
            tips.append('Increase iron-rich foods (spinach, beans, red meat).')
        if green_pixel > 40:
            tips.append('Stay hydrated; high green pixel may indicate plasma presence.')
        if prediction == 1:
            tips.append('Consult your doctor for supplements if needed.')
            tips.append('Get regular checkups.')
            learn_more = '[Learn more about anaemia (WHO)](https://www.who.int/news-room/fact-sheets/detail/anaemia)'
        else:
            tips.append('Maintain a balanced diet.')
            tips.append('Exercise regularly.')
            tips.append('Get enough sleep.')
            tips.append('Have regular health checkups.')
            learn_more = '[General health tips (Mayo Clinic)](https://www.mayoclinic.org/healthy-lifestyle)'
        # --- Feedback ---
        if prediction == 1:
            st.markdown('<span style="font-size:1.5em; color:#B22222;">🩸 <b>Prediction: Anaemic</b></span>', unsafe_allow_html=True)
            if confidence is not None:
                st.markdown(f'<span style="color:#B22222;">Confidence: <b>{confidence*100:.1f}%</b></span>', unsafe_allow_html=True)
            st.markdown('''<span style="color:#B22222;font-size:1.2em;">The patient is likely <b>Anaemic</b>. Please consult a healthcare professional for further advice.</span>''', unsafe_allow_html=True)
        else:
            st.markdown('<span style="font-size:1.5em; color:#228B22;">🟢 <b>Prediction: Not Anaemic</b></span>', unsafe_allow_html=True)
            if confidence is not None:
                st.markdown(f'<span style="color:#228B22;">Confidence: <b>{confidence*100:.1f}%</b></span>', unsafe_allow_html=True)
            st.markdown('''<span style="color:#228B22;font-size:1.2em;">The patient is likely <b>Not Anaemic</b>. Keep up the healthy habits!</span>''', unsafe_allow_html=True)
        # --- Health Tips Section ---
        st.markdown('<div class="health-tips"><b>💡 Personalized Health Tips:</b><ul>' + ''.join([f'<li>{tip}</li>' for tip in tips]) + f'</ul>{learn_more}</div>', unsafe_allow_html=True)
        # --- Downloadable Report ---
        report = io.StringIO()
        report.write('Anaemia Prediction Report\n')
        report.write('========================\n')
        report.write(f'Sex: {sex}\n')
        report.write(f'%Red Pixel: {red_pixel}\n')
        report.write(f'%Green pixel: {green_pixel}\n')
        report.write(f'%Blue pixel: {blue_pixel}\n')
        report.write(f'Hemoglobin (Hb): {hb}\n')
        report.write(f'Prediction: {"Anaemic" if prediction == 1 else "Not Anaemic"}\n')
        if confidence is not None:
            report.write(f'Confidence: {confidence*100:.1f}%\n')
        report.write('\nHealth Tips:\n')
        for tip in tips:
            report.write(f'- {tip}\n')
        report.write(f'\n{learn_more}\n')
        st.download_button('⬇️ Download Report', data=report.getvalue(), file_name='anaemia_report.txt', mime='text/plain')
        # --- Map Visualization ---
        if latitude is not None and longitude is not None:
            st.markdown('<div class="map-card">', unsafe_allow_html=True)
            st.markdown('#### 🗺️ Patient Location Map')
            st.markdown('<span style="color:#1e88e5;">This map shows the location you entered for the patient. You can use this to visualize where the prediction was made.</span>', unsafe_allow_html=True)
            map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
            st.map(map_df)
            st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown('<div class="footer">@2025 All rights reserved</div>', unsafe_allow_html=True)
