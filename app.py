import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Rain Prediction Dashboard", layout="wide")

# ---------------- COSMIC BACKGROUND ----------------
st.markdown("""
<style>

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #141E30);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    color: white;
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
    margin-bottom: 20px;
    transition: 0.3s;
}

.glass:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.5);
}

.title {
    font-size: 50px;
    text-align: center;
    font-weight: bold;
    margin-bottom: 20px;
}

div.stButton > button {
    background: linear-gradient(90deg,#ff512f,#dd2476);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px #ff512f;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("rain_model.pkl")

# ---------------- HEADER ----------------
st.markdown('<div class="title">🌧 Rain Prediction Dashboard</div>', unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    humidity = st.slider("💧 Humidity (3pm)", 0, 100, 50)
    pressure = st.slider("🌡 Pressure (3pm)", 980, 1050, 1000)

with col2:
    temp = st.slider("🌞 Temperature (3pm)", 0, 50, 25)
    wind = st.slider("💨 Wind Speed (3pm)", 0, 100, 20)

rain_today_option = st.selectbox("🌦 Did it rain today?", ["No", "Yes"])
rain_today = 1 if rain_today_option == "Yes" else 0

user_note = st.text_area("📝 Add optional note")

# ---------------- PREDICT ----------------
if st.button("🔮 Predict Weather"):

    with st.spinner("Analyzing atmospheric patterns..."):
        time.sleep(1.5)

        input_data = np.array([[humidity, pressure, temp, wind, rain_today]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]
        confidence = probability * 100

    # ---------------- DASHBOARD CARDS ----------------
    c1, c2, c3 = st.columns(3)

    # Card 1 - Confidence
    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📊 Model Confidence")
        st.progress(int(confidence))
        st.write(f"Confidence Level: {confidence:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Card 2 - Input Summary
    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🌡 Input Summary")
        st.write(f"Humidity: {humidity}%")
        st.write(f"Pressure: {pressure} hPa")
        st.write(f"Temperature: {temp} °C")
        st.write(f"Wind Speed: {wind} km/h")
        st.markdown('</div>', unsafe_allow_html=True)

    # Card 3 - Trend Graph
    with c3:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📈 Simulated Trend")

        x = np.arange(0,50)
        y = np.sin(x/5)+np.random.normal(0,0.2,50)+2

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x,y=y,mode='lines',
                                 line=dict(color='#00f5ff',width=3)))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=0,b=0),
            height=250
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- FINAL RESULT CARD ----------------
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if prediction == 1:
        st.error(f"🌧 Rain Tomorrow: YES ({confidence:.2f}% confidence)")
        st.image("https://cdn-icons-png.flaticon.com/512/1163/1163624.png", width=120)
    else:
        st.success(f"☀️ Rain Tomorrow: NO ({confidence:.2f}% confidence)")
        st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=120)

    if user_note:
        st.info(f"📝 Note: {user_note}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<br><center>Built with ❤️ using Streamlit | Machine Learning</center>", unsafe_allow_html=True)