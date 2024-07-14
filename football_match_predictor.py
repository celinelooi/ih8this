import os
from joblib import load
import streamlit as st
import pandas as pd
from datetime import datetime

# Function to load the trained model
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    return load(model_path)

# Define the model path and load the model
model_path = os.path.join(os.path.dirname(__file__), 'trained_model.joblib')
model = load_model(model_path)

if model is not None:
    st.write("Model loaded successfully")
else:
    st.stop()

# Title and description
st.title('Football Match Predictor')
st.write('Enter the details of the match to predict the outcome.')

# Display a football image
st.image('football.jpg', caption='Football Match Predictor', use_column_width=True)

# Team names from the CSV
teams = ['Arsenal', 'Liverpool', 'Manchester City', 'Aston Villa', 'Tottenham Hotspur', 'Manchester United', 
         'Newcastle United', 'West Ham United', 'Chelsea', 'Bournemouth', 'Brighton and Hove Albion', 
         'Wolverhampton Wanderers', 'Fulham', 'Crystal Palace', 'Brentford', 'Everton', 'Nottingham Forest', 
         'Luton Town', 'Burnley', 'Sheffield United', 'Leicester City', 'Leeds United', 'Southampton']

# Input fields for match details
team = st.selectbox('Team', options=teams)
opponent = st.selectbox('Opponent', options=teams)
if team == opponent:
    st.error("Team and opponent cannot be the same. Please select different teams.")
    st.stop()

venue = st.selectbox('Venue', options=['Home', 'Away'])
date = st.date_input('Date', min_value=datetime.now().date())
time = st.time_input('Kick-off Time')

# Convert inputs to model input format
team_code = teams.index(team)  # Encoding team based on their index
venue_code = 1 if venue == 'Home' else 0
opp_code = teams.index(opponent)  # Encoding opponents based on their index
hour = time.hour
day_code = date.weekday()  # Monday is 0 and Sunday is 6

# Prepare the features as expected by the model
features = pd.DataFrame({
    'team_code': [team_code],
    'venue_code': [venue_code],
    'opp_code': [opp_code],
    'hour': [hour],
    'day_code': [day_code]
})

# Prediction button
if st.button('Predict Result'):
    # Ensure the features match the expected input format
    expected_features = ['team_code', 'venue_code', 'opp_code', 'hour', 'day_code']
    if set(features.columns) != set(expected_features):
        st.error(f"Feature mismatch: Model expects 5 features but received {len(features.columns)}")
    else:
        # Predict
        prediction = model.predict(features)
        result = 'Win' if prediction[0] == 1 else 'Lose'
        
        # Display the prediction
        st.write(f'The prediction for {team} against {opponent} is: {result}')
