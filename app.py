###################################################################2
##########################33
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import requests
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
from dash import html
import plotly.graph_objects as go
from dash import Dash
import dash_leaflet as dl
import dash_leaflet.express as dlx
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import random

warnings.filterwarnings("ignore")
nlp = spacy.load("network_model")

# Load the solutions dataset
df = pd.read_csv("network_issues.csv", encoding='latin1', header=None, names=['issue', 'classification'], usecols=[0, 1])

app = Flask(__name__)

dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/', external_stylesheets=[dbc.themes.DARKLY, '/static/custom.css'])



sarimax_model = joblib.load('sarimax_model.pkl')
gbr_model = joblib.load('gbr_model.pkl')
svr_model = joblib.load('svr_model.pkl')
meta_model = joblib.load('meta_model.pkl')

# cell names / cities / LocalCell Ids
df = pd.read_csv('The_data.csv')
cell_names = df['Cell Name'].unique().tolist()
cell_to_city = dict(zip(df['Cell Name'], df['Cities']))
cell_to_local_id = dict(zip(df['Cell Name'], df['LocalCell Id']))

# Calculate Historical Interference for each Cell name
### kenou for each cell id
df['Historical Interference'] = df.groupby('Cell Name')['FT_UL.Interference'].transform(lambda x: x.shift(1).fillna(method='bfill'))
df['Historical Interference_Mean'] = df.groupby('Cell Name')['FT_UL.Interference'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
if 'Latitude' not in df.columns:
    df['Latitude'] = 0  # 7attit default value
if 'Longitude' not in df.columns:
    df['Longitude'] = 0  


api_key = '56bd3212a99808185e5d61c67a907279'

def get_weather_forecast(city):
    geo_url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
    geo_res = requests.get(geo_url)
    geo_data = geo_res.json()

    if geo_res.status_code == 200:
        lat = geo_data['coord']['lat']
        lon = geo_data['coord']['lon']

        cnt = 16  #3andi l7a9 bel API hedha ken fi 16 days weather forecast
        forecast_url = f'http://api.openweathermap.org/data/2.5/forecast/daily?lat={lat}&lon={lon}&cnt={cnt}&appid={api_key}&units=metric'
        forecast_res = requests.get(forecast_url)
        forecast_data = forecast_res.json()

        if forecast_res.status_code == 200:
            daily_forecasts = forecast_data['list']
            
            forecast_list = []
            for day in daily_forecasts:
                dt = datetime.utcfromtimestamp(day['dt']).strftime('%Y-%m-%d')
                temp_day = day['temp']['day']
                humidity = day['humidity']
                wind_speed = day['speed']

                forecast_list.append({
                    'City': city,
                    'Date': dt,
                    'Temperature (Â°C)': temp_day,
                    'Humidity (%)': humidity,
                    'Wind Speed (mph)': wind_speed
                })

            return forecast_list
        else:
            print(f"Error in getting forecast data for {city}: {forecast_data['message']}")
    else:
        print(f"Error in getting location data for {city}: {geo_data['message']}")
    return []

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html', cell_names=cell_names)

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')


###########################################################################################

tokenizer = BertTokenizer.from_pretrained('chatbot_model')
model = BertForSequenceClassification.from_pretrained('chatbot_model')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Predefined responses
responses = {
    "greeting": ["Hello! How can I assist you today?", "Hi there! How can I help you?"],
    "farewell": ["Goodbye! Have a great day!", "See you later!"],
    "issue_internal": "It seems like this is an internal issue. You might want to check the internal settings and configurations.",
    "issue_external": "It seems like this is an external issue. You should investigate external factors such as network interference.",
    "default": "I'm not sure how to respond to that. Can you please provide more details?"
}

def classify_issue(user_input):
    try:
        inputs = tokenizer(user_input, return_tensors='pt', max_length=128, padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()

        return "issue_internal" if pred_label == 1 else "issue_external"
    except Exception as e:
        print(f"Error in classify_issue: {e}")
        return "default"

def get_response(user_input):
    user_input = user_input.lower()

    # Pattern matching for greetings and farewells
    if any(greeting in user_input for greeting in ["hi", "hello", "hey"]):
        return random.choice(responses["greeting"])
    
    if any(farewell in user_input for farewell in ["bye", "goodbye", "see you"]):
        return random.choice(responses["farewell"])
    
    # Use BERT model for issue classification
    response_key = classify_issue(user_input)
    return responses.get(response_key, responses["default"])

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        if user_input:
            response = get_response(user_input)
            return jsonify({'response': response})
        return jsonify({'response': responses["default"]})
    except Exception as e:
        print(f"Error in /chat route: {e}")
        return jsonify({'response': "An error occurred while processing your request."})


###########################################################################################################

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        cell_name = data['Cell Name']
        city = cell_to_city[cell_name]
        local_cell_id = cell_to_local_id[cell_name]

        weather_forecast = get_weather_forecast(city)
        predictions = []

       
        cell_data = df[df['Cell Name'] == cell_name]

        for forecast in weather_forecast:
            historical_interference = cell_data['Historical Interference'].iloc[-1]
            historical_interference_mean = cell_data['Historical Interference_Mean'].iloc[-1]

            input_df = pd.DataFrame({
                'Temperature (Â°C)': [forecast['Temperature (Â°C)']],
                'Humidity (%)': [forecast['Humidity (%)']],
                'Wind Speed (mph)': [forecast['Wind Speed (mph)']],
                'LocalCell Id': [local_cell_id],
                'Historical Interference': [historical_interference],
                'Historical Interference_Mean': [historical_interference_mean]
            })

            # models predictions
            sarimax_prediction = sarimax_model.forecast(steps=1, exog=input_df[['Temperature (Â°C)', 'Humidity (%)', 'Wind Speed (mph)', 'LocalCell Id', 'Historical Interference', 'Historical Interference_Mean']])
            
            gbr_prediction = gbr_model.predict(input_df[['Temperature (Â°C)', 'Humidity (%)', 'Wind Speed (mph)', 'LocalCell Id', 'Historical Interference', 'Historical Interference_Mean']])
            
            svr_prediction = svr_model.predict(input_df[['Temperature (Â°C)', 'Humidity (%)', 'Wind Speed (mph)', 'LocalCell Id', 'Historical Interference', 'Historical Interference_Mean']])

            # Stack predictions
            stacked_manual = pd.DataFrame({
                'SARIMAX': sarimax_prediction,
                'GBR': gbr_prediction,
                'SVR': svr_prediction
            })

            # meta model
            meta_prediction = meta_model.predict(stacked_manual)
            rounded_prediction = round(meta_prediction[0], 5)

            predictions.append({
                'Date': forecast['Date'],
                'Prediction': rounded_prediction,
                'City': city,
                'Cell Name': cell_name,
                'Temperature (Â°C)': forecast['Temperature (Â°C)'],
                'Humidity (%)': forecast['Humidity (%)'],
                'Wind Speed (mph)': forecast['Wind Speed (mph)']
            })

        # Load existing predictions ken el file mawjoud
        try:
            existing_data = pd.read_csv('predictions.csv')
        except FileNotFoundError:
            existing_data = pd.DataFrame()

        #new predictions lel existing data
        global prediction_data
        prediction_data = pd.concat([existing_data, pd.DataFrame(predictions)], ignore_index=True)

        prediction_data.to_csv('predictions.csv', index=False)

        return jsonify({'predictions': predictions})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_date_range():
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    max_date = (pd.Timestamp.now() + pd.DateOffset(days=15)).strftime('%Y-%m-%d')
    return today, max_date
min_date, max_date = get_date_range()




######################################



# layout for Dash app
dash_app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Dropdown(id='city-dropdown', options=[
                {'label': city, 'value': city} for city in df['Cities'].unique()
            ], placeholder="Select a city", className="dropdown"),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                initial_visible_month=min_date,
                date=min_date,
                className="date-picker"
            ),
        ]), width=3, className="side-panel"),
        dbc.Col(html.Div([
            dl.Map(
                [
                    dl.TileLayer(),
                    dl.LayerGroup(id="marker-layer")
                ],
                center=[33.8869, 9.5375],
                zoom=6,
                id="leaflet-map",
                style={'height': '50vh'}
            )
        ]), width=9, className="main-panel"),
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Graph(id='temperature-graph')
        ]), width=4),
        dbc.Col(html.Div([
            dcc.Graph(id='humidity-graph')
        ]), width=4),
        dbc.Col(html.Div([
            dcc.Graph(id='wind-speed-graph')
        ]), width=4),
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Graph(id='pie-chart')
        ]), width=12),
    ]),
], className="container")




@dash_app.callback(
    Output('pie-chart', 'figure'),
    [Input('city-dropdown', 'value'),
     Input('date-picker', 'date')]
)
def update_colored_bar_chart(city, selected_date):
    if city and selected_date:
        try:
            print("Loading data from predictions.csv...")
            prediction_data = pd.read_csv('predictions.csv')
            
            print("Prediction Data Loaded Successfully.")  # Debugging statement
            print("Prediction Data Columns:", prediction_data.columns)  # Debugging statement
            print("Data Sample:\n", prediction_data.head())  # Debugging statement
            
            # Ensure date format 
            prediction_data['Date'] = pd.to_datetime(prediction_data['Date']).dt.strftime('%Y-%m-%d')
            print("Unique Dates in Prediction Data:", prediction_data['Date'].unique())  # Debugging statement
            
            # Filter predictions for selected city and date
            city_predictions = prediction_data[
                (prediction_data['City'] == city) &
                (prediction_data['Date'] == selected_date)
            ]
            
            print(f"Filtered Predictions:\n{city_predictions}")  # Debugging 
            
            # Get the top 3 worst cells
            top_cells = city_predictions.nlargest(3, 'Prediction')
            print(f"Top 3 Cells:\n{top_cells}")  # Debugging 
            
            if not top_cells.empty:
                fig = px.bar(
                    top_cells,
                    x='Cell Name',
                    y='Prediction',
                    color='Prediction',  
                    title=f'Top worst Cells in {city} on {selected_date}',
                    labels={'Cell Name': 'Cell Name', 'Prediction': 'Predicted Interference'},
                    height=400
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                    plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
                    font=dict(color='white'),        
                    title_font=dict(color='white'),   
                    xaxis_title_font=dict(color='white'),  
                    yaxis_title_font=dict(color='white')   
                )
                return fig
            else:
                print("No predictions available for the selected city and date.")  # Debugging 
        
        except FileNotFoundError:
            print("predictions.csv file not found.")  # Debugging 
        except Exception as e:
            print(f"Error loading or processing data: {e}")  # Debugging 
    
    return go.Figure()  # Return empty figure if no data



city_coordinates = {
    'Tunis': {'lat': 36.8065, 'lon': 10.1815},
    'Sfax': {'lat': 34.7404, 'lon': 10.7608},
    'Bizerte': {'lat': 37.2705, 'lon': 9.8739},
    'Ariana': {'lat': 36.8667, 'lon': 10.1667},
    'Monastir': {'lat': 35.7643, 'lon': 10.8113},
    'Nabeul': {'lat': 36.4561, 'lon': 10.7376},
}
@dash_app.callback(
    [Output('temperature-graph', 'figure'),
     Output('humidity-graph', 'figure'),
     Output('wind-speed-graph', 'figure')],
    [Input('city-dropdown', 'value'),
     Input('date-picker', 'date')]
)
def update_weather_graphs(city, selected_date):
    if city and selected_date:
        weather_forecast = get_weather_forecast(city)
        
        # Filter forecast data for selected date
        weather_df = pd.DataFrame(weather_forecast)
        selected_weather = weather_df[weather_df['Date'] == selected_date]
        
        if not selected_weather.empty:
            selected_weather = selected_weather.iloc[0]
            
            # Create gauges
            temp_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=selected_weather['Temperature (Â°C)'],
                gauge={
                    'axis': {'range': [None, 50], 'tickwidth': 1},
                    'bar': {'color': "red"},
                    'steps': [{'range': [0, 100], 'color': 'lightgrey'}],
                },
                number={'font': { 'color': 'lightgrey'}},
                title={'text': f"Temperature (°C)", 'font': {'color': 'lightgrey'}} 
            )).update_layout(
                height=200, 
                width=200, 
                margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                paper_bgcolor='rgba(0,0,0,0)',  
                plot_bgcolor='rgba(0,0,0,0)'    
            )

            humidity_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=selected_weather['Humidity (%)'],
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "blue"},
                    'steps': [{'range': [0, 100], 'color': 'lightgrey'}],
                },
                number={'font': { 'color': 'lightgrey'}},
                title={'text': f"Humidity (%)", 'font': {'color': 'lightgrey'}} 
            )).update_layout(
                height=200, 
                width=200, 
                margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)'    
            )

            wind_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=selected_weather['Wind Speed (mph)'],
                gauge={
                    'axis': {'range': [None, 35], 'tickwidth': 1},
                    'bar': {'color': "black"},
                    'steps': [{'range': [0, 100], 'color': 'lightgrey'}],
                },
                number={'font': { 'color': 'lightgrey'}},
                title={'text': f"Wind Speed (mph)", 'font': {'color': 'lightgrey'}} 
            )).update_layout(
                height=200, 
                width=200, 
                margin={'t': 0, 'b': 0, 'l': 0, 'r': 0},
                paper_bgcolor='rgba(0,0,0,0)',  
                plot_bgcolor='rgba(0,0,0,0)'   
            )
            
            return temp_fig, humidity_fig, wind_fig

   
    return go.Figure(), go.Figure(), go.Figure()


@dash_app.callback(
    Output('city-dropdown', 'value'),
    [Input('leaflet-map', 'clickData')]
)
def update_city_from_map_click(clickData):
    if clickData:
        lat = clickData['latlng']['lat']
        lon = clickData['latlng']['lng']
        closest_city = None
        min_distance = float('inf')

        for city, coords in city_coordinates.items():
            city_lat = coords['lat']
            city_lon = coords['lon']
            distance = ((lat - city_lat) ** 2 + (lon - city_lon) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_city = city

        return closest_city
    return None


if __name__ == '__main__':
    app.run(debug=True)
