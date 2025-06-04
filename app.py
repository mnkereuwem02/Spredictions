import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained models and preprocessing objects
elasticity_model = joblib.load('elasticity_model.pkl')
rf_model = joblib.load('model.pkl')
scaler_ml = joblib.load('scaler.pkl')
label_encoders = joblib.load('encoders.pkl')

def encode_feature(feature_name, value):
    try:
        # If the value is already numeric, return it as an int.
        return int(value)
    except (ValueError, TypeError):
        if feature_name in label_encoders:
            encoder = label_encoders[feature_name]
            # Preprocess value if needed
            value_processed = value.strip() if isinstance(value, str) else value
            # Check if the processed value is among the known classes:
            if value_processed not in encoder.classes_:
                raise ValueError(f"Value '{value_processed}' not seen during training for feature '{feature_name}'. Known classes: {encoder.classes_}")
            # Return the encoded value
            return int(encoder.transform([value_processed])[0])
        return value


# Sidebar for user input
# Dictionary mapping states to cities
states_cities = {'Alabama': ['Decatur', 'Montgomery', 'Florence', 'Mobile', 'Auburn', 'Huntsville', 'Hoover', 'Tuscaloosa'], 
 'Arizona': ['Gilbert', 'Phoenix', 'Scottsdale', 'Tucson', 'Mesa', 'Sierra Vista', 'Glendale', 'Tempe', 'Peoria', 'Bullhead City', 'Avondale', 'Yuma', 'Chandler'], 
 'Arkansas': ['Fayetteville', 'Jonesboro', 'Little Rock', 'Hot Springs', 'Texarkana', 'Rogers', 'Conway', 'Pine Bluff', 'Springdale'],
 'California': ['Los Angeles', 'San Francisco', 'Roseville', 'Pasadena', 'San Jose', 'Redlands', 'Whittier', 'Santa Clara', 'San Diego', 'Brentwood', 'Inglewood', 'Long Beach', 'Hesperia', 'Huntington Beach', 'Concord', 'Costa Mesa', 'Anaheim', 'Vallejo', 'Mission Viejo', 'Lancaster', 'Lake Elsinore', 'Santa Ana', 'Salinas', 'Riverside', 'Torrance', 'Oceanside', 'Murrieta', 'Oakland', 'Encinitas', 'Antioch', 'Escondido', 'Fresno', 'Fairfield', 'Pico Rivera', 'Westminster', 'Pomona', 'Laguna Niguel', 'San Bernardino', 'Ontario', 'Rancho Cucamonga', 'Stockton', 'Sunnyvale', 'Manteca', 'Sacramento', 'Thousand Oaks', 'Coachella', 'La Quinta', 'Vacaville', 'Bakersfield', 'Redondo Beach', 'Apple Valley', 'Woodland', 'San Mateo', 'Visalia', 'Temecula', 'Yucaipa', 'Chula Vista', 'Lakewood', 'Citrus Heights', 'San Gabriel', 'Danville', 'Moreno Valley', 'Santa Barbara', 'La Mesa', 'Lake Forest', 'Redding', 'Chico', 'Redwood City', 'Santa Maria', 'Oxnard', 'Montebello', 'El Cajon', 'Camarillo', 'Burbank', 'Modesto', 'Davis', 'Morgan Hill', 'San Clemente', 'Dublin', 'San Luis Obispo', 'Lodi'],
 'Colorado': ['Aurora', 'Denver', 'Colorado Springs', 'Arvada', 'Louisville', 'Parker', 'Pueblo', 'Broomfield', 'Littleton', 'Fort Collins', 'Englewood', 'Thornton', 'Loveland', 'Commerce City', 'Longmont', 'Greeley'], 
 'Connecticut': ['Fairfield', 'Manchester', 'Norwich', 'Middletown', 'Meriden', 'Bristol', 'Waterbury', 'Milford', 'Shelton', 'Danbury'],
 'Delaware': ['Dover', 'Wilmington', 'Newark'],
 'District of Columbia': ['Washington'], 
 'Florida': ['Fort Lauderdale', 'Melbourne', 'Tampa', 'Tamarac', 'Saint Petersburg', 'Pembroke Pines', 'Miami', 'Jacksonville', 'Lakeland', 'Palm Coast', 'Hialeah', 'Boca Raton', 'Homestead', 'Coral Springs', 'Boynton Beach', 'Deltona', 'Port Orange', 'Apopka', 'Ormond Beach', 'Tallahassee', 'Pompano Beach', 'Orlando', 'North Miami', 'Plantation', 'Port Saint Lucie', 'Miramar', 'Hollywood', 'Daytona Beach', 'Margate', 'Pensacola', 'Jupiter', 'Delray Beach', 'West Palm Beach', 'Kissimmee', 'Sanford', 'Coral Gables'],
 'Georgia': ['Columbus', 'Atlanta', 'Warner Robins', 'Roswell', 'Macon', 'Smyrna', 'Sandy Springs', 'Athens', 'Marietta', 'East Point', 'Woodstock'], 
 'Idaho': ['Meridian', 'Boise', 'Lewiston', 'Pocatello', 'Caldwell', 'Twin Falls'], 
 'Illinois': ['Naperville', 'Chicago', 'Orland Park', 'Bloomington', 'Decatur', 'Quincy', 'Peoria', 'Park Ridge', 'Aurora', 'Evanston', 'Rockford', 'Skokie', 'Elmhurst', 'Woodstock', 'Wheeling', 'Highland Park', 'Bolingbrook', 'Tinley Park', 'Carol Stream', 'Champaign', 'Freeport', 'Danville', 'Oak Park', 'Frankfort', 'Buffalo Grove', 'Des Plaines', 'Palatine', 'Glenview', 'Normal', 'Saint Charles', 'Romeoville', 'Arlington Heights', 'Oswego'], 
 'Indiana': ['New Albany', 'Columbus', 'Richmond', 'La Porte', 'Indianapolis', 'South Bend', 'New Castle', 'Lafayette', 'Mishawaka', 'Noblesville', 'Lawrence', 'Bloomington', 'Greenwood', 'Portage', 'Elkhart'], 
 'Iowa': ['Urbandale', 'Des Moines', 'Burlington', 'Cedar Rapids', 'Dubuque', 'Iowa City', 'Waterloo', 'Marion'],
 'Kansas': ['Olathe', 'Overland Park', 'Manhattan', 'Wichita', 'Garden City'], 
 'Kentucky': ['Henderson', 'Richmond', 'Louisville', 'Florence', 'Murray', 'Bowling Green', 'Georgetown', 'Owensboro'],
 'Louisiana': ['Monroe', 'Bossier City', 'Lafayette', 'Lake Charles', 'Kenner'], 
 'Maine': ['Bangor', 'Lewiston'], 
 'Maryland': ['Columbia', 'Clinton', 'Rockville', 'Baltimore', 'Gaithersburg', 'Laurel', 'Hagerstown'], 
 'Massachusetts': ['Lowell', 'Franklin', 'Lawrence', 'New Bedford', 'Everett', 'Leominster', 'Quincy', 'Malden', 'Holyoke', 'Revere', 'Cambridge', 'Beverly', 'Andover', 'Marlborough'],
 'Michigan': ['Westland', 'Jackson', 'Saginaw', 'Detroit', 'Taylor', 'Canton', 'Dearborn', 'Rochester Hills', 'Trenton', 'Lansing', 'Royal Oak', 'Lincoln Park', 'Midland', 'Sterling Heights', 'Grand Rapids', 'Dearborn Heights', 'Mount Pleasant', 'Roseville', 'Oak Park', 'Ann Arbor', 'Holland'], 
 'Minnesota': ['Eagan', 'Rochester', 'Minneapolis', 'Saint Paul', 'Lakeville', 'Cottage Grove', 'Moorhead', 'Apple Valley', 'Maple Grove', 'Saint Cloud', 'Roseville', 'Woodbury', 'Coon Rapids'], 
 'Mississippi': ['Jackson', 'Gulfport', 'Southaven', 'Hattiesburg'], 
 'Missouri': ['Independence', 'Gladstone', 'Jefferson City', 'Saint Peters', 'Springfield', 'Columbia', 'Saint Louis', 'Kirkwood', 'Saint Charles'],
 'Montana': ['Great Falls', 'Missoula', 'Bozeman', 'Billings', 'Helena'],
 'Nebraska': ['Fremont', 'Omaha', 'Grand Island', 'Norfolk'], 
 'Nevada': ['Las Vegas', 'Reno', 'North Las Vegas', 'Henderson', 'Sparks'], 
 'New Hampshire': ['Concord', 'Dover', 'Nashua'], 
 'New Jersey': ['Westfield', 'Morristown', 'Belleville', 'Lakewood', 'Hackensack', 'Plainfield', 'Linden', 'New Brunswick', 'Vineland', 'Bridgeton', 'Bayonne', 'Paterson', 'Perth Amboy', 'Passaic', 'Orange', 'Atlantic City', 'Clifton', 'East Orange'], 
 'New Mexico': ['Carlsbad', 'Farmington', 'Albuquerque', 'Las Cruces', 'Santa Fe', 'Clovis', 'Rio Rancho'], 
 'New York': ['New York City', 'Troy', 'New Rochelle', 'Auburn', 'Lindenhurst', 'Rochester', 'Mount Vernon', 'Oceanside', 'Yonkers', 'Buffalo', 'Long Beach', 'Freeport', 'Niagara Falls', 'Watertown', 'Jamestown', 'Hempstead', 'Rome', 'Utica'], 
 'North Carolina': ['Concord', 'Durham', 'Charlotte', 'Chapel Hill', 'Wilmington', 'Gastonia', 'Burlington', 'Cary', 'Asheville', 'Jacksonville', 'Monroe', 'Greenville', 'Thomasville', 'Fayetteville', 'Greensboro', 'Raleigh', 'Hickory', 'Wilson', 'Goldsboro'],
 'North Dakota': ['Fargo'], 'Ohio': ['Columbus', 'Newark', 'Hamilton', 'Akron', 'Medina', 'Dublin', 'Cincinnati', 'Springfield', 'Grove City', 'Cleveland', 'Lorain', 'Toledo', 'Cuyahoga Falls', 'Parma', 'Fairfield', 'Lakewood', 'Kent', 'Marion', 'Bowling Green', 'Lancaster', 'Troy', 'Mentor', 'Elyria', 'Mason'], 
 'Oklahoma': ['Edmond', 'Norman', 'Tulsa', 'Muskogee', 'Oklahoma City', 'Broken Arrow', 'Lawton'], 
 'Oregon': ['Portland', 'Salem', 'Tigard', 'Redmond', 'Medford', 'Springfield', 'Gresham', 'Eugene', 'Hillsboro'],
 'Pennsylvania': ['Philadelphia', 'Chester', 'Lancaster', 'Allentown', 'Reading', 'York', 'Altoona', 'Bethlehem'], 
 'Rhode Island': ['Warwick', 'Providence', 'Woonsocket', 'Cranston'], 
 'South Carolina': ['Columbia', 'Florence', 'North Charleston', 'Summerville', 'Mount Pleasant', 'Rock Hill'], 
 'South Dakota': ['Sioux Falls', 'Rapid City', 'Aberdeen'], 
 'Tennessee': ['Memphis', 'Bristol', 'Franklin', 'Columbia', 'Murfreesboro', 'Jackson', 'Smyrna', 'Johnson City', 'Knoxville', 'Clarksville', 'Nashville', 'Chattanooga', 'Bartlett', 'Lebanon', 'Hendersonville'],
 'Texas': ['Fort Worth', 'Houston', 'Richardson', 'San Antonio', 'Grand Prairie', 'Dallas', 'Pasadena', 'Austin', 'Harlingen', 'Amarillo', 'Huntsville', 'Laredo', 'Arlington', 'Tyler', 'Garland', 'Round Rock', 'Brownsville', 'Irving', 'Coppell', 'Allen', 'El Paso', 'Grapevine', 'Carrollton', 'Plano', 'Keller', 'Lubbock', 'Mesquite', 'College Station', 'San Angelo', 'Haltom City', 'Frisco', 'Corpus Christi', 'Pharr', 'Missouri City', 'Pearland', 'Conroe', 'Odessa', 'Edinburg', 'Baytown', 'Bedford', 'Deer Park', 'Mcallen', 'Port Arthur', 'The Colony', 'League City', 'Waco', 'Cedar Hill', 'Texas City', 'Beaumont', 'Abilene', 'San Marcos', 'Mansfield', 'Bryan', 'La Porte'], 
 'Utah': ['West Jordan', 'Orem', 'Layton', 'Provo', 'Pleasant Grove', 'Salt Lake City', 'Murray', 'Logan', 'Lehi', 'Draper'],
 'Vermont': ['Burlington'], 'Virginia': ['Springfield', 'Arlington', 'Waynesboro', 'Richmond', 'Alexandria', 'Virginia Beach', 'Harrisonburg', 'Salem', 'Suffolk', 'Charlottesville', 'Chesapeake', 'Newport News', 'Hampton'], 
 'Washington': ['Seattle', 'Des Moines', 'Marysville', 'Vancouver', 'Edmonds', 'Olympia', 'Bellevue', 'Kent', 'Auburn', 'Spokane', 'Bellingham', 'Covington', 'Redmond', 'Pasco', 'Longview', 'Renton', 'Everett'],
 'West Virginia': ['Wheeling'], 
 'Wisconsin': ['Madison', 'Franklin', 'Green Bay', 'Milwaukee', 'Appleton', 'Kenosha', 'Waukesha', 'Eau Claire', 'West Allis', 'Superior', 'La Crosse', 'Wausau', 'Sheboygan'], 
 'Wyoming': ['Cheyenne']}

# Dictionary mapping categories to subcategories
categories_subcategories = {
    "Furniture": ["Bookcases", "Chairs", "Tables", "Furnishings"],
    "Office Supplies":['Labels', 'Storage', 'Art', 'Binders', 'Appliances', 'Paper',
       'Envelopes', 'Fasteners', 'Supplies'],
    "Technology": ['Phones', 'Accessories', 'Machines', 'Copiers']
}

st.header("Price Optimization & Sales Prediction")
# Sidebar for user input
st.subheader("Price Optimization Inputs")
price = st.number_input("Your Price", min_value=0.0, format="%.2f", step=0.1)
competitor_price = st.number_input("Competitor Price", min_value=0.0, format="%.2f", step=0.1)
Segment = st.text_input("Customer_Segment (e.g., 0)", value="0")
State = st.selectbox("State", list(states_cities.keys()))
city = st.selectbox("City", states_cities[State])
Ship_Mode = st.selectbox("Ship Mode", ['Second Class', 'Standard Class', 'First Class', 'Same Day'])
Region = st.selectbox("Region", ['South', 'West', 'Central', 'East'])
category = st.selectbox("Category", list(categories_subcategories.keys()))
sub_category = st.selectbox("Sub-Category", categories_subcategories[category])
order_year = st.number_input("Order_Year (e.g., 2022)", min_value=1900, max_value=2100, step=1)

if st.button("Predict Results"):
    # Encoding categorical inputs using pre-trained label encoders.
    # (Make sure that in training, the label encoders were fit with these keys.)
    ship_mode_encoded = label_encoders['Ship Mode'].transform([Ship_Mode])[0]
    state_encoded = label_encoders['State_x'].transform([State])[0]
    city_encoded = label_encoders['City_x'].transform([city])[0]
    region_encoded = label_encoders['Region'].transform([Region])[0]
    category_encoded = label_encoders['Category'].transform([category])[0]
    sub_category_encoded = label_encoders['Sub-Category'].transform([sub_category])[0]

    # Compute Price Ratio
    # Check to prevent division by zero
    if competitor_price == 0:
        st.error("competitor price cannot be zero!")
    else:
        price_ratio = price / competitor_price

    # Preparing input for the demand elasticity model
    # We add a constant (1) for the regression intercept; the model was trained on [const, log_Price, Price_Ratio]
    

        elasticity_input = np.array([[1, np.log(price), price_ratio]])
        predicted_elasticity = elasticity_model.predict(elasticity_input)[0]
    
    # Preparing input for the ML model for quantity prediction.
    # The training features order is assumed to be:
    # ['Price_x', 'Price_y', 'Price_Ratio', 'Customer_Segment', 'Ship Mode', 'State_x', 'City_x', 'Region', 'Order Year']
    ml_input = np.array([[price, competitor_price, price_ratio, Segment, ship_mode_encoded, state_encoded, city_encoded, region_encoded,category_encoded,sub_category_encoded, order_year]])
    ml_input_scaled = scaler_ml.transform(ml_input)
    predicted_quantity = rf_model.predict(ml_input_scaled)[0]

    # Compute estimated revenue
    estimated_revenue = price * predicted_quantity

    # Display results
    st.subheader("Results")
    st.write(f"**Estimated Elasticity:** {predicted_elasticity:.2f}")
    st.write(f"**Predicted Quantity Sold:** {predicted_quantity:.0f}")
    st.write(f"**Projected Revenue:** ${estimated_revenue:,.2f}")

    # Revenue Optimization: Iterate through price options and compute the revenue
    st.subheader("Optimal Pricing Analysis")
    price_options = np.arange(price * 0.8, price * 1.2, price * 0.05)
    best_revenue = 0
    best_price = price
    for p in price_options:
        price_ratio_test = p / competitor_price
        elasticity_input_test = np.array([[1, np.log(p), price_ratio_test]])
        predicted_quantity_test = elasticity_model.predict(elasticity_input_test)[0]
        revenue_test = p * predicted_quantity_test
        if revenue_test > best_revenue:
            best_revenue = revenue_test
            best_price = p
    
    st.write(f"**Recommended Price for Maximum Revenue:** ${best_price:,.2f}")
    st.write(f"**Expected Revenue at Optimal Price:** ${best_revenue:,.2f}")
    
# Input for price change percentage
price_change_pct = st.number_input("Price Change (%)", value=0.0, step=0.1)


if st.button("Compute Elasticity Impact"):
        try:
        # Retrieving elasticity coefficient from my model
            elasticity_coefficient = elasticity_model.params["log_Price"]
        # Computing the estimated percentage change in quantity
            estimated_change_in_qty_pct = elasticity_coefficient * price_change_pct
            st.success(f"A {price_change_pct:.2f}% change in price is estimated to lead to a "
                   f"{estimated_change_in_qty_pct:.2f}% change in quantity demanded.")
        except Exception as e:
            st.error(f"Error retrieving elasticity coefficient: {e}")
