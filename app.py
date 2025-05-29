import streamlit as st
import joblib
import numpy as np
import pandas as pd
import gzip
import pickle

with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)

# Load model and preprocessing tools
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("encoders.pkl")  # dictionary of LabelEncoders

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

st.title("Sales Prediction Application")
# Input widgets
Ship_Mode = st.selectbox("Ship Mode", ['Second Class', 'Standard Class', 'First Class', 'Same Day'])
Segment = st.selectbox("Segment",['Consumer', 'Corporate', 'Home Office'])
State = st.selectbox("State", list(states_cities.keys()))
city = st.selectbox("City", states_cities[State])  
Region = st.selectbox("Region",['South', 'West', 'Central', 'East'])
category = st.selectbox("Category", list(categories_subcategories.keys()))
sub_category = st.selectbox("Sub-Category", categories_subcategories[category])
quantity = st.number_input("Quantity", min_value=1)
Discount = st.number_input('Discount',min_value=0.0, max_value=1.0, value=0.0, step=0.01)
profit = st.number_input('Profit', min_value=-1000.0, max_value=5000.0, value=0.0, step=0.01)




# Predict button
if st.button("Predict Sales"):
    input_data = pd.DataFrame({
        'Ship Mode': [Ship_Mode],
        'Segment':[Segment],
        'State': [State],
        'City': [city],
        'Region':[Region],
        'Category': [category],
        'Sub-Category': [sub_category],
        'Quantity': [quantity],
       'Discount':[Discount],
       'Profit':[profit]
    })

    # Encode each categorical column using its own LabelEncoder
    for col in label_encoders:
        if col in input_data.columns:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Sales: {prediction[0]}")

