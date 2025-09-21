import streamlit as st
import pandas as pd
import pickle
import os


# Load models
with open(os.path.join("Model", "disease_encoder.pkl"), "rb") as f:
    disease_encoder = pickle.load(f)

with open(os.path.join("Model", "xgb_pipeline.pkl"), "rb") as f:
    xgb_pipeline = pickle.load(f)

st.set_page_config(page_title="Disease Prediction System", layout="centered")
st.title("🦠 Disease Prediction System")
st.write("Enter the environmental and geographical details below to predict the disease risk.")

# States/UTs
states_and_uts = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry"
]

# Example state → districts dictionary
state_district_map = {
    "Andaman and Nicobar Islands": [
        "Andaman",
        "North and Middle Andaman"
    ],
    "Andhra Pradesh": [
        "Krishna",
        "Prakasam",
        "Nandyal",
        "Kurnool",
        "Anantapur",
        "Visakhapatnam",
        "Guntur",
        "Chittoor",
        "Kadapa",
        "Vizianagaram",
        "East Godavari",
        "Cuddapah",
        "Srikakulam",
        "Nellore",
        "West Godavari",
        "Ranga Reddy",
        "Godavari",
        "Chittor"
    ],
    "Arunachal Pradesh": [
        "West Kameng",
        "Upper Siang",
        "Lohit",
        "Kra Daadi",
        "East Siang",
        "East Kameng",
        "Papum Pare",
        "Upper Subansiri",
        "Shi-Yomi",
        "Dibang Valley",
        "Lower Dibang Valley",
        "Tirap",
        "West Siang",
        "Changlang",
        "Kurung Kumey",
        "Seppa Town",
        "Lower Subansiri"
    ],
    "Assam": [
        "Udalguri",
        "Baksa",
        "Dibrugarh",
        "Karbi Anglong",
        "Sonitpur",
        "Goalpara",
        "Dhemaji",
        "Darrang (Mangaldoi)",
        "Lakhimpur",
        "Dima Hasao",
        "Nagaon",
        "Kamrup",
        "Kokrajhar",
        "Jorhat",
        "Dhubri",
        "Sonitpur (Tezpur)",
        "Cachar",
        "Tinsukia",
        "Marigaon",
        "Karimganj",
        "Karbi-Anglong (Diphu)",
        "Chirang",
        "Nalbari",
        "Hailakandi",
        "Sibsagar",
        "Barpeta",
        "Karbi-Anglong",
        "Darrang",
        "Golaghat",
        "Sivasagar",
        "Karbi - Anglong",
        "Karbi-Anglong (DIPHU)",
        "Morigaon",
        "Bongaigaon",
        "Udalguri (Darrang)"
    ],
    "Bihar": [
        "Buxar",
        "Rohtas",
        "Siwan",
        "Jehanabad",
        "Munger",
        "West Champaran",
        "Nalanda",
        "Sheikhpura",
        "Muzaffarpur",
        "Saharsa",
        "Bhagalpur",
        "Araria",
        "Bhabua",
        "Arwal",
        "Patna",
        "Vaishali",
        "Aurangabad",
        "Madhubani",
        "Saran",
        "Nawada",
        "Jamui",
        "Purnia",
        "Darbhanga",
        "Samastipur",
        "Gaya",
        "Katihar",
        "Gopalganj",
        "Bhojpur",
        "Khagaria",
        "Banka",
        "East Champaran",
        "Lakhisarai",
        "Purnea"
    ],
    "Chandigarh": [
        "Chandigarh"
    ],
    "Chhattisgarh": [
        "Gariyaband",
        "Janjgir Champa",
        "Jashpur",
        "Balod",
        "Dhamtari",
        "Balodabazar",
        "Bastar",
        "Raigarh",
        "Rajnandgaon",
        "Durg",
        "Kanker",
        "Mahasamund",
        "Janjgir",
        "Raipur",
        "Kawardha",
        "Surajpur",
        "Balrampur",
        "Sukma",
        "Bemetara",
        "Korba",
        "Kondagaon",
        "Bilaspur",
        "Surguja",
        "Koriya",
        "Mungeli",
        "Narayanpur"
    ],
    "Dadra and Nagar Haveli": [
        "Dadra and Nagar Haveli"
    ],
    "Daman and Diu": [
        "Diu",
        "Daman"
    ],
    "Delhi": [
        "North East",
        "South",
        "South West",
        "North West",
        "West",
        "North",
        "Central",
        "East",
        "New Delhi"
    ],
    "Goa": [
        "South Goa",
        "North Goa"
    ],
    "Gujarat": [
        "Ahmedabad",
        "Navsari",
        "Rajkot",
        "Kutch",
        "Vadodara",
        "Panchmahal",
        "Surendranagar",
        "Bhavnagar",
        "Mehsana",
        "Jamnagar",
        "Amreli",
        "Narmada",
        "Sabarkantha",
        "Valsad",
        "Dahod",
        "Patan",
        "Porbandar",
        "Anand",
        "Junagadh",
        "Surat",
        "Kheda",
        "Bharuch",
        "Gandhinagar",
        "Banaskantha"
    ],
    "Dadra and Nagar Haveli": [
        "Dadra and Nagar Haveli"
    ],
    "Daman and Diu": [
        "Diu",
        "Daman"
    ],
    "Delhi": [
        "North East",
        "South",
        "South West",
        "North West",
        "West",
        "North",
        "Central",
        "East",
        "New Delhi"
    ],
    "Goa": [
        "South Goa",
        "North Goa"
    ],
    "Gujarat": [
        "Ahmedabad",
        "Navsari",
        "Rajkot",
        "Kutch",
        "Vadodara",
        "Panchmahal",
        "Surendranagar",
        "Bhavnagar",
        "Mehsana",
        "Jamnagar",
        "Amreli",
        "Narmada",
        "Sabarkantha",
        "Valsad",
        "Dahod",
        "Patan",
        "Porbandar",
        "Anand",
        "Junagadh",
        "Surat",
        "Kheda",
        "Bharuch",
        "Gandhinagar",
        "Banaskantha"
    ],"Haryana": [
        "Faridabad",
        "Gurgaon",
        "Rohtak",
        "Sonipat",
        "Hisar",
        "Panchkula",
        "Panipat",
        "Ambala",
        "Yamunanagar",
        "Karnal",
        "Kurukshetra",
        "Jhajjar",
        "Bhiwani",
        "Sirsa",
        "Jind",
        "Kaithal",
        "Rewari",
        "Mahendragarh",
        "Fatehabad",
        "Palwal",
        "Mewat"
    ],
    "Himachal Pradesh": [
        "Shimla",
        "Mandi",
        "Kangra",
        "Solan",
        "Sirmaur",
        "Una",
        "Chamba",
        "Hamirpur",
        "Kullu",
        "Bilaspur",
        "Kinnaur",
        "Lahaul and Spiti"
    ],
    "Jammu and Kashmir": [
        "Anantnag",
        "Baramulla",
        "Srinagar",
        "Jammu",
        "Kathua",
        "Udhampur",
        "Kupwara",
        "Pulwama",
        "Badgam",
        "Rajauri",
        "Poonch",
        "Doda"
    ],
    "Jharkhand": [
        "Ranchi",
        "Bokaro",
        "Dhanbad",
        "Deoghar",
        "Giridih",
        "East Singhbhum",
        "West Singhbhum",
        "Hazaribagh",
        "Palamu",
        "Chatra",
        "Garhwa",
        "Godda",
        "Pakur",
        "Dumka",
        "Koderma",
        "Sahibganj",
        "Lohardaga",
        "Simdega",
        "Latehar",
        "Saraikela Kharsawan",
        "Gumla",
        "Jamtara"
    ],
    "Karnataka": [
        "Bangalore Urban",
        "Mysuru",
        "Belagavi",
        "Tumakuru",
        "Ballari",
        "Dakshina Kannada",
        "Dharwad",
        "Shivamogga",
        "Vijayapura",
        "Bagalkot",
        "Bidar",
        "Chikkamagaluru",
        "Hassan",
        "Kolar",
        "Raichur",
        "Chitradurga",
        "Mandya",
        "Udupi",
        "Haveri",
        "Ramanagara",
        "Chikkaballapura",
        "Yadgir",
        "Kodagu",
        "Gadag",
        "Uttara Kannada",
        "Kalaburagi"
    ],
    "Kerala": [
        "Thiruvananthapuram",
        "Kollam",
        "Pathanamthitta",
        "Alappuzha",
        "Kottayam",
        "Idukki",
        "Ernakulam",
        "Thrissur",
        "Palakkad",
        "Malappuram",
        "Kozhikode",
        "Wayanad",
        "Kannur",
        "Kasaragod"
    ],
    "Madhya Pradesh": [
        "Bhopal",
        "Indore",
        "Jabalpur",
        "Gwalior",
        "Ujjain",
        "Sagar",
        "Dhar",
        "Satna",
        "Rewa",
        "Ratlam",
        "Chhindwara",
        "Shivpuri",
        "Murwara (Katni)",
        "Dewas",
        "Vidisha",
        "Shahdol",
        "Guna",
        "Khandwa",
        "Damoh",
        "Mandsaur",
        "Khargone",
        "Neemuch",
        "Panna",
        "Raisen",
        "Balaghat",
        "Sehore",
        "Singrauli",
        "Betul",
        "Datia",
        "Seoni",
        "Hoshangabad",
        "Chhatarpur",
        "Burhanpur",
        "Rajgarh",
        "Barwani",
        "Narsinghpur",
        "Sidhi",
        "Tikamgarh",
        "Shajapur",
        "Umaria",
        "Dindori",
        "Jhabua",
        "Alirajpur",
        "Anuppur",
        "Agar Malwa"
    ],
"Maharashtra": [
        "Mumbai",
        "Pune",
        "Nagpur",
        "Thane",
        "Nashik",
        "Aurangabad",
        "Solapur",
        "Amravati",
        "Kolhapur",
        "Sangli",
        "Jalgaon",
        "Akola",
        "Latur",
        "Dhule",
        "Chandrapur",
        "Parbhani",
        "Nanded",
        "Raigad",
        "Satara",
        "Buldhana",
        "Ahmednagar",
        "Wardha",
        "Beed",
        "Yavatmal",
        "Osmanabad",
        "Ratnagiri",
        "Gondia",
        "Hingoli",
        "Washim",
        "Gadchiroli",
        "Sindhudurg"
    ],
    "Manipur": [
        "Imphal East",
        "Imphal West",
        "Thoubal",
        "Churachandpur",
        "Bishnupur",
        "Ukhrul",
        "Senapati",
        "Tamenglong",
        "Chandel"
    ],
    "Meghalaya": [
        "East Khasi Hills",
        "West Garo Hills",
        "West Khasi Hills",
        "Jaintia Hills",
        "Ri Bhoi",
        "South Garo Hills"
    ],
    "Mizoram": [
        "Aizawl",
        "Lunglei",
        "Saiha",
        "Champhai",
        "Serchhip",
        "Kolasib",
        "Lawngtlai",
        "Mamit"
    ],
    "Nagaland": [
        "Dimapur",
        "Kohima",
        "Mokokchung",
        "Mon",
        "Tuensang",
        "Wokha",
        "Zunheboto",
        "Phek"
    ],
    "Odisha": [
        "Bhubaneswar",
        "Cuttack",
        "Ganjam",
        "Puri",
        "Balasore",
        "Bhadrak",
        "Sundargarh",
        "Jajpur",
        "Khurda",
        "Keonjhar",
        "Mayurbhanj",
        "Dhenkanal",
        "Bargarh",
        "Jagatsinghpur",
        "Kalahandi",
        "Nayagarh",
        "Angul",
        "Jharsuguda",
        "Rayagada",
        "Koraput",
        "Nabarangapur",
        "Kendrapara",
        "Sambalpur",
        "Malkangiri",
        "Nuapada",
        "Gajapati",
        "Debagarh",
        "Boudh",
        "Sonepur"
    ],
"Punjab": [
        "Amritsar",
        "Ludhiana",
        "Jalandhar",
        "Patiala",
        "Sangrur",
        "Gurdaspur",
        "Hoshiarpur",
        "Bathinda",
        "Firozpur",
        "Kapurthala",
        "Faridkot",
        "Rupnagar",
        "Mansa",
        "Moga",
        "Muktsar",
        "Shahid Bhagat Singh Nagar",
        "Barnala",
        "Tarn Taran",
        "Fatehgarh Sahib"
    ],
    "Rajasthan": [
        "Jaipur",
        "Jodhpur",
        "Kota",
        "Bikaner",
        "Ajmer",
        "Udaipur",
        "Bhilwara",
        "Alwar",
        "Sikar",
        "Nagaur",
        "Barmer",
        "Pali",
        "Jhunjhunu",
        "Churu",
        "Banswara",
        "Dholpur",
        "Rajsamand",
        "Karauli",
        "Tonk",
        "Chittorgarh",
        "Jhalawar",
        "Hanumangarh",
        "Dausa",
        "Bundi",
        "Sawai Madhopur",
        "Sri Ganganagar",
        "Bharatpur",
        "Jaisalmer",
        "Baran",
        "Pratapgarh"
    ],
    "Sikkim": [
        "East Sikkim",
        "West Sikkim",
        "North Sikkim",
        "South Sikkim"
    ],
    "Tamil Nadu": [
        "Chennai",
        "Coimbatore",
        "Madurai",
        "Tiruchirappalli",
        "Tirunelveli",
        "Salem",
        "Erode",
        "Vellore",
        "Thoothukudi",
        "Thanjavur",
        "Dindigul",
        "Tiruppur",
        "Virudhunagar",
        "Nagapattinam",
        "Ramanathapuram",
        "Cuddalore",
        "Kanchipuram",
        "Krishnagiri",
        "Kanyakumari",
        "Namakkal",
        "Sivaganga",
        "Pudukkottai",
        "Dharmapuri",
        "Theni",
        "Karur",
        "Villupuram",
        "Nilgiris",
        "Ariyalur",
        "Tiruvarur",
        "Perambalur",
        "Tiruvannamalai"
    ],
    "Telangana": [
        "Hyderabad",
        "Warangal",
        "Nizamabad",
        "Khammam",
        "Karimnagar",
        "Mahbubnagar",
        "Medak",
        "Adilabad",
        "Nalgonda"
    ],
    "Tripura": [
        "West Tripura",
        "South Tripura",
        "Dhalai",
        "North Tripura"
    ],
"Uttar Pradesh": [
        "Agra",
        "Aligarh",
        "Allahabad",
        "Ambedkar Nagar",
        "Amethi",
        "Auraiya",
        "Ayodhya",
        "Azamgarh",
        "Badaun",
        "Banda",
        "Barabanki",
        "Bareilly",
        "Basti",
        "Bhadohi",
        "Bijnor",
        "Budaun",
        "Bulandshahr",
        "Chandauli",
        "Chitrakoot",
        "Deoria",
        "Etah",
        "Etawah",
        "Farrukhabad",
        "Fatehpur",
        "Firozabad",
        "Gautam Buddha Nagar",
        "Ghaziabad",
        "Ghazipur",
        "Gonda",
        "Gorakhpur",
        "Hamirpur",
        "Hapur",
        "Hardoi",
        "Hathras",
        "Jalaun",
        "Jaunpur",
        "Jhansi",
        "Jyotiba Phule Nagar",
        "Kannauj",
        "Kanpur Dehat",
        "Kanpur Nagar",
        "Kanshiram Nagar",
        "Kaushambi",
        "Kushinagar",
        "Lakhimpur Kheri",
        "Lalitpur",
        "Lucknow",
        "Maharajganj",
        "Mahoba",
        "Mainpuri",
        "Mathura",
        "Mau",
        "Meerut",
        "Mirzapur",
        "Moradabad",
        "Muzaffarnagar",
        "Pilibhit",
        "Pratapgarh",
        "Raebareli",
        "Rampur",
        "Saharanpur",
        "Sant Kabir Nagar",
        "Shahjahanpur",
        "Shamli",
        "Shravasti",
        "Siddharth Nagar",
        "Sitapur",
        "Sonbhadra",
        "Sultanpur",
        "Unnao",
        "Varanasi"
    ],
    "Uttarakhand": [
        "Almora",
        "Bageshwar",
        "Chamoli",
        "Champawat",
        "Dehradun",
        "Haridwar",
        "Nainital",
        "Pauri Garhwal",
        "Pithoragarh",
        "Rudraprayag",
        "Tehri Garhwal",
        "Udham Singh Nagar",
        "Uttarkashi"
    ],
    "West Bengal": [
        "Alipurduar",
        "Bankura",
        "Bardhaman",
        "Birbhum",
        "Cooch Behar",
        "Dakshin Dinajpur",
        "Darjeeling",
        "Hooghly",
        "Howrah",
        "Jalpaiguri",
        "Jhargram",
        "Kalimpong",
        "Kolkata",
        "Malda",
        "Murshidabad",
        "Nadia",
        "North 24 Parganas",
        "Paschim Medinipur",
        "Purba Medinipur",
        "Purulia",
        "South 24 Parganas",
        "Uttar Dinajpur"
    ]



    # Add all your states and districts here
}

disease_precautions = {
    "Dengue": [
        " Use mosquito nets and repellents",
        " Avoid water stagnation around your home",
        " Wear long-sleeved clothes to reduce mosquito bites",
        " Seek medical attention if you have high fever or rashes"
    ],
    "Acute Diarrhoeal Disease": [  # Acute Diarrheal Disease
        " Drink only boiled or filtered water",
        " Eat freshly prepared and properly cooked food",
        " Wash hands with soap before eating",
        " Avoid street food and uncovered food items",
        " Visit a doctor if diarrhea persists or worsens"
    ],
    "Malaria": [
        "Use mosquito nets and repellents, especially during night",
        "Wear long-sleeved clothes to prevent mosquito bites",
        "Avoid areas with stagnant water",
        "Seek immediate medical attention if you have fever, chills, or fatigue"
    ],
    "Chikungunya": [
        "Use mosquito repellents and nets",
        "Wear full-sleeved clothing and cover exposed skin",
        "Avoid mosquito-prone areas, especially during daytime",
        "Stay hydrated and consult a doctor if fever or joint pain develops"
    ]
}


weeks = [f"{i}th week" for i in range(1, 53)]
months = list(range(1, 13))


# Layout
st.header(" Geographical Details")
col1, col2 = st.columns(2)

with col1:
    state_options = ["Select State/UT"] + states_and_uts
    state_ut = st.selectbox("State/UT", options=state_options, help="Choose the state or union territory",index=None)
    if state_ut == "Select State/UT":
        state_ut = ""

with col2:
    districts = state_district_map.get(state_ut, [])
    if districts:
        district_options = ["Select District"] + districts
        district = st.selectbox("District", options=district_options, help="Choose the district",index=None)
        if district == "Select District":
            district = ""
    else:
        district = st.text_input("District", placeholder="Select district")

st.header(" Environmental Data")
col3, col4 = st.columns(2)

with col3:
    temp_c = st.number_input("Temperature (°C)", step=0.1, placeholder="Range: -13.47 – 54.58")
    lai = st.number_input("Leaf Area Index (LAI)", step=1, placeholder="Range: 0 – 62")

with col4:
    preci = st.number_input("Precipitation", step=0.001, format="%.5f", placeholder="Range: 0.000002 – 5.68")
    week_options = ["Select Week"] + weeks
    week_of_outbreak = st.selectbox("Week of Outbreak", options=week_options,index=None)
    if week_of_outbreak == "Select Week":
        week_of_outbreak = ""

    month_options = ["Select Month"] + [str(m) for m in months]
    mon = st.selectbox("Month", options=month_options,index=None)
    if mon == "Select Month":
        mon = ""



# Convert Celsius → Kelvin
temp_k = temp_c + 273.15 if temp_c is not None else None

# Create dataframe
manual_input = pd.DataFrame([{
    "Temp": temp_k,
    "preci": preci,
    "LAI": lai,
    "state_ut": state_ut,
    "district": district,
    "week_of_outbreak": week_of_outbreak,
    "mon": mon
}])


st.subheader("Your Input Data")
st.write(manual_input)

if (not state_ut or not district
        or not week_of_outbreak
        or not mon):
    st.warning("⚠ Please complete all selections before predicting.")

# Prediction
if st.button("Predict Disease"):
    try:
        y_pred_encoded = xgb_pipeline.predict(manual_input)
        y_pred_encoded = y_pred_encoded.reshape(-1, 1)
        y_pred_label = disease_encoder.inverse_transform(y_pred_encoded)
        disease = y_pred_label[0][0]

        st.success(f"Predicted Disease: {disease}")

        # Show precautions if available
        if disease in disease_precautions:
            st.subheader("🛡️ Precautions")
            for precaution in disease_precautions[disease]:
                st.write(f"- {precaution}")
        else:
            st.info("No specific precautions available for this disease.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

