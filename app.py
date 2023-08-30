# Core Packages
import streamlit as st
from PIL import Image

# Data Viz Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# EDA Packages
import pandas as pd
import numpy as np

st.set_page_config(page_title='GAD Analysis',page_icon = 'logo.png', layout = 'wide', initial_sidebar_state = 'auto')
sns.set(rc={'figure.figsize':(20,15)})

DATA_URL = ('gad.csv')

st.markdown('# Graduate Admission Dataset')
st.markdown('### **Analysis of Graduate Admission Dataset**')

img = Image.open('gad.png')
st.image(img, width = 720, caption = 'Graduate Admission Dataset')

st.markdown('### **About the Dataset:**')
st.info('This dataset was built \
    with the purpose of helping students in \
        shortlisting universities with their profiles. \
            The predicted output gives them a fair \
                idea about their chances for a particular university. \
                    This dataset is inspired by the UCLA Graduate Dataset from Kaggle. \
                        The graduate studies dataset is a dataset which describes the probability of \
                            selections for Indian students dependent on the following parameters below.')

img = Image.open('univ.png')
st.image(img, width = 720, caption = "Top 5 Universities in the US")

st.markdown('### **Dataset Info:**')
st.markdown('##### **Attributes of the Dataset:**')
st.info('\t 1. GRE Score (out of 340), \
        \n\t 2. TOEFL Score (out of 120), \
        \n\t 3. University Rating (out of 5), \
        \n\t 4. Statement of Purpose/ SOP (out of 5), \
        \n\t 5. Letter of Recommendation/ LOR (out of 5), \
        \n\t 6. Research Experience (either 0 or 1), \
        \n\t 7. CGPA (out of 10), \
        \n\t 8. Chance of Admittance (ranging from 0 to 1)') 

img = Image.open('par.png')
st.image(img, width = 720, caption = "Influence of the Attributes on the Dataset")

def load_data(nrows):
    df = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    df.set_index('Serial No.', inplace=True)
    df.rename(lowercase, axis='columns', inplace=True)
    return df

st.title('Lets explore the Graduate Admission Dataset')
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading graduate admissions dataset...')
# Load 500 rows of data into the dataframe.
df = load_data(500)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading graduate admissions dataset...Completed!')

# Explore Dataset
st.header('Quick  Explore')
st.sidebar.subheader('Quick  Explore')
st.markdown("Tick the box on the side panel to explore the dataset.")

if st.sidebar.checkbox("Show Raw Data"):
    st.subheader('Raw data')
    st.write(df)
if st.sidebar.checkbox('Dataset Quick Look'):
    st.subheader('Dataset Quick Look:')
    st.write(df.head())
if st.sidebar.checkbox("Show Columns"):
    st.subheader('Show Columns List')
    all_columns = df.columns.to_list()
    st.write(all_columns)
if st.sidebar.checkbox('Statistical Description'):
    st.subheader('Statistical Data Descripition')
    st.write(df.describe())
if st.sidebar.checkbox('Missing Values?'):
    st.subheader('Missing values')
    st.write(df.isnull().sum())

st.header('Create Own Visualization')
st.markdown("Tick the box on the side panel to create your own Visualization.")
st.sidebar.subheader('Create Own Visualization')
if st.sidebar.checkbox('Count Plot'):
    st.subheader('Count Plot')
    st.info("If error, please adjust column name on side panel.")
    column_count_plot = st.sidebar.selectbox("Choose a column to plot count.", df.columns[:5])
    fig = sns.countplot(x=column_count_plot,data=df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
if st.sidebar.checkbox('Distribution Plot'):
    st.subheader('Distribution Plot')
    st.info("If error, please adjust column name on side panel.")
    column_dist_plot = st.sidebar.selectbox('Choose a column to plot density.', df.columns[:5])
    fig = sns.distplot(df[column_dist_plot])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Showing the Prediction Model
st.header('Building Prediction Model')
st.sidebar.subheader('Prediction Model')
st.markdown("Tick the box on the side panel to run Prediction Model.")

import pickle

if st.sidebar.checkbox('View Prediction Model'):
    st.subheader('Prediction Model')
pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)


#@st.cache()

# defining the function to predict the output
def convert_toefl_to_ielts(val):
    if val > 69 and val < 94:
        score = 6.5
    if val > 93 and val < 102:
        score = 7.0
    if val > 101 and val < 110:
        score = 7.5
    if val > 109 and val < 115:
        score = 8.0
    if val > 114 and val < 118:
        score = 8.5
    if val > 117 and val < 121:
        score = 9.0
    return score
    

def pred(gre, toefl, sop, lor, cgpa, resc):
    
    # Preprocessing user input
    ielts = convert_toefl_to_ielts(toefl)

    if resc == 'Yes':
        resc = 1
    else:
        resc = 0
    
    st.success("GRE Score = {} TOEFL Score = {} IELTS Score = {} CGPA = {} ".format(gre, toefl, ielts, cgpa))
    for univ in range(1, 6):
    # Predicting the output
        #prediction = model.predict([[gre, toefl, ielts, univ, sop, lor, cgpa, resc]])
        prediction = model.predict([[gre, toefl, univ, sop, lor, cgpa, resc]])
        
        st.info("Chance of Admittance for University Rank " + str(univ) + " = {0:.2f} %".format(prediction[0]*100))
        if prediction[0] >= 0.70:
            st.success('Congratulations! You are eligible to apply for this university!')
        else:
            st.caption('Better Luck Next Time :)')

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
      
    # following lines create boxes in which user can enter data required to make prediction 
    gre = st.slider("GRE Score (out of 340):", 0, 340, 0, step = 1)
    toefl = st.slider("TOEFL Score (out of 120):", 0, 120, 0, step = 1)
    sop = st.slider("SOP Score (out of 5):", value = 0.0, min_value = 0.0, max_value = 5.0, step = 0.5)
    lor = st.slider("LOR Score (out to 5):", value = 0.0, min_value = 0.0, max_value = 5.0, step = 0.5)
    resc = st.selectbox('Research Experience:', ("Yes", "No"))
    cgpa = st.number_input('Enter CGPA (out of 10):')
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = pred(gre, toefl, sop, lor, cgpa, resc)
     
if __name__=='__main__': 
    main()


import numpy as np
import streamlit as st
import pickle
import pandas as pd
if st.sidebar.checkbox('View Admission'):
    st.subheader('Admission')
dt = pd.read_csv("admins.csv")
college = np.unique(dt['College'])
clg_id = list(range(1, len(college) + 1))
dt['College_id'] = dt['College'].replace(college, clg_id)

model = pickle.load(open('modl.pkl', 'rb'))

def predict(init_features):
    final_features = [np.array(init_features)]
    pred = model.predict(final_features)
    clg_name = college[int(pred[0]) - 1]
    return clg_name

def main():
    st.title('College Prediction App')

    init_features = []
    for i in range(1):  # Assuming there are 4 input features
        feature = st.number_input(f'Enter Feature {i+1}')
        init_features.append(feature)

    if st.button('submit'):
        clg_name = predict(init_features)
        st.success(f'Your Predicted College is: {clg_name}')

if __name__ == "__main__":
    main()

def main():
    if st.button("Open Page 2"):
        page2()

def page2():
    
# import packages
 import pickle 
 import streamlit as st

import pandas as pd
import plotly.express as px

# import data and model

# Load the dataset
salaries = pd.read_csv('ds_salaries.csv')

# Load the saved model
with open(
    'model_2023.sav',
        'rb') as f:
    model =pickle.load(f)

# Preprocessing user input


def preprocess_inputs(title, experience, remoter, size, location):
    user_input_dict = {
        'job_title': [title],
        'experience_level': [experience],
        'remote_ratio': [remoter],
        'company_size': [size],
        'company_location_is_US': [1 if location == 'US' else 0]
    }

    user_input = pd.DataFrame(data=user_input_dict)

    cleaner_type = {
        'job_title': {
            'Data Analyst': 0,
            'Data Scientist': 1,
            'Data Engineer': 2,
            'Machine Learning Engineer': 3
        },
        'experience_level': {
            'Entry-level': 0,
            'Mid-Level': 1,
            'Senior': 2
        },
        'remote_ratio': {
            'No remote': 0,
            'Semi remote': 1,
            'Full remote': 2
        },
        'company_size': {
            'Small': 0,
            'Medium': 1,
            'Large': 2
        },
        'company_location_is_US': {
            1: 1,
            0: 0
        }
    }

    user_input = user_input.replace(cleaner_type)

    return user_input


# Function to create a bar chart


def create_bar_chart(data_frame, column, title):
    '''function to create a bar chart'''
    counts = data_frame[column].value_counts().head(5)
    fig = px.bar(
        counts,
        x=counts.index,
        y=counts.values,
        color=counts.index,
        text=counts.values,
        labels={
            'index': column,
            'y': 'count',
            'text': 'count'},
        template='seaborn',
        title=f'<b>{title}')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def predict_salary(ml_model, preprocessed_input):
    """ This function takes in a model and preprocessed input """
    pred = ml_model.predict(preprocessed_input)
    # extract the scalar value from the numpy array
    pred_value = pred[0][0]
    return pred_value

# Page layout


def main():
    '''function to create the main page'''

    # main page layout

    # Set Page Configuration

    # Define the page title text
    title_text = "<h1 style='text-align: center;'>Exploring Computer Science Careers:\
          Salaries, Jobs, and Global Trends</h1>"

    # Display the page title with centered styling
    st.markdown(title_text, unsafe_allow_html=True)

    # Set Header
    st.write(
        "The field of data science has witnessed a rapid expansion in recent years,\
             owing to the availability of vast amounts of data and the development of\
             sophisticated tools to analyze it. As a result, there is a growing demand\
             for data scientists across various industries. These professionals are\
             responsible for extracting insights and making data-driven decisions that\
             can impact a company's bottom line. To prepare for a data science job,\
             candidates need to have a strong foundation in statistics, programming,\
             and machine learning techniques. This app aims to provide candidates with\
             the necessary knowledge and understanding of the data science sector,\
             its requirements, and the skills needed to succeed in this field.")

    st.markdown(
        "<h2 style='text-align: center;'>Insights on computer science jobs</h2>",
        unsafe_allow_html=True)

    # Call the function for each tab
    tab1, tab2, tab3, tab4 = st.tabs(
        ['Most popular roles in Data Science',
         'Most represented company location',
         'Highest paid data science jobs',
         'Company Sizes in Data Science Field'])

    with tab1:
        create_bar_chart(
            salaries,
            'job_title',
            'Most popular roles in Data Science')

    with tab2:
        create_bar_chart(
            salaries,
            'company_location',
            'Most represented company location')

    with tab3:
        hp_jobs = salaries.groupby(
            'job_title', as_index=False)['salary_in_usd'].max().sort_values(
            by='salary_in_usd', ascending=False).head(10)
        fig = px.bar(
            hp_jobs,
            x='job_title',
            y='salary_in_usd',
            color='job_title',
            labels={
                'job_title': 'job title',
                'salary_in_usd': 'salary in usd'},
            template='seaborn',
            text='salary_in_usd',
            title='<b>Highest paid data science jobs')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        create_bar_chart(
            salaries,
            'company_size',
            'Company Sizes in Data Science Field')

    st.image(
        '1.jpg',
        use_column_width=True)

    # sidebar layout

    st.sidebar.image(
        'ab.png',
        width=150)
    st.sidebar.title('Predict your future salary')
    st.sidebar.write('Enter your profile information below:')

    # inputs
    title = st.sidebar.selectbox(
        'Please choose your job title',
        ('Data Analyst',
         'Data Scientist',
         'Data Engineer',
         'Machine Learning Engineer'))
    experience = st.sidebar.selectbox(
        'Please choose your experience level',
        ('Entry-level',
         'Mid-Level',
         'Senior'))
    remoter = st.sidebar.selectbox(
        'Please choose your remote ratio',
        ('No remote',
         'Semi remote',
         'Full remote'))
    size = st.sidebar.selectbox(
        'Please choose the company size',
        ('Small',
         'Medium',
         'Large'))
    location = st.sidebar.radio(
        'Select Location:',
        ('US',
         'Other'))

    # Pre-processing user_input
    user_input = preprocess_inputs(title, experience, remoter, size, location)

    # predict button
    if st.sidebar.button("ok"):
        pred_value = predict_salary(model, user_input)
        st.sidebar.write(f"Prediction: {pred_value:.2f} USD")


if __name__ == '__main__':
    main()


st.sidebar.subheader('Data Source')
st.sidebar.info("https://www.kaggle.com/graduate-admissions")
st.sidebar.subheader('Source Article')
st.sidebar.info("https://medium.com/analytics-vidhya/a-fresh-look-at-graduate-admissions-dataset-d39e4d20803e")
st.sidebar.subheader('Author Credits')
st.sidebar.info("BHARATH KUMAR REDDY\
    \n SAMEER SHAIK")
st.sidebar.subheader('Built with Streamlit')
st.sidebar.info("https://www.streamlit.io/")