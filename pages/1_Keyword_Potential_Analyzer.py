import streamlit as st
import searchconsole
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from pandas import ExcelWriter
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from sklearn.metrics import mean_absolute_error
import plotly.figure_factory as ff
import datetime

plt.rcParams['font.family'] = 'Poppins'

st.set_page_config(page_title="Keyword Potential Analyzer 🔍", page_icon="🔍", layout="wide")

st.markdown("# Keyword Potential Analyzer")
st.sidebar.header("Keyword Potential Analyzer")
st.write(
    """
    Welcome to the Keyword Potential Analyzer. This tool is designed for SEO consultants 
    to provide insights into keyword performance based on Google Search Console data. 
    It allows you to fetch and analyze data, providing valuable insights that can help in making data-driven decisions. 
    Enjoy exploring your data!
    """
)


# Cache the authentication object to avoid re-authenticating unless necessary
@st.cache_data()
def authenticate():
    """
    Authenticates and returns a searchconsole account instance.
    This function is cached across sessions to avoid re-authentication on every script rerun.
    """
    try:
        account = searchconsole.authenticate(client_config='key.json', serialize='credentials.json')
        return account
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None

def load_account_properties(account):
    if account:
        return [webproperty.url for webproperty in account.webproperties]
    else:
        return []
    


def app():
    # Only show the button if the user is not authenticated
    if 'account' not in st.session_state or not st.session_state.account:
        if 'button_clicked' not in st.session_state:
            st.session_state.button_clicked = False

        if not st.session_state.button_clicked:
            button_placeholder = st.empty()
            if button_placeholder.button('Login with your Search Console account'):
                # Authenticate and load properties
                st.session_state.account = authenticate()
                st.session_state.webproperty_urls = load_account_properties(st.session_state.account)
                st.session_state.button_clicked = True
                button_placeholder.empty()  # Remove the button
        else:
            st.session_state.account = authenticate()
            st.session_state.webproperty_urls = load_account_properties(st.session_state.account)

    # If the user is authenticated, proceed with the rest of the app
    if 'account' in st.session_state and st.session_state.account:
        if 'webproperty_urls' in st.session_state and st.session_state.webproperty_urls:
            # Rest of your code...
            # Create two columns
            left_column, right_column = st.columns(2)

            # Use a form to encapsulate the selectbox, date pickers, and filter input
            with st.form(key='my_form'):
                selected_url = st.selectbox(
                    "Select a web property URL:",
                    st.session_state.webproperty_urls,
                    index=st.session_state.webproperty_urls.index(st.session_state.selected_url) if 'selected_url' in st.session_state and st.session_state.selected_url in st.session_state.webproperty_urls else 0
                )

                date_ranges = ['1 month', '3 months', '6 months', '1 year', '16 months', 'Custom']
                date_range = st.radio('Select a date range:', date_ranges)

                if date_range == '1 month':
                    st.session_state.start_date = datetime.date.today() - datetime.timedelta(days=30)
                    st.session_state.end_date = datetime.date.today() - datetime.timedelta(days=1)
                elif date_range == '3 months':
                    st.session_state.start_date = datetime.date.today() - datetime.timedelta(days=90)
                    st.session_state.end_date = datetime.date.today() - datetime.timedelta(days=1)
                elif date_range == '6 months':
                    st.session_state.start_date = datetime.date.today() - datetime.timedelta(days=180)
                    st.session_state.end_date = datetime.date.today() - datetime.timedelta(days=1)
                elif date_range == '1 year':
                    st.session_state.start_date = datetime.date.today() - datetime.timedelta(days=365)
                    st.session_state.end_date = datetime.date.today() - datetime.timedelta(days=1)
                elif date_range == '16 months':
                    st.session_state.start_date = datetime.date.today() - datetime.timedelta(days=480)
                    st.session_state.end_date = datetime.date.today() - datetime.timedelta(days=1)

                # Always display start and end date fields
                st.session_state.start_date = st.date_input('Start date', value=st.session_state.start_date)
                st.session_state.end_date = st.date_input('End date', value=st.session_state.end_date)





                filter_query = st.text_input('Filter query', value='')
                min_impressions = st.number_input('Minimum impressions', value=0)
                min_clicks = st.number_input('Minimum clicks', value=0)
                max_position = st.number_input('Maximum position', value=10)
                new_position = st.number_input('Predicted position', value=1, min_value=1, max_value=100)
                # Get a list of all countries and their alpha-2 codes
                # Define your own list of countries and their codes
              # Define your own list of countries and their codes
                countries = [('Czech Republic', 'cze'), ('Slovakia', 'svk'), ('Poland', 'pol'), ('Germany', 'ger'), ('France', 'fra'), ('Italy', 'ita'), ('Spain', 'esp'), ('United States', 'usa'), ('United Kingdom', 'gbr'), ('Russia', 'rus')]

                # Use this list in your selectbox
                selected_country = st.selectbox('Select a country:', [country[0] for country in countries])

                # Get the corresponding country code
                country_code = next((country[1] for country in countries if country[0] == selected_country), None)
                submitted = st.form_submit_button(label='Submit')  # Place the submit button inside the form

            if submitted:
                # Display a spinner during data fetching and processing
                with st.spinner(text="In progress. I'm fetching and analyzing the data..."):
                    # Fetch and filter data
                    df = get_and_filter_data(st.session_state.account, selected_url, st.session_state.start_date, st.session_state.end_date, filter_query, min_impressions, min_clicks, max_position, country_code)  # Add country_code heredf = get_and_filter_data(st.session_state.account, selected_url, start_date, end_date, filter_query, min_impressions, min_clicks, max_position, country_code)  # Add country_code here

                    # Exclude string columns from the DataFrame
                    numeric_df = df.select_dtypes(include=['int64', 'float64'])

                    # Calculate the correlation matrix
                    correlation_matrix = numeric_df.corr()

                    # Create a heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, annot=True, cmap=sns.light_palette("#9B36F7", as_cmap=True))


                    # Display the heatmap
                    st.pyplot(plt)

                    # Continue with your code
                    X = numeric_df[['impressions', 'position']]
                    y = numeric_df.pop('clicks')

                    # Create the pairplot
                    sns.pairplot(numeric_df)
                    st.pyplot(plt)

                    # Train Test Split 
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                    # Standardise The Data 
                    scaler = StandardScaler()
                    scaler = scaler.fit(X_train)
                    X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns, index = X_train.index)
                    X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns, index = X_test.index)

                    RandomForestRegressorModel = RandomForestRegressor(n_estimators=100)
                    RandomForestRegressorModel.fit(X_train, y_train)
                    prediction_score = RandomForestRegressorModel.score(X_train, y_train)
                    test_score = RandomForestRegressorModel.score(X_test, y_test)
                    st.write(f"Prediction score: {prediction_score}, Test score: {test_score}")

                    # Plot feature importances
                    feature_importances = RandomForestRegressorModel.feature_importances_
                    indices = np.argsort(feature_importances)[::-1]
                    names = [X_train.columns[i] for i in indices]

                    plt.figure(figsize=(10, 5))
                    plt.title("Importance of individual factors for click prediction")
                    plt.bar(range(X_train.shape[1]), feature_importances[indices])
                    plt.xticks(range(X_train.shape[1]), names, rotation=90)
                    st.pyplot(plt)

                    # Plot predicted vs actual values
                    y_pred = RandomForestRegressorModel.predict(X_test)
                    plt.figure(figsize=(10, 5))
                    plt.scatter(y_test, y_pred)
                    plt.xlabel("Actual values")
                    plt.ylabel("Predicted values")
                    plt.title("Error rate of the prediction model")
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
                    st.pyplot(plt)

                    # Copy the DataFrame and store the original position
                    df_pos = df.copy()
                    df_pos['original_position'] = df_pos['position']
                    df_pos['position'] = new_position  # Use the user's input for the new position

                    # Standardise the data
                    X = pd.DataFrame(scaler.transform(df_pos[['impressions', 'position']]), columns = ['impressions', 'position'], index = df_pos.index)

                    # Use the trained model to predict clicks
                    predicted_clicks = RandomForestRegressorModel.predict(X)
                    df_predicted = df_pos.assign(predicted_clicks=predicted_clicks)

                    # Round 'original_position' to the nearest decimal and convert to float
                    df_predicted['original_position'] = df_predicted['original_position'].round(1)

                    df_predicted = df_predicted[df_predicted['original_position'] >= new_position + 1]

                    if df_predicted.empty:
                        st.error('No data to display after filtering.')
                    else:
                        # Sort df_predicted by 'predicted_clicks' in descending order and select top 30
                        top_keywords = df_predicted.sort_values('predicted_clicks', ascending=False).head(30)

                        # Round 'position' to the nearest integer and convert to int
                        top_keywords['position'] = top_keywords['position'].round().astype(int)

                        # Sort top_keywords by 'predicted_clicks' in descending order and reset index
                        top_keywords = top_keywords.sort_values('predicted_clicks', ascending=False).reset_index(drop=True)

                        # Create a color palette from dark to light
                        palette = sns.light_palette("#9B36F7", len(top_keywords))[::-1]

                        # Create a horizontal bar plot
                        fig, ax1 = plt.subplots(figsize=(10, 8), dpi=1000)
                        sns.barplot(x='predicted_clicks', y='query', hue='query', data=top_keywords, orient='h', palette=palette, ax=ax1, legend=False)

                        # Create a second y-axis
                        ax2 = ax1.twinx()

                        # Set the values of the second y-axis to 'original_position'
                        ax2.set_yticks(ax1.get_yticks())
                        ax2.set_ylim(ax1.get_ylim())
                        ax2.set_yticklabels(top_keywords['original_position'])  # Use 'original_position' here

                        # Continue with your code...

                        # Calculate R2 for the model
                        r2_train = RandomForestRegressorModel.score(X_train, y_train)
                        r2_test = RandomForestRegressorModel.score(X_test, y_test)

                        # Calculate RMSE and MAE for the model
                        y_pred_train = RandomForestRegressorModel.predict(X_train)
                        y_pred_test = RandomForestRegressorModel.predict(X_test)
                        rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
                        rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
                        mae_train = mean_absolute_error(y_train, y_pred_train)
                        mae_test = mean_absolute_error(y_test, y_pred_test)

                        # Add legend with R2, RMSE, MAE, the analyzed period, and the predicted position, and remove lines
                        legend_text = f'Predicted position: {new_position}\nR² train: {r2_train:.2f}, R² test: {r2_test:.2f}\nRMSE train: {rmse_train:.2f}, RMSE test: {rmse_test:.2f}\nMAE train: {mae_train:.2f}, MAE test: {mae_test:.2f}\nAnalyzované období: {st.session_state.start_date.strftime("%d.%m.%Y")} až {st.session_state.end_date.strftime("%d.%m.%Y")}'
                        ax1.legend([legend_text], 
                                loc='lower right', handlelength=0, title=' ')

                        # Set labels and title
                        ax1.set_xlabel('Predicted Clicks')
                        ax1.set_ylabel('Keywords')
                        ax2.set_ylabel('Average Position for the Analyzed Period')
                        plt.title(f'TOP 30 Brand Keywords for Optimization – {selected_url}', fontsize=10)

                        # Display the plot
                        st.pyplot(fig)

def get_and_filter_data(account, selected_url, start_date, end_date, filter_query, min_impressions, min_clicks, max_position, country_code):  # Add a parameter for the country code
    webproperty = account[selected_url]
    GSCdata = webproperty.query.range(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')).dimension('query').filter('country', country_code).get()  # Use the selected country code
    df = pd.DataFrame(GSCdata)
    if filter_query:  # Only apply the filter if filter_query is not empty
        df = df[~df['query'].str.contains(filter_query, flags=re.IGNORECASE, regex=True)]
    filtered_df = df[(df['impressions'] > min_impressions) & (df['clicks'] > min_clicks) & (df['position'] < max_position)]
    return filtered_df

app()