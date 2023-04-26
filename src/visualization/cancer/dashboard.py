
# Import the necessary libraries and load the data:
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from model import load_model, predict_diagnosis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the data
data = pd.read_csv("data/hold_out_cancer_data.csv")

# Prepare the data for visualization:

# Separate features and target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the model and make predictions
model = load_model()
y_pred = model.predict(X_test)



# Calculate model performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)


# Create the app layout with the visualizations:

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Cancer Diagnosis Prediction Dashboard"), className="text-center")
    ], className="mt-4"),

    # Feature Importance Visualization
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='feature-importance', figure=px.bar(x=X.columns, y=model.coef_[0], labels={'x': 'Features', 'y': 'Importance'}, title="Feature Importance"))
        ])
    ], className="mt-4"),

    # Feature Relationships Visualization
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='x-axis', options=[{'label': i, 'value': i} for i in X.columns], value='radius_mean', placeholder="Select Feature 1"),
            dcc.Dropdown(id='y-axis', options=[{'label': i, 'value': i} for i in X.columns], value='texture_mean', placeholder="Select Feature 2"),
            dcc.Graph(id='scatter-plot')
        ])
    ], className="mt-4"),

    # Model Performance Metrics Visualization
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='performance-metrics', figure=px.pie(names=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'], values=[accuracy, precision, recall, f1, roc_auc], title="Model Performance Metrics"))
        ])
    ], className="mt-4"),
])

# Update the scatter plot based on the selected features
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis', 'value'), Input('y-axis', 'value')]
)
def update_scatter(x_axis, y_axis):
    return px.scatter(data_frame=data, x=x_axis, y=y_axis, color='diagnosis', title="Feature Relationships", labels={'diagnosis': 'Diagnosis'}, hover_data=X.columns, height=500)

if __name__ == '__main__':
    app.run_server(debug=True)





