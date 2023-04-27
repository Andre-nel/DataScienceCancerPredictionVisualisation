
# Import the necessary libraries and load the data:
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from model import load_model    # , predict_diagnosis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the data
pre_pca_data = pd.read_csv(("C:/Users/candr/OneDrive/Desktop/Masters/DataScience/"
"DataScienceCancerPredictionVisualisation/src/visualization/cancer/"
"data/cancer_data.csv"))
pre_all_data = pd.read_csv(("C:/Users/candr/OneDrive/Desktop/Masters/DataScience/"
"DataScienceCancerPredictionVisualisation/src/visualization/cancer/"
"data/cancer_data_all_features.csv"))
# data = pd.read_csv("data/hold_out_cancer_data.csv")

# Prepare the data for visualization:

# Separate features and target
X_pca = pre_pca_data.drop(columns=['diagnosis'])
y_pca = pre_pca_data['diagnosis']

X_all = pre_all_data.drop(columns=['diagnosis'])
y_all = pre_all_data['diagnosis']

# Split the data into train and test sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, test_size=0.2, random_state=42)
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Load the model and make predictions
model_pca = load_model(("C:/Users/candr/OneDrive/Desktop/Masters/DataScience/"
                        "DataScienceCancerPredictionVisualisation/src/visualization/cancer"
                        "/model/logistic_regression_model.pkl"))
y_pred_pca = model_pca.predict(X_test_pca)

model_all = load_model(("C:/Users/candr/OneDrive/Desktop/Masters/DataScience/"
                        "DataScienceCancerPredictionVisualisation/src/visualization/cancer"
                        "/model/logistic_regression_model_all_features.pkl"))
y_pred_all = model_all.predict(X_test_all)


# Calculate model performance metrics
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
precision_pca = precision_score(y_test_pca, y_pred_pca)
recall_pca = recall_score(y_test_pca, y_pred_pca)
f1_pca = f1_score(y_test_pca, y_pred_pca)
roc_auc_pca = roc_auc_score(y_test_pca, y_pred_pca)

# Calculate model performance metrics
accuracy_all = accuracy_score(y_test_all, y_pred_all)
precision_all = precision_score(y_test_all, y_pred_all)
recall_all = recall_score(y_test_all, y_pred_all)
f1_all = f1_score(y_test_all, y_pred_all)
roc_auc_all = roc_auc_score(y_test_all, y_pred_all)


# Create the app layout with the visualizations:

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Cancer Diagnosis Prediction Dashboard"), className="text-center")
    ], className="mt-4"),

    # Feature Importance Visualization
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='feature-importance-pca', figure=px.bar(x=X_pca.columns,
                      y=model_pca.coef_[0], labels={'x': 'Features', 'y': 'Importance'},
                      title="PCA Pre Processed Feature Importance"))
        ])
    ], className="mt-4"),

    # Feature Importance Visualization
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='feature-importance-all', figure=px.bar(x=X_all.columns,
                      y=model_all.coef_[0], labels={'x': 'Features', 'y': 'Importance'},
                      title="Feature Importance (all)"))
        ])
    ], className="mt-4"),

    # Feature Relationships Visualization
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='x-axis-pca', options=[{'label': i, 'value': i}
                         for i in X_pca.columns], value='radius_mean', placeholder="Select Feature 1"),
            dcc.Dropdown(id='y-axis-pca', options=[{'label': i, 'value': i}
                         for i in X_pca.columns], value='texture_mean', placeholder="Select Feature 2"),
            dcc.Graph(id='scatter-plot-pca')
        ])
    ], className="mt-4"),
    # Feature Relationships Visualization
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='x-axis-all', options=[{'label': i, 'value': i}
                         for i in X_all.columns], value='radius_mean', placeholder="Select Feature 1"),
            dcc.Dropdown(id='y-axis-all', options=[{'label': i, 'value': i}
                         for i in X_all.columns], value='texture_mean', placeholder="Select Feature 2"),
            dcc.Graph(id='scatter-plot-all')
        ])
    ], className="mt-4"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='performance-metrics-pca', figure=px.bar(
                x=[accuracy_pca, precision_pca, recall_pca, f1_pca, roc_auc_pca],
                y=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'],
                orientation='h',
                labels={'x': 'Metric Value', 'y': 'Metrics'},
                title="PCA preprocessed Model Performance Metrics",
                text=[round(accuracy_pca, 2), round(precision_pca, 2), round(
                    recall_pca, 2), round(f1_pca, 2), round(roc_auc_pca, 2)]
            ))
        ])
    ], className="mt-4"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='performance-metrics-all', figure=px.bar(
                x=[accuracy_all, precision_all, recall_all, f1_all, roc_auc_all],
                y=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'],
                orientation='h',
                labels={'x': 'Metric Value', 'y': 'Metrics'},
                title="All features kept Model Performance Metrics",
                text=[round(accuracy_all, 2), round(precision_all, 2), round(
                    recall_all, 2), round(f1_all, 2), round(roc_auc_all, 2)]
            ))
        ])
    ], className="mt-4"),

])

# Update the scatter plot based on the selected features


@app.callback(
    Output('scatter-plot-pca', 'figure'),
    [Input('x-axis-pca', 'value'), Input('y-axis-pca', 'value')]
)
def update_scatter_pca(x_axis, y_axis):
    return px.scatter(data_frame=pre_pca_data, x=x_axis, y=y_axis, color='diagnosis',
                      title="Feature Relationships", labels={'diagnosis': 'Diagnosis'},
                      hover_data=X_pca.columns, height=500)


@app.callback(
    Output('scatter-plot-all', 'figure'),
    [Input('x-axis-all', 'value'), Input('y-axis-all', 'value')]
)
def update_scatter_all(x_axis, y_axis):
    return px.scatter(data_frame=pre_all_data, x=x_axis, y=y_axis, color='diagnosis',
                      title="Feature Relationships", labels={'diagnosis': 'Diagnosis'},
                      hover_data=X_all.columns, height=500)


if __name__ == '__main__':
    app.run_server(debug=True)
