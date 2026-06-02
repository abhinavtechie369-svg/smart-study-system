import pandas as pd
from sklearn.linear_model import LinearRegression

# Global variable (VERY IMPORTANT)
model = None


def load_data():
    data = pd.read_csv("data/student_data.csv")
    return data


def train_model():
    global model   #  tells Python we are using global variable
    
    data = load_data()
    
    X = data[['study_hours', 'sleep_hours', 'attendance', 'previous_score']]
    y = data['score']
    
    model = LinearRegression()
    model.fit(X, y)


def predict_score(study, sleep, attendance, previous):
    global model   # access same model
    
    # Train only if not trained
    if model is None:
        train_model()
    
    import pandas as pd

    input_data = pd.DataFrame([[study, sleep, attendance, previous]],
                          columns=['study_hours', 'sleep_hours', 'attendance', 'previous_score'])

    prediction = model.predict(input_data)
    return float(prediction[0])
def get_model_parameters():
    global model
    
    if model is None:
        train_model()
    
    m = model.coef_
    b = model.intercept_
    
    return m, b
