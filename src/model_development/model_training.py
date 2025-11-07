import pandas as pd
from sklearn import linear_model
# Functions to streamline training different kinds of models and getting predictions

def fit_calorie_model(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model: linear_model
):
    '''
    With a given model, fit it on training data and make predictions for the test data.

    Args:
        training_data
        test_data
        model : any sklearn model to be fitted

    Returns:
    {
        "fitted_model": initial model that has been fitted to training data,
        "test_predictions": predictions based on test data features
    }
    '''
    model.fit(training_data.drop(["Calories"], axis = 1),
        training_data["Calories"])

    predictions = model.predict(test_data.drop(["Calories"], axis = 1))

    return {
        "fitted_model": model,
        "test_predictions": predictions
    }



def postprocess_predictions(predictions: list):
    '''
    Apply postprocessing to generated predicted values to ensure sensible values.
    '''

    # We don't want to predict calorie expenditure of less than 0, 
    # this would also break the chosen metric calculator
    predictions[predictions < 0] = 0

    return predictions