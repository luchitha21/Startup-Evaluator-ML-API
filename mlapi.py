from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import shap
from fastapi.middleware.cors import CORSMiddleware

class Score(BaseModel):
    funding_total_usd:float
    funding_rounds:float
    first_funding_year:float
    last_funding_year:float
    founded_year:float
    months_bw_fundings:float
    average_funded_per_round:float
    is_software:float
    is_biotech: float
    is_curatedweb:float
    is_mobile:float
    is_Ecommerce:float
    is_usa:float

class Valuation(BaseModel):
    funding_rounds:float
    funding_total_usd:float
    number_of_members:float
    number_of_founders:float
    mean_funding:float
    max_funding:float
    seed:float
    number_of_invested_VCs:float
    total_investment_from_VCs: float
    year:float
    month:float
    day:float

with open("catboost.pkl",'rb') as f:
    model =  pickle.load(f)

with open("catboostRegressor.pkl",'rb') as g:
    model2 =  pickle.load(g)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/')
async def base_endpoint(item:Score):

    test_data =  [
        item.funding_total_usd,
        item.funding_rounds,
        item.first_funding_year,
        item.last_funding_year,
        item.founded_year,
        item.months_bw_fundings,
        item.average_funded_per_round,
        item.is_software,
        item.is_biotech,
        item.is_curatedweb,
        item.is_mobile,
        item.is_Ecommerce,
        item.is_usa
    ]


    # _scaled = scaler.transform(np.array(new_data).reshape(1, -1))
    # yhat = model.predict(test_data)[0]
    yprop = model.predict_proba(test_data)[0]
    yprop2 = model.predict_proba(test_data)[1]
    yprop3 = model.predict_proba(test_data)[2]
    yprop4 = model.predict_proba(test_data)[3]
    # graph  = shap.summary_plot(shap_values, test_data, plot_type="bar", feature_names = test_data, class_names=class_names)
    return {
        "probability":[yprop,yprop2,yprop3,yprop4],
        }

@app.post('/valuations')
async def base_endpoint(item:Valuation):

    test_data2 =  [
        item.funding_rounds,
        item.funding_total_usd,
        item.number_of_members,
        item.number_of_founders,
        item.mean_funding,
        item.max_funding,
        item.seed,
        item.number_of_invested_VCs,
        item.total_investment_from_VCs,
        item.year,
        item.month,
        item.day,
    ]


    # _scaled = scaler.transform(np.array(new_data).reshape(1, -1))
    # yhat = model.predict(test_data)[0]
    value = model2.predict(test_data2)
    # graph  = shap.summary_plot(shap_values, test_data, plot_type="bar", feature_names = test_data, class_names=class_names)
    return {
        "value":value,
        }