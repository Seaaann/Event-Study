import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st

fig = plt.figure()
sns.set_palette("GnBu_d")
import yfinance as yf

stock_list = [
    "AMZN",
    "AAPL",
    "TSLA",
    "GE",
    "GILD",
    "BA",
    "NFLX",
    "MS",
    "LNVGY",
    "BABA",
    "LK",
    "JOBS",
    "CEO",
    "TSM",
    "JD",
    "^GSPC",
]

prices = pd.DataFrame()
for stock in stock_list:
    symbol = stock
    tickerData = yf.Ticker(symbol)
    price_data = tickerData.history(period="1d", start="2019-1-1", end="2020-4-1")[
        "Close"
    ]
    prices[stock] = price_data
return_df = prices.pct_change().dropna()

date = [
    "2020-03-13",
    "2020-03-13",
    "2020-03-13",
    "2020-03-13",
    "2020-03-13",
    "2020-03-13",
    "2020-03-13",
    "2020-03-13",
    "2020-01-23",
    "2020-01-23",
    "2020-01-23",
    "2020-01-23",
    "2020-01-23",
    "2020-01-23",
    "2020-01-23",
]
date_df = pd.DataFrame(
    date,
    index=[
        "AMZN",
        "AAPL",
        "TSLA",
        "GE",
        "GILD",
        "BA",
        "NFLX",
        "MS",
        "LNVGY",
        "BABA",
        "LK",
        "JOBS",
        "CEO",
        "TSM",
        "JD",
    ],
    columns=["Date"],
)
date_df.index.name = "CompanyName"

firm_list = [
    "AMZN",
    "AAPL",
    "TSLA",
    "GE",
    "GILD",
    "BA",
    "NFLX",
    "MS",
    "LNVGY",
    "BABA",
    "LK",
    "JOBS",
    "CEO",
    "TSM",
    "JD",
]
return_df["RF"] = 0.0065
return_df["Mkt_RF"] = return_df["^GSPC"] - return_df["RF"]


def eventstudy(returndata, eventdata, stocklist):
    """
    returndata: is a dataframe with the market returns of the different firms
    eventdata: eventdata for the different firms
    stocklist: a list of the firms involved in the analysis

    Returns:
    abnreturn: a dictionary of the abnormal returns for each firm in their respective eventwindows -/+20
    """
    abnreturn = {}  # abnormal returns on the event window
    returndata = returndata.reset_index()
    Bse = []
    for stock in stocklist:

        eventindex = int(
            returndata[
                returndata["Date"] == str(eventdata.at[stock, "Date"])
            ].index.values
        )
        print(eventindex)
        event_df = returndata.loc[
            eventindex - 10 : eventindex + 10, ["Date", stock, "RF", "Mkt_RF"]
        ]
        estimation_df = returndata.loc[
            eventindex - 260 : eventindex - 11, ["Date", stock, "RF", "Mkt_RF"]
        ]
        formula = stock + " - RF ~ Mkt_RF"
        beta_Mkt = (
            sm.OLS.from_formula(formula, data=estimation_df).fit().params["Mkt_RF"]
        )

        alpha = (
            sm.OLS.from_formula(formula, data=estimation_df).fit().params["Intercept"]
        )

        standard_error = sm.OLS.from_formula(formula, data=estimation_df).fit().bse

        Bse.append(standard_error)
        print("{}, beta_Mkt= {},alpha= {}".format(stock, beta_Mkt, alpha))

        # expected returns for each firm in the estimation window
        expectedreturn_eventwindow = (event_df[["Mkt_RF"]].values * beta_Mkt) + alpha

        # abnormal returns on the event window - AR

        abnormal_return = event_df[stock].values - list(
            expectedreturn_eventwindow.flatten()
        )
        abnreturn[stock] = abnormal_return

    abnormalreturns_df = pd.DataFrame(abnreturn)
    abnormalreturns_df.index = abnormalreturns_df.index - 10
    return abnormalreturns_df


def CAR_se(Abnormal_return, stock_list):
    """
    To get the standard error of Cumulative Abnormal Return for each stock
    Input: the Abnormal Return datafram or matrix, a list of company names
    Output: a dataframe of cumulative standard error for each stock
    """
    residual_sigma_single = pd.DataFrame()
    residual_sigma_cum_single = pd.DataFrame()
    resi_single = []
    d = {}
    for x in stock_list:
        resistd = abnormalreturns_df[x].std() / 15
        d.update({x: resistd})

    residual_sigma_single = pd.DataFrame(d, index=Abnormal_return.index)
    residual_sigma_cum_single = np.sqrt(residual_sigma_single.cumsum())
    se_cum_single = np.sqrt(((residual_sigma_cum_single ** 2) / 15))

    return se_cum_single


def CAAR_se(Abnormal_return, stock_list):

    """
    To get the standard error of Cumulative Average Abnormal Return
    Input: the Abnormal Return datafram or matrix, a list of company names
    Output: a list of cumulative standard error
    """
    residual_sigma = pd.DataFrame()
    resi = []
    d = {}
    for x in stock_list:
        resistd = abnormalreturns_df[x].std() / 15
        d.update({x: resistd})

    residual_sigma = pd.DataFrame(d, index=Abnormal_return.index)
    residual_sigma_cum = np.sqrt(residual_sigma.cumsum())
    se_cum = np.sqrt(((residual_sigma_cum ** 2) / 15).mean(axis=1))

    return se_cum


abnormalreturns_df = eventstudy(
    returndata=return_df, eventdata=date_df, stocklist=firm_list
)

for i in range(1, 16):
    plt.subplot(4, 4, i)
    abnormalreturns_df[abnormalreturns_df.columns[i - 1]].plot()
    plt.xlabel("Event Window")
    plt.ylabel("Return")
    plt.axhline(
        y=(np.sqrt((abnormalreturns_df.iloc[:, i - 1].std() ** 2 / 21)) * 1.96),
        color="red",
        linestyle="--",
    )
    plt.axhline(
        y=(np.sqrt((abnormalreturns_df.iloc[:, i - 1].std() ** 2 / 21)) * -1.96),
        color="red",
        linestyle="--",
    )
    plt.title(abnormalreturns_df.columns[i - 1])
plt.show()

mean_AAR = abnormalreturns_df.mean(axis=1)
var_AAR = (abnormalreturns_df.std()) ** 2
var_matrix = pd.DataFrame(var_AAR)
var_matrix = var_matrix.T
var_AAR = sum(var_matrix.iloc[0]) / 15 ** 2
Std_AAR = np.sqrt(var_AAR)
mean_AAR.plot()
plt.axhline(y=Std_AAR * 1.96, color="red", linestyle="--")
plt.axhline(y=Std_AAR * -1.96, color="red", linestyle="--")

se_cum_single = CAR_se(abnormalreturns_df, firm_list)
CAR_df = abnormalreturns_df.cumsum()
plt.figure(figsize=(24, 12))
for i in range(1, 16):
    plt.subplot(4, 4, i)
    CAR_df[CAR_df.columns[i - 1]].plot()
    plt.plot(se_cum_single.iloc[:, i - 1] * 1.96, color="red", linestyle="--")
    plt.plot(se_cum_single.iloc[:, i - 1] * -1.96, color="red", linestyle="--")
    plt.xlabel("Event Window")
    plt.ylabel("CAR")
    plt.title(CAR_df.columns[i - 1])
plt.show()

se = CAAR_se(abnormalreturns_df, firm_list)
Var_AAR = ((CAR_df.mean(axis=1)) ** 2) / 15
Std_AAR = np.sqrt(Var_AAR)
# CAAR
CAAR = mean_AAR.cumsum()
# Plot CAAR
CAAR.plot(figsize=(12, 8))
plt.xlabel("Event Window")
plt.plot(se * 1.96, color="red", linestyle="--")
plt.plot(se * -1.96, color="red", linestyle="--")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Average Abnormal Return")
plt.show()