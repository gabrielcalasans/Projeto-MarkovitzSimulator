from altair.vegalite.v5.api import value
import streamlit as st
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import streamlit_nested_layout
from streamlit_tags import st_tags
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go



st.set_page_config(layout = 'wide')

memoria = st.session_state

# if 'numPortfolios' not in 'memoria':
#     memoria.numPortfolios = 10000

NUM_TRADING_DAYS = 252
# NUM_PORTFOLIOS = memoria.numPortfolios


# * Backend
class Portfolios:
    def __init__(self, dados, NUM_PORTFOLIOS):
        self.NUM_PORTFOLIOS = NUM_PORTFOLIOS
        self.stocks = dados.columns.tolist()
        self.dados = dados
        self.retornos = self.calculateReturn(self.dados)        
        self.portfolios = {}        
        self.portfolios['weights'], self.portfolios['means'], self.portfolios['risks'] = self.generatePortfolios(self.retornos)        
        self.otimo = self.optimizePort(self.portfolios['weights'], self.retornos)
        self.portOtimo = {
            'weights': self.otimo['x'],
            'vol': self.statistics(self.otimo['x'], self.retornos)[1],
            'returns': self.statistics(self.otimo['x'], self.retornos)[0],
            'sharpe': self.statistics(self.otimo['x'], self.retornos)[2]
        }     
    
    def calculateReturn(self, data):
        '''Calculate log return (Normalization)'''
        logReturn = np.log(data/data.shift(1))
        return logReturn[1:]
    
    def generatePortfolios(self, returns):
        '''Generate multiples portfolios'''
        pfMeans = []
        pfRisk = []
        pfWeights = []
        emojis = [':dollar:', ':moneybag:', ':euro:', ':briefcase:']
        msg = st.toast(f"**Generating portfolios...**")
        time.sleep(0.85)

        for _ in range(self.NUM_PORTFOLIOS):
            indice = _ % len(emojis)
            weight = np.random.random(len(self.stocks))
            weight /= np.sum(weight)
            pfWeights.append(weight)
            pfMeans.append(np.sum(returns.mean() * weight) * NUM_TRADING_DAYS)
            pfRisk.append(np.sqrt(np.dot(weight.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weight))))
            msg.toast(f"**Portfolio {_ + 1} generated!** {emojis[indice]}")
            

        return np.array(pfWeights), np.array(pfMeans), np.array(pfRisk)
        
    def optimalPortfolio(self):
        optimum = self.otimo
        returns = self.retornos
        print(f"Optimal portfolio for {self.dados.columns.tolist()} is {optimum['x'].round(3)}")
        print(f"Expected return, vol and Sharpe Ratio is {self.statistics(optimum['x'].round(3), returns)}")
   
    def figuraData(self):
        '''Plots a graph'''
        data = self.dados
        data.plot(figsize = (15, 8))
        plt.show()
        
    # def figuraPortfolio(self):
    #     returns = self.portfolios['means']
    #     vols = self.portfolios['risks']
    #     plt.figure(figsize = (15, 8))
    #     plt.scatter(vols, returns, c = returns/vols, marker = 'o')
    #     plt.grid(True)
    #     plt.xlabel('Expected Volatility')
    #     plt.ylabel('Expected Return')
    #     plt.colorbar(label = 'Sharpe Ratio')
    def figuraPortfolioPesos(self):
        portfolio = {'Weights': self.otimo['x'], 'Stocks': self.stocks}
        portfolio = pd.DataFrame(portfolio)
        fig = px.pie(portfolio, values = 'Weights', names = 'Stocks', title = 'Optimal Portfolio Stocks Weights')
        return fig

    def figuraOP(self):
        # opt = self.otimo
        # rets = self.retornos
        # pfRets = self.portfolios['means']
        # pfVols = self.portfolios['risks']
        # plt.figure(figsize = (15, 8))
        # plt.scatter(pfVols, pfRets, c = pfRets/pfVols, marker = 'o')
        # plt.grid(True)
        # plt.xlabel('Expected Volatility')
        # plt.ylabel('Expected Return')
        # plt.colorbar(label = 'Sharpe Ratio')
        # plt.plot(self.statistics(opt['x'], rets)[1], self.statistics(opt['x'], rets)[0], 'g*', markersize = 20)
        # plt.show()
        opt = self.otimo
        rets = self.retornos
        pfRets = self.portfolios['means']
        pfVols = self.portfolios['risks']
        fig = px.scatter(y = pfRets, x = pfVols, color = (pfRets/pfVols), labels = {
            'y': 'Expected Returns',
            'x': 'Expected Volatility',
            'color': 'Sharpe Ratio'
        }, title =  f'{self.NUM_PORTFOLIOS} Random Portfolios')
        fig.add_trace(
            go.Scatter(
                mode = 'markers',
                y = [self.statistics(opt['x'], rets)[0]],
                x = [self.statistics(opt['x'], rets)[1]],
                marker = dict(color = 'goldenrod'),
                name = 'Best Portfolio'
            )
        )
        return fig
    
    def optimizePort(self, weights, returns):
        '''Look for the portfolio with max sharpe ratio'''
        # The sum of weights is 1
        constraints = {
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1
        }
        bounds = [tuple((0, 1)) for _ in range(len(self.stocks))]
        return optimization.minimize(fun = self.minFunctionSharpe, x0 = weights[0],
                              args = returns, method = 'SLSQP', bounds = bounds,
                             constraints = constraints)

    def statistics(self, weights, returns):
        pfReturn = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
        pfVol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
        return np.array([pfReturn, pfVol, pfReturn/pfVol])

    # SciPy optimize can found the minimum of a given functions, in this case, we will find min volatility
    # the maximum of a f(x) is the minimum of -f(x)
    def minFunctionSharpe(self, weights, returns):
        '''Get the statistics of sharpe ratio'''
        return -self.statistics(weights, returns)[2]

def msgDownload(ativo):
    texto = "Downloading data from"
    msg = st.toast(f"{texto} {ativo} :briefcase:")
    for emoji in [':dollar:', ':moneybag:', ':euro:', ':briefcase:']*2:
        msg.toast(f"**{texto} {ativo} {emoji}**")
        time.sleep(0.15)
    msg.toast(f"**{ativo} downloaded!** :heavy_check_mark:")

def downloadData(stocks, inicio, fim):
    '''Download market data with Yahoo Finance'''
    stockData = {}
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            stockData[stock] = ticker.history(start = inicio, end = fim)['Close']
            msgDownload(stock)
        except:
            st.error(f"Historical data not found for {stock}")
        
    return pd.DataFrame(stockData)






    
def principal():
    def configuracoes():
        st.subheader(f"Settings :gear:")

        padrao = ['AAPL', 'WMT', 'TSLA', 'GE', 'AMZN', 'DB']
        stocks = st_tags(
            label='_Tickers_:dollar:',
            text='Press enter to add more',
            value = padrao,    
            suggestions=['ITUB4.SA', 'VALE3.SA', 'PETR4.SA', 'KNRI11.SA', 'AAPL', 'GOOG'],    
            key='tickers')


        colunasInternas = st.columns(2)
        with colunasInternas[0]:
            inicio = st.date_input(f"Start date :date:", value = datetime.date(2012, 1, 1), help = f"Start date of analysis :calendar:")

        with colunasInternas[1]:
            fim = st.date_input(f"End date :date:", value = datetime.date(2017, 1, 1), help = f"End date of analysis :calendar:")

        colunasInternas = st.columns(3)
        with colunasInternas[0]:
            numPortfolios = st.number_input(f"Nº of Random Portfolios :briefcase:", value = 10000, min_value = 100)
            
        
        
        return downloadData(stocks, inicio, fim), numPortfolios
        
        
        

    
    
    with st.sidebar:
        dados, numPortfolios = configuracoes()
    
    st.header(f"Markovitz Simulator :robot_face:")
    portfoliosGerados = Portfolios(dados, numPortfolios)
    colunas = st.columns(2)
    with colunas[0]:
        st.subheader(f"Graphs")
        st.plotly_chart(portfoliosGerados.figuraPortfolioPesos())        
        st.plotly_chart(portfoliosGerados.figuraOP())
        #st.dataframe(portfoliosGerados.portfolios)

    with colunas[1]:
        st.subheader(f"Statistics")
        st.write(f"Sharpe Ratio is {round(portfoliosGerados.portOtimo['sharpe'], 3)}")
        #st.dataframe(dados)        
        

        
        


principal()