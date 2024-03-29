import multiprocessing as mp
import yfinance as yf
import tf_TradingOperationsNew as to
import ai_TradingEnvironment as te
import ai_TDQN as tdqn
import pandas as pd
pd.options.mode.chained_assignment = None
import torch
import numpy as np
import random
import os
from cm_common import ReadFromFile, Grouper, DummyProcess
# device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

networkSettings = tdqn.networkSettings_(inputLayerSize=151, hiddenLayerSize=1024, outputLayerSize=2, dropout=0.4)

tdqnSettings = tdqn.tdqnSettings_(gamma=0.4, 
                                  epsilonStart=1.0, 
                                  epsilonEnd=0.01,
                                  epsilonDecay=10000,
                                  capacity=100000, 
                                  learningRate=0.0001,
                                  targetUpdateFrequency=500, 
                                  batchSize=32, 
                                  gradientClipping=1,
                                  targetNetworkUpdate = 1000, 
                                  numberOfEpisodes = 10,
                                  onlineNumberOfEpisodes = 3,
                                  rewardClipping = 1.5,
                                  device = device
                                  )

optimSettings = tdqn.optimSettings_(L2Factor=0.000001)


startingDate = '2015-01-01'
splittingDate = '2022-01-01'
endingDate = '2023-02-24'


trainingHorizon = te.Horizon(startingDate, splittingDate, "1d")
testingHorizon = te.Horizon(splittingDate, endingDate, "1d")


def InitializeTrainingTesting(stockName, identifierString, verbose = False):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    fileName = stockName + identifierString
    modelFileName = "./Models/" + fileName + "/" + stockName
    figureFileName = "./Figures/" + fileName + "/" + stockName

    if verbose == True:
        print("Stock Name: " + stockName + identifierString)
    
    StockTraining = to.StockHandler(stockName, ReadFromFile, ReadFromFile, trainingHorizon)
    StockTesting = to.StockHandler(stockName, ReadFromFile, ReadFromFile, testingHorizon)
    bistIndexTraining = to.StockHandler("XU100.IS", ReadFromFile, ReadFromFile, trainingHorizon)
    bistIndexTesting = to.StockHandler("XU100.IS", ReadFromFile, ReadFromFile, testingHorizon)

    PositionTraining = to.DummyPosition(StockTraining)
    PositionTesting = to.DummyPosition(StockTesting)
    
    TrainingEnvironment = te.TradingEnvironment(PositionTraining, bistIndexTraining)
    TestingEnvironment = te.TradingEnvironment(PositionTesting, bistIndexTesting)
    TrainingEnvironment.InitScaler(TrainingEnvironment.Position.dataFrame)
    TestingEnvironment.InitScaler(TrainingEnvironment.Position.dataFrame)
    
    Agent = tdqn.TDQNAgent(TrainingEnvironment, TestingEnvironment, tdqnSettings, networkSettings, optimSettings)
    Agent.Training(verbose=True)
    Agent.SaveModel(modelFileName)
    Agent.Testing()
    PositionTesting.PlotActionsCapital(figureFileName + "-Capital", showFlag=False)
    PositionTesting.PlotActionsPrice(figureFileName + "-Price", showFlag=False)
    Agent.PlotLoss(figureFileName + "-Loss", showFlag=False)

    if verbose == True:
        print("Stock Name: " + stockName + identifierString + " Done")


def InitializeTesting(stockName, identifierString, verbose = False):
    fileName = stockName + identifierString
    modelFileName = "./Models/" + fileName + "/" + stockName
    figureFileName = "./Figures/" + fileName + "/" + stockName

    if verbose == True:
        print("Stock Name: " + stockName)
    
    StockTesting = to.StockHandler(stockName, ReadFromFile, ReadFromFile, testingHorizon)
    PositionTesting = to.DummyPosition(StockTesting)
    TestingEnvironment = te.TradingEnvironment(PositionTesting)
    Agent = tdqn.TDQNAgent(TestingEnvironment, TestingEnvironment, tdqnSettings, networkSettings, optimSettings)
    Agent.LoadModel(modelFileName)
    Agent.Testing()
    PositionTesting.PlotActionsCapital(figureFileName + "-Capital", showFlag=False)
    PositionTesting.PlotActionsPrice(figureFileName + "-Price", showFlag=False)
    Agent.PlotLoss(figureFileName + "-Loss", showFlag=False)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    # listOfStocksNames = ["TSKB.IS"]
    # listOfStocksNames = ["BTC-USD"]
    # listOfStocksNames = ["AAPL", "ISCTR.IS", "DOHOL.IS", "ASELS.IS", "SISE.IS", "TSKB.IS"]
    listOfStocksNames = ["TSKB.IS", "ISCTR.IS", "DOHOL.IS"]

    listOfStocksNames = ["AKBNK.IS",
              "AKSEN.IS",
              "ALARK.IS",
              "ARCLK.IS",
              "ASELS.IS",
              "BIMAS.IS",
              "EKGYO.IS",
              "EREGL.IS",
              "FROTO.IS",
              "GUBRF.IS",
              "SAHOL.IS",
              "HEKTS.IS",
              "KRDMD.IS",
              "KCHOL.IS",
              "KOZAL.IS",
              "KOZAA.IS",
              "ODAS.IS",
              "PGSUS.IS",
              "PETKM.IS",
              "SASA.IS",
              "TAVHL.IS",
              "TKFEN.IS",
              "TOASO.IS",
              "TCELL.IS",
              "TUPRS.IS",
              "THYAO.IS",
              "GARAN.IS",
              "ISCTR.IS",
              "SISE.IS",
              "YKBNK.IS"]
    listOfStocksNames.sort()


    identifier = "-2123-{}E-{}B-{}U".format(tdqnSettings.numberOfEpisodes, tdqnSettings.batchSize, tdqnSettings.targetNetworkUpdate)

    paths = ["Models", "Figures"]

    for path in paths:
        for stockName in listOfStocksNames:
            dirs = path + "/" + stockName + identifier
            if not os.path.exists(dirs):
                os.makedirs(dirs, exist_ok=True)

    processList = [mp.Process(target=InitializeTrainingTesting, args = (listOfStocksNames[i], identifier, True)) for i in range(len(listOfStocksNames))]
    
    proc = DummyProcess()
    processGroup = Grouper(processList, 12, proc) 

    for processList in processGroup:
        for process in processList:
            process.start()
        for process in processList:
            process.join()

    # for process in processList:
    #     process.join()


