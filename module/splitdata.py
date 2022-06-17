from lib2to3.pgen2.pgen import DFAState
import pandas as pd

FIX_SIZE = 5000
DOMAIN_DICT = {
      0 : 'Imaginative',
      1 : 'Natural & Pure Science',
      2 : 'Applied Science',
      3 : 'Social Science ',
      4 : 'History',
      5 : 'Commerce & Finance',
      6 : 'Arts',
      7 : 'Belief & Thought'
  }

def setDF(path):
    global DF 
    DF = pd.read_csv('path')


def findProportion(countValues, trainPercent=0.7, valPercent=0.15, testPercent=0.15, reservePercent=0.1, size=None):
  output={}
  sortedSerial = countValues.index

  if size != None:
    overallSize = size 
  else: 
    overallSize = int(len(DF) / 8)

  reserveSize = int(overallSize * reservePercent)
  dataSize = overallSize -  reserveSize


  trainSize =  int(dataSize * trainPercent)
  valSize = int(dataSize * valPercent)
  testSize = int(dataSize * testPercent)

  limits = [trainSize,valSize,testSize]
  splitTags = ['Train','Validation','Test']
  
  start_index = 0

  for splitIndex in range(3):
    count = 0
    limit = limits[splitIndex]
    propDict = {}

    for i, serial in enumerate(sortedSerial[start_index:]):
      Index = start_index + i 

      if count + countValues[Index] < limit:
        count+=countValues[Index]
        propDict[serial] = countValues[Index]

      elif count + countValues[Index] == limit:
        propDict[serial] = countValues[Index]
        start_index = Index + 1
        break

      else:
        propDict[serial] = limit - count
        start_index = Index + 1
        break

    output[splitTags[splitIndex]] = propDict
  return output



def findProportion_all(Size=None):
  data = {}
  for i in range(8):
    category = DOMAIN_DICT[i]
    df = DF.loc[DF['Category'] == category]
    
    if Size != None:
      df.sample(frac=1,random_state=69).reset_index(drop=True)
      df = df[:Size]

    countValues = df['Serials'].value_counts()
    data[category] = findProportion(countValues,size=Size)
  return data

def getProportionPair(proportionDict):
  data = {}

  trainLST =  []
  valLST = []
  testLST = []
  
  for i in range(8):
    domain = DOMAIN_DICT[i]
    splitPorpDict = proportionDict[domain]
    
    trainLST += [pair for pair in list(splitPorpDict['Train'].items())]
    valLST += [pair for pair in list(splitPorpDict['Validation'].items())]
    testLST += [pair for pair in list(splitPorpDict['Test'].items())]

  data['Train'] = trainLST
  data['Validation'] = valLST
  data['Test'] = testLST

  return data

def splitData(portpotionPair):
  splitTags = ['Train','Validation','Test']
  main_df = pd.DataFrame(columns=DF.columns)
  for tag in splitTags:
    main_df = main_df.iloc[0:0]
    for pair in portpotionPair[tag]:
      serial = pair[0]
      amount = pair[1]

      add_df = DF.loc[DF['Serials'] == serial]

      add_df.drop(add_df.index[amount:], inplace=True)

      main_df = pd.concat([main_df,add_df]).reset_index(drop=True)
    main_df.to_csv(f'TNC_{tag}Set_{FIX_SIZE}.csv', index=False)
    print(f'Export {tag} complete!!!')




