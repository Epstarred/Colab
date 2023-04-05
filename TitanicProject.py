import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.linear_model

data_dir = '/Users/elisabethstarr/Desktop/'
train_data = pd.read_csv(data_dir + 'train.csv')
test_data = pd.read_csv(data_dir + 'test.csv')
#%%
df_train = train_data[['Age', 'SibSp', 'Fare']].astype('float64')
labels_train = train_data['Survived'].astype('float64')
#%%
#Problem: Each passenger must be described by a feature vector.
#Problem: Many ages are missing - how should we handle it?
#Solution: We decide to  fill in the NAs with the average passenger age

#check = (labels_pred == y_val)
#num_correct = sum(check)
#accuracy = num_correct/len(y_val)
#print('accuracy is:', accuracy)
mean_fare = train_data['Fare'].mean()
impute_vals = {'Age':mean_age, 'Fare':mean_fare, 'Cabin':'U'}
train_data = train_data.fillna(impute_vals)

#%%
cabins= [s for s in train_data['Cabin']]
decks = [c[0] for c in cabins]

#How many letters are in the decks?

d={}
for deck in decks:
    if deck not in d:
        d[deck] = 1
    else:
        d[deck] +=1


#%%

#Let's add a column for sex/gender
df_train['isFemale'] = (train_data['Sex'] =='female').astype('float64')
df_train['A'] = np.float64([deck =='A' for deck in decks]) #list comprehension syntax
df_train['B'] = np.float64([deck =='B' for deck in decks])
df_train['C'] = np.float64([deck =='C' for deck in decks])
df_train['D'] = np.float64([deck =='D' for deck in decks])
df_train['E'] = np.float64([deck =='E' for deck in decks])
df_train['F'] = np.float64([deck =='F' for deck in decks])
df_train['G'] = np.float64([deck =='G' for deck in decks])
df_train['T'] = np.float64([deck =='T' for deck in decks])
df_train['U'] = np.float64([deck =='U' for deck in decks])

#df_train = pd.get_dummies(train_data, columns = ['Pclass', 'Sex', 'Embarked'])
#the short coming with using get_dummies is if not all cabins are represented in the data

#%%
labels_train = train_data['Survived'].astype('float64')
model = sk.linear_model.LogisticRegression(max_iter = 500)
model.fit(df_train, labels_train)

#%%
test_data = test_data.fillna(impute_vals)
df_test = test_data[['Age', 'SibSp', 'Fare']].astype('float64')
df_test['isFemale'] = (test_data['Sex'] =='female').astype('float64')

cabins= [s for s in test_data['Cabin']]
decks = [c[0] for c in cabins]
df_test['isFemale'] = (test_data['Sex'] =='female').astype('float64')
df_test['A'] = np.float64([deck =='A' for deck in decks]) #list comprehension syntax
df_test['B'] = np.float64([deck =='B' for deck in decks])
df_test['C'] = np.float64([deck =='C' for deck in decks])
df_test['D'] = np.float64([deck =='D' for deck in decks])
df_test['E'] = np.float64([deck =='E' for deck in decks])
df_test['F'] = np.float64([deck =='F' for deck in decks])
df_test['G'] = np.float64([deck =='G' for deck in decks])
df_test['T'] = np.float64([deck =='T' for deck in decks])
df_test['U'] = np.float64([deck =='U' for deck in decks])

#%%
labels_pred = model.predict(df_test)

#%%
#Create a Dictionary for new Pandas Data frame
kaggle_sub_dict ={'PassengerId': test_data['PassengerId'], 'Survived':np.int64(labels_pred)}
kaggle_sub = pd.DataFrame(d_sub)
kaggle_sub.to_csv(data_dir+'myKaggleSubmission.csv', index=False)

#%% Feb 9th





#%%
