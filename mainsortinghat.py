from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('dataset_train.csv')
df_test = pd.read_csv('dataset_test.csv')

def standartize(df, names):
    tmp = df.copy()
    df = tmp[names]
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    return scaled_df, scaler

def preprocess_df(df):
    df = df.copy()
    # drop useless columns
    df = df.drop(columns=['Index', 'First Name', 'Last Name', 'Astronomy'])
    # fill missing values in features with mean
    df = df.fillna(df.mean())        
    # preprocess columns
    # convert categorical features to numerical
    df['Best Hand'] = df['Best Hand'].astype('category')
    df['Hogwarts House'] = df['Hogwarts House'].astype('category')
    # store mapping for later usage
    map_dict = dict(enumerate(df['Best Hand'].cat.categories)), dict(enumerate(df['Hogwarts House'].cat.categories))
    df['Best Hand'] = df['Best Hand'].cat.codes
    df['Hogwarts House'] = df['Hogwarts House'].cat.codes
    # convert string date to datetime
    df['Birthday'] = pd.to_datetime(df['Birthday'])
    # separate datetime into day, month, year features
    df['Birth_day'] = df['Birthday'].dt.day
    df['Birth_month'] = df['Birthday'].dt.month
    df['Birth_year'] = df['Birthday'].dt.year
    df = df.drop(columns=['Birthday'])
    
    X = df.drop(columns=['Hogwarts House'])
    y = df['Hogwarts House']
    
    # get train and val split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)  
    transformed_features = list(set(X_train.columns) - set(['Best Hand']))
    
    train_indxs = X_train.index
    test_indxs = X_test.index
    
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # standartize train
    X_train_scaled, scaler = standartize(X_train, transformed_features)
    X_train_scaled['Best Hand'] = X_train['Best Hand']
    
    # standartize test
    X_test_scaled = scaler.fit_transform(X_test[transformed_features])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=transformed_features)
    X_test_scaled['Best Hand'] = X_test['Best Hand']
    
    preprocessing_params = {'map_dict': map_dict,
                            'scaler': scaler,
                            'train_indxs':train_indxs,
                            'test_indxs':test_indxs}
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessing_params

def df_to_numpyarray(X, y):
    # Extract input & outupts as numpy arrays
    inputs_array = X.to_numpy()
    targets_array = y.to_numpy()
    return inputs_array, targets_array


