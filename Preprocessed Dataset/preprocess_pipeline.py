def load_data(path=r"C:\Users\Saket Dixit\Downloads\oqmd-v1_2-for-cgnn\oqmd_dataset\targets.csv"):
  df=pd.read_csv(path)
  return df

def audit_data(df):
  print("Shape: \n",df.shape)
  print("\nMissing Value:\n ",df.isnull().sum())
  print("\nData Types:\n ",df.dtypes)
  print("\nSkewness: \n",df.skew(numeric_only=True))


def handle_missing(df):
  num_cols=df.select_dtypes(include=np.number).columns
  cat_cols=df.select_dtypes(include='object').columns
  
  df[num_cols]=df[num_cols].fillna(df[num_cols].median())
  df[cat_cols]=df[cat_cols].fillna("missing")
  return df


def drop_low_variance(df,threshold=0.01):
  selector=VarianceThreshold(threshold)
  num_cols=df.select_dtypes(include=np.number)
  selector.fit(num_cols)
  reduced=num_cols[num_cols.columns[selector.get_support()]]
  return pd.concat([reduced,df.select_dtypes(exclude=np.number)],axis=1)


def feature_engineering(df):
  if 'energy_per_atom' in df.columns and 'volume_per_atom' in df.columns:
    df['energy_density']=df['energy_per_atom']/(df['volume_per_atom']+1e-5)
    
  if 'nsites' in df.columns and 'volume_per_atom' in df.columns:
    df['site_density']=df['nsites']/(df['volume_per_atom']+1e-5)
    
  return df


def remove_outliers(df,z_thresh=4.0):
  num_cols=df.select_dtypes(include=np.number).columns
  z_scores=np.abs(zscore(df[num_cols]))
  mask=(z_scores<z_thresh).all(axis=1)
  return df[mask]


def encode_features(df,onehot_limit=1000):
  cat_cols=df.select_dtypes(include='object').columns.tolist()
  onehot_cols=[col for col in cat_cols if df[col].nunique()<=onehot_limit]
  label_cols=[col for col in cat_cols if df[col].nunique()>onehot_limit]
  
  for col in label_cols:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col].astype(str))
  
  if onehot_cols:
    enc=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
    enc_df=pd.DataFrame(enc.fit_transform(df[onehot_cols]),columns=enc.get_feature_names_out(onehot_cols))
    df=df.drop(columns=onehot_cols)
    df=pd.concat([df.reset_index(drop=True),enc_df.reset_index(drop=True)],axis=1)
    
  return df


def scale_features(df):
  scaler=RobustScaler()
  scaled=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
  return scaled



def visualize_corr(df):
  corr=df.corr()
  plt.figure(figsize=(15,10))
  sns.heatmap(corr,cmap='coolwarm',annot=False)
  plt.title('Correlation HeatMap')
  plt.show()



def preprocess_pipeline(path=r"C:\Users\Saket Dixit\Downloads\oqmd-v1_2-for-cgnn\oqmd_dataset\targets.csv",save=True):
  df=load_data(path)
  audit_data(df)
  
  df=handle_missing(df)
  df=feature_engineering(df)
  df=drop_low_variance(df)
  df=remove_outliers(df)
  df=encode_features(df)
  df=scale_features(df)
  
  visualize_corr(df)
  
  if(save):
    os.makedirs(r"C:\Users\Saket Dixit\Downloads\oqmd-v1_2-for-cgnn\oqmd_dataset",exist_ok=True)
    df.to_csv(r"C:\Users\Saket Dixit\Downloads\oqmd-v1_2-for-cgnn\oqmd_dataset\cleantarget.csv")
    joblib.dump(df.columns.tolist(),r"C:\Users\Saket Dixit\Downloads\oqmd-v1_2-for-cgnn\oqmd_dataset\features_names.joblib")
    
  return df


if __name__ == "__main__":
  df=preprocess_pipeline()
