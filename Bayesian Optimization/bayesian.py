warnings.filterwarnings("ignore")

def initialize_botorch(df, target_column,max_train_size=10):
    X = df.drop(columns=[target_column]).values.astype(np.float32)
    y = df[target_column].values.astype(np.float32).reshape(-1, 1)
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    X_train, X_pool, y_train, y_pool = train_test_split(X, y_norm, test_size=0.8, random_state=42)
    
    if(len(X_train)>max_train_size):
        indices=np.random.choice(len(X_train),max_train_size,replace=False)
        X_train=X_train[indices]
        y_train=y_train[indices]

  train_X = torch.tensor(X_train)
    train_Y = torch.tensor(y_train)

    return train_X, train_Y, X_pool, y_pool, y_mean, y_std

def run_initial_bo_loop(train_X, train_Y, X_pool, y_pool, y_mean, y_std, target_column, n_suggestions=5):
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    EI = ExpectedImprovement(model=gp, best_f=train_Y.max(), maximize=True)

    X_pool_tensor = torch.tensor(X_pool, dtype=torch.float32)
    EI_vals = EI(X_pool_tensor.unsqueeze(1)).squeeze().detach().numpy()
    top_indices = EI_vals.argsort()[::-1][:n_suggestions]
    top_candidates = X_pool[top_indices]
    predicted_targets = gp.posterior(torch.tensor(top_candidates)).mean.detach().numpy() * y_std + y_mean

    df_suggested = pd.DataFrame(top_candidates, columns=[f"f_{i}" for i in range(top_candidates.shape[1])])
    df_suggested[target_column + "_predicted"] = predicted_targets
    df_suggested["rank"] = np.arange(1, n_suggestions + 1)

    return df_suggested,EI_vals

def plot_acquisition_values(EI_vals):
    plt.figure(figsize=(10, 4))
    plt.title("Acquisition Function (Expected Improvement) over Pool")
    plt.plot(sorted(EI_vals)[::-1])
    plt.xlabel("Ranked Candidate Index")
    plt.ylabel("EI Value")
    plt.grid(True)
    plt.show()
  
if __name__ =="__main__":
  train_X,train_Y,X_pool,y_pool,y_mean,y_std=initialize_botorch(df,target_column='energy_per_atom')
  suggestion,EI_vals=run_initial_bo_loop(train_X, train_Y,X_pool,y_pool,y_mean,y_std,'energy_per_atom',5)
  print(suggestion)
  plot_acquisition_values(EI_vals)
