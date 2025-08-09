def initialize_AL_BO(df, target_col,max_train_size=10000):
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    X_train, X_pool, y_train, y_pool = train_test_split(X, y_norm, test_size=0.95, random_state=42)
    
    if(len(X_train)>max_train_size):
        indices=torch.randperm(len(train_X))[:max_train_size]
        X_train=train_X[indices]
        y_train=y_train[indices]
    
    return torch.tensor(X_train), torch.tensor(y_train), X_pool, y_pool, y_mean, y_std

def retrain_gp(train_X, train_Y):
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model

def select_uncertain_points(model, X_pool_tensor, k=10):
    posterior = model.posterior(X_pool_tensor)
    mean = posterior.mean
    variance = posterior.variance
    scores = variance.detach().numpy().reshape(-1)
    topk_indices = scores.argsort()[::-1][:k]
    return topk_indices, scores

def visualize_loop(iteration, train_Y, y_pool, pred_means, uncertainties):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(pred_means, kde=True)
    plt.title(f"Predicted Values - Iteration {iteration}")
    
    plt.subplot(1, 2, 2)
    sns.histplot(uncertainties, kde=True)
    plt.title(f"Uncertainty Distribution - Iteration {iteration}")
    
    plt.show()
    
    print(f"[Iteration {iteration}] Training Size: {len(train_Y)} | Pool Size: {len(y_pool)}")
    print("Train Mean ± Std: ", train_Y.mean().item(), "±", train_Y.std().item())

def active_learning_loop(df, target_col, iterations=10, query_size=10):
    train_X, train_Y, X_pool, y_pool, y_mean, y_std = initialize_AL_BO(df, target_col)
    
    for i in range(iterations):
        print(f"\n Active Learning Iteration {i+1}")
        model = retrain_gp(train_X, train_Y)
        
        X_pool_tensor = torch.tensor(X_pool, dtype=torch.float32)
        posterior = model.posterior(X_pool_tensor)
        pred_means = posterior.mean.detach().numpy().reshape(-1)
        pred_vars = posterior.variance.detach().numpy().reshape(-1)

        visualize_loop(i+1, train_Y, y_pool, pred_means, pred_vars)
        topk_indices, scores = select_uncertain_points(model, X_pool_tensor, k=query_size)
        new_X = torch.tensor(X_pool[topk_indices])
        new_Y = torch.tensor(y_pool[topk_indices])
      
        train_X = torch.cat([train_X, new_X], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        X_pool = np.delete(X_pool, topk_indices, axis=0)
        y_pool = np.delete(y_pool, topk_indices, axis=0)

if __name__ =='__main__':
  active_learning_loop(df,target_col='energy_per_atom',iterations=10,query_size=10)
