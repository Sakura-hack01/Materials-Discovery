class UncertaintyNet(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1):
        super(UncertaintyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(dropout_rate)
        self.mean_out = nn.Linear(64, 1)      
        self.log_var_out = nn.Linear(64, 1)     

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        mean = self.mean_out(x)
        log_var = self.log_var_out(x)        
        return mean, log_var

def heteroscedastic_loss(y_pred_mean, y_log_var, y_true):
    precision = torch.exp(-y_log_var)
    return torch.mean(precision * (y_pred_mean - y_true)**2 + y_log_var)

def train_ensemble(df, target_column, n_models=5, epochs=200, lr=1e-3):
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    models = []
    for i in range(n_models):
        model = UncertaintyNet(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in loader:
                pred_mean, pred_log_var = model(batch_x)
                loss = heteroscedastic_loss(pred_mean, pred_log_var, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        models.append(model)
      
    preds_all = []
    vars_all = []
    for model in models:
        model.eval()
        pred_means, pred_log_vars = model(X_test)
        preds_all.append(pred_means.detach().numpy())
        vars_all.append(torch.exp(pred_log_vars).detach().numpy())

    preds_all = np.stack(preds_all)
    vars_all = np.stack(vars_all)
  
    ensemble_mean = np.mean(preds_all, axis=0)
    ensemble_variance = np.mean(vars_all + preds_all**2, axis=0) - ensemble_mean**2

    print("Test RMSE: ", np.sqrt(mean_squared_error(y_test, ensemble_mean)))
    print("Average Predictive Variance: ", np.mean(ensemble_variance))

    return ensemble_mean, ensemble_variance
  
if __name__ =="__main__":
    print(df.shape)
    train_ensemble(df,target_column='energy_per_atom',n_models=5,epochs=200)
