# Auto-converted from utils.ipynb.
# Cell boundaries are retained with # %% markers.

# %% [markdown]
# ### This notebook includes utility functions for Flash.


# %%
def Split_logs_into_chunks(df, r, start_id):
    """
    Splits a DataFrame of logs into chunks.

    This function divides a DataFrame into smaller chunks, each containing 'r' rows. 
    Each chunk is assigned a unique graph number starting from 'start_id'. The graph 
    number is incremented after every 'r' rows.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing log data.
    r (int): The number of rows in each chunk.
    start_id (int): The starting number for graph numbering.

    Returns:
    pandas.DataFrame: The modified DataFrame with an additional column 'graph_no' indicating 
                      the graph number for each row.
    """
    total_rows = len(df)
    df['graph_no'] = -1
    current_graph_no = start_id
    
    graph_ids = []
    for i in range(total_rows):
        graph_ids.append(current_graph_no)
        if (i + 1) % r == 0:
            current_graph_no += 1
            
    df.iloc[:len(graph_ids)]['graph_no'] = graph_ids
    df[df['graph_no'] == -1] = current_graph_no
    return df


# %%
def Modify_Structure(attack_df, benign_df, benign_structure_count):
    """
    Combines attack data with a subset of benign events.

    This function selects a specific number of unique benign processes from the benign 
    DataFrame and combines them with the attack DataFrame. Additionally, it creates 
    new events by pairing each malicious process with the selected benign processes 
    and appends these to the combined DataFrame.

    Parameters:
    attack_df (pandas.DataFrame): DataFrame containing attack data.
    benign_df (pandas.DataFrame): DataFrame containing benign data.
    benign_structure_count (int): The number of unique benign processes to select.

    Returns:
    pandas.DataFrame: A combined DataFrame containing both attack and selected benign data,
                      along with newly created events.
    """
    
    unique_benign_procs = list(set(benign_df['actorID']))
    selected_benign_procs = unique_benign_procs[:benign_structure_count]
    selected_benign_df = benign_df[benign_df['actorID'].isin(selected_benign_procs)]

    combined_df = pd.concat([attack_df, selected_benign_df], ignore_index=True)
                      
    event_template = {'actorID': None, 'objectID': None, 'action': '', 'timestamp': '', 'exec': '', 'path': ''}
    
    for malicious_proc in GT_mal:
        for benign_proc in selected_benign_procs:
            new_event = event_template.copy()
            new_event['actorID'] = malicious_proc
            new_event['objectID'] = benign_proc
            combined_df = combined_df.append(new_event, ignore_index=True)
    
    return combined_df

# %%
def train_xgb_model(word_encodings, gnn_encodings, labels, model_filename):
    """
    Trains an XGBoost classifier using word and GNN encodings and saves the model.

    This function trains an XGBoost classifier using provided word encodings and 
    graph neural network (GNN) encodings. It then saves the trained model to a file 
    and calculates the model's accuracy.

    Parameters:
    word_encodings (numpy.ndarray): The word encodings as a NumPy array.
    gnn_encodings (numpy.ndarray): The GNN encodings as a NumPy array.
    labels (numpy.ndarray): The labels for the training data.
    model_filename (str): The filename where the trained model will be saved.

    Returns:
    tuple: A tuple containing the trained XGBoost classifier and its accuracy.
    """

    x = np.hstack((word_encodings, gnn_encodings))
    y = labels

    xgb_cl = xgb.XGBClassifier()
    xgb_cl.fit(x, y)

    pickle.dump(xgb_cl, open(model_filename, "wb"))

    preds = xgb_cl.predict(x)
    accuracy = accuracy_score(y, preds)

    return xgb_cl, accuracy

# %%
def train_gnn(df, model, optimizer, device, num_epochs):
    """
    Trains a graph neural network (GNN) model using the provided dataset.

    This function prepares a graph from the given dataframe, trains the GNN model for 
    a specified number of epochs, and computes the encoding of the nodes after training.
    It uses CrossEntropyLoss with class weights for imbalance handling and calculates the 
    accuracy of the model.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the dataset for training.
    model: The GNN model to be trained.
    optimizer: The optimizer used for training the model.
    device: The device (CPU/GPU) on which the training is performed.
    num_epochs (int): The number of epochs for training the model.

    Returns:
    tuple: A tuple containing the trained GNN model and the node encodings as a NumPy array.
    """

    phrases, labels, edges, mapp = prepare_graph(df)
    word_encodings = [infer(x) for x in phrases]
    word_encodings = np.array(word_encodings)

    graph = Data(x=torch.tensor(word_encodings, dtype=torch.float).to(device),
                 y=torch.tensor(labels, dtype=torch.long).to(device),
                 edge_index=torch.tensor(edges, dtype=torch.long).to(device))

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = CrossEntropyLoss(weight=class_weights, reduction='mean')

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = criterion(out, graph.y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
            sorted, indices = out.sort(dim=1, descending=True)
            conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
            conf = (conf - conf.min()) / conf.max()
            pred = indices[:, 0]
            cond = pred == graph.y
            accuracy = cond.sum().item() / len(graph.y)

    model.eval()
    gnn_encodings = model.encode(graph.x, graph.edge_index).detach().cpu().numpy()

    return model, gnn_encodings

# %%
def Validate(nodes, labels, edges, model):
    """
    Validates a graph using the provided Gnn model and calculates confidence scores for each node.

    Parameters:
    nodes (List[List[float]]): Node features for the graph.
    labels (List[int]): labels for each node in the graph.
    edges (List[List[int]]): Edge indices of the graph.
    model (torch.nn.Module): The PyTorch model to be used for evaluation.

    Returns:
    List[float]: A list of confidence scores for each node in the graph where the model's prediction was incorrect.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = Data(x=torch.tensor(nodes, dtype=torch.float).to(device),
                 y=torch.tensor(labels, dtype=torch.long).to(device),
                 edge_index=torch.tensor(edges, dtype=torch.long).to(device))
    
    model.eval()
    out = model(graph.x, graph.edge_index)

    sorted, indices = out.sort(dim=1, descending=True)
    conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
    conf = (conf - conf.min()) / conf.max()

    pred = indices[:, 0]
    cond = (pred == graph.y)
    flag = ~torch.tensor(cond)

    return conf[flag].tolist()