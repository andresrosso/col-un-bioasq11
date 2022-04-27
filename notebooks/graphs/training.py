import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score, precision_score, accuracy_score

from src.models.model import GraphClassifier
from src.graphs.graph_loader import GraphDataset
from src.elastic_search_utils.elastic_utils import save_json


# ## Params

# In[2]:


LOAD_FOLDER = '/datasets/johan_tests_original_format_graphs/similarity_shape_100_20__score_threshold_006__similarity_relevance_07/training'


# In[3]:


SAVING_FOLDER = '/datasets/johan_tests_models/v1'


# In[4]:


SAVING_MODEL_PATH = f'{SAVING_FOLDER}/model.pth'
SAVING_METRICS_PATH = f'{SAVING_FOLDER}/metrics.json'


# In[20]:


DEBUG = False


# ## Dataset params

# In[5]:


BATCH_SIZE = 256


# In[6]:


VAL_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.15


# In[7]:


RELEVANCE_THRESHOLD = 0.04  # Keep 20% of elastic most relevant


# In[8]:


RANDOM_STATE = 42


# In[9]:


torch.manual_seed(RANDOM_STATE)


# ## Model params

# In[10]:


INPUT_DIM = 20
NODES_PER_GRAPH = 100
HIDDEN_CHANNELS = 16
OUTPUT_DIM = 2  # N_CLASSES
DROPOUT = 0.5


# In[11]:


DEVICE = torch.device('cuda')


# In[12]:


EPOCHS = 10


# ## Constants

# In[13]:
print('GENERATING DATASET')

dataset = GraphDataset(
    dataset_path=LOAD_FOLDER,
    batch_size=BATCH_SIZE,
    val_percentage=VAL_PERCENTAGE,
    test_percentage=TEST_PERCENTAGE,
    random_state=RANDOM_STATE,
    score_threshold=RELEVANCE_THRESHOLD,
    debug=DEBUG
)


# In[14]:


model = GraphClassifier(
    input_dims=INPUT_DIM,
    nodes_per_graph=NODES_PER_GRAPH,
    hidden_channels=HIDDEN_CHANNELS,
    output_dim=OUTPUT_DIM,
    dropout=DROPOUT
)

model = model.to(DEVICE)


# In[15]:


flat_labels = dataset.metadata['label'].tolist()

class_weights = compute_class_weight(
    'balanced', classes=[0,1], y=flat_labels
)
class_weights = torch.FloatTensor(class_weights)

# MU = 12
# class_weights = torch.FloatTensor([  # GDOT WEIGHTS
#     np.log((MU*7173.0/5886.0) + 1),
#     np.log((MU*7173.0/1287.0) + 1)
# ])
class_weights = class_weights.to(DEVICE)
class_weights


# In[16]:


criterion = torch.nn.NLLLoss(
    weight=class_weights
)
criterion = criterion.to(DEVICE)


# In[17]:

#1e-7 does nothing
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)#, weight_decay=5e-3)


# In[18]:


dataset.metadata['label'].value_counts()


# ## Training loop

# In[19]:


losses = []
accurracies = []
recalls = []
precisions = []
for epoch in range(EPOCHS):
    print('*' * 20)
    print(f'TRAINING EPOCH: {epoch}')
    model.train()

    epoch_loss = 0
    n_batches = 0
    for n_batch, batch in enumerate(dataset.get_batch('train')):
        batch = batch.to(DEVICE)
        optimizer.step()
        out = model(batch).float()
        batch_y_test = batch.y #F.one_hot(batch.y, OUTPUT_DIM).float()

        loss = criterion(out, batch_y_test)
        loss.backward()
        optimizer.zero_grad()
        batch.to('cpu')
        epoch_loss += loss.to('cpu').tolist()
        # FIXME JUST FOR MEMORY CLEANING
        del out
        del batch_y_test
        torch.cuda.empty_cache()
        n_batches += 1
        
    epoch_loss = epoch_loss/(n_batches)
    losses.append(epoch_loss)
    print(f'EPOCH {epoch} LOSS: {epoch_loss}')
    model.eval()
    y_val = []
    y_pred = []

    for test_batch in dataset.get_batch('val'):
        test_batch = test_batch.to(DEVICE)
        out_test = model(test_batch)
        test_gold = test_batch.y  # F.one_hot(test_batch.y, OUTPUT_DIM).float()
        y_val.append(test_gold.tolist())
        y_pred.append(out_test.tolist())
        test_batch.to('cpu')
        del out_test
        del test_gold
        torch.cuda.empty_cache()
    y_val = np.concatenate(y_val)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(
        y_val, y_pred.argmax(axis=1)
    )
    recall = recall_score(
        y_val, y_pred.argmax(axis=1), average='binary'
    )
    precision = precision_score(
        y_val, y_pred.argmax(axis=1), average='binary'
    )
    accurracies.append(acc)
    recalls.append(recall)
    precisions.append(precision)
    print("Value counts", pd.value_counts(np.array(y_pred.argmax(axis=1))))
    print("Recall", recall)
    print("Precision", precision)
    print(f'Accuracy: {acc:.4f}')


# In[ ]:


time_series = {
    'loss': losses,
    'accuracy': accurracies,
    'recall': recalls,
    'precision': precisions
}


# In[ ]:


save_json(time_series, SAVING_METRICS_PATH)


# In[ ]:


torch.save(model.state_dict(), SAVING_MODEL_PATH)
