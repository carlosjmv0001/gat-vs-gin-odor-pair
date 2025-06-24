import torch
from torch.utils.tensorboard import SummaryWriter
import torch_geometric as pyg
import torchmetrics
import tqdm
from pairing.data import PairData, Dataset, loader
import copy
import scipy
import scipy.stats
import uuid

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

dropout = 0

auroc = torchmetrics.classification.MultilabelAUROC(Dataset.num_classes())

def make_sequential(num_layers, input_dim, output_dim, is_last=False):
    layers = []
    layers.append(
        torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim),
                            torch.nn.ReLU(), torch.nn.Dropout(p=dropout)))
    while len(layers) < num_layers:
        layers.append(
            torch.nn.Sequential(torch.nn.Linear(output_dim, output_dim),
                                torch.nn.ReLU(), torch.nn.Dropout(p=dropout)))
    if is_last:
        if num_layers == 1:
            layers[-1] = torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim))
        else:
            layers[-1] = torch.nn.Sequential(
                torch.nn.Linear(output_dim, output_dim))
    return torch.nn.Sequential(*layers)

class GAT(torch.nn.Module):
    def __init__(self, num_convs, num_linear, embedding_size, aggr_steps, heads=2):
        super(GAT, self).__init__()
        self.layers = []
        self.task = "graph"
        self.pad = torch.nn.ZeroPad2d(
            (0, embedding_size - Dataset.num_node_features(), 0, 0))
        self.gat = pyg.nn.GATConv(embedding_size, embedding_size, heads=heads, concat=False)
        self.gat.to(device)
        self.num_convs = num_convs
        self.aggr_steps = aggr_steps
        self.readout = pyg.nn.aggr.Set2Set(embedding_size, aggr_steps)
        self.readout.to(device)
        self.post_mp = make_sequential(num_linear,
                                       2 * embedding_size,
                                       embedding_size,
                                       is_last=True)
        self.post_mp.to(device)
    def forward(self, x, edge_index, edge_attr, batch_index):
        x = self.pad(x)
        for _ in range(self.num_convs):
            x = self.gat(x, edge_index)
        pooled = torch.cat([
            pyg.nn.pool.global_add_pool(x, batch_index),
            pyg.nn.pool.global_mean_pool(x, batch_index)
        ], dim=1)
        if self.aggr_steps > 0:
            pooled = self.readout(x, index=batch_index)
        return self.post_mp(pooled)

class MixturePredictor(torch.nn.Module):
    def __init__(self, num_convs, num_linear, embedding_size, aggr_steps, num_classes, heads=2):
        super(MixturePredictor, self).__init__()
        self.gat = GAT(num_convs, num_linear, embedding_size, aggr_steps, heads=heads)
        self.out = make_sequential(num_linear,
                                   2 * embedding_size,
                                   num_classes,
                                   is_last=True)
    def forward(self, x_s, edge_index_s, edge_attr_s, x_s_batch, x_t, edge_index_t,
                edge_attr_t, x_t_batch, y, *args, **kwargs):
        emb_s = self.gat(x_s, edge_index_s, edge_attr_s, x_s_batch)
        emb_t = self.gat(x_t, edge_index_t, edge_attr_t, x_t_batch)
        embedding = torch.cat([emb_s, emb_t], dim=1)
        return self.out(embedding)

def do_train(params):
    print(params)
    model = MixturePredictor(num_convs=params["CONVS"],
                             num_linear=params["LINEAR"],
                             embedding_size=int(params["DIM"]),
                             aggr_steps=params["AGGR"],
                             num_classes=Dataset.num_classes(),
                             heads=2)
    model = model.to(device)
    bsz = int((2**14) / (params["DIM"]))
    print(f"BSZ={bsz}")
    train_loader = loader(train, batch_size=bsz)
    test_loader = loader(test, batch_size=bsz)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["LR"])
    end_step = .9 * params["STEPS"]
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                  start_factor=1,
                                                  end_factor=params["DECAY"],
                                                  total_iters=end_step)
    def do_train_epoch():
        model.train()
        losses = []
        for batch_data in train_loader:
            batch_data.to(device)
            optimizer.zero_grad()
            pred = model(**batch_data.to_dict())
            loss = loss_fn(pred, batch_data.y)
            loss.backward()
            losses.append(loss * len(batch_data.y))
            optimizer.step()
        return torch.stack(losses).sum() / len(train)
    def collate_test():
        model.eval()
        preds = []
        ys = []
        for batch_data in test_loader:
            batch_data.to(device)
            with torch.no_grad():
                pred = model(**batch_data.to_dict())
            preds.append(pred)
            ys.append(batch_data.y)
        return torch.cat(preds, dim=0), torch.cat(ys, dim=0)
    def get_test_loss():
        pred, y = collate_test()
        if pred.sum() == 0:
            print("HITTING ALL 0")
        return loss_fn(pred, y)
    def get_auroc():
        pred, y = collate_test()
        return auroc(pred, y.int())
    run_name = str(uuid.uuid1())[:8]
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    for s in tqdm.tqdm(range(int(params["STEPS"]))):
        loss = do_train_epoch()
        scheduler.step()
        tl = get_test_loss()
        writer.add_scalars('Loss', {'train': loss, 'test': tl}, s)
    torch.save(model, f"{log_dir}/model.pt")
    metrics = {"auroc": get_auroc(), "completed": s}
    print(run_name, metrics, params, sep="\n")
    writer.add_hparams(params, metrics)
    writer.close()

def generate_params():
    distributions = {
        'STEPS': scipy.stats.randint(100, 200),
        'LR': scipy.stats.loguniform(1e-5, 1e-1),
        'DIM': scipy.stats.randint(2**5, 2**10),
        "LINEAR": scipy.stats.randint(1, 6),
        "CONVS": scipy.stats.randint(1, 13),
        "AGGR": scipy.stats.randint(0, 13),
        "DECAY": scipy.stats.loguniform(1e-4, .1),
    }
    params = dict()
    for key, val in distributions.items():
        try:
            params[key] = val.rvs(1).item()
        except:
            params[key] = val.rvs(1)
    return params

if __name__ == "__main__":
    train = Dataset(is_train=True)
    test = Dataset(is_train=False)
    print(f"Training datapoints = {len(train)}. Test datapoints = {len(test)}.")
    for _ in range(1):
        do_train(generate_params())
