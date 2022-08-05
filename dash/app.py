import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA

import dash
from dash import dcc, html

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


def plot_allosteric_mutability():
    df = pd.read_csv("/scratch/ASD_Release_201909_AP.txt", sep="\t")
    mut = pd.read_csv("/scratch/mutations_gpcr.csv")
    uniprot_ids = [
        "Q14832",
        "P51686",
        "Q14416",
        "Q13467",
        "P21730",
        "P16473",
        "P47871",
        "Q14831",
        "Q9UBS5",
        "O75473",
        "P25116",
        "P43220",
        "O00222",
        "P28222",
        "Q9BXB1",
        "P61073",
        "P25024",
        "Q96RI0",
        "P41143",
        "P25103",
        "P29274",
        "P07550",
    ]
    df = df[df["uniprot_id"].isin(uniprot_ids)]

    data = df.apply(lambda row: [f"{row['uniprot_id']},{x}" for x in row["site_residue"].split(";")], axis=1).sum()

    data = pd.DataFrame([x.split(",") for x in data], columns=["uniprot_id", "from_big", "from", "pos", "chain"])
    data["pos"] = data["pos"].astype(str)
    mut["pos"] = mut["pos"].astype(str)

    merge = mut.merge(data.drop(["from_big", "chain"], axis=1), on=["pos", "uniprot_id"], how="left").drop_duplicates()
    allosteric_sites = merge[(merge["from_y"].notna()) & (merge["from_x"] == merge["from_y"])]
    non_allosteric_sites = merge[merge["from_y"].isna()]
    allosteric_sites["type"] = "allosteric"
    non_allosteric_sites["type"] = "non allosteric"

    col = ["#FF9900", "#f4080b"]
    fig = ff.create_distplot(
        [allosteric_sites["dist"], non_allosteric_sites["dist"]],
        ["allosteric", "non allosteric"],
        colors=col,
        show_rug=False,
        show_hist=False,
    )
    fig.update_layout(width=500, height=500)
    fig.update_layout(
        title="Allosteric site vs other parts of the protein mutation significance",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        # plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=[0, 0.5], title="ESM distance")
    fig.update_yaxes(title="Frequency")
    return fig


def get_mat_sim():
    mut = pd.read_csv("/scratch/human_prot_400_500/dist_data.csv")
    mat_sim = mut.groupby(["from", "to"]).agg("mean").pivot_table(index="from", columns="to", values="dist")
    mat_sim = mat_sim.fillna(0)
    mat_sim = mat_sim + mat_sim.T
    return mat_sim


def get_pam_blosum(mat_sim):
    def min_max(matrix):
        mn = matrix.min().min()
        mx = matrix.max().max()
        return (matrix - mn) / (mx - mn)

    colors = ["#0f6feb", "#f4080b"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PAM", "ESM"))
    pam = pd.read_csv("ftp://ftp.ncbi.nlm.nih.gov/blast/matrices/PAM30", sep="\s+", comment="#").iloc[:20, :20]
    mat_sim = mat_sim.loc[pam.index, pam.index]

    fig.add_trace(
        go.Heatmap(
            z=min_max(pam),
            x=pam.index,
            y=pam.columns,
            colorscale=[[0.0, "#413f6c"], [0.27, "#ab325a"], [0.6, "#f7d908"], [1.0, "#ffffff"]],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=min_max(1 - mat_sim),
            x=mat_sim.index,
            y=mat_sim.columns,
            colorscale=[[0.0, "#413f6c"], [0.27, "#ab325a"], [0.6, "#f7d908"], [1.0, "#ffffff"]],
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="PAM vs ESM mutation probability",
        paper_bgcolor="rgba(0,0,0,0)",
        # plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(showlegend=False, showscale=True)
    fig.update_layout(width=1000, height=500)
    return fig


def plot_corr(mat_sim):
    col = ["#ab325a"]
    corrs = pd.read_csv("/scratch/human_prot_400_500/pam_esm_corr.csv")
    fig = px.line(
        corrs,
        x="number",
        y="corr",
        color_discrete_sequence=col,
        width=500,
        height=500,
    )
    fig.update_layout(
        title="PAM and ESM Pearson correlation vs PAM number",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        # plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def tsne_mut(mut):
    red = PCA(n_components=2)
    reduced_mut = red.fit_transform([x.numpy() for x in mut["embedding"]])
    mut["uniprot_id"] = mut["name"].apply(lambda x: x.split("|")[0])
    mut["x"] = reduced_mut[:, 0]
    mut["y"] = reduced_mut[:, 1]
    fig = px.scatter(
        mut,
        "x",
        "y",
        color="uniprot_id",
        hover_name="name",
        width=900,
        height=700,
        title="t-SNE visualisation of human proteins and their mutations",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def tsne_rand(rand, wt):
    rand_sample = rand.sample(wt.shape[0])
    rand_sample["source"] = "random"
    wt["source"] = "wt"
    data = pd.concat([rand_sample, wt])
    red = PCA(n_components=2)
    reduced = red.fit_transform([x.numpy() for x in data["embedding"]])
    data["x"] = reduced[:, 0]
    data["y"] = reduced[:, 1]
    fig = px.scatter(data, "x", "y", color="source", hover_name="name", width=700, height=700)
    fig.update_layout(
        title="t-SNE visualisation of human vs randomly generated proteins",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

allosteric_mutability = plot_allosteric_mutability()
mat_sim = get_mat_sim()
pam_blosum = get_pam_blosum(mat_sim)
corr = plot_corr(mat_sim)
wt = pd.read_pickle("/scratch/human_prot_400_500/wt.pkl")
rand = pd.read_pickle("/scratch/human_prot_400_500/random.pkl")
mut = pd.read_pickle("/scratch/human_prot_400_500/mut.pkl")
tsne_wt = tsne_mut(mut)
tsne_rand = tsne_rand(rand, wt)

app.layout = html.Div(
    children=[
        html.H1(children="Show me what you got!"),
        html.Div(
            style={"display": "flex", "flex-direction": "row"},
            children=[
                dcc.Graph(id="allosteric-mutability", figure=allosteric_mutability),
                dcc.Graph(id="corr", figure=corr),
                dcc.Graph(id="pam-blosum", figure=pam_blosum),
            ],
        ),
        html.Div(
            style={"display": "flex", "flex-direction": "row"},
            children=[
                dcc.Graph(id="tsne-wt", figure=tsne_wt),
                dcc.Graph(id="tsne-rand", figure=tsne_rand),
            ],
        ),
    ]
)

# Only for running on development mode
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8080)
