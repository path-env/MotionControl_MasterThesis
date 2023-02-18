#%%
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px

# read report
ML_data = np.load('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/main/report/ML_results.npz', allow_pickle= True)
DL_data = np.load('/media/mangaldeep/HDD2/workspace/MotionControl_MasterThesis/DL_results.npz', allow_pickle= True)

ML_data = ML_data['arr_0'][0]

ML_data['cv_time'] = [ele for ele in ML_data['cv_time'] for i in range(10)]

ML_df = pd.DataFrame(ML_data)

DL_data = DL_data['arr_0'][0]

DL_df = pd.DataFrame(DL_data)

# df = df[df['dataset']=='OCIParams']
df = pd.concat([ML_df,DL_df])
#%%
fig = px.box(df, x="preproc", y="test_acc", color="Feat+Clf", notched=False)
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()
#%%
# procs = ['ica', 'ssp', 'car', 'car_ica', 'ssp_car', 'ssp_car_ica', 'ssp_ica']

# # Accuracy, timing, confusion matrix, F1_macro, ROC_AUC

# bar_plots = [
#     go.Bar(x=x, y=df['conservative'], name='Conservative', marker=go.bar.Marker(color='#0343df')),
#     go.Bar(x=x, y=df['labour'], name='Labour', marker=go.bar.Marker(color='#e50000')),
#     go.Bar(x=x, y=df['liberal'], name='Liberal', marker=go.bar.Marker(color='#ffff14')),
#     go.Bar(x=x, y=df['others'], name='Others', marker=go.bar.Marker(color='#929591')),
# ]

# layout = go.Layout(
#     title=go.layout.Title(text="Physionet", x=0.5),
#     yaxis_title="Accuracy",
#     xaxis_tickmode="array",
#     xaxis_tickvals=list(range(27)),
#     xaxis_ticktext=tuple(procs),
# )

# fig = go.Figure(data = bar_plots, layout = layout)

