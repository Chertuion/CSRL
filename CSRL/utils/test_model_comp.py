import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec



# Raw data
data_raw = [
    ['DIR', '57.8488±2.2244', '5.9096±4.5915', '31.8553±1.1245'],
    ['CIGA',  '58.0348±4.7996', '3.6505±7.0996', '31.4463±0.7233'],
    ['MolOOD',  '63.7375±3.3631', '8.71±11.454', '33.4075±2.9152'],
    ['CaiSi',  '67.8201±1.8487', '12.7496±8.8180', '34.9785±2.5691'],
    ['ERM', '60.6321±2.8074','5.5419±5.3109','31.5502±0.7590'],
    ['IRM', '58.0071±4.9469','5.6587±5.2684','31.5397±0.7699'],
    ['EIIL', '59.0861±4.2738','3.1901±4.0395','31.2167±0.5399'],
    
    ['DIR',  '59.1659±2.9646', '1.3885±3.3037', '22.1126±0.3075'],
    ['CIGA',  '60.0574±2.2668', '10.2931±10.5505', '23.9885±2.9484'],
    ['MolOOD',  '57.4225±2.6211', '7.17±9.0522', '22.9425±1.0769'],
    ['CaiSi',  '61.3305±1.7848', '20.4286±1.5309', '26.2997±0.5789'],
    ['ERM', '56.8460±1.1703','6.2802±7.5716','23.1385±1.5469'],
    ['IRM', '59.8975±1.4772','8.9550±10.6370','23.6790±2.1229'],
    ['EIIL', '59.2621±1.4127','7.7740±14.9408','24.2402±4.3792'],
    
    ['DIR',  '54.3040±0.9473', '4.0763±1.8272', '82.4663±0.3990'],
    ['CIGA', '53.4068±1.2358', '5.15188±4.1609', '82.6317±0.6020'],
    ['MolOOD', '50.625±4.6345', '2.99±1.8589', '82.36±0.3001'],
    ['CaiSi', '54.5810±0.8996', '4.7721±2.2695', '82.5987±0.3602'],
     ['ERM', '53.2990±0.1471','1.9575±0.7636','82.0967±0.1275'],
    ['IRM', '53.4612±1.1598','4.1669±1.9918','82.5158±0.3475'],
    ['EIIL', '53.4929±1.2763','2.7157±2.6993','82.3090±0.4782'],
    
    ['DIR',  '75.0275±1.1527', '22.8923±17.4263', '20.5906±7.1391'],
    ['CIGA', '71.3725±2.5800', '15.7386±27.3698', '20.3651±15.3095'],
    ['MolOOD', '73.4±3.1978', '30.545±20.6244', '25.4025±8.9199'],
    ['CaiSi', '76.2735±3.1370', '40.7714±5.8094', '29.6953±4.0910'],
    ['ERM', '70.9770±2.5768','3.3632±2.3513','12.7831±0.1888'],
    ['IRM', '71.8695±4.2190','14.8486±24.2400','31.5397±0.7699'],
    ['EIIL', '71.5565±5.0409','1.6728±5.3167','31.2167±0.5399'],
    
    ['DIR', '60.3508±3.4195', '11.9061±3.5892', '14.4009±1.0264'],
    ['CIGA', '59.1512±3.1836', '8.0314±5.6937', '13.1608±0.79'],
    ['MolOOD',  '56.5675±5.5512', '11.329±4.7907', '13.3075±1.0628'],
    ['CaiSi', '63.0390±2.3842', '15.2600±11.1878', '15.9918±3.5401'],
    ['ERM', '53.6490±1.9969','7.7801±3.2887','13.3954±0.6337'],
    ['IRM', '56.9921±4.1985','10.8591±10.1949','14.4578±1.6945'],
    ['EIIL', '57.0632±3.0717','6.8181±6.6666','13.2768±1.5362'],
    
    ['DIR', '58.1530±1.6697', '3.5640±1.5175', '97.1115±0.2370'],
    ['CIGA',  '57.8865±3.2713', '5.1917±1.2747', '97.2034±0.1653'],
    ['MolOOD',  '60.0±0.48', '4.26±1.07', '97.14±0.07'],
    ['CaiSi',  '63.5165±1.8709', '3.8964±1.9160', '97.0074±0.0748'],
    ['ERM', '60.9625±0.2609','5.5419±5.3109','97.2001±0.0888'],
    ['IRM', '58.8145±3.3094','4.6681±1.5920','97.1883±0.057'],
    ['EIIL', '58.3087±3.4399','5.5883±2.2104','97.1771±10.1236']
]


# Data size
datasets = 6
models = 7
metrics = 3

# Provide some labels for the models and evaluation criteria
model_names = ['DIR', 'CIGA', 'MolOOD', 'CHiMoGNN', 'ERM', 'IRM', 'EIIL']
metric_names = ['ROC-AUC', 'MCC', 'PR-AUC']
dataset_names = ['EC50-assay', 'KI-assay', 'potency-assay', 'EC50-scaffold', 'KI-scaffold', 'potency-scaffold']

# Initialize 3D arrays
data = np.zeros((datasets, models, metrics))
errors = np.zeros((datasets, models, metrics))

# Parse raw data and fill 3D arrays
# NOTE: You'll need to adjust the data_raw structure to be more Pythonic before this will work
# Parsing raw data to fill 3D arrays
for d in range(datasets):
    for m in range(models):
        for met in range(metrics):
            current_data = data_raw[(d*models) + m][met+1]
            value, error = current_data.split('±')
            data[d, m, met] = float(value)
            errors[d, m, met] = float(error)

# Colors
cr_sets = [
    [51/255, 60/255, 66/255],
    [49/255, 102/255, 88/255],
    [94/255, 166/255, 156/255],
    [102/255, 29/255, 23/255],
    [194/255, 207/255, 162/255],
    [164/255, 121/255, 158/255],
    [112/255, 102/255, 144/255]
]

# # Colors
# cr_sets = [
#     [10/255, 46/255, 87/255],
#     [83/255, 51/255, 140/255],
#     [174/255, 78/255, 137/255],
#     [59/255, 144/255, 217/255],
#     [234/255, 141/255, 158/255],
#     [236/255, 204/255, 236/255],
#     [230/255, 185/255, 117/255]
# ]

# cr_sets = [
#     [216/255, 217/255, 122/255],
#     [149/255, 195/255, 110/255],
#     [116/255, 200/255, 195/255],
#     [102/255, 29/255, 23/255],
#     [90/255, 151/255, 192/255],
#     [41/255, 83/255, 132/255],
#     [10/255, 46/255, 87/255]
# ]

cr_sets = [
    [63/255, 45/255, 107/255],
    [166/255, 32/255, 46/255],
    [174/255, 78/255, 137/255],
    [59/255, 144/255, 217/255],
    [234/255, 141/255, 158/255],
    [236/255, 204/255, 236/255],
    [230/255, 185/255, 117/255]
]

# cr_sets = [
#     [216/255, 217/255, 122/255],
#     [116/255, 200/255, 195/255],
#     [149/255, 195/255, 110/255],
#     [208/255, 188/255, 155/255],
#     [90/255, 151/255, 193/255],
#     [41/255, 83/255, 132/255],
#     [10/255, 46/255, 87/255]
# ]


208, 188, 155
102, 29, 23
bar_width = 0.35  # 条形的宽度
gap = 0.05  # 相同数据集中条形之间的间距
big_gap = 0.5  # 不同数据集之间的大间隔
# 定义截断范围
ylims_broken = [[(0, 40), (55, 80)],
                [(0, 30), (50, 65)],
                [(0, 20),(40, 60) ,(80, 90)],
                [(0, 60), (70, 100)],
                [(0, 25), (50, 70)],
                [(0,15), (55,65),(95, 100)],
                ]

ylims_sets = [[0, 80],
              [0, 65],
              [0, 90],
              [0, 90],
              [0, 70],
              [0, 100],
                ]

fig = plt.figure(figsize=(12.5, 7))
# 使用 GridSpec 定义子图布局
gs = GridSpec(2, 3)

label_font = {
    'family': 'Times New Roman',    # 例如 'serif', 'sans-serif', 'monospace'
    'color':  'black',
    'weight': 'bold',     # 字体加粗
    'size': 10,           # 字号大小
}

for i in range(2):
    for j in range(3):
        data_2d = data[i, :, :]
        errors_2d = errors[i, :, :]

        n_bars = len(model_names)
        bar_width = 0.15
        gap = 0.02  # 每个模型之间的间距
        big_gap = 0.2  # 在不同的数据集之间的大间隔
        total_width = (bar_width * n_bars) + (gap * (n_bars - 1)) + big_gap
        indices = np.arange(len(metric_names)) * total_width
        error_attributes = {'elinewidth': 1.5}
        bax = brokenaxes(ylims=ylims_broken[i * 3 + j], subplot_spec=gs[i * 3 + j],hspace=.16,  despine=False, d=0)
        data_2d = data[i * 3 + j, :, :]
        errors_2d = errors[i * 3 + j, :, :]
        for k in range(n_bars):
            # 更新误差线的颜色
            error_attributes['ecolor'] = cr_sets[k]
            bax.bar(indices + k * (bar_width + gap), data_2d[k, :], yerr=errors_2d[k, :], width=bar_width, capsize=2,
                    error_kw=error_attributes, align='center', label=model_names[k], color=cr_sets[k])
            bax.set_title(dataset_names[i * 3 + j], **label_font)
            bax.set_ylabel('Performance(%)', **label_font)
            bax.axs[0].set_xlim(indices[0] - bar_width, indices[-1] + total_width)
            bax.axs[1].set_xlim(indices[0] - bar_width, indices[-1] + total_width) 

        # 设置x轴的ticks和labels
        bax.axs[-1].set_xticks(indices + (n_bars - 1) * 0.5 * (bar_width + gap))
        bax.axs[-1].set_xticklabels(metric_names)
        first_element = ylims_broken[i * 3 + j]
        if len(bax.axs) == 2:
            start = first_element[0][1]
            end = first_element[1][0]
            bax.axs[1].set_ylim(ylims_sets[i * 3 + j][0], start)  # 第一部分
            bax.axs[0].set_ylim(end, ylims_sets[i * 3 + j][1])  # 第二部分

        else:
            bax.axs[2].set_ylim(ylims_sets[i * 3 + j][0], first_element[0][1])  # 第一部分
            bax.axs[1].set_ylim(first_element[1][0], first_element[1][1])  # 第二部分
            bax.axs[0].set_ylim(first_element[2][0], ylims_sets[i * 3 + j][1])  # 第三部分
            # 重新定位和调整条形图的位置
            for k in range(n_bars):
                bax.axs[0].set_xlim(indices[0] - bar_width, indices[-1] + total_width)
                bax.axs[1].set_xlim(indices[0] - bar_width, indices[-1] + total_width)
                bax.axs[2].set_xlim(indices[0] - bar_width, indices[-1] + total_width)
        



handles, labels = bax.axs[0].get_legend_handles_labels()
# 创建新的轴，用于放置图例
# legend_ax = fig.add_axes([0.15, 0.9, 0.7, 0.1])
legend_ax = fig.add_axes([0.15, 0.005, 0.7, 0.1])
legend_ax.axis('off')
# 在新的轴上放置图例
legend = legend_ax.legend(handles, labels, loc='center', ncol=len(model_names), borderaxespad=0.,fontsize='large', frameon=False)

plt.savefig("/data_1/wxl22/MIOOD/utils/difference_metrics.png", dpi=900)
# plt.savefig("/data_1/wxl22/drawer/exFigure_comp_other_color_4.png", format="png")
plt.show()

