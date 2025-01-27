

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import ot
import numpy as np
from scipy.stats import wasserstein_distance



def draw_tsne(train_features,valid_features,ty,vy, name, num_space, dim=3, type='final'):
    if dim != 2:
        # # 创建 t-SNE 实例
        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1200)
        # # 进行降维
        # train_results = tsne.fit_transform(tfps.detach().cpu())
        # valid_results = tsne.fit_transform(vfps.detach().cpu())
        train_results = tsne.fit_transform(train_features.cpu())
        valid_results = tsne.fit_transform(valid_features.cpu())
    else:
        train_results = train_features.cpu()
        valid_results = valid_features.cpu()



    train_colors = [[59/255, 144/255, 217/255],[230/255, 185/255, 117/255]]
    valid_colors = [[174/255, 78/255, 137/255],[63/255, 45/255, 107/255]]
    train_l = ['y = 0 (train)', 'y = 1 (train)']
    valid_l = ['y = 0 (valid)', 'y = 1 (valid)']

    # 绘制训练集点
    for label, color, idx in zip([0, 1], train_colors, train_l):
        indices = [i for i, l in enumerate(ty) if l == label]
        plt.scatter(train_results[indices, 0], train_results[indices, 1], color=color, label=idx, s=2)

    # 绘制验证集点
    for label, color, idx in zip([0, 1], valid_colors, valid_l):
        indices = [i for i, l in enumerate(vy) if l == label]
        plt.scatter(valid_results[indices, 0], valid_results[indices, 1], color=color, label=idx, s=2)

    # # 合并训练集和验证集的结果和标签
    # all_results = np.vstack((train_results, valid_results))
    # all_labels = np.hstack((ty, vy))  # 假设ty是训练集的标签，vy是验证集的标签

    # # 提取每个类别的坐标
    # coords_0 = all_results[all_labels == 0]
    # coords_1 = all_results[all_labels == 1]

    train_0 = train_results[ty == 0]
    train_1 = train_results[ty == 1]
    valid_0 = valid_results[vy == 0]
    valid_1 = valid_results[vy == 1]

    # 计算Wasserstein distance
    distance_x = wasserstein_distance(train_0[:, 0], valid_0[:, 0])
    distance_y = wasserstein_distance(train_1[:, 1], valid_1[:, 1])

    # 计算每个分布中样本的均匀权重
    n_train = train_results.shape[0]
    n_valid = valid_results.shape[0]
    train_weights = np.ones((n_train,)) / n_train
    valid_weights = np.ones((n_valid,)) / n_valid



    # 计算平均距离
    # average_distance = (distance_x + distance_y) / 2
    # 添加图例
    plt.legend()
    # 可选：添加标题和轴标签
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    # 添加文本显示Wasserstein distance
    plt.text(0, 1, f'Wasserstein distance: D(Y=0):{distance_x:.4f}, D(Y=1):{distance_y:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    if type == 'ex':
        plt.savefig('/data_1/wxl22/work2/utils/'+ name + str(num_space) + 'tsne_fusion_ex.png', dpi=900)
    else:
        plt.savefig('/data_1/wxl22/work2/utils/'+ name + str(num_space) + 'tsne_2d_fusion_ex.png', dpi=900)
    # plt.show()


def draw_tsne_3d(train_features, valid_features, ty, vy, name, num_space, labels=["Train", "Valid"], label_colors=[[(59/255, 144/255, 217/255), (230/255, 185/255, 117/255)], [(174/255, 78/255, 137/255), (63/255, 45/255, 107/255)]]):
    # 创建 t-SNE 实例，改为3维
    tsne = TSNE(n_components=3, verbose=1, perplexity=35, n_iter=750)
    # 进行降维
    train_results = tsne.fit_transform(train_features.cpu())
    valid_results = tsne.fit_transform(valid_features.cpu())

    # 创建3D图形
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    train_colors = [[59/255, 144/255, 217/255],[230/255, 185/255, 117/255]]
    valid_colors = [[174/255, 78/255, 137/255],[63/255, 45/255, 107/255]]
    train_l = ['y = 0 (train)', 'y = 1 (train)']
    valid_l = ['y = 0 (valid)', 'y = 1 (valid)']

    # 绘制训练集点
    for label, color, idx in zip([0, 1], train_colors, train_l):
        indices = [i for i, l in enumerate(ty) if l == label]
        plt.scatter(train_results[indices, 0], train_results[indices, 1], color=color, label=idx, s=2)

    # 绘制验证集点
    for label, color, idx in zip([0, 1], valid_colors, valid_l):
        indices = [i for i, l in enumerate(vy) if l == label]
        plt.scatter(valid_results[indices, 0], valid_results[indices, 1], color=color, label=idx, s=2)

    # 添加图例和标签
    ax.legend()
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title(f't-SNE 3D Visualization - {name}')

    # 设置视角
    ax.view_init(elev=0, azim=90)  # 修改视角为更具体的设置

    # 保存图形
    plt.savefig(f'/data_1/wxl22/work2/utils/{name}{num_space}_tsne_3d.png', dpi=300)  # 修改保存路径和文件名格式
    plt.show()
