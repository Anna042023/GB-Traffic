import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from GAF_op import build_image

# ======================
# 粒球节点
# ======================
class BallNode:
    def __init__(self, pixels):
        self.pixels = pixels
        self.centroid = np.mean(pixels, axis=0)
        self.feature = None

# ======================
# 粒球图生成器
# ======================
class GranularBallGraph:
    def __init__(self, P_thr=0.85, beta=0.2, theta_sim=0.8, alpha=1.5, eps=1e-6):
        self.P_thr = P_thr
        self.beta = beta
        self.theta_sim = theta_sim
        self.alpha = alpha
        self.eps = eps

    def init_balls(self, img):
        H, W, C = img.shape
        balls = []
        for i, j in product(range(H), range(W)):
            balls.append(BallNode(pixels=[np.array([i,j])]))
        return balls

    def joint_purity(self, ball1, ball2, img):
        union_pixels = np.vstack([ball1.pixels, ball2.pixels])
        C = img.shape[2]
        purities = []
        for c in range(C):
            vals = np.array([img[p[0],p[1],c] for p in union_pixels])
            mu = vals.mean()
            sigma = vals.std()
            purities.append(np.exp(- sigma**2 / (mu**2 + self.eps)))
        return np.mean(purities)

    def merge_balls(self, balls, img, tau_adapt):
        merged = True
        while merged:
            merged = False
            np.random.shuffle(balls)
            for i, ball in enumerate(balls):
                neighbors = [b for j,b in enumerate(balls) if j!=i and np.any(np.linalg.norm(np.array(ball.pixels)[:,None]-np.array(b.pixels)[None,:], axis=2)==1)]
                for n in neighbors:
                    purity = self.joint_purity(ball, n, img)
                    if purity >= self.P_thr * tau_adapt:
                        ball.pixels = np.vstack([ball.pixels, n.pixels])
                        balls.remove(n)
                        merged = True
                        break
                if merged:
                    break
        return balls

    def compute_node_features(self, balls, img):
        for ball in balls:
            pixels = np.array(ball.pixels)
            features = []
            for c in range(img.shape[2]):
                vals = np.array([img[p[0],p[1],c] for p in pixels])
                features.extend([vals.mean(), vals.std()])
            ball.feature = np.array(features)
        return balls

    def build_edges(self, balls):
        N = len(balls)
        edge_index = []
        features = np.array([b.feature for b in balls])
        cos_sim = cosine_similarity(features)
        for i in range(N):
            for j in range(i+1, N):
                spatial_adj = np.min(np.linalg.norm(np.array(balls[i].pixels)[:,None]-np.array(balls[j].pixels)[None,:], axis=2)) <= 1
                if spatial_adj or cos_sim[i,j] > self.theta_sim:
                    edge_index.append([i,j])
                    edge_index.append([j,i])
        return np.array(edge_index).T

    # ✅ 添加 __call__ 方法
    def __call__(self, img, sigma_s2=0.01):
        tau_adapt = 1 + self.beta * np.log(1 + sigma_s2)
        balls = self.init_balls(img)
        balls = self.merge_balls(balls, img, tau_adapt)
        balls = self.compute_node_features(balls, img)
        edges = self.build_edges(balls)
        return balls, edges

# ======================
# 可视化六张图
# ======================
def visualize_six_graphs(seq, img, physical_edges, gb_graphs, save_prefix="example"):
    fontsize = 20
    fig_size = (6,6)

    # Raw TS
    plt.figure(figsize=fig_size)
    plt.plot(seq, color='purple')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout(pad=0)
    plt.savefig(f"{save_prefix}_raw_ts.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Multi-scale Image
    plt.figure(figsize=fig_size)
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout(pad=0)
    plt.savefig(f"{save_prefix}_multi_scale.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Physical Graph
    plt.figure(figsize=fig_size)
    nodes = set(physical_edges.flatten())
    node_coords = {n:(np.random.rand(), np.random.rand()) for n in nodes}
    for i,j in physical_edges:
        xi, yi = node_coords[i]
        xj, yj = node_coords[j]
        plt.plot([xi,xj],[yi,yj],color='gray',alpha=0.5)
    for n,(x,y) in node_coords.items():
        plt.scatter(x,y,c='blue',s=50)
        plt.text(x+0.01,y+0.01,str(n),fontsize=fontsize-2)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout(pad=0)
    plt.savefig(f"{save_prefix}_physical.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 粒球图三粒度
    gran_labels = ['coarse','medium','fine']
    for (balls,edges), label in zip(gb_graphs, gran_labels):
        plt.figure(figsize=fig_size)
        coords = np.array([b.centroid for b in balls])
        for i,j in edges.T:
            plt.plot([coords[i,1], coords[j,1]], [coords[i,0], coords[j,0]], color='gray', alpha=0.5)
        plt.scatter(coords[:,1], coords[:,0], c='red', s=10)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.tight_layout(pad=0)
        plt.savefig(f"{save_prefix}_gb_{label}.png", dpi=300, bbox_inches='tight')
        plt.close()