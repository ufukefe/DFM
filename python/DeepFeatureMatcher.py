#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:46:43 2021

@author: kutalmisince
"""
import torch
import cv2.cv2 as cv
import numpy as np
from torchvision import models, transforms
from collections import namedtuple
import matplotlib.pyplot as plt

class DeepFeatureMatcher():
    
    def __init__(self, model: str = 'VGG19', padding_n = 16, enable_two_stage = True, ratio_th = [0.9, 0.9, 0.9, 0.9, 0.95, 1.0]):
        super(DeepFeatureMatcher, self).__init__()
        
        if model == 'VGG19':
            print('loading VGG19...')
            self.model = Vgg19()
            print('model is loaded.')
        else:
            print('Error: model ' + model + ' is not supported!')
            return
        
        self.padding_n = padding_n
        self.enable_two_stage = enable_two_stage
        self.ratio_th = np.sqrt(np.array(ratio_th))
           
    def match(self, img_A, img_B, *args):
        
        # transform into pytroch tensor and pad image to a multiple of 16
        inp_A, padding_A = self.transform(img_A) 
        inp_B, padding_B = self.transform(img_B) 
        
        # get acitvations
        #activations_AP = self.model.forward(inp_A)
        #activations_BP = self.model.forward(inp_B)
        
        activations_A = args[0][0]
        activations_B = args[0][1]
        
        points_AM = args[0][3]
        points_BM = args[0][4]
    
        if self.enable_two_stage:
            
            # initiate matches
            points_A, points_B = dense_feature_matching(activations_A[-1], activations_B[-1], self.ratio_th[-1])
        
            # upsample points
            points_A = (points_A + 0.5) * 16 - 0.5
            points_B = (points_B + 0.5) * 16 - 0.5
        
            # estimate homography for initial warping
            src = points_B.t().numpy()
            dst = points_A.t().numpy()
            
            H_initO, _ = cv.findHomography(src, dst, method=cv.RANSAC, ransacReprojThreshold=20, maxIters=5000, confidence=0.99)
        
            H_init = np.linalg.inv(np.array([[0.9331,0.0102,152.3175],[0.2392,0.9908,-2.6547],[0.0013,-0.0000,1.0000]]))
            
            # warp image B onto image A 
            img_C = cv.warpPerspective(img_B, H_init, (img_A.shape[1],img_A.shape[0]))
            
            # debug
            points_C = torch.from_numpy(H_init) @ torch.vstack((points_B + 0.5, torch.ones((1, points_B.size(1))))).double()
            points_C = points_C[0:2, :] / points_C[2, :] - 0.5 
            
            pA = (points_AM[-1] - 0.5) * 16 - 0.5
            pB = (points_BM[-1] - 0.5) * 16 - 0.5
        
            self.plot_keypoints(img_A, points_A, 'A init', pA,)
            self.plot_keypoints(img_B, points_B, 'B init', pB)
            self.plot_keypoints(img_C, points_C, 'B warp init')
            
            # transform into pytroch tensor and pad image to a multiple of 16
            inp_C, padding_C = self.transform(img_C)
            
            # get activations of the warped image
            #activations_C = self.model.forward(inp_C)
            activations_C = activations_B
            img_C = args[0][2]
            
        else:
            
            H_init = np.eye(3, dtype=np.double)
            
            img_C = img_B
            
            activations_C = activations_B
    
        # initiate matches
        points_A, points_C = dense_feature_matching(activations_A[-2], activations_C[-2], self.ratio_th[-1])
    
        #points_A = points_A[:, 752].unsqueeze(1) 17,25 i arayacagiz, hra sonucunda 34,50 ve 35,51 donmesi lazim ama donmuyor!
        #points_C = points_C[:, 752].unsqueeze(1)
        
        pA = (points_AM[-2] - 0.5) * 16 - 0.5
        pB = (points_BM[-2] - 0.5) * 16 - 0.5
        # upsample points
        self.plot_keypoints(img_A, (points_A + 0.5) * 16 - 0.5,  'A dense', pA)
        self.plot_keypoints(img_C, (points_C + 0.5) * 16 - 0.5,  'Bw dense', pB)
            
        return
        for k in range(len(activations_A) - 3, -1, -1):
            points_A, points_C = refine_points(points_A, points_C, activations_A[k], activations_C[k], self.ratio_th[k])
            pA = (points_AM[k] - 0.5) * (2**k) - 0.5
            pB = (points_BM[k] - 0.5) * (2**k) - 0.5
            self.plot_keypoints(img_A, (points_A + 0.5) * (2**k) - 0.5, 'A level: ' + str(k), pA)
            self.plot_keypoints(img_C, (points_C + 0.5) * (2**k) - 0.5, 'Bw level: ' + str(k), pB)
        
    
        # warp points form C to B
        points_B = torch.from_numpy(np.linalg.inv(H_init)) @ torch.vstack((points_C + 0.5, torch.ones((1, points_C.size(1))))).double()
        points_B = points_B[0:2, :] / points_B[2, :] - 0.5 
    
        points_A = points_A.double()
        
        # estimate homography
        src = points_B.t().numpy()
        dst = points_A.t().numpy()
        
        H, _ = cv.findHomography(src, dst, method=cv.RANSAC, ransacReprojThreshold=2.5, maxIters=5000, confidence=0.99)
        
        # warp image B onto image A
        img_R = cv.warpPerspective(img_B, H, (img_A.shape[1],img_A.shape[0]))
    
        points_R = torch.from_numpy(H) @ torch.vstack((points_B + 0.5, torch.ones((1, points_B.size(1))))).double()
        points_R = points_R[0:2, :] / points_R[2, :] - 0.5 
        
        # display results
        self.plot_keypoints(img_A, points_A, 'A')
        self.plot_keypoints(img_B, points_B, 'B')
        self.plot_keypoints(img_C, points_C, 'B initial warp')
        self.plot_keypoints(img_R, points_R, 'B final warp')
        
        return H, points_A.numpy(), points_B.numpy()
      
    def transform(self, img):
        
        '''
        Convert given numpy image in [0,1] to tensor, perform normalization and 
        pad right/bottom to make image canvas a multiple of self.padding_n

        Parameters
        ----------
        img : numpy array, input image

        Returns
        -------
        img_T : torch.tensor
        (pad_right, pad_bottom) : int tuple 

        '''

        [103.939, 116.779, 123.68] 
        # transform to tensor and normalize
        #T = transforms.Compose([transforms.ToTensor()])
        T = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])
        
        # zero padding to make image canvas a multiple of padding_n
        pad_right = 16 - img.shape[1] % self.padding_n if img.shape[1] % self.padding_n else 0
        pad_bottom = 16 - img.shape[0] % self.padding_n if img.shape[0] % self.padding_n else 0
        
        padding = torch.nn.ZeroPad2d([0, pad_right, 0, pad_bottom])
        
        # convert image
        img_T = padding(T(img.astype(np.float32))).unsqueeze(0)
        
        return img_T, (pad_right, pad_bottom)  
    
    @classmethod  
    def plot_keypoints(cls, img, pts, title='untitled', *args):
    
        f,a = plt.subplots()
        if len(args) > 0:
            pts2 = args[0]
            a.plot(pts2[0, :], pts2[1, :], marker='o', linestyle='none', color='green')
        
        a.plot(pts[0, :], pts[1, :], marker='+', linestyle='none', color='red')
        a.imshow(img)
        a.title.set_text(title)
        plt.pause(0.001)
        #plt.show() 
          
class Vgg19(torch.nn.Module):
    
    # modified from the original @ https://github.com/chenyuntc/pytorch-book/blob/master/chapter08-neural_style/PackedVGG.py
    
    def __init__(self, required_layers = [2, 7, 12, 21, 30, 32]):
        
        # features 2，7，12，21, 30, 32: conv1_2,conv2_2,relu3_2,relu4_2,conv5_2,conv5_3
        super(Vgg19, self).__init__()
        
        features = list(models.vgg19(pretrained = True).features)[:33] # get vgg features
        
        self.features = torch.nn.ModuleList(features).eval() # construct network in eval mode
        
        for param in self.features.parameters(): # we don't need graidents, turn them of to save memory
            param.requires_grad = False
                
        self.required_layers = required_layers # record required layers to save them in forward
        
        for layer in required_layers[:-1]:
            self.features[layer+1].inplace = False # do not overwrite in any layer
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in self.required_layers:
                print('appending layer ' +str(ii))
                results.append(x)
                
        vgg_outputs = namedtuple("VggOutputs", ['conv1_2', 'conv2_2', 'relu3_2', 'relu4_2', 'conv5_2', 'conv5_3'])
        
        return vgg_outputs(*results)
    
def dense_feature_matching(map_A, map_B, ratio_th):
    
    # normalize and reshape feature maps
    _, ch, h_A, w_A = map_A.size()
    _, _,  h_B, w_B = map_B.size()
    
    d1 = map_A.view(ch, -1).t()
    d1 /= torch.sqrt(torch.sum(torch.square(d1), 1)).unsqueeze(1)
    
    d2 = map_B.view(ch, -1).t()
    d2 /= torch.sqrt(torch.sum(torch.square(d2), 1)).unsqueeze(1)
    
    # perform matching
    matches, scores = mnn_ratio_matcher(d1, d2, ratio_th)
    
    # form a coordinate grid and convert matching indexes to image coordinates
    y_A, x_A = torch.meshgrid(torch.arange(h_A), torch.arange(w_A))
    y_B, x_B = torch.meshgrid(torch.arange(h_B), torch.arange(w_B))
    
    points_A = torch.stack((x_A.flatten()[matches[:, 0]], y_A.flatten()[matches[:, 0]]))
    points_B = torch.stack((x_B.flatten()[matches[:, 1]], y_B.flatten()[matches[:, 1]]))
    
    # discard the point on image boundaries
    discard = (points_A[0, :] == 0) | (points_A[0, :] == w_A-1) | (points_A[1, :] == 0) | (points_A[1, :] == h_A-1) \
            | (points_B[0, :] == 0) | (points_B[0, :] == w_B-1) | (points_B[1, :] == 0) | (points_B[1, :] == h_B-1)
    
    discard[:] = False
    points_A = points_A[:, ~discard]
    points_B = points_B[:, ~discard]
    
    return points_A, points_B
  
def refine_points(points_A: torch.Tensor, points_B: torch.Tensor, activations_A: torch.Tensor, activations_B: torch.Tensor, ratio_th = 0.8):

    # normalize and reshape feature maps
    
    d1 = activations_A.squeeze(0) / activations_A.squeeze(0).square().sum(0).sqrt().unsqueeze(0)
    d2 = activations_B.squeeze(0) / activations_B.squeeze(0).square().sum(0).sqrt().unsqueeze(0)
        
    # get number of points
    ch = d1.size(0)
    num_input_points = points_A.size(1)
    
    # upsample points
    points_A *= 2
    points_B *= 2
       
    # neighborhood to search
    neighbors = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # allocate space for scores
    scores = torch.zeros(num_input_points, neighbors.size(0), neighbors.size(0))
    
    # for each point search the refined matches in given [finer] resolution
    for i, n_A in enumerate(neighbors):
        
        #print(str(i) + ': ', n_A)
        
        for j, n_B in enumerate(neighbors):
            
            #print('    ' + str(j) + ': ', n_B)
            
            # if j < i:
            #     scores[:, i, j] = scores[:, j, i] 
            # else:
            # get features in the given neighborhood
            act_A = d1[:, points_A[1, :] + n_A[1], points_A[0, :] + n_A[0]].view(ch, -1)
            act_B = d2[:, points_B[1, :] + n_B[1], points_B[0, :] + n_B[0]].view(ch, -1)
            
            # compute mse
            scores[:, i, j] = torch.sum(act_A * act_B, 0)
            
    # retrieve top 2 nearest neighbors from A2B
    score_A, match_A = torch.topk(scores, 2, 2)
    score_A = torch.sqrt(2 - 2 * score_A)
    
    # compute lowe's ratio
    ratio_A2B = score_A[:, :, 0] / (score_A[:, :, 1] + 1e-8)
    
    # select the best match
    match_A2B = match_A[:, :, 0]
    score_A2B = score_A[:, :, 0]
    
    # retrieve top 2 nearest neighbors from B2A
    score_B, match_B = torch.topk(scores.transpose(2,1), 2, 2)
    score_B = torch.sqrt(2 - 2 * score_B)
    
    # compute lowe's ratio
    ratio_B2A = score_B[:, :, 0] / (score_B[:, :, 1] + 1e-8)
    
    # select the best match
    match_B2A = match_B[:, :, 0]
    #score_B2A = score_B[:, :, 0]
    
    # check for unique matches and apply ratio test
    ind_A = (torch.arange(num_input_points).unsqueeze(1) * neighbors.size(0) + match_A2B).flatten()
    ind_B = (torch.arange(num_input_points).unsqueeze(1) * neighbors.size(0) + match_B2A).flatten()
    
    ind = torch.arange(num_input_points * neighbors.size(0))
    
    mask = torch.logical_and(torch.min(ratio_A2B, ratio_B2A) < ratio_th,  (ind_B[ind_A] == ind).view(num_input_points, -1))
    
    # set a large SSE score for mathces above ratio threshold and not on to one (score_A2B <=4 so use 5)
    score_A2B[~mask] = 5
    
    # each input point can generate max two output points, so discard the two with highest SSE 
    _, discard = torch.topk(score_A2B, 2, 1)
    
    mask[torch.arange(num_input_points), discard[:, 0]] = 0
    mask[torch.arange(num_input_points), discard[:, 1]] = 0
    
    # x & y coordiates of candidate match points of A
    x = points_A[0, :].repeat(4, 1).t() + neighbors[:, 0].repeat(num_input_points, 1)
    y = points_A[1, :].repeat(4, 1).t() + neighbors[: ,1].repeat(num_input_points, 1)
    
    refined_points_A = torch.stack((x[mask], y[mask]))
    
    # x & y coordiates of candidate match points of A
    x = points_B[0, :].repeat(4, 1).t() + neighbors[:, 0][match_A2B]
    y = points_B[1, :].repeat(4, 1).t() + neighbors[:, 1][match_A2B]
    
    refined_points_B = torch.stack((x[mask], y[mask]))
    
    print('#input points: ' + str(points_A.size(1)))
    print('#output points: ' + str(refined_points_A.size(1)))
    
    return refined_points_A, refined_points_B
    
def mnn_ratio_matcher(descriptors1, descriptors2, ratio=0.8):
    
    # Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]
    match_sim = nns_sim[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))
    
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)
    match_sim = match_sim[mask]

    return (matches.data.cpu().numpy(),match_sim.data.cpu().numpy())

'''
from PIL import Image
import scipy.io as spio

img_A = Image.open('1.ppm')
inp_A = np.asarray(img_A.getdata()).reshape(img_A.size[1], img_A.size[0], -1).astype(np.float32) / 255.0

img_B = Image.open('4.ppm')
inp_B = np.asarray(img_B.getdata()).reshape(img_B.size[1], img_B.size[0], -1).astype(np.float32) / 255.0
        
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

fm = DeepFeatureMatcher()#.to(device)

H, points_A, points_B = fm.match(inp_A, inp_B)
'''


