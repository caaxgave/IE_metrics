# This is a sample Python script.
import argparse
import cv2
from skimage.metrics import mean_squared_error
import tensorflow as tf
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Adding arguments:
parser = argparse.ArgumentParser()
parser.add_argument("--gt_dir", type=str, default="LOL_gt/", help="path to dataset 1 or ground truth dataset")
parser.add_argument("--test_dir", type=str, default="LOL_results/", help="path to folder with folders with resulting "
                                                                         "datasets or LOLDataset")
parser.add_argument("--output_metrics", type=str, default="metric_results", help="json and resulting plots")
opt = parser.parse_args()
print(opt)

# Assign directories to variables and create directories
gt_dir = opt.gt_dir
test_dir = opt.test_dir
#try:
#    os.makedirs("%s" % opt.output_metrics, exist_ok=False)
#    print("Directory '%s' created successfully" % opt.output_metrics)
#except OSError as error:
#    print("Directory '%s' already exists" % opt.output_metrics)
os.makedirs("%s" % opt.output_metrics, exist_ok=True)

# Creating a dictionary for each model name and path which contain image enhancement results.
test_models = {}
for dir in os.listdir(test_dir):
  test_models[dir] = os.path.join(test_dir, dir)


def metrics(gt_dir, test_dir):

    gt_list = sorted(os.listdir(gt_dir))
    test_list = sorted(os.listdir(test_dir))  # In case names are equal

    # Particular case where our synthetic data has 'D_' or 'L_' before name
    # test_list = list(element[2:] for element in os.listdir(tests_dir))

    MSE, PSNR, SSIM, CC = [], [], [], []
    for item in range(len(gt_list)):
      gt_im = cv2.imread(gt_dir + '/' + gt_list[item])
      test_im = cv2.imread(test_dir + '/' + test_list[item])
      MSE.append(round(mean_squared_error(gt_im, test_im), 4))
      CC.append(round(measures.pearsonr(tf.image.convert_image_dtype(gt_im, tf.float32).numpy().flatten(), 
                                tf.image.convert_image_dtype(test_im, tf.float32).numpy().flatten())[0],4))
      PSNR.append(round(tf.image.psnr(gt_im, test_im, max_val=255).numpy(), 4))
      SSIM.append(tf.image.ssim(gt_im, test_im, max_val=1.0).numpy())

    return tf.linalg.normalize(MSE)[0].numpy().tolist(), CC, PSNR, SSIM


def metrics_df(gt_dir, test_models):
    i = 0
    for model in test_models:
        if i == 0:
            # create the dataframe
            metric_values = list(metrics(gt_dir, test_models[model]))
            df = pd.DataFrame(np.array(metric_values).T, 
                              columns=['MSE_{}'.format(model), 'CC_{}'.format(model), 
                                       'PSNR_{}'.format(model), 'SSIM_{}'.format(model)])
            i += 1
        else:
            metric_values = list(metrics(gt_dir, test_models[model]))
            df['MSE_{}'.format(model)] = np.array(metric_values[0]).T
            df['CC_{}'.format(model)] = np.array(metric_values[1]).T
            df['PSNR_{}'.format(model)] = np.array(metric_values[2]).T
            df['SSIM_{}'.format(model)] = np.array(metric_values[3]).T

    df.to_json(opt.output_metrics + '/' + 'metrics.json')
    df.to_csv(opt.output_metrics + '/' + 'metrics.csv')

    return df


def metric_visualization(df):
    rows = round(len(test_models)/2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 10), sharey=True, sharex=True)
    # plt.figure(figsize=(8, 6))
    # plt.title('RetinexNet')
    i,j=0,0
    for model in test_models:
        if j%2 == 0:
            sns.lineplot(data=df.loc[:, ['MSE_{}'.format(model), 'PSNR_{}'.format(model), 'SSIM_{}'.format(model)]],
                         color="g", ax=axes[int(i), 0])
            axes[int(i), 0].set_title('%s' % model)
        else:
            sns.lineplot(data=df.loc[:, ['MSE_{}'.format(model), 'PSNR_{}'.format(model), 'SSIM_{}'.format(model)]],
                         color="g", ax=axes[int(i), 1])
            axes[int(i), 1].set_title('%s' % model)
        j+=1
        i+=0.5

    # Saving first plot (linear plot)
    plt.savefig(opt.output_metrics + '/' + 'linear_plot.png')

    sns.set(font_scale=0.7)
    sns.catplot(kind="box", data=df, height=4, aspect=2.5, palette='RdYlBu')
    sns.stripplot(data=df, jitter=0.25, color='k', alpha=0.6)
    plt.xticks(rotation=55)

    # Saving first plot (box-dots plot)
    plt.tight_layout()
    plt.savefig(opt.output_metrics + '/' + 'boxdots_plot.png')


# Running program.
if __name__ == '__main__':
    metric_visualization(metrics_df(gt_dir, test_models))
