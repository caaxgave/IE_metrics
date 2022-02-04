# This is a sample Python script.
import argparse
import cv2
from skimage.metrics import mean_squared_error
import scipy.stats as measures
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
parser.add_argument("--test_dir", type=str, default="LOL_results/",
                    help="path to folder with folders with resulting "
                         "datasets or LOLDataset")
parser.add_argument("--output_metrics", type=str, default="metric_results", help="json and resulting plots")
opt = parser.parse_args()
print(opt)

# Assign directories to variables and create directories
gt_dir = opt.gt_dir
test_dir = opt.test_dir

os.makedirs("%s" % opt.output_metrics, exist_ok=True)

# Creating a dictionary for each model name and path which contain image enhancement results.
test_models = {}
for dir in sorted(os.listdir(test_dir)):
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
      MSE.append(mean_squared_error(gt_im, test_im))
      CC.append(round(measures.pearsonr(tf.image.convert_image_dtype(gt_im, tf.float32).numpy().flatten(),
                                        tf.image.convert_image_dtype(test_im, tf.float32).numpy().flatten())[0], 4))
      PSNR.append(round(tf.image.psnr(gt_im, test_im, max_val=255).numpy(), 4))
      SSIM.append(round(tf.image.ssim(gt_im, test_im, max_val=255).numpy(), 4))

    return tf.linalg.normalize(MSE)[0].numpy().tolist(), CC, PSNR, SSIM


def metrics_df(gt_dir, test_models):
    global df
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


def metric_visualization(df, plot_type=None):
    i, j = 0, 0
    metric_list = ['MSE', 'CC', 'PSNR', 'SSIM']
    metric_model = []
    rows = round(len(metric_list) / 2)

    if plot_type=='linear':
        fig, axes = plt.subplots(rows, 2, figsize=(15, 10), sharey=False, sharex=True)
        for metric in metric_list:
            if j % 2 == 0:
                metric_model = ['{}_{}'.format(metric, model) for model in test_models]
                sns.lineplot(data=df.loc[:, metric_model],
                             color="g", ax=axes[int(i), 0])
                axes[int(i), 0].set_title('%s' % metric, fontsize=12)
                axes[int(i), 0].legend(fontsize=8.5)
                if metric == 'PSNR':
                    axes[int(i), 0].set(ylim=(0, 30))
                else:
                    axes[int(i), 0].set(ylim=(0, 1))
            else:
                metric_model = ['{}_{}'.format(metric, model) for model in test_models]
                sns.lineplot(data=df.loc[:, metric_model],
                             color="g", ax=axes[int(i), 1])
                axes[int(i), 1].set_title('%s' % metric, fontsize=12)
                axes[int(i), 1].legend(fontsize=8.5)
                if metric == 'PSNR':
                    axes[int(i), 1].set(ylim=(0, 30))
                else:
                    axes[int(i), 1].set(ylim=(0, 1))
            j += 1
            i += 0.5

        # Saving first plot (linear plot)
        plt.tight_layout()
        plt.savefig(opt.output_metrics + '/' + 'linear_plot.png')

    else:
        fig, axes = plt.subplots(rows, 2, figsize=(15, 10), sharey=False, sharex=False)
        for metric in metric_list:
            if j % 2 == 0:
                metric_model = ['{}_{}'.format(metric, model) for model in test_models]
                sns.boxplot(data=df.loc[:, metric_model], palette="Blues",
                            ax=axes[int(i), 0], boxprops=dict(alpha=0.2),
                            medianprops=dict(color="blue", alpha=0.8))
                sns.stripplot(data=df.loc[:, metric_model], jitter=0.3, color='k', alpha=0.6,
                              ax=axes[int(i), 0], size=4)
                axes[int(i), 0].set_title('%s' % metric, fontsize=12)
                if metric == 'PSNR':
                    axes[int(i), 0].set(ylim=(0, 30))
                else:
                    axes[int(i), 0].set(ylim=(0, 1))
            else:
                metric_model = ['{}_{}'.format(metric, model) for model in test_models]
                sns.boxplot(data=df.loc[:, metric_model], palette="Blues",
                            ax=axes[int(i), 1], boxprops=dict(alpha=0.2),
                            medianprops=dict(color="blue", alpha=0.8))
                sns.stripplot(data=df.loc[:, metric_model], jitter=0.3, color='k', alpha=0.6,
                              ax=axes[int(i), 1], size=4)
                axes[int(i), 1].set_title('%s' % metric, fontsize=12)
                if metric == 'PSNR':
                    axes[int(i), 1].set(ylim=(0, 30))
                else:
                    axes[int(i), 1].set(ylim=(0, 1))
            j += 1
            i += 0.5

        # Saving first plot (box-dots plot)
        plt.tight_layout()
        plt.savefig(opt.output_metrics + '/' + 'boxdots_plot.png')


def quality_cases(df):

    SSIM_list = [column for column in df.columns if column.startswith("SSIM")]
    SSIM_df = df.loc[:, SSIM_list]
    max_SSIM = [max(list(df.loc[:, SSIM_col])) for SSIM_col in df.loc[:, SSIM_list]]
    max_indx = list(map(lambda x, y: list(SSIM_df.iloc[:, y]).index(x), max_SSIM, range(0,len(SSIM_list))))
    min_SSIM = [min(list(df.loc[:, SSIM_col])) for SSIM_col in df.loc[:, SSIM_list]]
    min_indx = list(map(lambda x, y: list(SSIM_df.iloc[:, y]).index(x), min_SSIM, range(0,len(SSIM_list))))
    #mean_SSIM = [np.mean(list(df.loc[:, SSIM_col])) for SSIM_col in df.loc[:, SSIM_list]]

    #for choosing mean quality images:
    low_list, high_list = [], []
    for j in range(len(SSIM_df.columns)):
        SSIM_sorted = sorted(list(SSIM_df.iloc[:, j]))
        indx_low = SSIM_sorted[int(len(SSIM_sorted)/2) - 1]
        indx_high = SSIM_sorted[round(len(SSIM_sorted)/2 + 1) - 1]
        low_list.append(indx_low)
        high_list.append(indx_high)


    lowmean_indx = list(map(lambda x, y: list(SSIM_df.iloc[:, y]).index(x), low_list, range(0,len(SSIM_list))))
    highmean_indx = list(map(lambda x, y: list(SSIM_df.iloc[:, y]).index(x), high_list, range(0,len(SSIM_list))))

    return max_indx, min_indx, lowmean_indx, highmean_indx


def image_compare(quality_cases):
    indexes = list(quality_cases)
    i = 0
    images_list = []
    for model in test_models:
        im_sorted = sorted(os.listdir(test_models[model]))
        max_SSIM_im = im_sorted[indexes[0][i]]
        min_SSIM_im = im_sorted[indexes[1][i]]
        lowmean_im = im_sorted[indexes[2][i]]
        highmean_im = im_sorted[indexes[3][i]]

        i += 1
        images_list.append([max_SSIM_im, highmean_im, lowmean_im, min_SSIM_im])

    ver = []
    image = 0
    for model in test_models:
        path = test_models[model]
        # read first max
        grid_row = [cv2.imread(os.path.join(path, images_list[image][j])) for j in range(4)]
        ver.append(np.vstack(tuple(grid_row)))
        image += 1

    hor = np.hstack((ver[0], ver[1], ver[2], ver[3], ver[4], ver[5]))

    #cv2.imshow(hor)
    cv2.imwrite(opt.output_metrics + '/' + "Image_comparison.png", hor)


# Running program.
if __name__ == '__main__':
    metrics_df(gt_dir, test_models)
    metric_visualization(df, plot_type="linear")
    metric_visualization(df)
    image_compare(quality_cases(df))