# IE_metrics

## Running demo

When running the script as demo, it will take 15 normal-light images from LOLDataset ("_LOL_gt_") as first-images dataset or as groung truth dataset. Then, as a second-images dataset ("_LOL_results_") it will take previously enhanced images from same 15 low-light LOLDataset (with 6 different models: RLBHE, FHSABP, LIME, DUAL, RetinexNet and F-RetinexNet). \
The metrics to compare the quality of enhanced images against ground truth, we implement Mean Square Error (MSE), Peak Signal to Noise Ratio (PSNR) and SSIM (Structural Similarity Index Measure). \
As result of runinning *IE_metrics_main.py* with no arguments, you will get a folder called "metric_results" with a _.json_ and a _.csv_ files with a data frame containing all the metrics for each image and each model. Further, you will get a linear-plot for each model and a single box-dots-plot.

By running in a terminal first clone whole repository:

*git clone https://github.com/caaxgave/IE_metrics*

Then run requirements:

*pip install -r requirements.txt*

Then you can run the script:

*python IE_metrics_main.py*

If you get an error regarding _opencv-python_ or _cv2_, you should run the script in a virtual enviroment. For this, you must first create the virtual enviroment as follows:

*python3 -m venv DIR_NAME*

Now, you should be able to run the script in the virtual enviroment:

*python3 IE_metrics_main.py*


## Running your own dataset

For running metrics with your own dataset it is nessesary to specify --gt_dir and --test_dir arguments, with a string-type input for each. 

* gt_dir: path to dataset 1 or ground truth dataset 
* test_dir: path to folder with folders with resulting datasets.

Further, you can also add path where resulting files will be located, by using _--output_metrics_ argument.

Example:

*python IE_metrics_main.py --gt_dir user/desktop/my_gt_dataset --test_dir user/desktop/my_test_dataset --output_metrics user/desktop/my_results*


Please, bear in mind that the names, you use for folders with images as _test_ dataset, will be printed in plots, so keep a short name or an abbreviation for each. Also, every pair of images, i.e, image *dog.png* in ground truth dataset, must have the same name in test folders:

*gt_dir/dog.png >>> test_dir/DUAL/dog.png*
