Executing prep2.sh runs the image preprocessing, inference in the reinforcement learning algorithm, and ostium detection postprocessing (conversion to world coordinates)

Input: CT image. Saved in /pre_segmentation_test, --landmark_GT (if you have the GT of ostium detection, not needed), --RAS2LPS (1 if RAS CT-image orientation)

Outputs:
- results_job.csv with voxel prediction (saved in /output_case_101)
- pred_plot.png is a plot of the prediction
- ostium_pred.xlsx with world coordinate prediction (saved in /output_case_101/post)
