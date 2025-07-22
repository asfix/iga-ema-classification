# Deep Learning for Celiac Diagnosis: Recognizing IgA-Class EMA Patterns on Monkey Liver Substrate Through EfficientNet Architectures

This study comprehensively evaluates the performance of the EfficientNet and EfficientNetV2 architectures in binary, three-class and four-class classification
scenarios using immunofluorescence images. Our experiments on 254 clinical samples show exceptional performance, with EfficientNetV2-S achieving 99.37% accuracy in binary classification (positive/negative), 95.28% in three-class classification (negative, weak-positive, strong-positive), and 86.98% in the challenging four-class scenario involving ambiguous grey zone samples.

# Code Usage

This code is ready to be used without any command line parameters. Just change wandb_project_name for "wandb" project, data_directory for the 5-fold cross-validation train data folder. You can also change the epoch count and the target models  ['efficientnet-b0' to 'efficientnet-b7', 'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l']

# Benchmarking

You can also do a benchmark with the study of Caetano dos Santos, F. L., Michalek, I. M., Laurila, K., Kaukinen, K., Hyttinen, J., and Lindfors, K. (2019). Automatic classification of iga endomysial antibody test for celiac disease: a new method deploying machine learning. Scientific reports, 9(1):9217. The code for it was produced and deployed into santos_code.
