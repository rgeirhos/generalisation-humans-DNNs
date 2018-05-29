"""
Script to copy all relevant .csv files, i.e. those with 
the correct suffix, to a new directory.
"""

import os
import shutil

def main():

    sourcedir = "../raw-data/fine-tuning/"
    targetdir = "../raw-data/training/"

    # these are the model identifiers of exactly
    # those models that are used in the experiments
    model_prefix_list = ["01v4", "26", "07", "08",
                           "11", "16", "09", "21", 
                           "03", "30", "31", "32",
                           "33", "34", "35", "36",
                           "22", "27", "19", "18"]
    for i, m in enumerate(model_prefix_list):
        model_prefix_list[i] = "sixteen"+m
    
    # add two additional prefixes
    model_prefix_list.append("specialised")
    model_prefix_list.append("all-noise")


    experiments = sorted(os.listdir(sourcedir))

    for e in experiments:
        target_expt_path = os.path.join(targetdir, e)
        if not os.path.exists(target_expt_path):
            os.makedirs(target_expt_path)

       
        csv_file_list = sorted(os.listdir(os.path.join(sourcedir, e)))

        for c in csv_file_list:
            for m in model_prefix_list:
                if m in c: # prefix contained in name 
                    source = os.path.join(sourcedir, e, c)
                    target = os.path.join(targetdir, e, c)
                    shutil.copyfile(source, target)
                    break
 

if __name__ == "__main__":
    main()
